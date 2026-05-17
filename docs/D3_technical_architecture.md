# D3 — Technical Architecture
**AML Intelligence Platform | Team Bazooka | PS3**

---

## 1. System Overview

The platform is composed of three independent systems that communicate only via CSV files on disk. No system imports from another. This separation enables independent development, testing, and deployment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SYSTEM 1 — AML ECOSYSTEM GENERATOR                                         │
│                                                                             │
│  G1 AccountGen → G2 TransactionGen → G3 TypologyInjector → G4 GroundTruth  │
│                                                                             │
│  Outputs:  data/raw/historical_transactions.csv  (500K rows)                │
│            data/raw/stream_transactions.csv      (50K rows, time-ordered)   │
│            data/raw/hidden_ground_truth.csv      (SEALED — System 3 only)  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ CSV
┌─────────────────────────────────────▼───────────────────────────────────────┐
│  SYSTEM 2 — REAL-TIME DETECTION PLATFORM                                    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  OFFLINE PIPELINE  (runs once; artifacts written to data/artifacts/) │   │
│  │                                                                      │   │
│  │  S1 FeatureEng → L3A BehavioralTrainer (IF+LOF+AE)                  │   │
│  │  S2 GraphBuilder → L4A GNNTrainer (TGN + MegaGNN, 15 epochs)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ONLINE PIPELINE  (FastAPI; loads artifacts at startup)              │   │
│  │                                                                      │   │
│  │  POST /stream/transaction                                            │   │
│  │       │                                                              │   │
│  │       ├─ S4 StreamIngestion  (parse + validate TransactionEvent)     │   │
│  │       ├─ S5 FeatureUpdate    (compute LiveFeatureVector, then ingest) │   │
│  │       ├─ S6 GraphUpdate      (seed graph + BFS cycle detection)      │   │
│  │       │                                                              │   │
│  │       ├─ L1 RuleEngine       (13 FATF rules → RuleOutput)           │   │
│  │       ├─ L2B GraphAnalytics  (PageRank, community risk)              │   │
│  │       ├─ L3B BehavioralInfer (ISO+LOF+AE → BehavioralOutput)        │   │
│  │       ├─ L4B GNNInference    (TGN memory + MegaGNN → GNNOutput)     │   │
│  │       │                                                              │   │
│  │       ├─ P1 RiskFusion       (weighted combiner → FusedRiskOutput)  │   │
│  │       └─ P2 AlertManager     (threshold 31.0 → Alert or None)       │   │
│  │                                                                      │   │
│  │  WebSocket /stream/ws  →  Live event broadcast to dashboard          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ generated_alerts.csv
┌─────────────────────────────────────▼───────────────────────────────────────┐
│  SYSTEM 3 — BLIND EVALUATOR                                                 │
│                                                                             │
│  E1 Joiner → E2 TxMetrics → E3 CommunityEval → E4 PatternCoverage          │
│           → E5 ReportGenerator → data/evaluation/evaluation_report.json     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer-by-Layer Technical Design

### Layer 1 — Deterministic Rule Engine (`system2_platform/layer1_rules/`)

13 FATF-aligned rules implemented as deterministic Python functions over `LiveFeatureVector`:

| Rule | Description |
|------|-------------|
| R01 | CTR threshold (≥ ₹5 lakh equivalent) |
| R02 | Structuring (9 sub-threshold transactions in 24h) |
| R03 | Rapid movement (3+ hops within 2h) |
| R04 | Shell entity (no KYC + international + high amount) |
| R05 | Dormant activation (365+ days inactive, large first tx) |
| R06 | Velocity burst (≥5 tx in 1h) |
| R07 | High-risk jurisdiction with elevated amount |
| R08 | Round number clustering |
| R09 | Fan-in concentration (≥5 senders in 24h) |
| R10 | Fan-out distribution (≥5 receivers in 24h) |
| R11 | New account sudden large transfer |
| R12 | Cross-border amount mismatch |
| R13 | Failed transaction ratio anomaly |

Each rule returns a boolean + base score contribution. Any triggered rule sets `rule_triggered=True`.

### Layer 2 — Graph Analytics (`system2_platform/layer2_graph/`)

- **Offline (L2A):** Builds a `networkx.MultiDiGraph` from historical transactions; computes PageRank, betweenness centrality, community partitions (Louvain/Infomap).
- **Online (L2B):** Lightweight BFS cycle detection up to 3 hops; community risk score from pre-computed partition; returns `LiveGraphFeatureVector`.

### Layer 3 — Behavioral Anomaly Detection (`system2_platform/layer3_behavioral/`)

- **Offline (L3A):** Trains on 45-feature vectors derived from historical transactions per account. Three models:
  - `IsolationForest` (contamination=0.05) — global anomaly isolation
  - `LocalOutlierFactor` (k=20, novelty=True) — density-based outlier
  - `AutoEncoder` (128→64→32→64→128, MSE threshold at 95th percentile)
- **Online (L3B):** Scores incoming feature vectors; normalises to 0–100 anomaly scale. Top-3 feature contributors extracted by SHAP-like perturbation for explainability.

### Layer 4 — GNN (TGN + MegaGNN) (`system2_platform/layer4_gnn/`)

- **Model:** Temporal Graph Network (TGN) with 64-dimensional node memory + MegaGNN bidirectional aggregation head.
  - TGN MEMORY_DIM = 64; MESSAGE_DIM = 64; TIME_DIM = 64
  - MegaGNN: INPUT = 208 (64 + 64 + 16 structural), OUTPUT = 64, 3 SAGEConv layers
- **Training (L4A):** Self-supervised link prediction. Positive edges = real transactions; negative edges sampled at 1:3 ratio. BCE loss, Adam, 15 epochs, 200K edge subsample (evenly-spaced to preserve temporal order). Training time ~5 min on CPU.
- **Online (L4B):** Dot-product similarity of sender/receiver embeddings → sigmoid → link probability. `gnn_anomaly_score = (1 - link_score) × 100`. Falls back to 50.0 for unknown nodes.

**Critical implementation detail:** `TGN._set_memory()` uses `.clone()` on each memory tensor to prevent pickle serialisation bloat (shared base storage would cause a 421 MB pickle instead of 901 KB).

### Risk Fusion (`system2_platform/post_detection/`)

```
fused_score = 0.35 × rule_score
            + 0.30 × behavioral_score
            + 0.25 × gnn_anomaly_score
            + 0.10 × graph_score
            + 20.0  (if cycle detected in transaction graph)
```

Alert threshold: **31.0** → Medium risk. Alerts stored in-memory with `threading.Lock`; community-level aggregation tracks highest scorer per cluster.

---

## 3. Data Flow & Contracts

All cross-module types defined in `system2_platform/contracts/`:

| Contract | Contents |
|----------|----------|
| `TransactionEvent` | Pydantic model for incoming stream transaction |
| `LiveFeatureVector` | 45 float features + metadata |
| `LiveGraphFeatureVector` | PageRank, betweenness, community_id, community_risk, cycle_detected |
| `RuleOutput` | triggered_rules list, rule_score, rule_triggered bool |
| `BehavioralOutput` | iso_score, lof_score, ae_score, overall, top_drivers |
| `GNNInferenceOutput` | link_score, gnn_anomaly_score, embedding dims |
| `FusedRiskOutput` | fused_score, risk_level, component breakdown |
| `Alert` | Alert ID, timestamp, all component outputs, alert_type |

---

## 4. Frontend Architecture (`frontend/`)

- **Framework:** Next.js 14 App Router (TypeScript)
- **State:** TanStack Query v5 for REST, custom `AMLWebSocket` hook for streaming
- **Visualisation:** Recharts 2 (time-series, bar), Cytoscape 3.28 (graph explorer)
- **Pages:** `/` (live dashboard), `/alerts` (queue), `/alerts/[id]` (detail), `/analytics` (communities), `/graph` (explorer), `/upload` (CSV replay)

---

## 5. Technology Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Python runtime | CPython | 3.11 |
| ML/DL | PyTorch | 2.1+ |
| Graph DL | PyG (torch-geometric) | 2.4+ |
| Graph analytics | NetworkX | 3.2+ |
| Classical ML | scikit-learn, XGBoost | 1.3+, 2.0+ |
| API framework | FastAPI + uvicorn | 0.110+, 0.27+ |
| Frontend | Next.js | 14.2 |
| Package mgmt | uv | latest |
| Data format | Parquet (internal), CSV (cross-system) | — |

---

*System diagram: `docs/architecture_diagram.svg`*
