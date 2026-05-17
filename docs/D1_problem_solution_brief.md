# D1 — Problem + Solution Brief
**Team Bazooka | PS3: Fund Flow Tracking | iDEA 2.0 Hackathon**

---

## 1. The Problem

Money laundering costs the global economy an estimated **$800 billion – $2 trillion per year** (UN Office on Drugs and Crime). Banks file Suspicious Activity Reports (SARs) reactively — after a compliance officer manually reviews flagged transactions days or weeks after the fact. Three structural failures make this hard:

1. **Rule-based alerting produces 90–95% false positives.** Threshold rules (e.g., "flag any transfer > ₹50 lakh") generate noise that buries real threats and burns analyst capacity.
2. **Transaction-level views miss network-level fraud.** Money mules, smurfing cells, and circular layering schemes span dozens of accounts. No single transaction looks anomalous; the pattern only emerges across the graph.
3. **Monitoring is batch, not real-time.** Traditional AML systems run nightly batch jobs. Launderers exploit the 12–24 hour blind window to move funds through multiple hops before detection.

**PS3 specifically asks:** Can a team build a fund-flow tracking system that detects AML typologies in real time, explains its decisions, and is demonstrably better than existing baselines?

---

## 2. Our Solution

**AML Intelligence Platform** is a three-system, four-layer ML pipeline that scores every incoming transaction in **< 200 ms** with an explainable risk score.

### Three Systems

| System | Purpose |
|--------|---------|
| **System 1 — Generator** | Synthesises a realistic AML ecosystem: 5,000 accounts, 500K transactions, 10 typologies (structuring, circular laundering, layering chains, fan-in/fan-out, fraud rings, dormant activation, velocity bursts, cross-border layering, round-tripping). Produces `historical_transactions.csv`, `stream_transactions.csv`, and a sealed `hidden_ground_truth.csv` for blind evaluation. |
| **System 2 — Platform** | The real-time detection engine. Four scoring layers (see below) fused by a weighted combiner. FastAPI backend + WebSocket streaming + Next.js dashboard. |
| **System 3 — Evaluator** | Blind evaluation against ground truth: AUROC, PR-AUC, per-typology detection coverage, ring detection rate, and benchmark comparison. |

### Four Detection Layers

```
Transaction event
      │
      ▼
Layer 1 — Rule Engine (13 FATF rules)
      │   Deterministic, < 1 ms, zero false negatives on hard rules
      ▼
Layer 2 — Graph Analytics (NetworkX + community detection)
      │   Cycle detection, PageRank, community risk aggregation
      ▼
Layer 3 — Behavioral Anomaly (IsolationForest + LOF + AutoEncoder)
      │   45-dimensional feature vector, per-account baseline deviation
      ▼
Layer 4 — GNN (TGN + MegaGNN, 551K historical edges)
      │   Temporal Graph Network memory + bidirectional aggregation
      ▼
Risk Fusion (weights: rules 0.35, behavioral 0.30, GNN 0.25, graph 0.10)
      │
      ▼
Alert Manager → Dashboard → Analyst
```

### Key Claims

- **AUROC > 0.92** on the synthetic blind test set (verified; see `data/evaluation/evaluation_report.json`).
- **Real-time scoring < 200 ms** per transaction on a single CPU server.
- **10 AML typologies** covered with per-typology detection breakdown.
- **Explainable alerts**: every alert includes triggered rules, top anomaly drivers by feature name, GNN anomaly score, and graph structural evidence (cycles, community risk).
- **Zero new data sharing**: entire dataset is synthetic, generator is open-source, no real customer data anywhere in the pipeline.

---

## 3. Why Our Approach Beats the Baseline

| Metric | Tide (Low-Intensity) | MEGA-GNN (literature) | **Ours** |
|--------|---------------------|-----------------------|---------|
| AUROC  | 0.801 | 0.951 | **≥ 0.92** |
| F1     | 0.574 | 0.809 | **≥ 0.78** |
| Latency | batch (hours) | batch | **< 200 ms** |
| Explainability | none | embedding similarity | **rule + feature + graph** |

The key architectural advantage is **temporal graph memory**: TGN maintains a 64-dimensional memory vector per node updated at each transaction, allowing the model to recognise returning mule accounts even after long dormancy periods — something static GNNs and rule engines fundamentally cannot do.

---

## 4. Limitations & Honesty

- Data is **fully synthetic**. Real-world performance will differ (concept drift, missing features, reporting bias).
- The demo runs on a **single machine** (no distributed runtime, no Kafka). For a 50-account POC the latency target holds; at 50M accounts, the graph layer would need sharding.
- **No privacy-enhancing technologies** (federated learning, differential privacy). These are out of scope for this POC.
- No authentication on the dashboard — this is a hackathon demo, not a production deployment.

---

*For technical architecture, see `docs/D3_technical_architecture.md`. For reproduction steps, see `README.md`.*
