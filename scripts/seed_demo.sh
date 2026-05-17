#!/usr/bin/env bash
# seed_demo.sh — End-to-end demo pipeline
# Usage: bash scripts/seed_demo.sh [--rate N]
#
# Runs: generate -> train -> serve -> replay -> evaluate
# Estimated time: ~10 minutes on a modern laptop (CPU only)

set -euo pipefail

RATE="50"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rate) RATE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=============================================="
echo "  AML Intelligence Platform - Demo Pipeline"
echo "=============================================="
echo ""

# Step 1: Clean previous data
echo "[1/5] Cleaning previous data..."
rm -rf data/raw/*.csv data/artifacts/* data/evaluation/* 2>/dev/null || true
mkdir -p data/raw data/artifacts data/evaluation logs

# Step 2: Generate synthetic data
echo "[2/5] Generating synthetic AML dataset..."
python -m scripts.run_generator --config config/generator_config.json
echo "      OK - Generated historical + stream CSVs"

# Step 3: Train offline models
echo "[3/5] Training offline models (L3 + L4)..."
python -m scripts.run_offline_pipeline
echo "      OK - Behavioral models + GNN artifacts written"

# Step 4: Start API server
echo "[4/5] Starting FastAPI on port 8000..."
uvicorn system2_platform.api.main:app --port 8000 > logs/api.log 2>&1 &
API_PID=$!
echo "      PID $API_PID - waiting for startup..."

for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "      OK - API ready"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "      FAIL - API failed to start (check logs/api.log)"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Step 5: Replay stream + evaluate
echo "[5/5] Replaying stream at ${RATE} tx/sec..."
python scripts/replay_stream.py \
    --csv   data/raw/stream_transactions.csv \
    --out   data/evaluation/replay_scores.jsonl \
    --alerts-csv data/evaluation/generated_alerts.csv \
    --rate  "$RATE"
echo "      OK - Stream replay complete"

echo ""
echo "Running blind evaluation..."
python -m scripts.run_evaluator \
    --scores data/evaluation/replay_scores.jsonl \
    --alerts data/evaluation/generated_alerts.csv \
    --truth  data/raw/hidden_ground_truth.csv \
    --out-dir data/evaluation

echo ""
echo "=============================================="
echo "  Demo complete. Results:"
echo "  data/evaluation/evaluation_report.json"
echo "=============================================="

kill $API_PID 2>/dev/null || true
