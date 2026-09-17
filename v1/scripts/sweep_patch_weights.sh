#!/bin/bash
# 介入強度 (Patch Weight) を下げた全層スイープ実行用スクリプト

source .venv/bin/activate

RUN_ID="20260825T023809Z_a1078cb_cb9ad1b8"
LAYERS=(0 4 8 12 16 20 24 27)
WEIGHTS=(0.5 0.1)

for weight in "${WEIGHTS[@]}"; do
    echo "========================================================="
    echo "Starting Sweep with Patch Weight: $weight"
    echo "========================================================="
    for layer in "${LAYERS[@]}"; do
        echo "Running patching for layer $layer with weight $weight..."
        python scripts/run_causal_intervention.py \
            --run-id $RUN_ID \
            --limit 0 \
            --target-layer $layer \
            --patch-weight $weight
    done
done

echo "All sweeping tasks completed."
