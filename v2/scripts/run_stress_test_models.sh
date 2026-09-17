#!/bin/bash

# 包括的ストレステスト用のバッチスクリプト
# 使用例: bash v2/scripts/run_stress_test_models.sh

MODELS=("Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-3B")
INSTRUCT_MODELS=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct")

echo "Starting Stress Test for generalization across model sizes..."

for i in "${!MODELS[@]}"; do
    BASE_MODEL="${MODELS[$i]}"
    INST_MODEL="${INSTRUCT_MODELS[$i]}"
    
    echo "================================================="
    echo " Evaluating pair: $BASE_MODEL & $INST_MODEL"
    echo "================================================="
    
    # 1. 抽出と尤度計算
    # python v2/scripts/run_extract_and_likelihood.py --model "$BASE_MODEL"
    # python v2/scripts/run_extract_and_likelihood.py --model "$INST_MODEL" --is_instruct
    
    # 2. Steering 検証
    # python v2/scripts/run_steering_and_likelihood.py --model "$BASE_MODEL" --layers 20 24 27 --alphas -3.0 0.0 3.0
    # python v2/scripts/run_steering_and_likelihood.py --model "$INST_MODEL" --is_instruct --layers 20 24 27 --alphas -3.0 0.0 3.0
    
    echo "Done with $BASE_MODEL pair."
done

echo "All stress tests completed."
