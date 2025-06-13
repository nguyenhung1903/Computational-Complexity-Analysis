#!/bin/bash

MODELS=("Restormer" "PromptIR" "HAIR" "AdaIR" "QuaHaarIR")
SIZES=("128 128" "256 256" "512 512")

for model in "${MODELS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Running model: $model with input size: $size"
        python benchmark_vram_usage.py --model "$model" --input_size $size
        echo "--------------------------------------"
    done
done