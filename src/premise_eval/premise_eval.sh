#!/bin/bash


# ===============
# ===============
# ===============

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"


INPUT_FILE="/home/sagnikm3/PARC/datasets/math/math_positives.json"
OUTPUT_FILE="/home/sagnikm3/PARC/outputs/math_positives_qwen7b.json"

python premise_eval.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4

INPUT_FILE="/home/sagnikm3/PARC/datasets/math/math_negatives.json"
OUTPUT_FILE="/home/sagnikm3/PARC/outputs/math_negatives_qwen7b.json"

python premise_eval.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4

INPUT_FILE="/home/sagnikm3/PARC/datasets/math/math_positives_perturbed.json"
OUTPUT_FILE="/home/sagnikm3/PARC/outputs/math_positives_perturbed_qwen7b.json"

python premise_eval.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4

