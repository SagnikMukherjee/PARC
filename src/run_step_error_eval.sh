#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"

# Input and output files

INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/metamathqa/metamathqa_positives.json"
OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_results/metamathqa/metamathqa_positives_qwen72b.json"

# Run the error classification script
python "step_error_eval.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 8


INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/metamathqa/metamathqa_positives_perturbed.json"
OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_results/metamathqa/metamathqa_positives_perturbed_qwen72b.json"

# Run the error classification script
python "step_error_eval.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 8

INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/metamathqa/metamathqa_negatives.json"
OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_results/metamathqa/metamathqa_negatives_qwen72b.json"

# Run the error classification script
python "step_error_eval.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 8
