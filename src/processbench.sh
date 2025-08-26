#!/bin/bash

INPUT_FILE_1=/home/sagnikm3/PARC/datasets/process_bench/math.json
INPUT_FILE_2=/home/sagnikm3/PARC/datasets/process_bench/gsm8k.json

python processbench.py \
    --input-file $INPUT_FILE_1 \
    --output-file /home/sagnikm3/dag-llm-temp/dag-llm/eval/merged_eval/math_qwen7b.json \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --batch-size 1 \
    --use-local-vllm \
    --tensor-parallel-size 4

python processbench.py \
    --input-file $INPUT_FILE_2 \
    --output-file /home/sagnikm3/dag-llm-temp/dag-llm/eval/merged_eval/gsm8k_qwen7b.json \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --batch-size 1 \
    --use-local-vllm \
    --tensor-parallel-size 4

