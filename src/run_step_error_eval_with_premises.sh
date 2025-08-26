# #!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# Input and output files

INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives.json"
OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_qwen32b.json"

# Run the error classification script
python "step_error_eval_with_premises.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives_perturbed.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_perturbed_qwen32b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_negatives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_negatives_qwen32b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_qwen32b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives_perturbed.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_perturbed_qwen32b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_negatives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_negatives_qwen32b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4







# MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"

# # Input and output files

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_llama70b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives_perturbed.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_perturbed_llama70b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_negatives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_negatives_llama70b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_llama70b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives_perturbed.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_perturbed_llama70b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_negatives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_negatives_llama70b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4











# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# # Input and output files

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_llama8b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives_perturbed.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_perturbed_llama8b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_negatives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_negatives_llama8b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_llama8b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4


# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives_perturbed.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_perturbed_llama8b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_negatives.json"
# OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_negatives_llama8b.json"

# # Run the error classification script
# python "step_error_eval_with_premises.py" \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4




# # MODEL_PATH="gpt-4o-mini"

# # # Input and output files

# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_gpt4omini.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4


# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives_perturbed.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_perturbed_gpt4omini.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4

# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_negatives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_negatives_gpt4omini.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4


# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_gpt4omini.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4


# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives_perturbed.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_perturbed_gpt4omini.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4

# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_negatives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_negatives_gpt4omini.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4








# # MODEL_PATH="gpt-4o"

# # # Input and output files

# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_gpt4o.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4


# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_positives_perturbed.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_positives_perturbed_gpt4o.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4

# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/gsm8k/gsm8k_negatives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k/gsm8k_negatives_gpt4o.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4


# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_gpt4o.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4


# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_positives_perturbed.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_positives_perturbed_gpt4o.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4

# # INPUT_FILE="/home/sagnikm3/dag-llm/final_datasets/error/orca_math/orcamath_negatives.json"
# # OUTPUT_FILE="/home/sagnikm3/dag-llm/eval/error_eval_with_premises/orca_math/orcamath_negatives_gpt4o.json"

# # # Run the error classification script
# # python "step_error_eval_with_premises.py" \
# #     --input-file "$INPUT_FILE" \
# #     --output-file "$OUTPUT_FILE" \
# #     --model-path "$MODEL_PATH" \
# #     --tensor-parallel-size 4






