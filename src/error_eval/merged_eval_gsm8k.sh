MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"


INPUT_FILE="/home/sagnikm3/PARC/datasets/gsm8k/gsm8k_negatives.json"
OUTPUT_FILE="/home/sagnikm3/PARC/outputs/gsm8k_negatives_llama8b.json"

python merged_premise_error_eval.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/PARC/datasets/gsm8k/gsm8k_negatives_cleaned.json"
# OUTPUT_FILE="/home/sagnikm3/PARC/outputs/gsm8k_negatives_cleaned_llama8b.json"

# python merged_premise_error_eval.py \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/PARC/datasets/gsm8k/gsm8k_positives.json"
# OUTPUT_FILE="/home/sagnikm3/PARC/outputs/gsm8k_positives_llama8b.json"

# python merged_premise_error_eval.py \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

# INPUT_FILE="/home/sagnikm3/PARC/datasets/gsm8k/gsm8k_positives_cleaned.json"
# OUTPUT_FILE="/home/sagnikm3/PARC/outputs/gsm8k_positives_cleaned_llama8b.json"

# python merged_premise_error_eval.py \
#     --input-file "$INPUT_FILE" \
#     --output-file "$OUTPUT_FILE" \
#     --model-path "$MODEL_PATH" \
#     --tensor-parallel-size 4

