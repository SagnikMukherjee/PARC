# Premise-Augmented Reasoning Chains Improve Error Identification in Math Reasoning with LLMs

## 📝 Abstract

Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.

## Dataset Format

Each entry in the datasets (see the datasets folder) follows this schema:

- **question**: The math word problem posed to the model.  
- **ground_truth_solution**: Step-by-step human-verified solution.  
- **ground_truth_answer**: Final numeric/string answer from the ground truth solution.  
- **model_answer**: The model’s predicted final answer.  
- **steps**: The sequence of reasoning steps produced by the model.  
- **is_correct**: Boolean indicating whether the model’s final answer matches the ground truth.  
- **premise_annotation**: Mapping of each model step to its underlying premises.  
- **error_annotation**: Labeled errors in the model’s reasoning, including error types and descriptions at the step level.  

### Example Entry

```json
{
  "question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?",
  "ground_truth_solution": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000",
  "ground_truth_answer": "70000",
  "model_answer": "78000",
  "steps": [
    "Step 1: Calculate the total amount Josh spent on the house and repairs.\nTotal amount spent = Cost of the house + Cost of repairs\nTotal amount spent = $80,000 + $50,000\nTotal amount spent = $130,000",
    "Step 2: The value of the house increased by 150% after the repairs. This means the new value of the house is 250% of the original value (100% + 150% increase).\nLet's denote the original value of the house as x. Then, the new value of the house is 2.5x (since 250% is the same as 2.5 times the original value).",
    "Step 3: We know that the new value of the house is $130,000 (the total amount spent on the house and repairs). We can set up an equation to represent this:\n2.5x = $130,000",
    "Step 4: Solve for x, the original value of the house.\nx = $130,000 / 2.5\nx = $52,000",
    "Step 5: Now that we know the original value of the house, we can find the profit Josh made.\nProfit = New value of the house - Original value of the house\nProfit = $130,000 - $52,000\nProfit = $78,000\n\nTherefore, Josh made a profit of $78,000."
  ],
  "is_correct": false,
  "premise_annotation": { ... },
  "error_annotation": { ... }
}
```

## 🔧 Installation
```bash
git clone https://github.com/SagnikMukherjee/PARC.git
cd PARC
conda create -n parc
pip install -r requirements.txt
```

## Run Evaluations
For OpenAI models, we run inference with AzureOpenAI. In that case set the following environment variables - AZURE_ENDPOINT and AZURE_OPENAI_KEY. 
### Premise Mapping

```bash
cd src/premise_eval
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"


export INPUT_FILE="<path to PARC folder>/datasets/gsm8k/gsm8k_negatives.json"
export OUTPUT_FILE="<path to PARC folder>/outputs/gsm8k_negatives_llama8b.json"

python merged_premise_error_eval.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4
```

### Baseline error evaluation
```bash
cd src/baseline
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# Input and output files

export INPUT_FILE="<path to PARC folder>/datasets/gsm8k/gsm8k_negatives.json"
export OUTPUT_FILE="<path to PARC folder>/outputs/gsm8k_negatives_llama8b.json"

# Run the error classification script
python "step_error_eval.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4
```

### Error Eval with Premises

```bash
cd src/error_eval
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"


export INPUT_FILE="<path to PARC folder>/datasets/gsm8k/gsm8k_negatives.json"
export OUTPUT_FILE="<path to PARC folder>/outputs/gsm8k_negatives_llama8b.json"

python merged_premise_error_eval.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 4
```

## 📄 Citation

```bibtex
@misc{mukherjee2025premiseaugmentedreasoningchainsimprove,
      title={Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs}, 
      author={Sagnik Mukherjee and Abhinav Chinta and Takyoung Kim and Tarun Anoop Sharma and Dilek Hakkani-Tür},
      year={2025},
      eprint={2502.02362},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02362}, 
}
```

## 📧 Contact

For questions or support, please contact:
- Sagnik Mukherjee: sagnikm3@illinois.edu
- Abhinav Chinta: achinta3@illinois.edu

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
