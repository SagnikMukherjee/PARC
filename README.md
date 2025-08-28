# Premise-Augmented Reasoning Chains Improve Error Identification in Math Reasoning with LLMs

## 📝 Abstract

Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.


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
