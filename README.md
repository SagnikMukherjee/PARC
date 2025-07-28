# Premise-Augmented Reasoning Chains Improve Error Identification in Math Reasoning with LLMs

## 📝 Abstract

Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.

## 🏆 Key Contributions

- **Novel Framework**: Transform Linear Reasoning Chains (LRC) into Premise-Augmented Reasoning Chains (PARC) by identifying premises for each reasoning step
- **Enhanced Error Detection**: Improve error identification accuracy by 6% to 16% absolute when step-by-step verification is carried out under premises
- **Refined Error Taxonomy**: Introduce "accumulation errors" - steps that are locally correct but inherit upstream errors
- **PERL Dataset**: A comprehensive dataset of reasoning chains annotated with premises and error types

## 🔧 Installation

### Prerequisites

```bash
# Create a virtual environment with uv
uv venv parc_env
source parc_env/bin/activate  # On Windows: parc_env\Scripts\activate

# Install required packages
uv pip install torch transformers loguru tqdm python-dotenv
```

### For Local vLLM Usage (Optional)
```bash
uv pip install vllm
```

### For Azure/OpenAI Usage
```bash
uv pip install openai
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:
```bash
API_KEY=your_openai_or_azure_api_key_here
```

### 2. Prepare Your Data

Your input JSON file should contain mathematical reasoning problems in the following format:

```json
[
  {
    "question": "Solve for x: 2x + 5 = 15",
    "steps": [
      "Subtract 5 from both sides: 2x + 5 - 5 = 15 - 5",
      "Simplify: 2x = 10", 
      "Divide both sides by 2: x = 5"
    ]
  }
]
```

### 3. Run ProcessBench Component

This script runs the ProcessBench component for premise identification and error annotation on mathematical reasoning problems.

#### Using OpenAI/Azure Models:
```bash
python src/processbench.py \
  --input-file data/problems.json \
  --output-file results/annotated_problems.json \
  --model-name gpt-4o \
  --batch-size 32
```

#### Using Local vLLM Models:
```bash
python src/processbench.py \
  --input-file data/problems.json \
  --output-file results/annotated_problems.json \
  --model-name /path/to/your/local/model \
  --use-local-vllm \
  --tensor-parallel-size 4 \
  --batch-size 16
```

### 4. Monitor Progress

The ProcessBench script provides detailed logging including:
- Progress bars for batch processing
- Token usage statistics (for API-based models)
- Error handling for failed inference calls
- Final cost estimates for API usage


## 📋 ProcessBench Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input-file` | str | ✅ | Path to the input JSON file containing problems |
| `--output-file` | str | ✅ | Path to save the annotated results |
| `--model-name` | str | ✅ | Model name (e.g., 'gpt-4o') or local model path |
| `--use-local-vllm` | flag | ❌ | Use local vLLM instead of Azure/OpenAI |
| `--batch-size` | int | ❌ | Batch size for parallel inference (default: 128) |
| `--tensor-parallel-size` | int | ❌ | Number of GPUs for tensor parallelism (default: 4) |

## 📊 ProcessBench Output Format

The ProcessBench script generates annotated data with the following structure:

```json
{
  "question": "Original problem statement",
  "steps": ["List of solution steps"],
  "premise_annotation": {
    "steps": [
      {
        "step_number": 1,
        "original_step": "Step text",
        "premises": [[0, "Question premise"], [1, "Previous step"]],
        "conclusion": "What this step concludes",
        "reasoning": "How premises lead to conclusion"
      }
    ]
  },
  "error_annotation": {
    "step_annotations": [
      {
        "step_index": 1,
        "standalone_verdict": "correct",
        "contextual_verdict": "correct",
        "standalone_explanation": "Mathematical analysis",
        "contextual_explanation": "Logical consistency analysis"
      }
    ]
  },
  "predicted_error_step_indices": [2, 4]
}
```

## 🔍 Error Types

The framework identifies three types of errors:

1. **Mathematical Error**: Incorrect calculations or formula applications
2. **Logical Inconsistency**: Steps that violate logical principles or make unjustified conclusions  
3. **Accumulation Error**: Steps that are locally correct but built upon erroneous premises

## 📈 Performance

Our experiments show that:
- LLMs achieve **90%+ recall** in premise identification
- Error identification accuracy improves by **6-16%** absolute when using PARC
- The framework works effectively across multiple model scales and types

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.


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