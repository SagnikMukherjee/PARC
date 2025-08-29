import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from openai import AzureOpenAI
from utils_ref import *

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse the JSON file containing solution steps."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_error_classification_prompt(question: str, solution: str, step: str, tokenizer: AutoTokenizer) -> str:
    """Create a zero-shot prompt for evaluating step correctness."""
    system_prompt = """You are a helpful AI assistant that analyzes mathematical solution steps. 
    Your task is to determine if each statement is COMPLETELY correct by carefully analyzing its validity.
    Focus ONLY on whether the current step is valid - do not consider whether it helps reach the final answer or whether better steps could have been taken.
    Mark a statement as CORRECT unless you find a specific error - if you're unsure, lean towards marking it as CORRECT.
    """
    
    user_message = f"""Question:
{question}

Solution so far:
{solution}


### 1. "Logical_Inconsistency"
- **Definition**: Steps that violate logical principles or make unjustified conclusions
- **Examples**:
  - False equivalences
  - Invalid deductions
  - Unsupported assumptions
  - Note that incorrect use of previous information (example the step uses a wrong value of a variable) is a Logical_Inconsistency
- **Impact**: Breaks the logical flow of the solution

### 2. "Mathematical_Error"
- **Definition**: Incorrect calculations, misuse of formulas, or mathematical operations
- **Examples**:
  - Arithmetic mistakes
  - Incorrect algebraic manipulations
  - Wrong formula application
  - Note that Mathematical_Error can only appear when there is an error in calculation
- **Impact**: Produces incorrect numerical or algebraic results

### 3. "Accumulation_Error"
- **Definition**: Errors that propagate from previous incorrect steps
- **Examples**:
  - Using wrong intermediate results
  - Building upon previously miscalculated values
- **Impact**: Compounds previous mistakes into larger errors

### 4. "Other"
- **Definition**: Any error that doesn't fit into the above categories
- **Examples**:
  - Notation mistakes
  - Unclear explanations
  - Formatting issues
- **Impact**: Varies depending on the specific error

Statement to analyze:
{step}

Format your response as:
Reasoning: [detailed analysis of the statement's validity]
Verdict: [CORRECT,  Mathematical_Error, Logical_Inconsistency, or Accumulation_Error]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
    return prompt

def extract_steps(problem_data: Dict[str, Any]) -> List[str]:
    """Extract individual steps from the problem data."""
    steps = []
    if 'steps' in problem_data:
        steps = problem_data['steps']
    return steps

def process_model_output(output: str) -> Dict[str, str]:
    """Process model output to extract error classification and explanation."""
    # Convert to lowercase for field matching
    output_lower = output.lower()
    lines = output_lower.strip().split('\n')
    
    result = {
        'verdict': 'Unknown',
        'reasoning': '',
        'explanation': output  # Keep the entire original output
    }
    
    # Find reasoning and verdict
    for i, line in enumerate(lines):
        if line.startswith('reasoning:'):
            # Collect all lines until we hit 'verdict:'
            reasoning_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith('verdict:'):
                if lines[j].strip():  # Skip empty lines
                    reasoning_lines.append(lines[j].strip())
                j += 1
            result['reasoning'] = ' '.join(reasoning_lines)
        elif line.startswith('verdict:'):
            result['verdict'] = line.replace('verdict:', '').strip().upper()
    
    return result

def run_gpt_inference(messages: List[Dict[str, str]], model_name: str) -> str:
    """Run inference using Azure OpenAI GPT models."""
    client = AzureOpenAI(
        api_key="",
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_ENDPOINT")
    )
    
    deployment_name = model_name
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=0.0
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='Evaluate step-level errors using LLM')
    parser.add_argument('--input-file', type=str, required=True,
                      help='Path to the JSON file containing solution steps')
    parser.add_argument('--output-file', type=str, required=True,
                      help='Path to save the error classification results')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the LLM model')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                      help='Number of GPUs to use for tensor parallelism')
    args = parser.parse_args()

    # Initialize model based on type
    print("Loading model...")
    is_gpt = "gpt" in args.model_path.lower()
    
    if not is_gpt:
        # Initialize the tokenizer and model for non-GPT models
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        sampling_params = SamplingParams(
            temperature=0.0,  # Use greedy decoding
            max_tokens=512,
            stop=None
        )
        llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    else:
        tokenizer = None
        llm = None

    # Load data
    print("Loading data...")
    data = load_json_data(args.input_file)
    
    # Process each problem
    results = []
    for problem in tqdm(data, desc="Processing problems"):
        question = problem.get('question', '')
        steps = extract_steps(problem)
        
        step_results = []
        for step_idx,step in enumerate(steps):
            solution = '\n'.join(steps[:step_idx])

            if is_gpt:
                # Create messages for GPT model
                messages = [
                    {"role": "system", "content": """You are a helpful AI assistant that analyzes mathematical solution steps. 
                    Your task is to determine if each statement is COMPLETELY correct by carefully analyzing its validity.
                    Focus ONLY on whether the current step is valid - do not consider whether it helps reach the final answer or whether better steps could have been taken.
                    Mark a statement as CORRECT unless you find a specific error - if you're unsure, lean towards marking it as CORRECT."""},
                    {"role": "user", "content": f"""Question:
                    {question}

                    Solution so far:
                    {solution}


                    ### 1. "Logical_Inconsistency"
                    - **Definition**: Steps that violate logical principles or make unjustified conclusions
                    - **Examples**:
                    - False equivalences
                    - Invalid deductions
                    - Unsupported assumptions
                    - Note that incorrect use of previous information (example the step uses a wrong value of a variable) is a Logical_Inconsistency
                    - **Impact**: Breaks the logical flow of the solution

                    ### 2. "Mathematical_Error"
                    - **Definition**: Incorrect calculations, misuse of formulas, or mathematical operations
                    - **Examples**:
                    - Arithmetic mistakes
                    - Incorrect algebraic manipulations
                    - Wrong formula application
                    - Note that Mathematical_Error can only appear when there is an error in calculation
                    - **Impact**: Produces incorrect numerical or algebraic results

                    ### 3. "Accumulation_Error"
                    - **Definition**: Errors that propagate from previous incorrect steps
                    - **Examples**:
                    - Using wrong intermediate results
                    - Building upon previously miscalculated values
                    - **Impact**: Compounds previous mistakes into larger errors

                    ### 4. "Other"
                    - **Definition**: Any error that doesn't fit into the above categories
                    - **Examples**:
                    - Notation mistakes
                    - Unclear explanations
                    - Formatting issues
                    - **Impact**: Varies depending on the specific error

                    Statement to analyze:
                    {step}

                    Format your response as:
                    Reasoning: [detailed analysis of the statement's validity]
                    Verdict: [CORRECT,  Mathematical_Error, Logical_Inconsistency, or Accumulation_Error]"""}
                ]
                output = run_gpt_inference(messages, args.model_path)
            else:
                # Use local model
                prompt = create_error_classification_prompt(question, solution, step, tokenizer)
                output = llm.generate([prompt], sampling_params, use_tqdm=False)[0].outputs[0].text

            result = process_model_output(output)

            if (len(problem['error_annotation']['error_annotations']['step_level'][step_idx]['errors'])==0):
                ground_truth="correct"
            else:
                errors = problem['error_annotation']['error_annotations']['step_level'][step_idx]['errors']
                errors = [error['error_type'] for error in errors]
                ground_truth = errors

            step_results.append({
                'step': step,
                'classification': result,
                'ground_truth': ground_truth
            })
        
        results.append({
            'question': question,
            'solution': solution,
            'step_evaluations': step_results
        })
    
    # Save results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
