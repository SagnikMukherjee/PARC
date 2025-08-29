import os
import json
import random
import argparse
from typing import List, Dict, Any, Union, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from openai import AzureOpenAI


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse the SciBench error annotated JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_premise_prompt(
        question: str, 
        solution: str, 
        step: str, 
        tokenizer: AutoTokenizer = None, 
        samples_fewshot: Optional[List[Dict[str, str]]] = None, 
        is_gpt: bool = False
    ) -> Union[str, List[Dict[str, str]]]:

    if (samples_fewshot is not None) and (len(samples_fewshot) > 0):
        fewshot_template = "\n\nYou can refer to the following example for guidance:"
        for i, sample in enumerate(samples_fewshot):
            fewshot_template += f"\n\nExample {i+1}:\n\n"
            fewshot_template += f"Question (Step 0):\n{sample['question']}\n\n"
            fewshot_template += f"Solution so far:\n{sample['context']}\n\n"
            fewshot_template += f"Next step to analyze:\n{sample['step']}\n\n"
            fewshot_template += f"In this example, you need to generate:\n{sample['justification']}"
    else:
        fewshot_template = ""

    """Zero/Few-shot prompt for premise extraction."""
    user_message = f"""You are provided with a question, a partial solution, and the next step in the solution. Your task is to identify the steps that serve as premises for the given next step.
A step qualifies as a premise if the next step directly relies on information from that step. Based on the identified premises, the correctness of the next step should be fully verifiable.

Question (Step 0):
{question}

Solution so far:
{solution}

Next step to analyze:
{step}

For the step above, identify which previous steps (including Step 0 - the question) are premises and explain why each one is necessary. Remember:
1. A step cannot be a premise to itself
2. The question (Step 0) can be a premise if used directly

Generate **ONLY** the premises and nothing else.
Format your response with one premise per line as:
Step X: [explanation of why this step is necessary for the current step]{fewshot_template}"""

    if is_gpt:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_message}
        ]
        return messages
    else:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_message}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)

def extract_steps(problem_data: Dict[str, Any]) -> List[int]:
    """Extract individual steps from the problem data."""
    if 'steps' in problem_data:
        return [step.strip() for step in problem_data['steps']]
    return []

def process_model_output(output: str) -> List[str]:
    """Process model output to extract premises."""
    # Split the output into lines and clean them
    lines = output.strip().split('\n')
    
    # Filter out empty lines and clean each line
    premises = [re.findall(r"Step \d+", line.strip()) for line in lines if line.strip()]
    # Remove empty lists
    premises = [p for p in premises if p]
    premises = [p[0] for p in premises]
    premises = [int(p.split()[1]) for p in premises]
    return premises

def get_ground_truth_premises(problem_data: Dict[str, Any], step_idx: int) -> List[int]:
    """Extract ground truth premises for a given step from premise annotation."""
    if 'premise_annotation' in problem_data and 'steps' in problem_data['premise_annotation']:
        steps = problem_data['premise_annotation']['steps']
        if 0 <= step_idx < len(steps):
            step_data = steps[step_idx+1]
            # Extract premise step numbers and remove duplicates
            premises = list(set(premise[0] for premise in step_data.get('premises', [])))
            return sorted(premises)  # Return sorted list of unique premise step numbers
    return []

def run_gpt_inference(messages: List[Dict[str, str]], model_name: str) -> str:
    """Run inference using Azure OpenAI GPT models."""
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
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

def ensure_output_directory(output_file: str) -> None:
    """Ensure the output directory exists, create it if it doesn't."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate premise generation using Llama model')
    parser.add_argument('--input-file', type=str, required=True,
                      help='Path to the SciBench error annotated JSON file')
    parser.add_argument('--output-file', type=str, required=True,
                      help='Path to save the evaluation results')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the model')
    parser.add_argument('--fewshot-path', type=str, help='Path to the few-shot data')
    parser.add_argument('--fewshot-num', type=int, default=0, help='Number of few-shot examples to use')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                      help='Number of GPUs to use for tensor parallelism')
    args = parser.parse_args()

    # Initialize model based on type
    print("Loading model...")
    is_gpt = "gpt" in args.model_path.lower()
    
    if not is_gpt:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=None
        )
        llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=8192)
    else:
        tokenizer = None
        llm = None

    # Load data
    print("Loading data...")
    problems = load_json_data(args.input_file)

    # Prepare few-shot data
    fewshot_samples = []
    if args.fewshot_num > 0:
        print("Loading few-shot data...")
        fewshot_data = load_json_data(args.fewshot_path)
        random.seed(42)
        fewshot_indice = random.sample(range(len(fewshot_data)), args.fewshot_num)
        for fs_idx in fewshot_indice:
            example_per_data = []
            for step in fewshot_data[fs_idx]['step_premises']:
                if step['generated_premises'] == step['ground_truth_premises']:
                    example_per_data.append({
                        'question': fewshot_data[fs_idx]['question'],
                        'step': step['step'],
                        'premises': step['generated_premises'],
                        'context': step['context'],
                        'justification': step['justification']
                    })
            fewshot_samples.append(random.choice(example_per_data))
        print(f"Selected {len(fewshot_samples)} few-shot examples...")

    results = []
    
    if is_gpt:
        # Process sequentially for GPT models
        for problem_idx, problem in enumerate(tqdm(problems)):
            steps = extract_steps(problem)
            step_results = []
            
            for step_idx, step in enumerate(steps):
                if step_idx == 0:
                    continue

                solution = "\n".join(steps[:step_idx])
                prompt = create_premise_prompt(
                    problem.get('question', ''), solution, step, tokenizer, fewshot_samples, is_gpt
                )
                
                if problem_idx == 0 and step_idx == 1:
                    print("\nFirst input sent to model:")
                    print(prompt[1]["content"])
                
                output_text = run_gpt_inference(prompt, args.model_path)
                premises = process_model_output(output_text)
                ground_truth_premises = get_ground_truth_premises(problem, step_idx)
                
                step_results.append({
                    'step': step,
                    'generated_premises': premises,
                    'ground_truth_premises': ground_truth_premises,
                    'context': solution,
                    'justification': output_text
                })
            
            results.append({
                'question': problem.get('question', ''),
                'step_premises': step_results
            })
    else:
        # Process in parallel for non-GPT models
        all_prompts = []
        prompt_metadata = []  # Store metadata to reconstruct results
        
        print("Preparing prompts for batch processing...")
        for problem_idx, problem in enumerate(problems):
            steps = extract_steps(problem)
            
            for step_idx, step in enumerate(steps):
                # if step_idx == 0:
                #     continue
                    
                solution = "\n".join(steps[:step_idx])
                prompt = create_premise_prompt(
                    problem.get('question', ''), solution, step, tokenizer, fewshot_samples, is_gpt
                )
                
                if problem_idx == 0 and step_idx == 1:
                    print("\nFirst input sent to model:")
                    print(prompt)
                
                all_prompts.append(prompt)
                prompt_metadata.append({
                    'problem_idx': problem_idx,
                    'step_idx': step_idx,
                    'step': step,
                    'solution': solution,
                    'problem': problem
                })
        
        print(f"Processing {len(all_prompts)} prompts in parallel...")
        outputs = llm.generate(all_prompts, sampling_params)
        
        # Initialize results structure
        results = [{
            'question': problem.get('question', ''),
            'steps': problem['steps'],
            # 'label': problem['label'],
            'premise_annotation': {"steps":[]}
        } for problem in problems]
        
        # Process outputs and reconstruct results
        for output, metadata in zip(outputs, prompt_metadata):
            output_text = output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
            premises = process_model_output(output_text)
            ground_truth_premises = get_ground_truth_premises(metadata['problem'], metadata['step_idx'])
            
            step_result = {
                # 'step_number': int(metadata['step'].split(":")[0].replace("step", "").strip()),
                'original_step': metadata['step'],
                'premises': [[premise, ""] for premise in premises],
                'ground_truth_premises': ground_truth_premises,
                'context': metadata['solution'],
                'justification': output_text
            }
            
            results[metadata['problem_idx']]['premise_annotation']['steps'].append(step_result)

    # Modify output file path if using few-shot
    if args.fewshot_num > 0:
        args.output_file = args.output_file.replace(".json", f"_{args.fewshot_num}shot.json")
    
    # Ensure output directory exists
    ensure_output_directory(args.output_file)
    
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()