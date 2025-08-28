import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import random
import argparse
from typing import List, Dict, Any, Union, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from openai import AzureOpenAI
from utils_ref import *
from utils.inference_engine import InferenceEngine, Hyperparameters, get_total_usage
from dotenv import load_dotenv

load_dotenv()

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_premise_prompt(
        question: str, 
        solution: str, 
        step: str, 
        tokenizer: AutoTokenizer = None, 
        samples_fewshot: Optional[List[Dict[str, str]]] = None, 
        is_gpt: bool = False
    ) -> Union[str, List[Dict[str, str]]]:
    """Zero/Few-shot prompt for premise extraction."""
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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def process_model_output_premises(output: str) -> List[str]:
    """Process model output to extract premises."""
    lines = output.strip().split('\n')
    premises = [re.findall(r"Step \d+", line.strip()) for line in lines if line.strip()]
    premises = [p for p in premises if p]
    premises = [p[0] for p in premises]
    premises = [int(p.split()[1]) for p in premises]
    return premises

def process_model_output_errors(output: str) -> Dict[str, str]:
    """Process model output to extract error classification and explanation."""
    if output is None:
        return {'verdict': 'Unknown', 'reasoning': '', 'explanation': ''}
    output_lower = output.lower()
    lines = output_lower.strip().split('\n')
    
    result = {
        'verdict': 'Unknown',
        'reasoning': '',
        'explanation': output
    }
    
    for i, line in enumerate(lines):
        if line.startswith('reasoning:'):
            reasoning_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith('verdict:'):
                if lines[j].strip():
                    reasoning_lines.append(lines[j].strip())
                j += 1
            result['reasoning'] = ' '.join(reasoning_lines)
        elif line.startswith('verdict:'):
            result['verdict'] = line.replace('verdict:', '').strip().upper()
    
    return result

def get_inference_engine(model_name: str) -> InferenceEngine:
    """Initialize the appropriate InferenceEngine based on model name."""
    return InferenceEngine(
        inference_strategy="azure_openai",
        connection_details={
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "base_url": os.getenv("AZURE_ENDPOINT"),
            "api_version": "2024-02-15-preview"
        },
        model_name=model_name
    )

def run_gpt_inference(messages: List[Dict[str, str]], model_name: str) -> str:
    """Run inference using Azure OpenAI GPT models."""
    # Initialize inference engine
    inference_engine = get_inference_engine(model_name)
    
    # Setup hyperparameters
    hyperparameters = Hyperparameters(
        max_tokens=4096,
        temperature=0.0
    )
    
    # Run inference
    responses = inference_engine.parallel_messages_inference([messages], hyperparameters)
    return responses[0] if responses else None

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import random
import argparse
from typing import List, Dict, Any, Union, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from openai import AzureOpenAI
from utils_ref import *
from utils.inference_engine import InferenceEngine, Hyperparameters, get_total_usage
from dotenv import load_dotenv

load_dotenv()

def process_problems_in_batches(
    problems: List[Dict], 
    llm: Optional[LLM], 
    tokenizer: Optional[AutoTokenizer], 
    sampling_params: Optional[SamplingParams], 
    args: argparse.Namespace,
    fewshot_samples: List[Dict[str, str]], 
    batch_size: int = 1
) -> List[Dict]:
    """Process problems in batches for faster inference."""
    results = []
    num_batches = (len(problems) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(problems))
        batch_problems = problems[start_idx:end_idx]
        
        batch_results = []
        # Initialize inference engine and hyperparameters for Azure
        if args.use_azure:
            inference_engine = get_inference_engine(args.model_path)
            hyperparameters = Hyperparameters(
                max_tokens=4096,
                temperature=0.0
            )
            
        for problem_idx, problem in enumerate(tqdm(batch_problems, 
                                                 desc=f"Processing batch {batch_idx + 1}/{num_batches}")):
            # Extract problem components
            question = problem.get('question', '')
            steps = extract_steps(problem)
            solution = '\n'.join(steps)
            
            # Process premises for all steps in parallel
            premise_messages = []
            for step_idx, step in enumerate(steps):
                if step_idx == 0:  # Skip question
                    continue
                    
                solution_so_far = '\n'.join(steps[:step_idx])
                messages = create_premise_prompt(question, solution_so_far, step, 
                                              None, fewshot_samples, True)
                premise_messages.append(messages)
            
            # Run batch inference for premises
            if args.use_azure:
                premise_outputs = inference_engine.parallel_messages_inference(premise_messages, hyperparameters)
            else:
                prompts = [create_premise_prompt(question, '\n'.join(steps[:i]), steps[i], 
                          tokenizer, fewshot_samples, False) 
                          for i in range(len(steps))]
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                premise_outputs = [output.outputs[0].text for output in outputs]
            premise_results = []
            for step_idx, output in enumerate(premise_outputs):
                premises = process_model_output_premises(output)
                premises = [i for i in premises if i<step_idx + 1]
                premise_results.append({
                    'step_number': step_idx+1,
                    'original_step': steps[step_idx],
                    'premises': [[premise, ""] for premise in premises],
                    'conclusion': "",
                    'reasoning': "",
                    'ground_truth_premises': get_ground_truth_premises(problem, step_idx)
                })
            

            problem["premise_annotations_generated"] = {}
            problem["premise_annotations_generated"]["steps"] = premise_results
            ## TODO: make premise results as the same schema for problem['premise_annotation']


            # Process errors for all steps in parallel
            steps_with_premises = extract_steps_and_premises(problem)

            standalone_messages = []
            contextual_messages = []
            
            for step_info in steps_with_premises:
                if args.use_azure:
                    standalone_msg = create_error_classification_prompt_gpt_standalone(
                        question, solution, step_info['step_text']
                    )
                    contextual_msg, premises_text = create_error_classification_prompt_gpt(
                        question, solution, step_info['step_text'],
                        step_info['formatted_premises'], step_info['premise_reasons']
                    )
                else:
                    standalone_msg = create_error_classification_prompt_standalone(
                        question, solution, step_info['step_text'], tokenizer
                    )
                    contextual_msg, premises_text = create_error_classification_prompt(
                        question, solution, step_info['step_text'],
                        step_info['formatted_premises'], step_info['premise_reasons'],
                        tokenizer
                    )
                standalone_messages.append(standalone_msg)
                contextual_messages.append((contextual_msg, premises_text))
            
            # Run batch inference for error classification
            if args.use_azure:
                standalone_outputs = inference_engine.parallel_messages_inference(standalone_messages, hyperparameters)
                contextual_outputs = inference_engine.parallel_messages_inference([msg for msg, _ in contextual_messages], hyperparameters)
            else:
                standalone_prompts = [msg for msg in standalone_messages]
                contextual_prompts = [msg for msg, _ in contextual_messages]
                standalone_outputs = llm.generate(standalone_prompts, sampling_params, use_tqdm=False)
                contextual_outputs = llm.generate(contextual_prompts, sampling_params, use_tqdm=False)
                standalone_outputs = [output.outputs[0].text for output in standalone_outputs]
                contextual_outputs = [output.outputs[0].text for output in contextual_outputs]
            
            error_results = []
            for step_idx, (step_info, standalone_output, contextual_output, (_, premises_text)) in enumerate(
                zip(steps_with_premises, standalone_outputs, contextual_outputs, contextual_messages)):
                
                # Get ground truth errors
                if len(problem['error_annotation']['error_annotations']['step_level'][step_idx]['errors']) == 0:
                    ground_truth = "correct"
                else:
                    errors = problem['error_annotation']['error_annotations']['step_level'][step_idx]['errors']
                    ground_truth = [error['error_type'] for error in errors]
                
                verdict_standalone = process_model_output_errors(standalone_output)
                verdict_contextual = process_model_output_errors(contextual_output)
                
                error_results.append({
                    'step_index': step_idx,
                    'step_text': step_info['step_text'],
                    'premises': premises_text,
                    'verdict_standalone': verdict_standalone['verdict'].lower(),
                    'explanation_standalone': verdict_standalone['explanation'],
                    'verdict_contextual': verdict_contextual['verdict'].lower(),
                    'explanation_contextual': verdict_contextual['explanation'],
                    'ground_truth': ground_truth
                })
            
            # Combine results for this problem
            predicted_premises = [i["premises"] for i in premise_results]
            predicted_premises_restructured = [[elem[0] for elem in sublist] for sublist in predicted_premises]

            batch_results.append({
                'problem_id': problem.get('id', start_idx + problem_idx),
                'question': question,
                'solution': solution,
                'premise_evaluation': premise_results,
                'step_classifications': error_results,
                'premise_list': predicted_premises_restructured
            })
        
        results.extend(batch_results)
        
        # Save intermediate results and print statistics
        if args.use_azure and (batch_idx + 1) % 1 == 0:  # Save after each batch for Azure
            usage_stats = get_total_usage()
            print(f"\nCurrent token usage - Prompt: {usage_stats['total_prompt_tokens']}, "
                  f"Completion: {usage_stats['total_completion_tokens']}")
            print(f"Current total cost: ${usage_stats['total_cost']:.4f}")
            
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate premises and errors in solution steps')
    parser.add_argument('--input-file', type=str, required=True,
                      help='Path to the input JSON file')
    parser.add_argument('--output-file', type=str, required=True,
                      help='Path to save the evaluation results')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the model or model name for Azure')
    parser.add_argument('--fewshot-path', type=str, help='Path to the few-shot data')
    parser.add_argument('--fewshot-num', type=int, default=0, help='Number of few-shot examples')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                      help='Number of GPUs for tensor parallelism')
    parser.add_argument('--use-azure', action='store_true', default=False,
                      help='Use Azure OpenAI instead of model path')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for parallel processing')
    args = parser.parse_args()

    # Initialize model and parameters
    print("Loading model...")
    is_gpt = args.use_azure or "gpt" in args.model_path.lower()
    
    if not is_gpt:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=None
        )
        llm = LLM(model=args.model_path,
                  tensor_parallel_size=args.tensor_parallel_size, download_dir="/shared/storage-01/huggingface/models/")
    else:
        tokenizer = None
        llm = None
        sampling_params = None  # Initialize as None for Azure/GPT models

    # Load data
    print("Loading data...")
    problems = load_json_data(args.input_file)

    # Prepare few-shot data if needed
    fewshot_samples = []
    if args.fewshot_num > 0:
        print("Loading few-shot data...")
        fewshot_data = load_json_data(args.fewshot_path)
        random.seed(42)
        fewshot_indices = random.sample(range(len(fewshot_data)), args.fewshot_num)
        for fs_idx in fewshot_indices:
            # [Few-shot data processing code remains the same]
            pass

    # Process problems in batches
    results = process_problems_in_batches(
        problems, llm, tokenizer, sampling_params, args, 
        fewshot_samples, args.batch_size
    )

    # Save final results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final usage statistics
    if is_gpt:
        usage_stats = get_total_usage()
        print("\n=== Final Usage Statistics ===")
        print(f"Total Prompt Tokens: {usage_stats['total_prompt_tokens']}")
        print(f"Total Completion Tokens: {usage_stats['total_completion_tokens']}")
        print(f"Total Cost: ${usage_stats['total_cost']:.4f}")

if __name__ == "__main__":
    main()