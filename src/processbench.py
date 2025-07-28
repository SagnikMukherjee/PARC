#!/usr/bin/env python

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

# ========== Adjust project_root to your local path if needed ==========
project_root = ""
sys.path.append(project_root)

from utils.inference_engine import InferenceEngine, Hyperparameters, get_total_usage

from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    logger.warning("vLLM not installed; local usage will not work unless installed.")

############################################################
# HELPER PROMPT FUNCTIONS
############################################################

STANDALONE_SYS_PROMPT = """Your task is to determine whether a given sentence contains any mathematical errors. 
For mathematical error, check if the sentence contains mathematical calculations (arithmetic or algebraic), and whether they are incorrect. If there are such errors, mark the sentence as "mathematical_error"
-   Mathematical errors can only come from incorrect results of mathematical operations
If no such errors exist, mark it as "correct".

Note: mathematical error can only come from incorrect numerical or algebraic calculations (i.e. wrong multiplication, wrong addition etc.)
if there are no numerical or algebraic calculations done, you can mark it as correct
"""

STANDALONE_USR_PROMPT = """
Statement to analyze:
{step}
Format your response as:

Reasoning: [detailed analysis of the statement's validity]
Verdict: [correct or  mathematical_error]
"""

CONTEXTUAL_SYS_PROMPT = """You are provided with a math question, a statement which is a step in the solution to the question and the premises to this step (the question is also a premise). Your task is to identify whether the step follows naturally from the premises or not. 
If the current step contradicts the premises, mark it as a logical_inconsistency
If the step can be directly inferred from the premises, mark it as correct.
You should not check whether the premises are correct, assume they are correct. Only check the sentence given.
 """

CONTEXTUAL_USR_PROMPT = """
Given Premises:
Question:
{question}
Previous steps as premise:
{premises}
Statement to analyze:
{step}

Guidelines:
1. for logical_inconsistency check if the step was performed under misinterpretation of the premises, made invalid deductions or had unsupported assumptions
2. Don't check for correctness of the premises, your only task is to check correctness of the given sentence


Format your response as:           
Reasoning: [detailed analysis of the statement's validity]
Verdict: [correct, logical_inconsistency]
"""

############################################################
# NEW PREMISE ANNOTATION PROMPTS
############################################################

def create_premise_prompt(
        question: str, 
        solution_so_far: str, 
        step: str, 
        tokenizer: AutoTokenizer = None, 
        samples_fewshot: Optional[List[Dict[str, str]]] = None, 
        is_gpt: bool = False
    ) -> Any:
    """Create a prompt for premise extraction."""
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
{solution_so_far}

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
        return tokenizer.apply_chat_template(messages, tokenize=False)

def process_model_output_premises(output: str) -> List[int]:
    """Process model output to extract premises."""
    if output is None:
        return []
        
    lines = output.strip().split('\n')
    premises = []
    
    for line in lines:
        if not line.strip():
            continue
            
        match = re.search(r"Step\s+(\d+)", line)
        if match:
            try:
                step_num = int(match.group(1))
                premises.append(step_num)
            except ValueError:
                pass
    
    return premises

def create_error_classification_prompt_gpt_standalone(question: str, step: str) -> List[Dict[str, str]]:
    """Create a prompt for GPT models that focuses on standalone correctness (mathematical error or correct)."""
    system_prompt = STANDALONE_SYS_PROMPT
    user_message = STANDALONE_USR_PROMPT.format(step=step)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    return messages

def create_error_classification_prompt_gpt(question: str, premises_text: str, step: str) -> List[Dict[str, str]]:
    """Create a prompt for GPT models that focuses on contextual correctness (logical_inconsistency or correct)."""
    system_prompt = CONTEXTUAL_SYS_PROMPT
    user_message = CONTEXTUAL_USR_PROMPT.format(
        question=question,
        premises=premises_text,
        step=step
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    return messages

def parse_verdict(text: str) -> str:
    """
    Find a line starting with 'Verdict:' (case-insensitive).
    Return the remainder. Lowercase it. If none found, return 'unknown'.
    """
    if not text:
        return "unknown"
    lines = text.strip().split('\n')
    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("verdict:"):
            return lower.replace("verdict:", "").strip()
    return "unknown"

############################################################
# LOCAL vs. AZURE INFERENCE: We define a LocalInferenceEngine
############################################################

class LocalInferenceEngine:
    """
    A local vLLM 'engine' that replicates the .parallel_messages_inference(messages, hyperparams)
    method of your Azure InferenceEngine for consistent usage within this script.

    MAJOR CHANGE: In _messages_to_prompt(), we now call `tokenizer.apply_chat_template`
    with `add_generation_prompt=True`, so vLLM sees the correct formatting (System/User roles).
    """
    def __init__(self, model_name: str, tensor_parallel_size: int = 4):
        if not _HAS_VLLM:
            raise RuntimeError("vLLM or transformers not installed, cannot use LocalInferenceEngine.")

        logger.info(f"Loading local model from {model_name} with vLLM.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0

    def parallel_messages_inference(
        self,
        messages_batch: List[List[Dict[str, str]]],
        hyperparameters: Hyperparameters
    ) -> List[str]:
        """
        - messages_batch: list of dialogues, each a list of {"role": "...", "content": "..."}
        - hyperparameters: e.g. temperature, max_tokens, etc.

        Convert each list of messages into a text prompt using `_messages_to_prompt`,
        then run batch generation with vLLM.
        Return a list of response strings, one per input.
        """
        prompts = [self._messages_to_prompt(msgs) for msgs in messages_batch]

        sp = SamplingParams(
            temperature=hyperparameters.temperature if hyperparameters.temperature is not None else 0.0,
            max_tokens=hyperparameters.max_tokens if hyperparameters.max_tokens else 512
        )

        logger.info(f"Running local vLLM batch of size {len(prompts)}.")
        outputs = self.llm.generate(prompts, sp, use_tqdm=True)

        results = []
        for out in outputs:
            if not out.outputs:
                results.append("")
            else:
                text = out.outputs[0].text
                results.append(text)

        return results

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts a list of ChatCompletion-style messages to a single text prompt using the
        HuggingFace 'apply_chat_template' (if your tokenizer supports it). This ensures we
        get a proper System/User/Assistant chat style in a single string for vLLM.

        Note: For many HF models, 'apply_chat_template' is model-specific. If your model's
        tokenizer doesn't have it, you might need a custom approach. The key is to replicate
        a chat-style format. The crucial piece is `add_generation_prompt=True`, so that the
        final user message is followed by something like: '\nAssistant:' 
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            prompt = ""
            for msg in messages:
                role = msg["role"].lower()
                if role == "system":
                    prompt += f"[System]\n{msg['content'].strip()}\n\n"
                elif role == "user":
                    prompt += f"[User]\n{msg['content'].strip()}\n\n"
                else:
                    prompt += f"[{role.capitalize()}]\n{msg['content'].strip()}\n\n"
            prompt += "Assistant:"
            return prompt
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

############################################################
# get_engine() for either Azure or local usage
############################################################

def get_engine(model_name: str, use_local_vllm: bool, tensor_parallel_size: int = 4) -> Optional[object]:
    if use_local_vllm:
        if not _HAS_VLLM:
            raise RuntimeError("vLLM not installed or failed import. Cannot use local usage.")
        return LocalInferenceEngine(model_name=model_name, tensor_parallel_size=tensor_parallel_size)
    else:
        # Azure usage
        logger.info(f"Using Azure InferenceEngine for model {model_name}.")
        if model_name in ["o1-mini", "o1-preview"]:
            return InferenceEngine(
                inference_strategy="azure_openai",
                connection_details={
                    "api_key": os.getenv("API_KEY"),
                    "base_url": "https://api.openai.com/v1",
                    "api_version": ""
                },
                model_name=model_name
            )
        else:
            return InferenceEngine(
                inference_strategy="openai",
                connection_details={
                    "api_key": os.getenv("API_KEY"),
                    "base_url": "https://api.openai.com/v1",
                    "api_version": ""
                },
                model_name=model_name
            )

############################################################
# LOADING & SAVING
############################################################

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records from {file_path}")
    return data

def save_json_data(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} records to {file_path}")

############################################################
# REVISED PREMISE ANNOTATION IMPLEMENTATION
############################################################

def run_premise_annotation(
    problems: List[Dict[str, Any]],
    engine: object,
    batch_size: int = 128
) -> None:
    """
    Run premise annotation for all steps in all problems.
    """
    all_steps = []
    
    for prob_idx, prob in enumerate(problems):
        question = prob.get("question", prob.get("problem", ""))
        steps = prob.get("steps", [])
        
        # For each step in the problem (starting from step 1, since step 0 is the question)
        for step_idx in range(1, len(steps)):
            solution_so_far = "\n".join(steps[:step_idx])
            current_step = steps[step_idx]
            
            all_steps.append({
                "problem_idx": prob_idx,
                "step_idx": step_idx,
                "question": question,
                "solution_so_far": solution_so_far,
                "current_step": current_step
            })
    
    logger.info(f"Total steps for premise annotation: {len(all_steps)}")
    
    # Initialize premise annotation data structure for all problems
    for prob in problems:
        # Create a structure that matches the expected format for evaluation
        premise_steps = [
            {
                "step_number": 0,
                "original_step": prob.get("question", prob.get("problem", "")),
                "premises": [],
                "conclusion": "key information from the problem",
                "reasoning": "organization of problem information"
            }
        ]
        
        # Add an entry for each solution step
        for step_idx, step_text in enumerate(prob.get("steps", []), 1):
            premise_steps.append({
                "step_number": step_idx,
                "original_step": step_text,
                "premises": [],  # Will be filled in by the batch processing
                "conclusion": "",
                "reasoning": ""
            })
        
        prob["premise_annotation"] = {"steps": premise_steps}
    
    # Process in batches
    num_batches = (len(all_steps) + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(all_steps))
        batch = all_steps[start_idx:end_idx]
        
        logger.info(f"[Premise] Processing batch {b+1}/{num_batches}, size={len(batch)}")
        
        # Create prompts for all steps in this batch
        prompts = []
        for item in batch:
            # Create the prompt using the function from the reference script
            prompt = create_premise_prompt(
                question=item["question"],
                solution_so_far=item["solution_so_far"],
                step=item["current_step"],
                is_gpt=True
            )
            prompts.append(prompt)
        
        # Run inference in batch
        hyperparams = Hyperparameters(max_tokens=2048, temperature=0.0)
        try:
            batch_responses = engine.parallel_messages_inference(prompts, hyperparams)
        except Exception as e:
            logger.error(f"Error in premise batch {b+1}: {e}")
            batch_responses = [None] * len(batch)
        
        # Process each response and update the premise_annotation in the problems
        for item, response in zip(batch, batch_responses):
            prob_idx = item["problem_idx"]
            step_idx = item["step_idx"]
            
            if response is not None:
                # Process the model's output to extract premises
                premises = process_model_output_premises(response)
                
                # Make sure the premises are valid (no self-reference, only previous steps)
                premises = [p for p in premises if p < step_idx]
                
                # Format the premises in the expected format: [[step_number, reason]]
                formatted_premises = [[p, f"Premise from step {p}"] for p in premises]
                
                # Update the premise_annotation for this step
                problems[prob_idx]["premise_annotation"]["steps"][step_idx]["premises"] = formatted_premises
        
        # Log usage statistics
        usage_stats = get_total_usage()
        logger.info(
            f"Usage so far: Prompt={usage_stats['total_prompt_tokens']} "
            f"Completion={usage_stats['total_completion_tokens']} "
            f"Cost=${usage_stats['total_cost']:.4f}"
        )

############################################################
# HELPER FUNCTIONS FOR PREMISE-BASED ERROR ANNOTATION
############################################################

def extract_steps_with_premises(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract steps with their specific premises based on premise annotation.
    This follows the approach from the reference script but adapted to our data structure.
    """
    all_steps_with_premises = []
    
    for p_idx, prob in enumerate(problems):
        question = prob.get("question", prob.get("problem", ""))
        steps = prob.get("steps", [])
        
        # Get premise annotations if available
        premise_annotations = prob.get("premise_annotation", {}).get("steps", [])
        
        # Start from step 1 (skip the question, which is step 0)
        for s_idx, step_text in enumerate(steps):
            step_info = {
                "problem_idx": p_idx,
                "step_idx": s_idx,
                "question": question,
                "step_text": step_text,
                "premises": [],
                "formatted_premises": [],
                "premise_reasons": []
            }
            
            # Find premise annotation for this step
            premise_info = None
            if s_idx < len(premise_annotations):
                premise_info = premise_annotations[s_idx]
            
            # If we have premise info, extract the specific premises
            if premise_info and "premises" in premise_info:
                premises = premise_info["premises"]
                for premise in premises:
                    if len(premise) >= 2:
                        premise_step_num, reason = premise[0], premise[1]
                        
                        # Skip if premise is the step itself or step 0 (the question)
                        if premise_step_num == s_idx:
                            continue
                            
                        # For step 0 (question), use the question text
                        if premise_step_num == 0:
                            original_step_text = question
                        else:
                            # For other steps, find the original step text
                            if premise_step_num < len(steps):
                                original_step_text = steps[premise_step_num - 1]
                            else:
                                continue  # Invalid premise step number
                        
                        step_info["premises"].append([premise_step_num, original_step_text])
                        step_info["premise_reasons"].append(reason)
            
            # Format premises for the prompt
            if step_info["premises"]:
                step_info["formatted_premises"] = [f"Premise {i} - {p[1]}" for i, p in enumerate(step_info["premises"])]
            else:
                # If no specific premises were identified, fall back to using previous steps
                if s_idx > 0:
                    # Just use all previous steps as premises
                    step_info["formatted_premises"] = [f"Premise {i} - {steps[i]}" for i in range(s_idx)]
                    step_info["premise_reasons"] = ["Previous step"] * s_idx
            
            all_steps_with_premises.append(step_info)
    
    return all_steps_with_premises

############################################################
# RUN ERROR ANNOTATION (revised to use premise information)
############################################################

def run_error_annotation(
    problems: List[Dict[str, Any]],
    engine: object,
    batch_size: int = 128
) -> List[List[Dict[str, Any]]]:
    """
    For each step in each problem, run:
      1) standalone check (correct or mathematical_error)
      2) contextual check (correct or logical_inconsistency)

    Uses specifically identified premises for contextual checks.
    Returns a list of lists, one sublist per problem, each sublist is step-level results.
    """
    # Extract steps with their premises
    all_steps = extract_steps_with_premises(problems)
    
    results_for_all = [None]*len(all_steps)
    num_batches = (len(all_steps) + batch_size - 1) // batch_size
    logger.info(f"Total steps for error annotation: {len(all_steps)}")

    for b in range(num_batches):
        batch_data = all_steps[b*batch_size : (b+1)*batch_size]

        # Build the message arrays:
        #  1) Standalone: check if there's a math error
        #  2) Contextual: check if there's a logical inconsistency vs. premises
        standalone_messages = []
        contextual_messages = []
        for item in batch_data:
            # Create standalone prompt (mathematical correctness)
            st_msgs = create_error_classification_prompt_gpt_standalone(
                question=item["question"],
                step=item["step_text"]
            )
            
            # Create contextual prompt (logical consistency with premises)
            premises_text = "\n".join(item["formatted_premises"]) if item["formatted_premises"] else ""
            
            ct_msgs = create_error_classification_prompt_gpt(
                question=item["question"],
                premises_text=premises_text,
                step=item["step_text"]
            )
            
            standalone_messages.append(st_msgs)
            contextual_messages.append(ct_msgs)

        hyperparams = Hyperparameters(max_tokens=1024, temperature=0.0)

        logger.info(f"[Error] Processing batch {b+1}/{num_batches}, size={len(batch_data)}")
        # Run standalone
        try:
            stand_resps = engine.parallel_messages_inference(standalone_messages, hyperparams)
        except Exception as e:
            logger.error(f"Error in standalone error batch {b+1}: {e}")
            stand_resps = [None]*len(batch_data)

        # Run contextual
        try:
            cont_resps = engine.parallel_messages_inference(contextual_messages, hyperparams)
        except Exception as e:
            logger.error(f"Error in contextual error batch {b+1}: {e}")
            cont_resps = [None]*len(batch_data)

        # Combine results
        for i, (sresp, cresp) in enumerate(zip(stand_resps, cont_resps)):
            global_idx = b*batch_size + i
            item = batch_data[i]
            if sresp is None and cresp is None:
                results_for_all[global_idx] = None
                continue

            # parse verdict lines
            s_verdict = parse_verdict(sresp) if sresp else "unknown"
            c_verdict = parse_verdict(cresp) if cresp else "unknown"

            # store
            results_for_all[global_idx] = {
                "problem_idx": item["problem_idx"],
                "step_idx": item["step_idx"],
                "standalone_verdict": s_verdict,
                "standalone_explanation": sresp or "",
                "contextual_verdict": c_verdict,
                "contextual_explanation": cresp or ""
            }

        # If Azure, track usage
        usage_stats = get_total_usage()
        logger.info(
            f"Usage after error batch {b+1}: Prompt={usage_stats['total_prompt_tokens']} Completion={usage_stats['total_completion_tokens']}"
            f" Cost=${usage_stats['total_cost']:.4f}"
        )

    # Group the results by problem
    error_ann = [[] for _ in range(len(problems))]
    for rec in results_for_all:
        if rec is None:
            continue
        p_idx = rec["problem_idx"]
        error_ann[p_idx].append({
            "step_index": rec["step_idx"],
            "standalone_verdict": rec["standalone_verdict"],
            "standalone_explanation": rec["standalone_explanation"],
            "contextual_verdict": rec["contextual_verdict"],
            "contextual_explanation": rec["contextual_explanation"]
        })

    # sort each sublist by step_index
    for i in range(len(error_ann)):
        error_ann[i].sort(key=lambda x: x["step_index"])

    return error_ann

def integrate_error_annotation(problems: List[Dict[str, Any]], error_data: List[List[Dict[str, Any]]]) -> None:
    for i, step_list in enumerate(error_data):
        problems[i]["error_annotation"] = {
            "step_annotations": step_list
        }
        # Gather which steps are flagged as error if either verdict != "correct"/"unknown"
        flagged_steps = []
        for step in step_list:
            s_v = step["standalone_verdict"].lower()
            c_v = step["contextual_verdict"].lower()
            if (s_v not in ["correct", "unknown"]) or (c_v not in ["correct", "unknown"]):
                flagged_steps.append(step["step_index"])
        problems[i]["predicted_error_step_indices"] = flagged_steps

############################################################
# MAIN
############################################################

def main():
    parser = argparse.ArgumentParser(description="Integrated premise + error annotation with local vLLM or Azure.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the annotated JSON.")
    parser.add_argument("--model-name", type=str, required=True, help="Local model path or Azure model name.")
    parser.add_argument("--use-local-vllm", action="store_true", help="Use local vLLM instead of Azure/OpenAI.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for parallel inference calls.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Number of GPUs to use for tensor parallelism in vLLM.")
    args = parser.parse_args()

    load_dotenv(os.path.join(project_root, ".env"))
    logger.info(f"Starting integrated annotation. Model={args.model_name}, local_vllm={args.use_local_vllm}")

    # 1) Load data
    problems = load_json_data(args.input_file)
    logger.info(f"Total problems: {len(problems)}")

    # 2) Get the appropriate engine
    engine = get_engine(args.model_name, args.use_local_vllm, tensor_parallel_size=args.tensor_parallel_size)

    # 3) Premise annotation
    run_premise_annotation(problems, engine, batch_size=args.batch_size)

    # 4) Error annotation (unchanged)
    error_data = run_error_annotation(problems, engine, batch_size=args.batch_size)
    integrate_error_annotation(problems, error_data)

    # 5) Final usage stats
    usage_stats = get_total_usage()
    logger.info("=== Final Usage Stats ===")
    logger.info(f"Prompt tokens: {usage_stats['total_prompt_tokens']}, "
                f"Completion tokens: {usage_stats['total_completion_tokens']}, "
                f"Cost: ${usage_stats['total_cost']:.4f}")

    # 6) Save results
    save_json_data(problems, args.output_file)

    # Summaries
    annotated_prem = sum(1 for p in problems if "premise_annotation" in p)
    annotated_err = sum(1 for p in problems if "error_annotation" in p)
    logger.info(f"Done. premise_annotation for {annotated_prem}/{len(problems)} problems. "
                f"error_annotation for {annotated_err}/{len(problems)} problems.")


if __name__ == "__main__":
    main()