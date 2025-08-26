#!/usr/bin/env python

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

# ========== Adjust project_root to your local path if needed ==========
project_root = "/home/sagnikm3/dag-llm-temp/dag-llm"
sys.path.append(project_root)

from utils.inference_engine import InferenceEngine, Hyperparameters, get_total_usage

# ---- IMPORT THE HELPER FUNCTIONS YOU PROVIDED ----
# For simplicity, I've pasted them inline below. If you have them in a separate module, import from there.
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    logger.warning("vLLM not installed; local usage will not work unless installed.")

############################################################
# HELPER PROMPT FUNCTIONS (from your snippet)
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
        # For some models, you might need trust_remote_code=True, etc.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

        # We'll store usage stats ourselves if needed
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
            if not out.outputs:  # some error case
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
        # If your tokenizer does *not* implement apply_chat_template, you can replicate
        # the logic manually. If it does, we can do:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            # fallback to a naive manual approach
            # (example: "[System]\n{content}\n\n[User]\n{content}\n\nAssistant:")
            prompt = ""
            for msg in messages:
                role = msg["role"].lower()
                if role == "system":
                    prompt += f"[System]\n{msg['content'].strip()}\n\n"
                elif role == "user":
                    prompt += f"[User]\n{msg['content'].strip()}\n\n"
                else:
                    prompt += f"[{role.capitalize()}]\n{msg['content'].strip()}\n\n"
            # add a final "Assistant:" so model knows to produce an answer
            prompt += "Assistant:"
            return prompt
        else:
            # Use HF's chat template if available
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
        # Adapt logic as needed for your environment:
        if model_name in ["o1-mini", "o1-preview"]:
            return InferenceEngine(
                inference_strategy="azure_openai",
                connection_details={
                    "api_key": os.getenv("AZURE_OPENAI_KEY_O1"),
                    "base_url": "https://uiuc-convai-sweden.openai.azure.com/",
                    "api_version": "2024-09-01-preview"
                },
                model_name=model_name
            )
        else:
            return InferenceEngine(
                inference_strategy="azure_openai",
                connection_details={
                    "api_key": os.getenv("AZURE_OPENAI_KEY"),
                    "base_url": "https://uiuc-convai.openai.azure.com/",
                    "api_version": "2024-02-15-preview"
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
# PROMPTS FOR PREMISE ANNOTATION (unchanged from your snippet)
############################################################

def create_premise_prompt(problem: Dict[str, Any]) -> str:
    question = problem.get("question", problem.get("problem", ""))
    steps_joined = "\n".join(problem.get("steps", []))

    json_template = '''{
    "steps": [
        {
            "step_number": 0,
            "original_step": "{question}",
            "premises": [[0, "premise from problem statement"]],
            "conclusion": "key information from the problem",
            "reasoning": "organization of problem information"
        },
        {
            "step_number": 1,
            "original_step": "copy the exact step text from student's solution",
            "premises": [
                [0, "premise from problem"],
                [1, "premise from step 1"]
            ],
            "conclusion": "result of this step",
            "reasoning": "how premises combine to reach the conclusion"
        }
    ]
}'''

    prompt = f"""Given this math word problem and its solution steps, identify the key premises and their relationships.

Problem: {question}

Solution Steps:
{steps_joined}

Return your analysis in this exact JSON format:
{json_template}

Critical Rules for Premises:
1. A step can NEVER use itself as a premise.
2. Premises can only come from Step 0 (the problem statement) or previous steps.
3. The number of steps in your output (excluding step 0) must match exactly the number of steps in the student's solution.
4. Only produce valid JSON, with no extra text or formatting."""

    return prompt

def format_prompts_premise(problems: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    messages_batch = []
    for prob in problems:
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert in mathematical reasoning. "
                "Output only valid JSON describing premises, conclusion, and reasoning. "
                "No extra text."
            )
        }
        user_msg = {
            "role": "user",
            "content": create_premise_prompt(prob)
        }
        messages_batch.append([system_msg, user_msg])
    return messages_batch

############################################################
# RUN PREMISE ANNOTATION
############################################################

def run_premise_annotation(
    problems: List[Dict[str, Any]],
    engine: object,
    batch_size: int = 128
) -> List[str]:
    num_batches = (len(problems) + batch_size - 1) // batch_size
    all_responses = []

    for b in range(num_batches):
        batch = problems[b*batch_size : (b+1)*batch_size]
        prompts = format_prompts_premise(batch)
        hyperparams = Hyperparameters(max_tokens=4096, temperature=0.0)

        logger.info(f"[Premise] Processing batch {b+1}/{num_batches}, size={len(batch)}")
        try:
            batch_resps = engine.parallel_messages_inference(prompts, hyperparams)
            all_responses.extend(batch_resps)
        except Exception as e:
            logger.error(f"Error in premise batch {b+1}: {e}")
            all_responses.extend([None]*len(batch))

        usage_stats = get_total_usage()
        logger.info(
            f"Usage so far: Prompt={usage_stats['total_prompt_tokens']} Completion={usage_stats['total_completion_tokens']}"
            f" Cost=${usage_stats['total_cost']:.4f}"
        )

    return all_responses

def integrate_premises(problems: List[Dict[str, Any]], premise_texts: List[str]) -> None:
    for prob, annotation_str in zip(problems, premise_texts):
        if annotation_str is None:
            continue
        try:
            annotation_json = json.loads(annotation_str)
            prob["premise_annotation"] = annotation_json
        except json.JSONDecodeError:
            logger.warning("Could not parse premise annotation as JSON. Skipping.")

############################################################
# RUN ERROR ANNOTATION (updated to use your GPT prompt helpers)
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

    Returns a list of lists, one sublist per problem, each sublist is step-level results.
    """
    # Flatten all steps
    all_steps = []
    for p_idx, prob in enumerate(problems):
        question = prob.get("question", prob.get("problem", ""))
        steps = prob.get("steps", [])
        for s_idx, step_text in enumerate(steps):
            # We'll supply the text of previous steps as "premises" in a single block
            premises_text = "\n".join(
                f"- {st}" for st in steps[:s_idx]
            ) if s_idx > 0 else ""
            all_steps.append({
                "problem_idx": p_idx,
                "step_idx": s_idx,
                "question": question,
                "premises_text": premises_text,
                "step_text": step_text
            })

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
            # We do not strictly need 'solution' argument; pass item["question"] if needed
            st_msgs = create_error_classification_prompt_gpt_standalone(
                question=item["question"],
                step=item["step_text"]
            )
            ct_msgs = create_error_classification_prompt_gpt(
                question=item["question"],
                premises_text=item["premises_text"],
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
    premise_responses = run_premise_annotation(problems, engine, batch_size=args.batch_size)
    integrate_premises(problems, premise_responses)

    # 4) Error annotation
    error_data = run_error_annotation(problems, engine, batch_size=args.batch_size)
    integrate_error_annotation(problems, error_data)

    # 5) Final usage stats (only meaningful if using Azure)
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
