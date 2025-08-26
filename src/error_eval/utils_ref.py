import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from openai import AzureOpenAI
import re
from typing import List, Dict


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

CONTEXTUAL_SYS_PROMPT = """You are provided with a math question, a statement which is a step in the solution to the question and the premises to this steps (the question is also a premise). Your task is to identify whether the step follow naturally from the premises or not. 
If the current step contradicts the premises, mark is as a logical_inconsistency
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

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse the JSON file containing solution steps."""   
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_steps_and_premises(problem_data: Dict[str, Any], recursive_premises: bool = False) -> List[Dict[str, Any]]:
    """Extract steps and their premises from the problem data."""
    steps_with_premises = []
    
    def get_step_premises(step_number: int, visited: set) -> List[Dict[str, Any]]:
        """Recursively get premises for a step including premises of premises."""
        if step_number in visited:
            return []  # Avoid cycles
            
        visited.add(step_number)
        premises = []
        
        # Find the step data
        step_data = None
        for s in steps:
            if s.get('step_number') == step_number:
                step_data = s
                break
                
        if not step_data:
            return []
            
        # Get direct premises
        for premise in step_data.get('premises', []):
            if len(premise) >= 2:  # Premise should have [step_number, reason]
                premise_step_number, reason = premise[0], premise[1]
                # Skip step 0 as premise
                if premise_step_number == 0:
                    continue
                # Find the original step text for this premise
                original_step = None
                for s in steps:
                    if s.get('step_number') == premise_step_number:
                        original_step = s.get('original_step', '')
                        break
                        
                if original_step:
                    # Get premises of this premise recursively if enabled
                    sub_premises = get_step_premises(premise_step_number, visited.copy()) if recursive_premises else []
                    # Sort sub_premises by step number
                    sub_premises.sort(key=lambda x: x['step_number'])
                    # Remove "Premise X - " if it exists at the start of original_step
                    if original_step.startswith("Premise "):
                        original_step = original_step.split(" - ", 1)[1] if " - " in original_step else original_step
                    premises.append({
                        'step_number': premise_step_number,
                        'step_text': original_step,
                        'reason': reason,
                        'sub_premises': sub_premises
                    })
                else:
                    # If we can't find original step, use reason as both step and info
                    # Remove "Premise X - " if it exists at the start of reason
                    if reason.startswith("Premise "):
                        reason = reason.split(" - ", 1)[1] if " - " in reason else reason
                    premises.append({
                        'step_number': -1,
                        'step_text': reason,
                        'reason': reason,
                        'sub_premises': []
                    })
        
        # Sort premises by step number
        premises.sort(key=lambda x: x['step_number'])
        return premises
    
    if 'premise_annotations_generated' in problem_data:
        premise_data = problem_data['premise_annotations_generated']
        if 'steps' in premise_data:
            steps = premise_data['steps']
            for step_data in steps:
                # Skip step 0 as it's usually just restating the question
                if step_data.get('step_number', -1) == 0:
                    continue
                    
                step_number = step_data.get('step_number', -1)
                step_info = {
                    'step_text': step_data.get('original_step', ''),
                    'premises': [],
                    'premise_reasons': []
                }
                
                if recursive_premises:
                    # Get premises recursively with hierarchy
                    step_info['premises'] = get_step_premises(step_number, set())
                    
                    # Format premises for the prompt
                    formatted_premises = []
                    premise_reasons = []
                    seen_premises = set()  # Track seen premises to avoid duplicates
                    all_premises = []  # Collect all premises first
                    
                    def collect_premises(premises_list: List[Dict[str, Any]], depth: int = 0):
                        for p in premises_list:
                            if p['step_text'] not in seen_premises:
                                seen_premises.add(p['step_text'])
                                all_premises.append((p['step_number'], depth, p))
                            collect_premises(p['sub_premises'], depth + 1)
                    
                    # First collect all premises
                    collect_premises(step_info['premises'])
                    
                    # Sort all premises by step number
                    all_premises.sort(key=lambda x: x[0])  # Sort by step_number
                    
                    # Now format them in order
                    for step_num, depth, p in all_premises:
                        prefix = "  " * depth + f"Premise {len(formatted_premises)} - "
                        # Remove "Premise X - " if it exists at the start of step_text
                        step_text = p['step_text']
                        if step_text.startswith("Premise "):
                            step_text = step_text.split(" - ", 1)[1] if " - " in step_text else step_text
                        formatted_premises.append(prefix + step_text)
                        premise_reasons.append(p['reason'])
                    
                    step_info['formatted_premises'] = formatted_premises
                    step_info['premise_reasons'] = premise_reasons
                else:
                    # Original non-recursive behavior
                    for premise in step_data.get('premises', []):
                        if len(premise) >= 2:  # Premise should have [step_number, reason]
                            step_number, reason = premise[0], premise[1]
                            if step_number == 0:
                                continue
                            # Find the original step text for this premise
                            original_step = None
                            for s in steps:
                                if s.get('step_number') == step_number:
                                    original_step = s.get('original_step', '')
                                    break
                            if original_step:
                                # Remove "Premise X - " if it exists at the start of original_step
                                if original_step.startswith("Premise "):
                                    original_step = original_step.split(" - ", 1)[1] if " - " in original_step else original_step
                                step_info['premises'].append([step_number, original_step])
                                step_info['premise_reasons'].append(reason)
                            else:
                                # If we can't find original step, use reason as both step and info
                                # Remove "Premise X - " if it exists at the start of reason
                                if reason.startswith("Premise "):
                                    reason = reason.split(" - ", 1)[1] if " - " in reason else reason
                                step_info['premises'].append([-1, reason])  # Use -1 for non-step premises
                                step_info['premise_reasons'].append(reason)
                    
                    # Sort premises by step number
                    step_info['premises'].sort(key=lambda x: x[0])
                    # Extract just the text after sorting
                    step_info['formatted_premises'] = [p[1] for p in step_info['premises']]
                
                steps_with_premises.append(step_info)
                
    return steps_with_premises


def create_error_classification_prompt(question: str, solution: str, step: str, premises: List[str], premise_reasons: List[str], tokenizer: AutoTokenizer) -> tuple[str, str]:
    """Create a zero-shot prompt for evaluating step correctness. Returns (prompt, premises_text)."""
    system_prompt = CONTEXTUAL_SYS_PROMPT
    
    # Format premises with contextual information, handling duplicates
    unique_premises = []
    premise_to_reasons = {}  # Map premise to list of reasons
    
    # Build mapping of premises to their reasons
    for premise, reason in zip(premises, premise_reasons):
        if premise not in premise_to_reasons:
            premise_to_reasons[premise] = []
            unique_premises.append(premise)
        premise_to_reasons[premise].append(reason)
    
    # Format unique premises
    premises_text = "\n"
    for i, premise in enumerate(unique_premises):
        # Determine if premise is from question or previous step
        is_question = premise == question
        premises_text += f"{premise}\n"
    
    premises_text += "\n"
    
    user_message = CONTEXTUAL_USR_PROMPT.format(
        question=question,
        premises=premises_text,
        step=step
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, premises_text






def create_error_classification_prompt_gpt(question: str, solution: str, step: str, premises: List[str], premise_reasons: List[str]) -> tuple[List[Dict[str, str]], str]:
    """Create a prompt for GPT models. Returns (messages, premises_text)."""
    system_prompt = CONTEXTUAL_SYS_PROMPT
    
    # Format premises with contextual information, handling duplicates
    unique_premises = []
    premise_to_reasons = {}  # Map premise to list of reasons
    
    # Clean and build mapping of premises to their reasons
    for premise, reason in zip(premises, premise_reasons):
        cleaned_premise = premise
        if cleaned_premise not in premise_to_reasons:
            premise_to_reasons[cleaned_premise] = []
            unique_premises.append(cleaned_premise)
        premise_to_reasons[cleaned_premise].append(reason)
    
    # Format unique premises
    premises_text = "Premises:\n"
    for i, premise in enumerate(unique_premises):
        premises_text += f"{premise}\n"
    
    premises_text += "\n"
    
    user_message = CONTEXTUAL_USR_PROMPT.format(
        question=question,
        premises=premises_text,
        step=step
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    return messages, premises_text


def create_error_classification_prompt_gpt_standalone(question: str, solution: str, step: str) -> tuple[List[Dict[str, str]], str]:
    """Create a prompt for GPT models that focuses on standalone mathematical/logical correctness.
    Returns (messages, premises_text)."""
    system_prompt = STANDALONE_SYS_PROMPT
    
    user_message = STANDALONE_USR_PROMPT.format(step=step)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    return messages


def create_error_classification_prompt_standalone(question: str, solution: str, step: str, tokenizer: AutoTokenizer):
    """Create a zero-shot prompt for evaluating step correctness. Returns (prompt, premises_text)."""
    system_prompt = STANDALONE_SYS_PROMPT
    user_message = STANDALONE_USR_PROMPT.format(step=step)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_steps(problem: dict) -> list:
    """Extract individual steps from the problem data."""
    if 'steps' in problem:
        return [step.strip() for step in problem['steps']]
    return []

def get_ground_truth_premises(problem: dict, step_idx: int) -> list:
    """Extract ground truth premises for a given step."""
    if 'premise_annotation' in problem and 'steps' in problem['premise_annotation']:
        steps = problem['premise_annotation']['steps']
        if 0 <= step_idx < len(steps):
            step_data = steps[step_idx+1]  # +1 because step_idx 0 is the question
            premises = list(set(premise[0] for premise in step_data.get('premises', [])))
            return sorted(premises)
    return []

def process_model_output_premises(output: str) -> List[int]:
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