import json
from collections import defaultdict
import numpy as np
import os

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_step_accuracy(data):
    correct, error, acc = [], [], []
    
    for problem_id,problem in enumerate(data[:50]):
        if problem_id == 13:
            continue
        total_steps = 0
        correct_steps = 0
        total_error_steps = 0
        correct_error_steps = 0
        total_accumulation_error_steps = 0
        correct_accumulation_error_steps = 0
        step_eval = []

        for step_id, step_class in enumerate(problem['step_classifications']):
            ground_truth = [gt.lower() for gt in step_class['ground_truth']] if isinstance(step_class['ground_truth'], list) else [step_class['ground_truth'].lower()]
            verdict_standalone = step_class['verdict_standalone'].lower()
            verdict_contextual = step_class['verdict_contextual'].lower()
            if (verdict_standalone == 'correct' or verdict_standalone == "unknown") and (verdict_contextual == 'correct' or verdict_contextual == "unknown"):
                step_eval.append("correct")
            else:
                step_eval.append("incorrect")
            
            if 'correct' in ground_truth:
                total_steps += 1
                if (verdict_standalone == 'correct' or verdict_standalone == 'unknown') and (verdict_contextual == 'correct' or verdict_contextual == 'unknown'):

                    premises = problem['premise_list'][step_id]
                    if 0 in premises:
                        premises.remove(0)
                    premises = [i-1 for i in premises]
                    for i in premises:
                        if i>=step_id:
                            premises.remove(i)
                    premises = [i for i in premises if i < len(step_eval)]
                    if premises and 'incorrect' in [step_eval[i] for i in premises]:
                        step_eval.pop()
                        step_eval.append('incorrect')
                        continue
                    correct_steps += 1

            elif not (len(ground_truth) == 1 and ground_truth[0] == 'accumulation_error'):
                total_error_steps += 1
                if verdict_standalone == 'correct' and verdict_contextual == 'correct':
                    continue
                else:
                    correct_error_steps += 1
            else:
                total_accumulation_error_steps += 1
                premises = problem['premise_list'][step_id]
                if 0 in premises:
                    premises.remove(0)
                premises = [i-1 for i in premises]
                premises = [i for i in premises if i < len(step_eval)]
                if (verdict_standalone == 'correct' or verdict_standalone == 'unknown') and (verdict_contextual == 'correct' or verdict_contextual == 'unknown') and 'incorrect' in[step_eval[i] for i in premises]:
                    step_eval.pop()
                    step_eval.append('incorrect')
                    correct_accumulation_error_steps += 1
        if total_steps > 0:
            correct.append(correct_steps/total_steps)
        if total_error_steps > 0:
            error.append(correct_error_steps/total_error_steps)
        if total_accumulation_error_steps > 0:
            acc.append(correct_accumulation_error_steps/total_accumulation_error_steps)
    
    print(f"\nAccuracy for correct steps: {np.mean(np.array(correct)):.2%}", len(correct))
    print(f"Accuracy for error steps: {np.mean(np.array(error)):.2%}", len(error))
    print(f"Accuracy for accumulation error steps: {np.mean(np.array(acc)):.2%}", len(acc))
    # return {
    #     'correct_accuracy': correct_steps/total_steps,
    #     'error_accuracy': correct_error_steps/total_error_steps
    # }

def main():
    dir_path = '/home/sagnikm3/dag-llm/eval/error_eval_results/test/'

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        data = load_json_data(file_path)
        print(filename)
        analyze_step_accuracy(data)

    # file_path = '/home/sagnikm3/dag-llm/eval/error_eval_with_premises/gsm8k_positives_perturbed_gpt4o.json'
    # data = load_json_data(file_path)
    # analyze_step_accuracy(data)

if __name__ == "__main__":
    main()
