import json
import sys
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process evaluation data')
parser.add_argument('--input-file', required=True, help='Path to input JSON file')
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    data = json.load(f)

# Process correct answers
data_correct = [i for i in data if i["final_answer_correct"]]
incorrect = 0
for j,i in enumerate(data_correct):
    predictions_standalone = [j['standalone_verdict'] for j in i["error_annotation"]["step_annotations"]]
    predictions_contextual = [j['contextual_verdict'] for j in i["error_annotation"]["step_annotations"]]
    print(j, set(predictions_contextual), set(predictions_standalone))

    if 'logical_inconsistency' in predictions_contextual or 'mathematical_error' in predictions_contextual:
        incorrect += 1
    
    elif 'logical_inconsistency' in predictions_standalone or 'mathematical_error' in predictions_standalone:
        incorrect += 1
correct_accuracy = 1 - incorrect/len(data_correct)
print(f"Accuracy for correct answers: {correct_accuracy:.3f}")

# Process incorrect answers
data_wrong = [i for i in data if not i["final_answer_correct"]]
correct = 0
for i in data_wrong:
    predictions_standalone = [j['standalone_verdict'] for j in i["error_annotation"]["step_annotations"]]
    predictions_contextual = [j['contextual_verdict'] for j in i["error_annotation"]["step_annotations"]]

    label = i['label']

    try:
        pred_contextual = next(i for i, pred in enumerate(predictions_contextual) if pred != 'correct')
    except Exception as e:
        pred_contextual = len(predictions_contextual)

    try:
        predictions_standalone = next(i for i, pred in enumerate(predictions_standalone) if pred != 'correct')
    except Exception as e:
        predictions_standalone = len(predictions_standalone)

    pred = min(pred_contextual, predictions_standalone)

    if pred == label:
        correct += 1

wrong_accuracy = correct/len(data_wrong)
print(f"Accuracy for incorrect answers: {wrong_accuracy:.3f}")