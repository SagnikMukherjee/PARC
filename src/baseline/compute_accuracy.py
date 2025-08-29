## averaging for each qs
import json

import os, argparse


parser = argparse.ArgumentParser(description='Evaluate premises and errors in solution steps')
parser.add_argument('--input-folder', type=str, required=True,
                    help='Path to the input JSON file')
args = parser.parse_args()

for filename in os.listdir(args.input_folder):
    file_path = os.path.join(args.input_folder, filename)
    if os.path.isfile(file_path):
        PATH = file_path

        print(f"Processing file: {PATH}")
        with open(PATH, 'r') as f:
            data = json.load(f)
        errors = {}
        for i in range(len(data)):
            ground_truth, predictions = [], []
            for step_evaluation in data[i]['step_evaluations']:
                ground_truth.append(step_evaluation['ground_truth'])
                predictions.append(step_evaluation['classification']['verdict'])

            ground_truth = [[i] if i =='correct'  else i for i in ground_truth] 

            ground_truth = [['error' if item in ('Logical_Inconsistency', 'Mathematical_Error') else item.lower() for item in sublist] for sublist in ground_truth]
            error_types = list(set([item.lower() for sublist in ground_truth for item in sublist]))
            predictions = [i.lower() for i in predictions]
            predictions = ['error' if item in ('logical_inconsistency', 'mathematical_error') else item for item in predictions]

            for error in error_types:
                accuracy = 0
                total = 0
                for i in range(len(ground_truth)):
                    gt = ground_truth[i]
                    pred = predictions[i]
                    if error in gt:
                        total+=1
                        if pred in gt:
                            accuracy += 1
                if error not in errors:
                    errors[error] = []
                errors[error].append(accuracy/total)
    for error in errors:
        print(error, sum(errors[error])/len(errors[error]), len(errors[error]))