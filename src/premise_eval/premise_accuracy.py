import json
# from sklearn import metrics
# from typing import List, Dict
import os

def compute_precision_recall(ground_truth, prediction):
    # Convert to sets for easier computation
    gt_set = set(ground_truth)
    pred_set = set(prediction)


    # Calculate True Positives, False Positives, False Negatives
    true_positives = len(gt_set & pred_set)  # Intersection
    false_positives = len(pred_set - gt_set) # Prediction but not in ground truth
    false_negatives = len(gt_set - pred_set) # Ground truth but not in prediction

    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


dir_path = '/home/sagnikm3/PARC/outputs'

for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)
    if "gpt" in file_path:
        continue

    with open(file_path, 'r') as f:
        data = json.load(f)

    precision, recall, f1, prune_coeff = [], [], [], []

    for i in range(len(data)):
        ground_truth, predictions = [], []
        question = data[i]

        precision_temp, recall_temp, prune_coeff_temp = [], [], []
        for step in question['premise_annotation']['steps']:
            ground_truth.append(set(step['ground_truth_premises']))

            predicted_premises = [i[0] for i in step["premises"]]
            predictions.append(set(predicted_premises))


        for j, (gt, pred) in enumerate(zip(ground_truth, predictions)):
            if (j<1):
                continue
            pred = set([x for x in pred if x <= j])
            p, r = compute_precision_recall(gt, pred)
            precision_temp.append(p)
            recall_temp.append(r)
            # print(gt, pred, len(pred), j)
            prune_coeff_temp.append(1-(len(pred)/(j+1)))
        
        if precision_temp:
            precision.append(sum(precision_temp) / len(precision_temp))
            recall.append(sum(recall_temp) / len(recall_temp))
            prune_coeff.append(sum(prune_coeff_temp)/ len(recall_temp))

            

    print(file_path)

    print("Precision:", sum(precision) / len(precision))
    print("Recall:", sum(recall) / len(recall))
    print("pruned by:", sum(prune_coeff)/len(prune_coeff))