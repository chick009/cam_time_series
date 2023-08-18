import torch
from sklearn.metrics import f1_score, recall_score, precision_score

def evaluate_metrics(pred_label, test_label):
    prediction_tensor = torch.tensor(pred_label)
    real_tensor = torch.tensor(test_label)

    metrics = dict()

    metrics['F1 Score'] = f1_score(real_tensor, prediction_tensor, average='binary')
    metrics['recall'] = recall_score(real_tensor, prediction_tensor, average='binary')
    metrics['precision'] = precision_score(real_tensor, prediction_tensor, average='binary')

    print("The F1 Score is: ", metrics['F1 Score'])
    print("The Recall Rate is: ", metrics['recall'])
    print("The Precision Rate is: ", metrics['precision'])

    return metrics