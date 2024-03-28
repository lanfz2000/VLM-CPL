"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_evaluate(targets, predictions):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
    # print(reordered_preds)
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    recall = metrics.recall_score(targets.cpu().numpy(), reordered_preds.cpu().numpy(), average='macro')
    f1 = metrics.f1_score(targets.cpu().numpy(), reordered_preds.cpu().numpy(), average='macro')

    return {'ACC': acc, 'f1': f1, 'recall': recall, 'hungarian_match': match, 'reordered_preds':reordered_preds}

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
    # print(num_correct)
    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    print(res)
    return res