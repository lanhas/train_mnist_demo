import numpy as np
import torch


class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.vals = []

    def add(self, val):
        self.vals.append(val)

    def get_avg(self):
        val = np.array(self.vals)
        if len(val.shape) == 1:
            return np.nanmean(val)
        else:
            return np.nanmean(val, axis=0)


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()