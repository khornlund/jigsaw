import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def BCEWithLogitsLossMean(output, target):
    return nn.BCEWithLogitsLoss(reduction='mean')
