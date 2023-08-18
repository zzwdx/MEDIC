import torch 
from torch import nn
from torch.nn import functional as F

class CrossEntropyOH(nn.Module):
    def __init__(self):
        super(CrossEntropyOH, self).__init__()

    def forward(self, input, label):
        log_prob = F.log_softmax(input, dim=1)
        loss = -torch.sum(log_prob * label) / len(input)
        return loss