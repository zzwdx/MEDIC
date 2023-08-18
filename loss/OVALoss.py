import torch 
from torch import nn
from torch.nn import functional as F

class OVALoss(nn.Module):
    def __init__(self):
        super(OVALoss, self).__init__()

    def forward(self, input, label):
        assert len(input.size()) == 3
        assert input.size(1) == 2

        input = F.softmax(input, 1)
        label_p = torch.zeros((input.size(0),
                           input.size(2))).long().cuda()
        label_range = torch.range(0, input.size(0) - 1).long()
        label_p[label_range, label] = 1
        label_n = 1 - label_p
        open_loss_pos = torch.mean(torch.sum(-torch.log(input[:, 1, :]
                                                    + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(input[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
        return 0.5*(open_loss_pos + open_loss_neg)
