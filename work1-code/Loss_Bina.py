import torch
import torch.nn as nn

class My_Loss(nn.Module):
    def __init__(self):
        super(My_Loss, self).__init__()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, pred):
        # pred = self.sigmoid(pred)
        bs,ch,h,w = pred.shape
        pred = pred.contiguous().view(bs, -1)
        loss_1 = pred
        loss_2 = 1-pred
        loss = torch.sum(loss_1*loss_2, dim=1)
        return torch.mean(loss)