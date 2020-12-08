import torch
from torch import nn
import numpy as np


def neg(idx):
    if idx == 0:
        return None
    else:
        return -idx


class warpLoss(nn.Module):
    def __init__(self, net, loss_fun, margin, emlist=None):
        super(warpLoss, self).__init__()
        self.net = net
        self.loss_fun = loss_fun
        self.emlist = emlist
        self.margin = margin
    def forward(self, x, *args, calc_loss=True):
        logit = self.net(x)
        # print(logit[0].shape)
        # print(args[0][0].shape)
        # exit()
        
        logit = [l.float() for l in logit]
        if not calc_loss:
             if isinstance(logit, torch.Tensor):
                 return logit.detach()
             else:
                 return [l.detach() for l in logit]

        outputs, weights, pred_prob_list = self.loss_fun(logit,*args)
        return outputs, weights, pred_prob_list


if __name__ == "__main__":
    from perceptual_loss import ploss1
    from sklosses import DiceLossNew
    import numpy as np
    net = lambda x: x
    warp = warpLoss(net, ploss1(), DiceLossNew())

    truth = np.zeros([3,1,20,20,20], dtype=np.float32)
    truth[:,:,3:7,3:7,3:7] = 1
    pred = np.copy(truth)
    pred[pred==0] = -10
    pred[pred==1] = 1
    truth = torch.from_numpy(truth)
    pred = torch.from_numpy(pred)

    print(warp(pred, truth, runem=True))
