import torch
from torch import nn
from torch.nn import functional as F
from .sklosses import *
import numpy as np
import os


class feat_fun1(nn.Module):
    def __init__(self, npool = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, 64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.npool = npool

    def __load__para(self):
        path = '/'+os.path.join(*(__file__.split('/')[:-1]))
        ckpt = torch.load(path+'/res_feat1.ckpt')
        self.load_state_dict(ckpt)

    def forward(self, x):
        output = []
        f = self.conv1(x)
        output.append(f)
        for i in range(self.npool):
            x = F.avg_pool3d(x, 2, 2)
            f = self.conv1(x)
            output.append(f)
        return output

    def __len__(self):
        return self.npool+1


class feat_fun2(nn.Module):
    def __init__(self):
        super().__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.conv1 = nn.Conv3d(1, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(24, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(24, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(24, 32, kernel_size=1)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.load()
        self.relu = nn.ReLU(inplace=True)

    def load(self):
        path = '/'+os.path.join(*(__file__.split('/')[:-1]))
        ckpt = torch.load(path+'/dsbfeat.ckpt')
        self.load_state_dict(ckpt)

    def forward(self, x):
        x = x*2-1
        outs = []
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        outs.append(x)

        x1 = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x1) + self.conv5(x))
        x1 = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x1) + x)
        x = self.maxpool1(x)
        outs.append(x)

        return outs

    def __len__(self):
        return 2


class percep_loss(nn.Module):
    def __init__(self, feat_fun, dis_fun = nn.MSELoss(), pix_loss = BinaryDiceLoss, weights = None, ignore_index=None):
        super().__init__()
        self.feat_fun = feat_fun
        self.dis_fun = dis_fun
        self.ignore_index = ignore_index
        self.pix_loss = pix_loss(ignore_index=ignore_index)
        if weights is None:
            weights = [1]*len(feat_fun)
        if isinstance(weights, int):
            weights = [weights]*len(feat_fun)
        self.weights = weights
        assert  len(weights) == len(feat_fun)

    def forward(self, logit, lab):
        assert logit.shape[1] <= 2
        self.feat_fun.eval()
        ls = ()
        pxloss = self.pix_loss(logit, lab)

        if self.ignore_index is not None:
            lab[lab == self.ignore_index] = 0
        ls += (pxloss.data,)
        total_loss = pxloss+0
        if logit.shape[1] == 1:
            pred = F.sigmoid(logit)
        else:
            pred = F.softmax(logit, dim=1)[:,1:]

        f_pred = self.feat_fun(pred)
        # print(f_pred[0][0,0])
        with torch.no_grad():
            f_lab = self.feat_fun(lab.float())
            # f_lab = [f.detach() for f in f_lab]

        for i, (f1, f2) in enumerate(zip(f_pred, f_lab)):
            floss = self.dis_fun(f1,f2)
            total_loss += floss*self.weights[i]
            ls += (floss.data*self.weights[i],)
            # print(total_loss)
        return total_loss, ls

def ploss1(ignore_index):
    return percep_loss(feat_fun1(), weights=[100,100], pix_loss=BinaryDiceLoss2, ignore_index = ignore_index)

def ploss2(ignore_index):
    return percep_loss(feat_fun2(), weights=[.2,0.04], pix_loss=CELoss, ignore_index = ignore_index)


if __name__ == "__main__":
    truth = np.zeros([1,1,20,20,20], dtype=np.float32)
    truth[:,:,3:7,3:7,3:7] = 1
    pred = np.copy(truth)
    pred[pred==0] = -1
    pred[pred==1] = 1
    truth = torch.from_numpy(truth)
    pred = torch.from_numpy(pred)

    loss = ploss1()
    print(loss(pred, truth))
    pred = pred*100
    print(loss(pred, truth))
