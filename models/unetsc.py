import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
try:
    from .inplace_abn import ABN, InPlaceABN, InPlaceABNSync
    from .layer import Conv3d_ABN,Bottleneck_ABN
except:
    from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
    from layer import Conv3d_ABN,Bottleneck_ABN


def build_layer(f1, f2, N_layer=0, stride=1, skip=True):
    if stride ==1:
        if f1 == f2 and skip:
            layers = []
        else:
            layers = [Conv3d_ABN(f1, f2, kernel_size=3, stride=stride, padding=1)]
    elif stride == 2:
        layers = [Conv3d_ABN(f1, f2, kernel_size=4, stride=stride, padding=1)]
    else:
        layers = [Conv3d_ABN(f1, f2, kernel_size=3, stride=stride, padding=1)]

    for i in range(N_layer):
        layers.append(Bottleneck_ABN(f2))
    return nn.Sequential(*layers)


class unet_type2(nn.Module):
    def __init__(self, n_inp = 1, feats = [32, 64, 64, 128, 128, 256, 256], blocks=[2,2,2,2,2,2,2], n_pred_p=[1,1,1], n_pred_d=[4,4,4], L_output = [0,1,2], abn = 1, dropout_ratio = 0.0):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.n_inp = n_inp
        if abn==0:
            abnblock = ABN
        elif abn==1:
            abnblock = InPlaceABN
        else:
            abnblock = InPlaceABNSync

        Conv3d_ABN.define_abn(abnblock)
        Bottleneck_ABN.define_abn(abnblock)

        self.strides = np.array([[1,1,1], [2,2,2], [4,4,4], [8,8,8], [16,16,16]])
        for i in range(6):
            if i==0:
                setattr(self, 'l'+str(i), build_layer(n_inp, feats[i], blocks[i]))
                setattr(self, 'r'+str(i), build_layer(feats[i], feats[i], blocks[i], skip=True))
            else:
                setattr(self, 'l'+str(i), build_layer(feats[i-1], feats[i], blocks[i], stride=2))
                setattr(self, 'r'+str(i), build_layer(feats[i], feats[i], blocks[i], skip=True))
                setattr(self, 'rt'+str(i), build_layer(feats[i], feats[i-1], blocks[i]))


        self.r0 = Conv3d_ABN(feats[0], feats[0], kernel_size=3, padding=1)
        self.out_layer = nn.Conv3d(feats[0], 1, kernel_size=1, stride=1)
        self.drop_out = nn.Dropout3d(p=dropout_ratio)
        for c_out1,c_out2, out in zip(n_pred_p, n_pred_d, L_output):
            setattr(self, 'out'+str(out)+'p', nn.Conv3d(feats[out], c_out1, kernel_size=3, padding=1))
            outd = nn.Conv3d(feats[out], c_out2, kernel_size=3, padding=1)
            #outd.weight.data.fill_(0.0)
            outd.bias.data.fill_(0.0)
            setattr(self, 'out'+str(out)+'d', outd)

        self.L_output = L_output
        self.strides = self.strides[np.array(L_output)]

    def forward(self, x):

        outputs = []
        x0 = self.l0(x)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)


        x5 = self.r5(x5)
        x5 = self.rt5(self.upsample(x5))

        x4 = torch.add(x5, x4)
        x4 = self.r4(x4)
        if 4 in self.L_output:
            x4 = self.drop_out(x4)
            outputs.append(self.out4p(x4))
            outputs.append(self.out4d(x4))
        x4 = self.rt4(self.upsample(x4))

        x3 = torch.add(x4, x3)
        x3 = self.r3(x3)
        if 3 in self.L_output:
            x3 = self.drop_out(x3)
            outputs.append(self.out3p(x3))
            outputs.append(self.out3d(x3))
        x3 = self.rt3(self.upsample(x3))

        x2 = torch.add(x3, x2)
        x2 = self.r2(x2)
        if 2 in self.L_output:
            x2 = self.drop_out(x2)
            outputs.append(self.out2p(x2))
            outputs.append(self.out2d(x2))
        x2 = self.rt2(self.upsample(x2))

        x1 = torch.add(x2, x1)
        x1 = self.r1(x1)
        if 1 in self.L_output:
            x1 = self.drop_out(x1)
            outputs.append(self.out1p(x1))
            outputs.append(self.out1d(x1))
        x1 = self.rt1(self.upsample(x1))

        x0 = torch.add(x1, x0)
        x0 = self.r0(x0)
        if 0 in self.L_output:
            x0 = self.drop_out(x0)
            outputs.append(self.out0p(x0))
            outputs.append(self.out0d(x0))

        return tuple(outputs)



class fpn_c(unet_type2):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [32, 32, 64, 64, 128, 128, 128], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

class fpn_c_sync(fpn_c):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

class fpn_c_c2_sync(fpn_c):
    def __init__(self, config, abn=2):
        super().__init__(config=config, n_inp=2, abn = abn)



class fpn_d(unet_type2):
    def __init__(self, config, abn=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        super().__init__(n_inp = 1, feats = [48, 64, 128, 128, 256, 256, 256], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output)

class fpn_d_sync(fpn_d):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)


