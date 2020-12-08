import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .inplace_abn import ABN, InPlaceABN, InPlaceABNSync
from .deform_conv import  Deform_conv2d,Deform_conv3d
from .layer import Conv3d_ABN


class unet_type1(nn.Module):
    def __init__(self, n_inp = 1, feats = [32, 32, 64, 64, 128, 128, 128], n_pred_p=[1,1,1], n_pred_d=[4,4,4], L_output = [0,1,2],abn = 1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.n_inp = n_inp
        if abn==0:
            abnblock = ABN
        elif abn==1:
            abnblock = InPlaceABN
        else:
            abnblock = InPlaceABNSync

        Conv3d_ABN.define_abn(abnblock)
        self.in_layer = Conv3d_ABN(n_inp, feats[0], kernel_size=5, padding=2)  # 32,320
        self.strides = np.array([[1,1,1], [2,2,2], [4,4,4], [8,8,8], [16,16,16]])
        self.l1 = Conv3d_ABN(feats[0], feats[1], kernel_size=4, stride=2, padding=1)  # 32,160
        self.l2 = Conv3d_ABN(feats[1], feats[2], kernel_size=4, stride=2, padding=1)  # 32,80
        self.l3 = Conv3d_ABN(feats[2], feats[3], kernel_size=4, stride=2, padding=1)  # 16,40
        self.l4 = Conv3d_ABN(feats[3], feats[4], kernel_size=4, stride=2, padding=1)  # 8,20
        self.l5 = Conv3d_ABN(feats[4], feats[5], kernel_size=4, stride=2, padding=1)  # 4,10
        self.l6 = Conv3d_ABN(feats[5], feats[6], kernel_size=4, stride=2, padding=1)  # 2, 5
        # self.lconvlayer7 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # 2, 5
        # self.lconvlayer7_bn = InPlaceABN(256)
        #
        # self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.r6 = Conv3d_ABN(feats[6], feats[6], kernel_size=3, padding=1)
        self.rt6 = Conv3d_ABN(feats[6], feats[5], kernel_size=3, padding=1)

        self.r5 = Conv3d_ABN(feats[5], feats[5], kernel_size=3, padding=1)
        self.rt5 = Conv3d_ABN(feats[5], feats[4], kernel_size=3, padding=1)

        self.r4 = Conv3d_ABN(feats[4], feats[4], kernel_size=3, padding=1)
        self.rt4 = Conv3d_ABN(feats[4], feats[3], kernel_size=3, padding=1)

        self.r3 = Conv3d_ABN(feats[3], feats[3], kernel_size=3, padding=1)
        self.rt3 = Conv3d_ABN(feats[3], feats[2], kernel_size=3, padding=1)

        self.r2 = Conv3d_ABN(feats[2], feats[2], kernel_size=3, padding=1)
        self.rt2 = Conv3d_ABN(feats[2], feats[1], kernel_size=3, padding=1)

        self.r1 = Conv3d_ABN(feats[1], feats[1], kernel_size=3, padding=1)
        self.rt1 = Conv3d_ABN(feats[1], feats[0], kernel_size=3, padding=1)

        self.r0 = Conv3d_ABN(feats[0], feats[0], kernel_size=3, padding=1)
        self.out_layer = nn.Conv3d(feats[0], 1, kernel_size=1, stride=1)

        for c_out1,c_out2, out in zip(n_pred_p, n_pred_d, L_output):
            setattr(self, 'out'+str(out)+'p', nn.Conv3d(feats[out], c_out1, kernel_size=3, padding=1))
            setattr(self, 'out'+str(out)+'d', nn.Conv3d(feats[out], c_out2, kernel_size=3, padding=1))
        self.L_output = L_output
        self.strides = self.strides[np.array(L_output)]

    def forward(self, x):

        outputs = []
        x0 = self.in_layer(x)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)
        #x6 = self.l6(x5)

        #x6 = self.r6(x6)
        #x6 = self.rt6(self.upsample(x6))

        #x5 = torch.add(x6, x5)
        x5 = self.r5(x5)
        x5 = self.rt5(self.upsample(x5))

        x4 = torch.add(x5, x4)
        x4 = self.r4(x4)
        if 4 in self.L_output:
            outputs.append(self.out4p(x4))
            outputs.append(self.out4d(x4))
        x4 = self.rt4(self.upsample(x4))

        x3 = torch.add(x4, x3)
        x3 = self.r3(x3)
        if 3 in self.L_output:
            outputs.append(self.out3p(x3))
            outputs.append(self.out3d(x3))
        x3 = self.rt3(self.upsample(x3))

        x2 = torch.add(x3, x2)
        x2 = self.r2(x2)
        if 2 in self.L_output:
            outputs.append(self.out2p(x2))
            outputs.append(self.out2d(x2))
        x2 = self.rt2(self.upsample(x2))

        x1 = torch.add(x2, x1)
        x1 = self.r1(x1)
        if 1 in self.L_output:
            outputs.append(self.out1p(x1))
            outputs.append(self.out1d(x1))
        x1 = self.rt1(self.upsample(x1))

        x0 = torch.add(x1, x0)
        x0 = self.r0(x0)
        if 0 in self.L_output:
            outputs.append(self.out0p(x0))
            outputs.append(self.out0d(x0))

        return tuple(outputs)
#         out_layer = self.out_layer(x0)



#         return out_layer



class fpn_a(unet_type1):
    def __init__(self, config, abn=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        super().__init__(n_inp = 1, feats = [32, 32, 64, 64, 128, 128, 128], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output)

class fpn_a_sync(fpn_a):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)




