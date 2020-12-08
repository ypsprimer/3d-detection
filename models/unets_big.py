import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
try:
    from .inplace_abn import ABN, InPlaceABN, InPlaceABNSync
    from .layer import Conv3d_ABN,Bottleneck_ABN,Conv3d_Mish,Bottleneck_Mish,Mish
except:
    from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
    from layer import Conv3d_ABN,Bottleneck_ABN,Conv3d_Mish,Bottleneck_Mish,Mish


def build_layer(f1, f2, N_layer=0, stride=1, skip=True):
    """
    :param f1: 输入通道数
    :param f2: 输入通道数
    :param N_layer: residual blocks数量
    :param skip: 
    
    """
    if stride ==1:
        if f1 == f2 and skip:
            layers = []
        else:
            # conv3x3x3 stride=1
            layers = [Conv3d_ABN(f1, f2, kernel_size=3, stride=stride, padding=1)]
    elif stride == 2:
        # conv4x4x4 stride=2
        layers = [Conv3d_ABN(f1, f2, kernel_size=4, stride=stride, padding=1)]
    else:
        layers = [Conv3d_ABN(f1, f2, kernel_size=3, stride=stride, padding=1)]

    # residual bottleneck * N_layers
    for i in range(N_layer):
        layers.append(Bottleneck_ABN(f2))

    return nn.Sequential(*layers)

def build_layer_Mish(f1, f2, N_layer=0, stride=1, skip=True):
    if stride ==1:
        if f1 == f2 and skip:
            layers = []
        else:
            layers = [Conv3d_Mish(f1, f2, kernel_size=3, stride=stride, padding=1)]
    elif stride == 2:
        layers = [Conv3d_Mish(f1, f2, kernel_size=4, stride=stride, padding=1)]
    else:
        layers = [Conv3d_Mish(f1, f2, kernel_size=3, stride=stride, padding=1)]

    for i in range(N_layer):
        layers.append(Bottleneck_Mish(f2))
    return nn.Sequential(*layers)


# main model
class unet_type3(nn.Module):
    def __init__(self, n_inp = 1, feats = [32, 64, 64, 128, 128, 256, 256], blocks=[2,2,2,2,2,2,2], n_pred_p=[1,1,1], n_pred_d=[4,4,4], L_output = [0,1,2],abn = 1, dropout_ratio=0.0):
        """
        :param n_inp: 输入通道数量 [bs, n_inp, z, y, x]
        :param n_pred_p: 预测类别数量 [1,1,1] 二分类
        :param n_pred_d: 预测位置信息 [4,4,4] 四个坐标值
        :param feats: 通道数量 [32, 64, 64, 128, 128, 256, 256]
        :param blocks: residual block的数量 [2,2,2,2,2,2]
        :param L_output: 输出预测结果的层 [0, 1, 2]

        """
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

        # not used
        self.strides = np.array([[1,1,1], [2,2,2], [4,4,4], [8,8,8], [16,16,16]])
        
        # 6 stages
        for i in range(6):
            # input layer
            if i==0:
                # 4 blocks
                # 3x3 * 3
                setattr(self, 'l'+str(i), build_layer(n_inp, feats[i], blocks[i]))
                # 3x3 * 2
                setattr(self, 'r'+str(i), build_layer(feats[i], feats[i], blocks[i], skip=True))
            
            # other layers
            else:
                # 6 blocks
                # stride 2 downsample
                setattr(self, 'l'+str(i), build_layer(feats[i-1], feats[i], blocks[i], stride=2))
                setattr(self, 'r'+str(i), build_layer(feats[i], feats[i], blocks[i], skip=True))
                # add after upsampled (keep dim with previous block)
                setattr(self, 'rt'+str(i), build_layer(feats[i], feats[i-1], blocks[i]))

        self.r0 = Conv3d_ABN(feats[0], feats[0], kernel_size=3, padding=1)
        self.out_layer = nn.Conv3d(feats[0], 1, kernel_size=1, stride=1)
        self.drop_out = nn.Dropout3d(p=dropout_ratio)

        # 不同layer的预测类别和位置信息
        for c_out1,c_out2, out in zip(n_pred_p, n_pred_d, L_output):
            # outp = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outp = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'p', outp)
            # outd = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outd = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'d', outd)

        for c_out1,c_out2, out in zip(n_pred_p, n_pred_d, L_output):
            setattr(self, 'out'+str(out)+'p', nn.Conv3d(feats[out], c_out1, kernel_size=3, padding=1))
            outd = nn.Conv3d(feats[out], c_out2, kernel_size=3, padding=1)
            # outd.weight.data.fill_(0.0)
            outd.bias.data.fill_(0.0)
            setattr(self, 'out'+str(out)+'d', outd)

        self.L_output = L_output

        # not used
        self.strides = self.strides[np.array(L_output)]

    def forward(self, x):
        """
        越靠近top，感受野越大，检测物体越大
        共有5个level: [0,1,2,3,4,]

        感受野计算：
        """
        # bottom -> top: 1 -> ... -> 5
        outputs = []
        x0 = self.l0(x)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)

        # top -> bottom: 5 -> upsample add -> ... -> unsample add -> 1
        x5 = self.r5(x5)
        x5 = self.rt5(self.upsample(x5))

        x4 = torch.add(x5, x4)
        x4 = self.r4(x4)
        if 4 in self.L_output:
            x4 = self.drop_out(x4)
            x4p = self.before_out3p(x4)
            x4d = self.before_out3d(x4)
            outputs.append(self.out4p(F.relu(x4p)))
            outputs.append(self.out4d(F.relu(x4d)))
        x4 = self.rt4(self.upsample(x4))

        x3 = torch.add(x4, x3)
        x3 = self.r3(x3)
        if 3 in self.L_output:
            x3 = self.drop_out(x3)
            x3p = self.before_out3p(x3)
            x3d = self.before_out3d(x3)
            outputs.append(self.out3p(F.relu(x3p)))
            outputs.append(self.out3d(F.relu(x3d)))
        x3 = self.rt3(self.upsample(x3))

        x2 = torch.add(x3, x2)
        x2 = self.r2(x2)
        if 2 in self.L_output:
            x2 = self.drop_out(x2)
            x2p = self.before_out2p(x2)
            x2d = self.before_out2d(x2)
            outputs.append(self.out2p(F.relu(x2p)))
            outputs.append(self.out2d(F.relu(x2d)))
        x2 = self.rt2(self.upsample(x2))

        x1 = torch.add(x2, x1)
        x1 = self.r1(x1)
        if 1 in self.L_output:
            x1 = self.drop_out(x1)
            x1p = self.before_out1p(x1)
            x1d = self.before_out1d(x1)
            outputs.append(self.out1p(F.relu(x1p)))
            outputs.append(self.out1d(F.relu(x1d)))
        x1 = self.rt1(self.upsample(x1))

        x0 = torch.add(x1, x0)
        # 3x3 * 2
        x0 = self.r0(x0)
        if 0 in self.L_output:
            x0 = self.drop_out(x0)
            # 3x3
            x0p = self.before_out0p(x0)
            x0d = self.before_out0d(x0)
            # 3x3
            outputs.append(self.out0p(F.relu(x0p)))
            outputs.append(self.out0d(F.relu(x0d)))

        return tuple(outputs)
    
    
class unet_type3_4l(nn.Module):
    def __init__(self, n_inp = 1, feats = [32, 64, 64, 128, 128, 256, 256], blocks=[2,2,2,2,2,2,2], n_pred_p=[1,1,1], n_pred_d=[4,4,4], L_output = [0,1,2],abn = 1, dropout_ratio=0.0):
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
        for i in range(4):
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
            #outp = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outp = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'p', outp)
            #outd = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outd = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'d', outd)

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
#         x4 = self.l4(x3)
#         x5 = self.l5(x4)


#         x5 = self.r5(x5)
#         x5 = self.rt5(self.upsample(x5))

#         x4 = torch.add(x5, x4)
#         x4 = self.r4(x4)
#         if 4 in self.L_output:
#             x4 = self.drop_out(x4)
#             x4p = self.before_out3p(x4)
#             x4d = self.before_out3d(x4)
#             outputs.append(self.out4p(F.relu(x4p)))
#             outputs.append(self.out4d(F.relu(x4d)))
#         x4 = self.rt4(self.upsample(x4))

#         x3 = torch.add(x4, x3)
        x3 = self.r3(x3)
        if 3 in self.L_output:
            x3 = self.drop_out(x3)
            x3p = self.before_out3p(x3)
            x3d = self.before_out3d(x3)
            outputs.append(self.out3p(F.relu(x3p)))
            outputs.append(self.out3d(F.relu(x3d)))
        x3 = self.rt3(self.upsample(x3))

        x2 = torch.add(x3, x2)
        x2 = self.r2(x2)
        if 2 in self.L_output:
            x2 = self.drop_out(x2)
            x2p = self.before_out2p(x2)
            x2d = self.before_out2d(x2)
            outputs.append(self.out2p(F.relu(x2p)))
            outputs.append(self.out2d(F.relu(x2d)))
        x2 = self.rt2(self.upsample(x2))

        x1 = torch.add(x2, x1)
        x1 = self.r1(x1)
        if 1 in self.L_output:
            x1 = self.drop_out(x1)
            x1p = self.before_out1p(x1)
            x1d = self.before_out1d(x1)
            outputs.append(self.out1p(F.relu(x1p)))
            outputs.append(self.out1d(F.relu(x1d)))
        x1 = self.rt1(self.upsample(x1))

        x0 = torch.add(x1, x0)
        x0 = self.r0(x0)
        if 0 in self.L_output:
            x0 = self.drop_out(x0)
            x0p = self.before_out0p(x0)
            x0d = self.before_out0d(x0)
            outputs.append(self.out0p(F.relu(x0p)))
            outputs.append(self.out0d(F.relu(x0d)))

        return tuple(outputs)    
    
class unet_HR(nn.Module):
    def __init__(self, n_inp = 1, feats = [16, 24, 40, 80, 112, 192, 320], blocks=[4,3,3,3,3,3,3], n_pred_p=[1,1,1], n_pred_d=[4,4,4], L_output = [0,1,2],abn = 1, dropout_ratio=0.0):
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

        self.strides = np.array([[1,1,1], [2,2,2], [4,4,4], [8,8,8]])
        for i in range(4):
            if i==0:
                setattr(self, 'l'+str(i), build_layer(n_inp, feats[i], blocks[i]))
                setattr(self, 'r'+str(i), build_layer(feats[i], feats[i], blocks[i]))
            else:
                setattr(self, 'l'+str(i), build_layer(feats[i-1], feats[i], blocks[i], stride=2))
                setattr(self, 'r'+str(i), build_layer(feats[i], feats[i], blocks[i], skip=True))
                setattr(self, 'rt'+str(i), build_layer(feats[i], feats[i-1], blocks[i]))


#         self.r0 = Conv3d_ABN(feats[0], feats[0], kernel_size=3, padding=1)
        self.out_layer = nn.Conv3d(feats[0], 1, kernel_size=1, stride=1)
        self.drop_out = nn.Dropout3d(p=dropout_ratio)
        for c_out1,c_out2, out in zip(n_pred_p, n_pred_d, L_output):
            #outp = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outp = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'p', outp)
            #outd = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outd = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'d', outd)

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
#         x4 = self.l4(x3)
#         x5 = self.l5(x4)


#         x5 = self.r5(x5)
#         x5 = self.rt5(self.upsample(x5))

#         x4 = torch.add(x5, x4)
#         x4 = self.r4(x4)
#         if 4 in self.L_output:
#             x4 = self.drop_out(x4)
#             x4p = self.before_out3p(x4)
#             x4d = self.before_out3d(x4)
#             outputs.append(self.out4p(F.relu(x4p)))
#             outputs.append(self.out4d(F.relu(x4d)))
#         x4 = self.rt4(self.upsample(x4))

#         x3 = torch.add(x4, x3)
        x3 = self.r3(x3)
        if 3 in self.L_output:
            x3 = self.drop_out(x3)
            x3p = self.before_out3p(x3)
            x3d = self.before_out3d(x3)
            outputs.append(self.out3p(F.relu(x3p)))
            outputs.append(self.out3d(F.relu(x3d)))
        x3 = self.rt3(self.upsample(x3))

        x2 = torch.add(x3, x2)
        x2 = self.r2(x2)
        if 2 in self.L_output:
            x2 = self.drop_out(x2)
            x2p = self.before_out2p(x2)
            x2d = self.before_out2d(x2)
            outputs.append(self.out2p(F.relu(x2p)))
            outputs.append(self.out2d(F.relu(x2d)))
        x2 = self.rt2(self.upsample(x2))

        x1 = torch.add(x2, x1)
        x1 = self.r1(x1)
        if 1 in self.L_output:
            x1 = self.drop_out(x1)
            x1p = self.before_out1p(x1)
            x1d = self.before_out1d(x1)
            outputs.append(self.out1p(F.relu(x1p)))
            outputs.append(self.out1d(F.relu(x1d)))
        x1 = self.rt1(self.upsample(x1))

        x0 = torch.add(x1, x0)
        x0 = self.r0(x0)
        if 0 in self.L_output:
            x0 = self.drop_out(x0)
            x0p = self.before_out0p(x0)
            x0d = self.before_out0d(x0)
            outputs.append(self.out0p(F.relu(x0p)))
            outputs.append(self.out0d(F.relu(x0d)))

        return tuple(outputs)
    
class unet_type3_Mish(nn.Module):
    def __init__(self, n_inp = 1, feats = [32, 64, 64, 128, 128, 256, 256], blocks=[2,2,2,2,2,2,2], n_pred_p=[1,1,1], n_pred_d=[4,4,4], L_output = [0,1,2],abn = 1, dropout_ratio=0.0):
        super().__init__()

        self.relu = Mish()
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
                setattr(self, 'l'+str(i), build_layer_Mish(n_inp, feats[i], blocks[i]))
                setattr(self, 'r'+str(i), build_layer_Mish(feats[i], feats[i], blocks[i], skip=True))
            else:
                setattr(self, 'l'+str(i), build_layer_Mish(feats[i-1], feats[i], blocks[i], stride=2))
                setattr(self, 'r'+str(i), build_layer_Mish(feats[i], feats[i], blocks[i], skip=True))
                setattr(self, 'rt'+str(i), build_layer_Mish(feats[i], feats[i-1], blocks[i]))


        self.r0 = Conv3d_Mish(feats[0], feats[0], kernel_size=3, padding=1)
        self.out_layer = nn.Conv3d(feats[0], 1, kernel_size=1, stride=1)
        self.drop_out = nn.Dropout3d(p=dropout_ratio)
        for c_out1,c_out2, out in zip(n_pred_p, n_pred_d, L_output):
            #outp = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outp = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'p', outp)
            #outd = build_layer(feats[out], feats[out] // 4, 0, skip=False)
            outd = nn.Conv3d(feats[out], feats[out], kernel_size=3, padding=1)
            setattr(self, 'before_out'+str(out)+'d', outd)

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
            x4p = self.before_out3p(x4)
            x4d = self.before_out3d(x4)
            outputs.append(self.out4p(self.relu(x4p)))
            outputs.append(self.out4d(self.relu(x4d)))
        x4 = self.rt4(self.upsample(x4))

        x3 = torch.add(x4, x3)
        x3 = self.r3(x3)
        if 3 in self.L_output:
            x3 = self.drop_out(x3)
            x3p = self.before_out3p(x3)
            x3d = self.before_out3d(x3)
            outputs.append(self.out3p(self.relu(x3p)))
            outputs.append(self.out3d(self.relu(x3d)))
        x3 = self.rt3(self.upsample(x3))

        x2 = torch.add(x3, x2)
        x2 = self.r2(x2)
        if 2 in self.L_output:
            x2 = self.drop_out(x2)
            x2p = self.before_out2p(x2)
            x2d = self.before_out2d(x2)
            outputs.append(self.out2p(self.relu(x2p)))
            outputs.append(self.out2d(self.relu(x2d)))
        x2 = self.rt2(self.upsample(x2))

        x1 = torch.add(x2, x1)
        x1 = self.r1(x1)
        if 1 in self.L_output:
            x1 = self.drop_out(x1)
            x1p = self.before_out1p(x1)
            x1d = self.before_out1d(x1)
            outputs.append(self.out1p(self.relu(x1p)))
            outputs.append(self.out1d(self.relu(x1d)))
        x1 = self.rt1(self.upsample(x1))

        x0 = torch.add(x1, x0)
        x0 = self.r0(x0)
        if 0 in self.L_output:
            x0 = self.drop_out(x0)
            x0p = self.before_out0p(x0)
            x0d = self.before_out0d(x0)
            outputs.append(self.out0p(self.relu(x0p)))
            outputs.append(self.out0d(self.relu(x0d)))

        return tuple(outputs)



class fpn_HR(unet_HR):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [16, 24, 40, 80, 112, 192, 320], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

class fpn_HR_sync(fpn_HR):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

        
class fpn_big(unet_type3):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']

        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1

        # 每种scale下anchor的数量
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [32, 32, 64, 64, 128, 128, 128], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

# bn sync
class fpn_big_sync(fpn_big):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

class fpn_big_4l(unet_type3_4l):
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

class fpn_big_4l_sync(fpn_big_4l):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)
        
class fpn_small_ori(unet_type3):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [16, 32, 64, 64, 128, 128, 128], blocks=[0,1,2,2,2,2,2], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

class fpn_small_ori_sync(fpn_small_ori):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

class fpn_big_subw(unet_type3):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [16, 32, 64, 64, 128, 128, 128], blocks=[2,2,2,2,2,2,2], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

class fpn_big_subw_sync(fpn_big_subw):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

class fpn_big_subd(unet_type3):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [32, 32, 64, 64, 128, 128, 128], blocks=[1,2,2,2,2,2,2], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

class fpn_big_subd_sync(fpn_big_subd):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

class fpn_small_Mish(unet_type3_Mish):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        dropout_ratio = config.net['dropout']
        super().__init__(n_inp = n_inp, feats = [8, 8, 16, 16, 32, 32, 32], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output, dropout_ratio=dropout_ratio)

class fpn_small_Mish_sync(fpn_small_Mish):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)


class fpn_big3(unet_type3):
    def __init__(self, config, abn=1, n_inp=1):
        L_output = config.rpn['layer_output']
        N_cls = config.classifier['N_cls']
        if config.classifier['activation'] == 'softmax':
            N_cls+=1
        n_anchors = [len(a) for a in config.rpn['anchors']]
        n_pred_p = [a*N_cls for a in n_anchors]
        n_pred_d = [a*4 for a in n_anchors]
        super().__init__(n_inp = n_inp, feats = [32, 32, 64, 64, 128, 128, 128], blocks=[3,3,3,3,3,3,3], abn = abn, n_pred_p= n_pred_p, n_pred_d=n_pred_d, L_output=L_output)

class fpn_big3_sync(fpn_big3):
    def __init__(self, config, abn=2):
        super().__init__(config=config, abn = abn)

