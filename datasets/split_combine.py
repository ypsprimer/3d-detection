import torch
import numpy as np
from torchvision.ops import nms3d

def mypad(x, pad, padv):
    def neg(b):
        if b==0:
            return None
        else:
            return -b
    pad = np.array(pad)
    x2 = np.zeros(np.array(x.shape)+np.sum(pad,1), dtype = x.dtype)+padv
    x2[pad[0,0]:neg(pad[0,1]), pad[1,0]:neg(pad[1,1]), pad[2,0]:neg(pad[2,1]), pad[3,0]:neg(pad[3,1])] = x
    return x2


class Data_bysplit():
    def __init__(self, data, nzhw, crop_size, side_len, shape_post):
        self.data = data
        self.nzhw = nzhw
        self.crop_size = crop_size
        self.side_len = side_len
        self.shape_post = shape_post
        izhw_list = []
        for iz in range(nzhw[0]):
            for ih in range(nzhw[1]):
                for iw in range(nzhw[2]):
                    izhw_list.append([iz, ih, iw])
        self.izhw_list = izhw_list
    def getse(self, izhw):
        se = []
        for i, n ,crop, side, shape in zip(izhw, self.nzhw, self.crop_size, self.side_len, self.shape_post):
            if i == n-1 and i > 0:
                e = shape
                s = e - crop
            else:
                s = i * side
                e = s + crop
            se += [s,e]
        return se

    def __getitem__(self, id):
        if isinstance(id, slice):
            result = []
            for idx in list(range(self.__len__())[id]):
                result.append(self.__getitem__(idx))
            return result
        iz,ih,iw = self.izhw_list[id]
        sz, ez, sh, eh, sw, ew = self.getse([iz,ih,iw])
        split = self.data[:, sz:ez, sh:eh, sw:ew]
        return split

    def __len__(self):
        return np.prod(self.nzhw)

class SplitComb():
    def __init__(self, config):
        margin = np.array(config.prepare['margin'])
        side_len = np.array(config.prepare['crop_size']) - margin * 2
        stride = config.prepare['seg_stride']
        self.pad_mode = config.prepare['pad_mode']
        self.pad_value = config.prepare['pad_value']

        if isinstance(side_len, int):
            side_len = [side_len] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(margin, int):
            margin = [margin] * 3

        self.side_len = np.array(side_len)
        self.stride = np.array(stride)
        self.margin = np.array(margin)

    @staticmethod
    def getse(izhw, nzhw, crop_size, side_len, shape_post):
        se = []
        for i, n ,crop, side, shape in zip(izhw, nzhw, crop_size, side_len, shape_post):
            if i == n-1 and i > 0:
                e = shape
                s = e - crop
            else:
                s = i * side
                e = s + crop
            se += [s,e]
        return se

    @staticmethod
    def getse2(izhw, nzhw, crop_size, side_len, shape_len):
        se = []
        for i, n ,crop, side, shape in zip(izhw, nzhw, crop_size, side_len, shape_len):
            if i == n-1 and i > 0:
                e = shape
                s = e - side
            else:
                s = i * side
                e = s + side
            se += [s,e]
        return se


    def split(self, data, side_len=None, margin=None):
        if side_len is None:
            side_len = self.side_len
        if margin is None:
            margin = self.margin
        crop_size = side_len+margin*2

        assert (np.all(side_len > margin))

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len[0]))
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        shape_pre = [z, h, w]

        pad = [[0, 0],
               [margin[0], np.max([margin[0], crop_size[0]-z-margin[0]])],
               [margin[1], np.max([margin[1], crop_size[1]-h-margin[1]])],
               [margin[2], np.max([margin[2], crop_size[2]-w-margin[2]])]]
        # print(data.shape)
        # print(side_len[1])
        # print(side_len[1]-h-margin[1])
        # print(pad)
        if self.pad_mode == 'constant':
            data = mypad(data, pad, self.pad_value)
        else:
            data = np.pad(data, pad, self.pad_mode)
        shape_post = list(data.shape[1:])
        shapes = np.array([shape_pre, shape_post])
        self.shapes = shapes
        # split_data = Data_bysplit(data, [nz,nh,nw], crop_size, side_len, shape_post)
        # splits = np.zeros((nz*nh*nw, data.shape[1], crop_size[0], crop_size[1], crop_size[2]), dtype = data.dtype)
        splits = []
        id = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz, ez, sh, eh, sw, ew = self.getse([iz,ih,iw], [nz,nh,nw], crop_size, side_len, shape_post)
                    # print(sz, ez, sh, eh, sw, ew)
                    splits.append(data[:, sz:ez, sh:eh, sw:ew])
                    id += 1
        splits = (np.array(splits))
        return splits, shapes

    def combine(self, output, shapes=None, side_len=None, stride=None, margin=None):
        comb_result = []
        if side_len is None:
            side_len = self.side_len
        if stride is None:
            stride = self.stride
        if margin is None:
            margin = self.margin
        if shapes is None:
            shape = self.shapes

        shape_pre, shape_post = shapes
        shape_pre = shape_pre.numpy()
        z,h,w = shape_pre
        nz = int(np.ceil(float(z) / side_len[0]))
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        assert (np.all(side_len % stride == 0))
        assert (np.all(margin % stride == 0))

        newshape = (np.array([z, h, w])/ stride).astype(np.int)
        side_len = (self.side_len / stride).astype(np.int)
        margin = (self.margin / stride).astype(np.int)
        crop_size = side_len+margin*2

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz, ez, sh, eh, sw, ew = self.getse2([iz,ih,iw], [nz,nh,nw], crop_size, side_len, shape_pre)
#                     print(sz, ez, sh, eh, sw, ew)
                    bias = torch.tensor([sz, sh, sw, sz, sh, sw]).float().unsqueeze(0).float().cuda()
                    result_split = output[idx]
                    if len(result_split)>0:
                        result_split = result_split.clone()
                        result_split[:,:6] += bias
                        comb_result.append(result_split)
                    
                    idx += 1
        if len(comb_result)>0:
            comb_result = torch.cat(comb_result, dim=0)
            keep = nms3d(comb_result[:,:6], comb_result[:,6],iou_threshold=0.1)
            comb_result = comb_result[keep]
        
        return comb_result
