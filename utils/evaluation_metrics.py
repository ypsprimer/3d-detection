import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
#evalution metrics of Coronary Arteries's segmentation in CT scans
#based on Dice similarity coefficient(DSC) of whole Coronary Arteries and Part Structures
#Part Structures include Aorta Coronary(AC)，Left Anterior Descending Artery(LAD),Left Circumflex Artery(LCX) and Right Coronary Artery(RCA)
#Definition: DSC = 2TP/(2TP + FP +FN) where:
#True Positive(TP):The network predicts that the voxel is part of class ci. The voxel is labelled ci in the ground truth.
#False Positive(FP):The network predicts that the voxel is part of class ci. The voxel is not labelled ci in the ground truth.
#False Negative(FN):The network predicts that the voxel is not part of class ci. The voxel is labelled ci in the ground truth.
class DSCEvaluator():
    def __init__(self,threshold = None):
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(0)
        self.threshold = threshold or 0.5

    def evaluate(self, pre_labels, ori_labels):
        batch_size = pre_labels.size(0)
        class_num = pre_labels.size(1)
        dsc = []
        if class_num == 1:
            for i in range(batch_size):
                out_idx = pre_labels[i]
                label_wh = ori_labels.data[i, 0, :]
                output_sm = self.sigmoid(out_idx)
                ps_wh = output_sm.data[0]
                ps_wh[ps_wh >= self.threshold] = 1
                ps_wh[ps_wh < self.threshold] = 0
                TP_wh = torch.sum(torch.mul(ps_wh,label_wh))
                FP_wh = torch.sum(ps_wh) - TP_wh
                FN_wh = torch.sum(label_wh) - TP_wh
                dsc_wh = 2*TP_wh/(2*TP_wh + FP_wh + FN_wh)
                dsc.append(dsc_wh)
        elif class_num == 2:
            for i in range(batch_size):
                out_idx = pre_labels[i]
                label_wh = ori_labels.data[i, 0, :]
                output_sm = self.softmax(out_idx)
                ps_wh = output_sm.data[1]
                ps_wh[ps_wh >= self.threshold] = 1
                ps_wh[ps_wh < self.threshold] = 0
                TP_wh = torch.sum(torch.mul(ps_wh,label_wh))
                FP_wh = torch.sum(ps_wh) - TP_wh
                FN_wh = torch.sum(label_wh) - TP_wh
                dsc_wh = 2*TP_wh/(2*TP_wh + FP_wh + FN_wh)
                dsc.append(dsc_wh)
        else:
            for i in range(batch_size):
                out_idx = pre_labels[i]
                output_sm = self.softmax(out_idx)
                ps_wh = output_sm.data[0]
                ps_wh[ps_wh <= self.threshold] = 1
                ps_wh[ps_wh > self.threshold] = 0
                label_wh = ori_labels.data[i, 0, :]
                label_wh[label_wh > 0] = 1
                TP_wh = torch.sum(torch.mul(ps_wh,label_wh))
                FP_wh = torch.sum(ps_wh) - TP_wh
                FN_wh = torch.sum(label_wh) - TP_wh
                dsc_wh = 2*TP_wh/(2*TP_wh + FP_wh + FN_wh)
                dsc.append(dsc_wh)
                for j in range(class_num-2):
                    ps_idx = output_sm.data[j+1]
                    ps_idx[ps_idx >= self.threshold] = 1
                    ps_idx[ps_idx < self.threshold] = 0
                    label_idx = ori_labels.data[i, 0, :]
                    label_idx[label_idx == (j+1)] = 1
                    label_idx[label_idx != (j+1)] = 0
                    TP_idx = torch.sum(torch.mul(ps_idx,label_idx))
                    FP_idx = torch.sum(ps_idx) - TP_idx
                    FN_idx = torch.sum(label_idx) - TP_idx
                    dsc_idx = 2*TP_idx/(2*TP_idx + FP_idx + FN_idx)
                    dsc.append(dsc_idx)          
        return dsc

        
    

