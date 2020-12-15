# %load losses/rpn_loss.py
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

def neg(x):
    if x==0:
        return None
    else:
        return -x

class Focal_loss(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178

        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]
#         pos_ms = []
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
            prob_pred = F.sigmoid(logit_pred)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            if fpn_connects is not None:
                connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)

            negmask = (prob_gt==-1).float()
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                negmask = negmask*max_filter
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
#             print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
            bi = 0            
            for p_pred, c_gt, p_gt in zip(logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if level == 2 and False:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-2:c_gt[c,1]+2,
                                                                  c_gt[c,2]-2:c_gt[c,2]+2,
                                                                  c_gt[c,3]-2:c_gt[c,3]+2])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
#                     if level == N_level-1:
#                         pos_ms.append(prob_pos)
#                     
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None and False:
                        connects_tags = connects_gt[bi][c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            weight[weight_select] = torch.mean(weight[weight_select])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                bi += 1
#             if level == N_level-1:
#         print(pos_ms,torch.std_mean(pos_ms))
#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)

class Focal_loss_level(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.ones(1,device=output[0].device)
        count_pos = torch.ones(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
#         fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
#         fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178

        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]
#         pos_ms = []
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
            prob_pred = F.sigmoid(logit_pred)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            if fpn_connects is not None:
                connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)

            negmask = (prob_gt==-1).float()
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                negmask = negmask*max_filter
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += (torch.sum(nll*weight)/torch.sum(weight))
#             print('neg',level,torch.sum((negmask>0).float()))
#             count_neg+=torch.sum(negmask)
            bi = 0            
            for p_pred, c_gt, p_gt in zip(logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if level == 2 and False:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-2:c_gt[c,1]+2,
                                                                  c_gt[c,2]-2:c_gt[c,2]+2,
                                                                  c_gt[c,3]-2:c_gt[c,3]+2])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
#                     if level == N_level-1:
#                         pos_ms.append(prob_pos)
#                     
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
#                     weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None and False:
                        connects_tags = connects_gt[bi][c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            weight[weight_select] = torch.mean(weight[weight_select])
                    cls_loss_pos += (-torch.sum(F.logsigmoid(logit_pos)*weight)/torch.sum(weight))
#                     print('pos',level,c_gt.shape[0])
#                     count_pos+=c_gt.shape[0]
                bi += 1
#             if level == N_level-1:
#         print(pos_ms,torch.std_mean(pos_ms))
#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]

class Focal_loss_ignoreneg(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
#         fpn_pos_factor = [1.0, 1.0, 1.0, 1.0, 1.0][-N_level:] #nms_size
#         fpn_neg_factor = [1.0, 1.0, 1.0, 1.0, 1.0][-N_level:] #nms_size
        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
#             print(level,logit_pred.shape)
            prob_pred = F.sigmoid(logit_pred)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            if fpn_connects is not None:
                connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)
            bi = 0
            prob_pos = []
            for p_pred, c_gt, p_gt in zip(logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if level == 2 and False:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-2:c_gt[c,1]+2,
                                                                  c_gt[c,2]-2:c_gt[c,2]+2,
                                                                  c_gt[c,3]-2:c_gt[c,3]+2])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
#                     print(level,'pos',logit_pos.shape)
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None and False:
                        connects_tags = connects_gt[bi][c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            weight[weight_select] = torch.mean(weight[weight_select])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                bi += 1
            negmask = (prob_gt==-1).float()
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                if len(prob_pos) > 0:
                    pos_std,pos_mean = torch.std_mean(prob_pos)
                    factor = (torch.where(prob_pred>pos_mean+1.5*pos_std, torch.zeros_like(prob_pred), torch.ones_like(prob_pred))).detach().float()
#                     max_filter = (F.max_pool3d(prob_pred, padding=1+3*level, kernel_size=3+6*level,stride=1)==prob_pred).detach().float()
                    max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                    negmask = negmask*max_filter*factor
#                     logit_pos_add = logit_pred[prob_pred>pos_mean+1.5*pos_std]
# #                     print(logit_pos_add,logit_pos_add.shape)
#                     prob_pos_add = F.sigmoid(logit_pos_add)
# #                     print(level,'pos',logit_pos.shape)
#                     weight = torch.pow(1-prob_pos_add.detach(),self.alpha)
#                     cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos_add)*weight)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
#                     count_pos+=torch.sum(weight)
                else:
#                     max_filter = (F.max_pool3d(prob_pred, padding=1+3*level, kernel_size=3+6*level,stride=1)==prob_pred).detach().float()
                    max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                    negmask = negmask*max_filter
#             print(level,'neg',torch.sum(negmask))
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
#             print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
    
#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)

class Focal_loss_ignoreneg_max_pos(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms

    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None):
        """
        用于分类
        
        """
        # 
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
#         fpn_pos_factor = [1.0, 1.0, 1.0, 1.0, 1.0][-N_level:] #nms_size
#         fpn_neg_factor = [1.0, 1.0, 1.0, 1.0, 1.0][-N_level:] #nms_size
        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]
        for level in range(N_level):
            # 不同输出层的结果
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            # 置信度
            logit_pred = output[level*2]
            # print(level,logit_pred.shape)
            prob_pred = F.sigmoid(logit_pred)
            # 预测框
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            if fpn_connects is not None:
                connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)
            bi = 0
            prob_pos = []
            
            # 计算阳性
            for p_pred, c_gt, p_gt in zip(logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
                    # print(p_pred.shape, c_gt.shape,p_gt.shape)
                    # print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if True:
                        max_range = level+1
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                # print(c_gt[c,:])
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-max_range:c_gt[c,1]+max_range,
                                                                  c_gt[c,2]-max_range:c_gt[c,2]+max_range,
                                                                  c_gt[c,3]-max_range:c_gt[c,3]+max_range])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
                    # print(level,'pos',logit_pos.shape)
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None and False:
                        connects_tags = connects_gt[bi][c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            weight[weight_select] = torch.mean(weight[weight_select])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
                    # print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                bi += 1
            
            # 阴性框
            negmask = (prob_gt==-1).float()
            # [-,0], 转换完成后全部为+，
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                if len(prob_pos) > 0:
                    pos_std,pos_mean = torch.std_mean(prob_pos)
                    factor = (torch.where(prob_pred>pos_mean+1.5*pos_std, torch.zeros_like(prob_pred), torch.ones_like(prob_pred))).detach().float()
                    # max_filter = (F.max_pool3d(prob_pred, padding=1+3*level, kernel_size=3+6*level,stride=1)==prob_pred).detach().float()
                    max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                    negmask = negmask*max_filter*factor
                    # logit_pos_add = logit_pred[prob_pred>pos_mean+1.5*pos_std]
                    # print(logit_pos_add,logit_pos_add.shape)
                    # prob_pos_add = F.sigmoid(logit_pos_add)
                    # print(level,'pos',logit_pos.shape)
                    # weight = torch.pow(1-prob_pos_add.detach(),self.alpha)
                    # cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos_add)*weight)*fpn_pos_factor[level]
                    # print('pos',level,c_gt.shape[0])
                    # count_pos+=torch.sum(weight)
                else:
                    # max_filter = (F.max_pool3d(prob_pred, padding=1+3*level, kernel_size=3+6*level,stride=1)==prob_pred).detach().float()
                    max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                    negmask = negmask*max_filter
            # print(level,'neg',torch.sum(negmask))
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            # weighted loss
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
            # print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
    
    #    print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]
    #    return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)

class Focal_loss_ignoreneg_max_pos_nms_kernel(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
#         fpn_pos_factor = [1.0, 1.0, 1.0, 1.0, 1.0][-N_level:] #nms_size
#         fpn_neg_factor = [1.0, 1.0, 1.0, 1.0, 1.0][-N_level:] #nms_size
        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
#             print(level,logit_pred.shape)
            prob_pred = F.sigmoid(logit_pred)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            if fpn_connects is not None:
                connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)
            bi = 0
            prob_pos = []
            for p_pred, c_gt, p_gt in zip(logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if True:
                        max_range = level+1
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-max_range:c_gt[c,1]+max_range,
                                                                  c_gt[c,2]-max_range:c_gt[c,2]+max_range,
                                                                  c_gt[c,3]-max_range:c_gt[c,3]+max_range])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
#                     print(level,'pos',logit_pos.shape)
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None and False:
                        connects_tags = connects_gt[bi][c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            weight[weight_select] = torch.mean(weight[weight_select])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                bi += 1
            negmask = (prob_gt==-1).float()
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                if len(prob_pos) > 0:
                    pos_std,pos_mean = torch.std_mean(prob_pos)
                    factor = (torch.where(prob_pred>pos_mean+1.5*pos_std, torch.zeros_like(prob_pred), torch.ones_like(prob_pred))).detach().float()
                    max_filter = (F.max_pool3d(prob_pred, padding=1+3*level, kernel_size=3+6*level,stride=1)==prob_pred).detach().float()
#                     max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                    negmask = negmask*max_filter*factor
#                     logit_pos_add = logit_pred[prob_pred>pos_mean+1.5*pos_std]
# #                     print(logit_pos_add,logit_pos_add.shape)
#                     prob_pos_add = F.sigmoid(logit_pos_add)
# #                     print(level,'pos',logit_pos.shape)
#                     weight = torch.pow(1-prob_pos_add.detach(),self.alpha)
#                     cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos_add)*weight)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
#                     count_pos+=torch.sum(weight)
                else:
                    max_filter = (F.max_pool3d(prob_pred, padding=1+3*level, kernel_size=3+6*level,stride=1)==prob_pred).detach().float()
#                     max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                    negmask = negmask*max_filter
#             print(level,'neg',torch.sum(negmask))
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
#             print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
    
#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)

class Focal_loss2(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False, strides=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.strides = strides

    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None,fnames=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        weight_neg_sum = torch.zeros(1,device=output[0].device)
        weight_pos_sum = torch.zeros(1,device=output[0].device)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178

        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]


        pred_prob_list = [[-1, -1, -1]]
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
            prob_pred = F.sigmoid(logit_pred)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)

            negmask = (prob_gt==-1).float()
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                if self.strides is None or True:
                    kernel_size = 2 + 1
                else:
                    kernel_size = 8 / self.strides[N_level-1-level][0] + 1
                max_filter = (F.max_pool3d(prob_pred, padding=int((kernel_size - 1) / 2), \
                                           kernel_size=int(kernel_size),stride=1)==prob_pred).detach().float()
                negmask = negmask*max_filter
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
#             print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
            weight_neg_sum+=torch.sum(weight*fpn_neg_factor[level])
            for bi, p_pred, c_gt, p_gt, cc_gt, fname in zip(range(logit_pred.shape[0]), logit_pred, coord_gt, prob_gt, connects_gt, fnames):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if level == 2 and False:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-2:c_gt[c,1]+2,
                                                                  c_gt[c,2]-2:c_gt[c,2]+2,
                                                                  c_gt[c,3]-2:c_gt[c,3]+2])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None:
                        connects_tags = cc_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            region_reweight = torch.min(weight[weight_select])
                            #weight[weight_select] = torch.mean(weight[weight_select])

                            #nodule_name = fnames[bi] + '___' + str(int(ctag)+1)
                            pred_prob_list.append([float(fnames[bi]), float(ctag), float(region_reweight.detach().cpu().numpy())])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                    weight_pos_sum+=torch.sum(weight*weight2*fpn_pos_factor[level])

#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        pred_prob_list = np.array(pred_prob_list)
        pred_prob_list = torch.Tensor(pred_prob_list).cuda()
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg], [weight_pos_sum, weight_neg_sum], pred_prob_list
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)



class Focal_loss3(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False, strides=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.strides = strides

    def forward(self, output, fpn_prob, fpn_coord,fpn_connects=None,fnames=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        weight_neg_sum = torch.zeros(1,device=output[0].device)
        weight_pos_sum = torch.zeros(1,device=output[0].device)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178

        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]


        pred_prob_list = [[-1, -1, -1]]
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
            prob_pred = F.sigmoid(logit_pred)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)

            negmask = (prob_gt==-1).float()
            nll = -F.logsigmoid(-logit_pred)
            if self.nms:
                if self.strides is None or True:
                    kernel_size = 2 + 1
                else:
                    kernel_size = 8 / self.strides[N_level-1-level][0] + 1
                max_filter = (F.max_pool3d(prob_pred, padding=int((kernel_size - 1) / 2), \
                                           kernel_size=int(kernel_size),stride=1)==prob_pred).detach().float()
                negmask = negmask*max_filter
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
#             print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
            weight_neg_sum+=torch.sum(weight*fpn_neg_factor[level])

            for bi, w, cc_gt, fname in zip(range(weight.shape[0]), weight, connects_gt, fnames):
                neg_tags = torch.unique(cc_gt)
                for neg_tag in neg_tags:
                    if neg_tag >= 0:
                        continue
                    weight_select = (cc_gt == neg_tag)
                    region_reweight = torch.max(w[weight_select])
                    pred_prob_list.append([float(fnames[bi]), float(neg_tag), float(region_reweight.detach().cpu().numpy())])


            for bi, p_pred, c_gt, p_gt, cc_gt, fname in zip(range(logit_pred.shape[0]), logit_pred, coord_gt, prob_gt, connects_gt, fnames):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if level == 2 and False:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-2:c_gt[c,1]+2,
                                                                  c_gt[c,2]-2:c_gt[c,2]+2,
                                                                  c_gt[c,3]-2:c_gt[c,3]+2])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None:
                        connects_tags = cc_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            assert ctag > 0
                            weight_select = (connects_tags == ctag)
                            region_reweight = torch.min(weight[weight_select])
                            #weight[weight_select] = torch.mean(weight[weight_select])

                            #nodule_name = fnames[bi] + '___' + str(int(ctag)+1)
                            pred_prob_list.append([float(fnames[bi]), float(ctag), float(region_reweight.detach().cpu().numpy())])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                    weight_pos_sum+=torch.sum(weight*weight2*fpn_pos_factor[level])

#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        pred_prob_list = np.array(pred_prob_list)
        pred_prob_list = torch.Tensor(pred_prob_list).cuda()
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg], [weight_pos_sum, weight_neg_sum], pred_prob_list
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)




class LabelSmooth_loss(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
    def forward(self, output, fpn_prob,fpn_coord,fpn_connects=None):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        #fpn_pos_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        #fpn_neg_factor = [6.0, 6.0, 6.0, 4.0, 2.0][-N_level:] #209
        fpn_pos_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178
        fpn_neg_factor = [4.0, 4.0, 4.0, 2.0, 1.0][-N_level:] #178

        anchor_pos_factor = torch.Tensor([1, 1]).cuda()

        #fpn_pos_factor = [pow(2, item) for item in range(N_level)][::-1]
        #fpn_neg_factor = [pow(2, item) for item in range(N_level)][::-1]
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
            prob_pred = torch.clamp(F.sigmoid(logit_pred),0.001, 0.999)
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            if fpn_connects is not None:
                connects_gt = fpn_connects[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)

            negmask = (prob_gt==-1).float()
            factor = torch.where(prob_pred>0.1, 0.1*torch.ones_like(prob_pred), torch.zeros_like(prob_pred))
            nll = -torch.log(1-prob_pred)*(1-factor) + (-torch.log(prob_pred)*factor)
            if self.nms:
                max_filter = (F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)==prob_pred).detach().float()
                negmask = negmask*max_filter
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)*fpn_neg_factor[level]
#             print('neg',level,torch.sum((negmask>0).float()))
            count_neg+=torch.sum(weight)
            bi = 0
            for p_pred, c_gt, p_gt in zip(logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    if level == 2 and False:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for c in range(len(logit_pos)):
                            try:
                                logit_pos[c] = torch.max(p_pred[c_gt[c,0],
                                                                  c_gt[c,1]-2:c_gt[c,1]+2,
                                                                  c_gt[c,2]-2:c_gt[c,2]+2,
                                                                  c_gt[c,3]-2:c_gt[c,3]+2])
                            except:
                                logit_pos[c] = p_pred[c_gt[c,0],c_gt[c,1],c_gt[c,2],c_gt[c,3]]
                    else:
                        logit_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]

                    prob_pos = F.sigmoid(logit_pos)
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    weight2 = anchor_pos_factor[c_gt[:, 0]]
                    if fpn_connects is not None and False:
                        connects_tags = connects_gt[bi][c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                        for ctag in torch.unique(connects_tags):
                            weight_select = (connects_tags == ctag)
                            weight[weight_select] = torch.mean(weight[weight_select])
                    cls_loss_pos += -torch.sum(F.logsigmoid(logit_pos)*weight*weight2)*fpn_pos_factor[level]
#                     print('pos',level,c_gt.shape[0])
                    count_pos+=torch.sum(weight)
                bi += 1
#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)




class LabelSmooth_localmax_loss(nn.Module):
    def __init__(self, ignore_index, alpha=1, nms=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
    def forward(self, output, fpn_prob,fpn_coord):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = torch.zeros(1,device=output[0].device)
        count_pos = torch.zeros(1,device=output[0].device)
        for level in range(N_level):
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x a x w x h x d (1 x 2 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            logit_pred = output[level*2]
            prob_pred = torch.clamp(F.sigmoid(logit_pred),0.001, 0.999)
            #print(torch.mean(prob_pred))
            prob_gt = fpn_prob[N_level-1-level] # N x a x w x h x d
            coord_gt = fpn_coord[N_level-1-level] # N x 4(number) x 4 (coord)

            negmask = (prob_gt==-1).float()
            factor = torch.where(prob_pred>0.1, 0.1*torch.ones_like(prob_pred), torch.zeros_like(prob_pred))
            nll = -torch.log(1-prob_pred)*(1-factor) + (-torch.log(prob_pred)*factor)
            prob_pred_max = F.max_pool3d(prob_pred, padding=1, kernel_size=3,stride=1)
            if self.nms:
                max_filter = (prob_pred_max==prob_pred).detach().float()
                negmask = negmask*max_filter*(prob_pred>0.001).float()
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            cls_loss_neg += torch.sum(nll*weight)
            count_neg+=torch.sum(weight)
            for p_pred, c_gt, p_gt in zip(prob_pred_max, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
#                     print(p_pred.shape, c_gt.shape,p_gt.shape)
#                     print(p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]])
                    c_gt = c_gt[c_gt[:,0]>-1]
                    prob_pos = p_pred[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    weight = torch.pow(1-prob_pos.detach(),self.alpha)
                    cls_loss_pos += -torch.sum((torch.log(prob_pos)*0.9 + torch.log(1-prob_pos)*0.1)*weight)
                    count_pos+= torch.sum(weight)
#        print('neg loss in card', cls_loss_neg/(count_neg+1e-3))
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)


class Focal_loss_softmax(nn.Module):
    def __init__(self, ignore_index, N_cls, alpha=1, nms=False, weight_cls=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.N_cls = N_cls
        if weight_cls is None:
            self.weight_cls = torch.tensor([1]*N_cls)
        else:
            self.weight_cls = torch.tensor(weight_cls)
    def forward(self,output, fpn_prob,fpn_coord):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = 0
        count_pos = 0
        for level in range(N_level):
            # 对每个fpn level 计算损失函数
            logit_pred = output[level*2]
            prob_gt = fpn_prob[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level]
            # batchsize,class,anchor,x,y,z
            logit_pred = logit_pred.view([logit_pred.shape[0],self.N_cls, logit_pred.shape[1]//self.N_cls,
                    logit_pred.shape[2],logit_pred.shape[3], logit_pred.shape[4]])
            # 计算概率和negtive log likelihood
            nll_pred = -F.log_softmax(logit_pred, dim=1)
            prob_pred = F.softmax(logit_pred, dim=1)


            # 负样本的损失函数
            nll_negtive = nll_pred[:,0,:]
            prob_negtive = prob_pred[:,0,:]
            negmask = (prob_gt==-1).float()
            if self.nms:
                # 找到为负概率的局部极小点
                max_filter = (F.max_pool3d(nll_negtive, padding=1, kernel_size=3,stride=1)==nll_negtive).detach().float()
                # 既是负样本，又是局部极小点，即难负样本
                negmask = negmask*max_filter
            # 每个样本的权重
            weight = torch.pow(1-prob_negtive.detach(),self.alpha)*negmask
            # 总的 negtive log likelihood
            cls_loss_neg += torch.sum(nll_negtive*weight)
            # 总的权重
            count_neg+=torch.sum(weight)

            # 正样本的损失函数
            for p_pred, n_pred, c_gt, p_gt in zip(prob_pred, nll_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
                    # 把补位的坐标去掉
                    c_gt = c_gt[c_gt[:,0]>-1]
                    # 拿到每个坐标对应的结节种类
                    cls_gt = p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]].long()
                    # 预测概率和nll
                    p_target = p_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    nll_target = n_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 计算每个样本的权重，由两部分组成，第一部分是focal loss，第二部分是人为指定的各种类权重
#                     print(cls_gt)
#                     print(self.weight_cls[cls_gt])
                    weight = torch.pow(1-p_target.detach(),self.alpha) *self.weight_cls[cls_gt].cuda()
                    cls_loss_pos += torch.sum(nll_target*weight)
                    count_pos+=torch.sum(weight)
#             print(cls_loss_neg, count_neg, cls_loss_pos, count_pos)
#        return cls_loss_pos/(count_pos+1e-3) , cls_loss_neg/(count_neg+1e-3)
        return [cls_loss_pos, cls_loss_neg], [count_pos, count_neg]


# class Focal_loss_sigmoid(nn.Module):
#     def __init__(self, ignore_index, N_cls, alpha=1, nms=False, weight_cls=None):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.alpha = alpha
#         self.nms = nms
#         self.N_cls = N_cls
#         if weight_cls is None:
#             self.weight_cls = torch.tensor([1]*N_cls)
#         else:
#             self.weight_cls = torch.tensor(weight_cls)
#     def forward(self,output, fpn_prob,fpn_coord):
#         cls_loss_neg = torch.zeros(1,device=output[0].device)
#         cls_loss_pos = torch.zeros(1,device=output[0].device)
#         cls_loss_other = torch.zeros(1,device=output[0].device)
#         N_level = len(fpn_prob)
#         count_neg = 0
#         count_pos = 0
#         count_other = 0
#         for level in range(N_level):
#             # 对每个fpn level 计算损失函数
#             logit_pred = output[level*2]
#             prob_gt = fpn_prob[N_level-1-level]
#             coord_gt = fpn_coord[N_level-1-level]
#             # batchsize,class,anchor,x,y,z
# #             logit_pred = logit_pred.view([logit_pred.shape[0],self.N_cls, logit_pred.shape[1]//self.N_cls,
# #                     logit_pred.shape[2],logit_pred.shape[3], logit_pred.shape[4]])
#             # 计算概率和negtive log likelihood
#             prob_pred = torch.clamp(F.sigmoid(logit_pred),0.001,0.999)
#             prob_pos = prob_pred.view([prob_pred.shape[0],self.N_cls, prob_pred.shape[1]//self.N_cls,
#                     prob_pred.shape[2],prob_pred.shape[3], prob_pred.shape[4]])
#             prob_pos,_ = torch.max(prob_pos, dim=1)

#             # 负样本的损失函数
#             negmask = (prob_gt==-1).float()
#             if self.nms:
#                 # 找到为负概率的局部极小点
#                 max_filter = (F.max_pool3d(prob_pos, padding=1, kernel_size=3,stride=1)==prob_pos).detach().float()
#                 # 既是负样本，又是局部极小点，即难负样本
#                 negmask = negmask*max_filter
#             # 每个样本的权重
#             weight = torch.pow(prob_pos.detach(),self.alpha)*negmask
#             # 总的 negtive log likelihood
#             cls_loss_neg += -torch.sum(torch.log(1-prob_pos)*weight)
#             # 总的权重
#             count_neg+=torch.sum(weight)

#             # 正样本的损失函数
#             shape = prob_pred.shape
#             prob_pred = prob_pred.view([shape[0],self.N_cls, shape[1]//self.N_cls,
#                      shape[2],shape[3], shape[4]])
#             for p_pred, c_gt, p_gt in zip(prob_pred, coord_gt, prob_gt):
#                 if c_gt[0,0]>-1:
#                     # 把补位的坐标去掉
#                     c_gt = c_gt[c_gt[:,0]>-1]
# #                     break
#                     # 拿到每个坐标对应的结节种类
#                     cls_gt = p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]].long()-1
#                     # 预测概率
#                     p_target = p_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
#                     # 计算每个样本的权重，由两部分组成，第一部分是focal loss，第二部分是人为指定的各种类权重
#                     weight = torch.pow(1-p_target.detach(),self.alpha) *self.weight_cls[cls_gt].float().cuda()
#                     cls_loss_pos += -torch.sum(torch.log(p_target)*weight)
#                     count_pos+=torch.sum(weight)

#                     p_others = p_pred[:, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
#                     # 如果一个其他类的概率足够大，那就需要惩罚它
#                     weight = (p_others>(p_target-0.1)).float() * (p_others>0.5).float() * (p_target>0.5).float()
#                     weight[cls_gt] = 0
#                     cls_loss_other += torch.sum(-torch.log(1-p_others)*weight)
#                     count_other += torch.sum(weight)

#         return cls_loss_pos/(count_pos+1e-3), cls_loss_neg/(count_neg+1e-3), cls_loss_other/(count_other+1e-3)

class Focal_loss_sigmoid(nn.Module):
    def __init__(self, ignore_index, N_cls, alpha=1, nms=False, weight_cls=None, eps=1e-4):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.N_cls = N_cls
        self.eps = eps
        if weight_cls is None:
            self.weight_cls = torch.tensor([1]*N_cls)
        else:
            self.weight_cls = torch.tensor(weight_cls)

    def forward(self,output, fpn_prob,fpn_coord):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        cls_loss_other = torch.zeros(1,device=output[0].device)

        N_level = len(fpn_prob)
        count_neg = 0
        count_pos = 0
        count_other = 0
        for level in range(N_level):
            # 对每个fpn level 计算损失函数
            logit_pred = output[level*2]
            prob_gt = fpn_prob[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level]
            # output list: [s/4 prob, s/4 bbox, s/2 prob, s/2 bbox, s/1 prob, s/1 bbox]
            # s/4 prob: N x ca x w x h x d (1 x 10 x 40 x 40 x 40)
            # s/4 bbox: N x 4a x w x h x d (1 x 8 x 40^3)

            # batchsize,class,anchor,x,y,z
#             logit_pred = logit_pred.view([logit_pred.shape[0],self.N_cls, logit_pred.shape[1]//self.N_cls,
#                     logit_pred.shape[2],logit_pred.shape[3], logit_pred.shape[4]])
            # 计算概率和negtive log likelihood
            prob_pred = torch.clamp(F.sigmoid(logit_pred),self.eps,1-self.eps)


            # 负样本的损失函数
            # prob_gt N x a x w x h x d
            negmask = (prob_gt==-1).float().unsqueeze(1).repeat(1,self.N_cls,1,1,1,1).view(prob_pred.shape)
            #negmask N x ca x w x h x d
            if self.nms:
                # 找到为负概率的局部极小点
                # N x ca x w x h x d
                max_filter = (F.max_pool3d(logit_pred, padding=1, kernel_size=3,stride=1)==logit_pred).detach().float()
                # 概率要大于eps 才行
                thresh_filter = (prob_pred>self.eps).float()
                # 既是负样本，又是局部极小点，即难负样本
                negmask = negmask*max_filter*thresh_filter
            # 每个样本的权重
            weight = torch.pow(prob_pred.detach(),self.alpha)*negmask
            # 总的 negtive log likelihood
            cls_loss_neg += -torch.sum(torch.log(1-prob_pred)*weight)
            # 总的权重
            count_neg+=torch.sum(weight)

            # 正样本的损失函数
            # N x ca x w x h x d
            shape = prob_pred.shape
            prob_pred = prob_pred.view([shape[0],self.N_cls, shape[1]//self.N_cls,
                     shape[2],shape[3], shape[4]])
            # N x c x a x w x h x d
            for p_pred, c_gt, p_gt in zip(prob_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
                    # 把补位的坐标去掉
                    c_gt = c_gt[c_gt[:,0]>-1]
#                     break
                    # 拿到每个坐标对应的结节种类
                    cls_gt = p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]].long()-1
                    # 预测概率
                    p_target = p_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 计算每个样本的权重，由两部分组成，第一部分是focal loss，第二部分是人为指定的各种类权重
                    weight = torch.pow(1-p_target.detach(),self.alpha) *self.weight_cls[cls_gt].float().cuda()
                    cls_loss_pos += -torch.sum(torch.log(p_target)*weight)
                    count_pos+=torch.sum(weight)

                    p_others = p_pred[:, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]] # 5 vector
                    # 如果一个其他类的概率足够大，那就需要惩罚它
                    weight = (F.relu(p_others-(p_target-0.1)).float() * (p_others>0.5).float() * (p_target>0.5).float()).detach()
                    weight[cls_gt] = 0
                    cls_loss_other += torch.sum(-torch.log(1-p_others)*weight)
                    count_other += torch.sum(weight)

#        return cls_loss_pos/(count_pos+1e-3), cls_loss_neg/(count_neg+1e-3), cls_loss_other/(count_other+1e-3)
        return [cls_loss_pos, cls_loss_neg, cls_loss_other], [count_pos, count_neg, count_other]


class Focal_loss_sigmoid2(nn.Module):
    def __init__(self, ignore_index, N_cls, alpha=1, nms=False, weight_cls=None, eps=1e-4):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.N_cls = N_cls
        self.eps = eps
        if weight_cls is None:
            self.weight_cls = torch.tensor([1]*N_cls)
        else:
            self.weight_cls = torch.tensor(weight_cls)

    def forward(self,output, fpn_prob,fpn_coord):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        cls_loss_other = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = 0
        count_pos = 0
        count_other = 0
        for level in range(N_level):
            # 对每个fpn level 计算损失函数
            logit_pred = output[level*2]
            prob_gt = fpn_prob[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level]
            # batchsize,class,anchor,x,y,z
            logit_pred = logit_pred.view([logit_pred.shape[0],self.N_cls, logit_pred.shape[1]//self.N_cls,
                    logit_pred.shape[2],logit_pred.shape[3], logit_pred.shape[4]])
            # 计算概率和negtive log likelihood
            prob_pred = torch.clamp(F.sigmoid(logit_pred),self.eps,1-self.eps)
#             prob_pred = prob_pred.view([prob_pred.shape[0],self.N_cls, prob_pred.shape[1]//self.N_cls,
#                     prob_pred.shape[2],prob_pred.shape[3], prob_pred.shape[4]])

            # 负样本的损失函数
            negmask = (prob_gt==-1).float()
            if self.nms:
                prob_pred_max,_ = torch.max(prob_pred, dim=1)
                # 找到为负概率的局部极小点
                max_filter = (F.max_pool3d(prob_pred_max, padding=1, kernel_size=3,stride=1)==prob_pred_max).detach().float()
                thresh_filter = (prob_pred_max>self.eps).float()
                # 既是负样本，又是局部极小点，即难负样本
                negmask = negmask*max_filter*thresh_filter
            # 每个样本的权重
            weight = torch.pow(prob_pred_max.detach(),self.alpha)*negmask
            # 总的 negtive log likelihood
            cls_loss_neg += -torch.sum(torch.log(1-prob_pred_max)*weight)
            # 总的权重
            count_neg+=torch.sum(weight)

            # 正样本的损失函数

#             prob_pred = prob_pred.view([shape[0],self.N_cls, shape[1]//self.N_cls,
#                      shape[2],shape[3], shape[4]])
            for p_pred,l_pred, c_gt, p_gt in zip(prob_pred,logit_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
                    # 把补位的坐标去掉
                    c_gt = c_gt[c_gt[:,0]>-1]
#                     break
                    # 拿到每个坐标对应的结节种类
                    cls_gt = p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]].long()-1
                    # 预测概率
                    p_target = p_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    l_target = l_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 计算每个样本的权重，由两部分组成，第一部分是focal loss，第二部分是人为指定的各种类权重
                    weight = torch.pow(1-p_target.detach(),self.alpha) *self.weight_cls[cls_gt].float().cuda()
                    cls_loss_pos += -torch.sum(F.logsigmoid(l_target)*weight)
                    count_pos+=torch.sum(weight)

                    p_others = p_pred[:, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    l_others = l_pred[:, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 如果一个其他类的概率足够大，那就需要惩罚它
                    weight = (F.relu(p_others-(p_target-0.05)).float() * (p_others>0.5).float() * (p_target>0.5).float()).detach()
                    weight[cls_gt] = 0
                    cls_loss_other += torch.sum(-F.logsigmoid(-l_others)*weight)
                    count_other += torch.sum(weight)
        return [cls_loss_pos, cls_loss_neg, cls_loss_other], [count_pos, count_neg, count_other]
#        return cls_loss_pos/(count_pos+1e-3), cls_loss_neg/(count_neg+1e-3), cls_loss_other/(count_other+1e-3)

class Focal_loss_sigmoid3(nn.Module):
    def __init__(self, ignore_index, N_cls, alpha=1, nms=False, weight_cls=None, eps=1e-4):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.N_cls = N_cls
        self.eps = eps
        if weight_cls is None:
            self.weight_cls = torch.tensor([1]*N_cls)
        else:
            self.weight_cls = torch.tensor(weight_cls)

    def forward(self,output, fpn_prob,fpn_coord):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        cls_loss_other = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = 0
        count_pos = 0
        count_other = 0
        for level in range(N_level):
            # 对每个fpn level 计算损失函数
            logit_pred = output[level*2]
            prob_gt = fpn_prob[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level]
            # batchsize,class,anchor,x,y,z
#             logit_pred = logit_pred.view([logit_pred.shape[0],self.N_cls, logit_pred.shape[1]//self.N_cls,
#                     logit_pred.shape[2],logit_pred.shape[3], logit_pred.shape[4]])
            # 计算概率和negtive log likelihood
            prob_pred = torch.clamp(F.sigmoid(logit_pred),self.eps,1-self.eps)
            prob_pred = prob_pred.view([prob_pred.shape[0],self.N_cls, prob_pred.shape[1]//self.N_cls,
                    prob_pred.shape[2],prob_pred.shape[3], prob_pred.shape[4]])

            # 负样本的损失函数
            negmask = (prob_gt==-1).float()
            if self.nms:
                # N x c x a x w x h x d
                prob_pred_max,_ = torch.max(prob_pred, dim=1)
                # N x a x w x h x d
                # 找到为负概率的局部极小点
                max_filter = (F.max_pool3d(prob_pred_max, padding=1, kernel_size=3,stride=1)==prob_pred_max).detach().float()
                thresh_filter = (prob_pred_max>self.eps).float()
                # 既是负样本，又是局部极小点，即难负样本
                negmask = negmask*max_filter*thresh_filter
            # 每个样本的权重
            weight = torch.pow(prob_pred_max.detach(),self.alpha)*negmask
            # 总的 negtive log likelihood
            cls_loss_neg += -torch.sum(torch.log(1-prob_pred_max)*weight)
            # 总的权重
            count_neg+=torch.sum(weight)

            # 正样本的损失函数

#             prob_pred = prob_pred.view([shape[0],self.N_cls, shape[1]//self.N_cls,
#                      shape[2],shape[3], shape[4]])
            for p_pred, c_gt, p_gt in zip(prob_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
                    # 把补位的坐标去掉
                    c_gt = c_gt[c_gt[:,0]>-1]
#                     break
                    # 拿到每个坐标对应的结节种类
                    cls_gt = p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]].long()-1
                    # 预测概率
                    p_target = p_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 计算每个样本的权重，由两部分组成，第一部分是focal loss，第二部分是人为指定的各种类权重
                    weight = torch.pow(1-p_target.detach(),self.alpha) *self.weight_cls[cls_gt].float().cuda()
                    cls_loss_pos += -torch.sum(torch.log(p_target)*weight)
                    count_pos+=torch.sum(weight)

                    p_others = p_pred[:, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 如果一个其他类的概率足够大，那就需要惩罚它
                    weight = (F.relu(p_others-(p_target-0.05)).float() * (p_others>0.5).float() * (p_target>0.5).float()).detach()
                    weight[cls_gt] = 0
                    cls_loss_other += torch.sum(-torch.log(1-p_others)*weight)
                    count_other += torch.sum(weight>0).float()
        return [cls_loss_pos, cls_loss_neg, cls_loss_other], [count_pos, count_neg, count_other]
#        return cls_loss_pos/(count_pos+1e-3), cls_loss_neg/(count_neg+1e-3), cls_loss_other/(count_other+1e-3)


class Focal_loss_sigmoid4(nn.Module):
    def __init__(self, ignore_index, N_cls, alpha=1, nms=False, weight_cls=None, eps=1e-4):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.nms = nms
        self.N_cls = N_cls
        self.eps = eps
        if weight_cls is None:
            self.weight_cls = torch.tensor([1]*N_cls)
        else:
            self.weight_cls = torch.tensor(weight_cls)

    def forward(self,output, fpn_prob,fpn_coord):
        cls_loss_neg = torch.zeros(1,device=output[0].device)
        cls_loss_pos = torch.zeros(1,device=output[0].device)
        cls_loss_other = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_prob)
        count_neg = 0
        count_pos = 0
        count_other = 0
        for level in range(N_level):
            # 对每个fpn level 计算损失函数
            logit_pred = output[level*2]
            prob_gt = fpn_prob[N_level-1-level]
            coord_gt = fpn_coord[N_level-1-level]
            # batchsize,class,anchor,x,y,z
#             logit_pred = logit_pred.view([logit_pred.shape[0],self.N_cls, logit_pred.shape[1]//self.N_cls,
#                     logit_pred.shape[2],logit_pred.shape[3], logit_pred.shape[4]])
            # 计算概率和negtive log likelihood
            prob_pred = torch.clamp(F.sigmoid(logit_pred),self.eps,1-self.eps)
            prob_pred = prob_pred.view([prob_pred.shape[0],self.N_cls, prob_pred.shape[1]//self.N_cls,
                    prob_pred.shape[2],prob_pred.shape[3], prob_pred.shape[4]])

            # 负样本的损失函数
            negmask = (prob_gt==-1).float()
            if self.nms:
                prob_pred_max,_ = torch.max(prob_pred, dim=1)
                # 找到为负概率的局部极小点
                max_filter = (F.max_pool3d(prob_pred_max, padding=1, kernel_size=3,stride=1)==prob_pred_max).detach().float()
                thresh_filter = (prob_pred_max>self.eps).float()
                # 既是负样本，又是局部极小点，即难负样本
                negmask = negmask*max_filter*thresh_filter
            # 每个样本的权重
            weight = torch.pow(prob_pred_max.detach(),self.alpha)*negmask
            # 总的 negtive log likelihood
            cls_loss_neg += -torch.sum(torch.log(1-prob_pred_max)*weight)
            # 总的权重
            count_neg+=torch.sum(weight>0).float()

            # 正样本的损失函数

#             prob_pred = prob_pred.view([shape[0],self.N_cls, shape[1]//self.N_cls,
#                      shape[2],shape[3], shape[4]])
            for p_pred, c_gt, p_gt in zip(prob_pred, coord_gt, prob_gt):
                if c_gt[0,0]>-1:
                    # 把补位的坐标去掉
                    c_gt = c_gt[c_gt[:,0]>-1]
#                     break
                    # 拿到每个坐标对应的结节种类
                    cls_gt = p_gt[c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]].long()-1
                    # 预测概率
                    p_target = p_pred[cls_gt, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 计算每个样本的权重，由两部分组成，第一部分是focal loss，第二部分是人为指定的各种类权重
                    weight = torch.pow(1-p_target.detach(),self.alpha) *self.weight_cls[cls_gt].float().cuda()
                    cls_loss_pos += -torch.sum(torch.log(p_target)*weight)
                    count_pos+=torch.sum(weight>0).float()

                    p_others = p_pred[:, c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    # 如果一个其他类的概率足够大，那就需要惩罚它
                    weight = (F.relu(p_others-(p_target-0.05)).float() * (p_others>0.5).float() * (p_target>0.5).float()).detach()
                    weight[cls_gt] = 0
                    cls_loss_other += torch.sum(-torch.log(1-p_others)*weight)
                    count_other += torch.sum(weight>0).float()
        return [cls_loss_pos, cls_loss_neg, cls_loss_other], [count_pos, count_neg, count_other]
#        return cls_loss_pos/(count_pos+1e-3), cls_loss_neg/(count_neg+1e-3), cls_loss_other/(count_other+1e-3)

class Bbox_loss(nn.Module):
    def __init__(self, lossfun):
        super().__init__()
        self.lossfun = lossfun

    def forward(self, output, fpn_coord, fpn_diff):
        """
        修改为立方体的reg loss
        :param output: 网络输出
        :param fpn_coord: 计算reg loss所在的anchor的位置
        :param fpn_diff: anchor和gt的diff
        """
        reg_loss = torch.zeros(1,device=output[0].device)
        reg_weight = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_diff)
        for level in range(N_level):
            diff_pred = output[level*2+1]
            coord_gt = fpn_coord[N_level-1-level]
            diff_gt = fpn_diff[N_level-1-level].cuda().float()
            diff_pred = diff_pred.view([diff_pred.shape[0],6, diff_pred.shape[1]//6,
                                        diff_pred.shape[2],diff_pred.shape[3], diff_pred.shape[4]])

            for d_pred, c_gt, d_gt in zip(diff_pred, coord_gt, diff_gt):
                if c_gt[0,0]>-1:
                    d_gt = d_gt[c_gt[:,0]>-1]
                    c_gt = c_gt[c_gt[:,0]>-1]
                    d_pred_sample = d_pred[:,c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    loss_raw = self.lossfun(d_pred_sample.transpose(0,1), d_gt)
                    loss_regress = torch.sum(loss_raw)
                    reg_loss += loss_regress
                    reg_weight += c_gt.shape[0]
                    #print(loss_regress, c_gt.shape[0])
#        reg_loss /= (reg_weight+1e-3)
        return [reg_loss],[reg_weight]

class Bbox_loss2(nn.Module):
    def __init__(self, lossfun):
        super().__init__()
        self.lossfun = lossfun

    def forward(self, output, fpn_coord, fpn_diff):
        loss_weight = torch.Tensor([[1, 1, 1, 0.1]]).cuda()
        reg_loss = torch.zeros(1,device=output[0].device)
        reg_weight = torch.zeros(1,device=output[0].device)
        N_level = len(fpn_diff)
        for level in range(N_level):
            diff_pred = output[level*2+1]
            coord_gt = fpn_coord[N_level-1-level]
            diff_gt = fpn_diff[N_level-1-level].cuda().float()
            diff_pred = diff_pred.view([diff_pred.shape[0],4, diff_pred.shape[1]//4,
                                        diff_pred.shape[2],diff_pred.shape[3], diff_pred.shape[4]])

            for d_pred, c_gt, d_gt in zip(diff_pred, coord_gt, diff_gt):
                if c_gt[0,0]>-1:
                    d_gt = d_gt[c_gt[:,0]>-1]
                    c_gt = c_gt[c_gt[:,0]>-1]
                    d_pred_sample = d_pred[:,c_gt[:,0],c_gt[:,1],c_gt[:,2],c_gt[:,3]]
                    loss_raw = self.lossfun(d_pred_sample.transpose(0,1), d_gt)
                    loss_raw = loss_raw * loss_weight
                    loss_regress = torch.sum(loss_raw)
                    reg_loss += loss_regress
                    reg_weight += c_gt.shape[0]
                    #print(loss_regress, c_gt.shape[0])
#        reg_loss /= (reg_weight+1e-3)
        return [reg_loss],[reg_weight]


def l1_loss(input, target):
    return torch.abs(input-target)

def diou_loss(input, target):
    pred_bbox = torch.cat([input[:, :3] - input[:, 3:]/2, input[:, :3] + input[:, 3:]/2], dim=1)
    target_bbox = torch.cat([target[:, :3] - target[:, 3:]/2, target[:, :3] + target[:, 3:]/2], dim=1)
    bbox_diou = Diou(pred_bbox, target_bbox)
    return 1 - bbox_diou


def Diou(bboxes1, bboxes2):
    d1 = bboxes1[:, 3] - bboxes1[:, 0]
    d2 = bboxes2[:, 3] - bboxes2[:, 0]

    area1 = d1 * d1 * d1
    area2 = d2 * d2 * d2

    center_z1 = (bboxes1[:, 3] + bboxes1[:, 0]) / 2
    center_x1 = (bboxes1[:, 4] + bboxes1[:, 1]) / 2
    center_y1 = (bboxes1[:, 5] + bboxes1[:, 2]) / 2
    center_z2 = (bboxes2[:, 3] + bboxes2[:, 0]) / 2
    center_x2 = (bboxes2[:, 4] + bboxes2[:, 1]) / 2
    center_y2 = (bboxes2[:, 5] + bboxes2[:, 2]) / 2

    inter_max_zxy = torch.min(bboxes1[:, 3:], bboxes2[:, 3:])
    inter_min_zxy = torch.max(bboxes1[:, :3], bboxes2[:, :3])
    out_max_zxy = torch.max(bboxes1[:, 3:], bboxes2[:, 3:])
    out_min_zxy = torch.min(bboxes1[:, :3], bboxes2[:, :3])

    inter = torch.clamp((inter_max_zxy - inter_min_zxy), min=0)
    inter_area = inter[:, 0] * inter[:, 1] * inter[:, 2]
    inter_diag = (center_z2 - center_z1) ** 2 + (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_zxy - out_min_zxy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2) + (outer[:, 2] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    return dious


def l2_loss(input, target):
    return torch.pow(input-target,2)*20


def smooth_l1_loss(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    return loss

class Loss_comb2(nn.Module):
    def __init__(self,config, lamb1=1, lamb2=1):
        super().__init__()
        self.ignore_index = config.prepare['label_para']['ignore_index']
        self.margin = np.array(config.prepare['margin'])
        self.cropsize = np.array(config.prepare['crop_size'])
        self.focal = Focal_loss(self.ignore_index)
        self.lbbox = Bbox_loss(l1_loss)
        self.lamb1 = lamb1
        self.lamb2 = lamb2

    def clipmargin(self,tensors):
        for i,o in enumerate(tensors):
            stride = np.array(self.cropsize)/np.array(o.shape[2:])
            m = (self.margin/stride).astype('int')
            tensors[i] = o[:,:,m[0]:neg(m[0]), m[1]:neg(m[1]),m[2]:neg(m[2])]
        return tensors

    def forward(self, output, fpn_prob, fpn_coord_prob, fpn_coord_diff, fpn_diff, fpn_connects=None, fnames=None):
        """
        
        Return: list()
            loss: 
            weight: 
        """
        output = self.clipmargin(list(output))

        loss1, weight1 = self.focal(output, fpn_prob, fpn_coord_prob, fpn_connects)
        loss2, weight2 = self.lbbox(output, fpn_coord_diff, fpn_diff)
        loss1[1] = loss1[1]*self.lamb1 # 负样本权重
        loss2[0] = loss2[0]*self.lamb2 # bbox 回归权重
        loss = loss1+loss2
        weight = weight1+weight2
        #print(loss, [l.shape for l in loss])
        #print(weight, [l.shape for l in weight])
        return torch.cat(loss).unsqueeze(0), torch.cat(weight).unsqueeze(0), None
    

class Loss_comb2_v2(nn.Module):
    def __init__(self,config, lamb1=1, lamb2=1):
        super().__init__()
        self.ignore_index = config.prepare['label_para']['ignore_index']
        self.margin = np.array(config.prepare['margin'])
        self.cropsize = np.array(config.prepare['crop_size'])
        self.strides = np.array(config.rpn['strides'])
        self.focal = Focal_loss(self.ignore_index)
        self.lbbox = Bbox_loss(l1_loss)
        self.lamb1 = lamb1
        self.lamb2 = lamb2

    def clipmargin(self,tensors):
        for i,o in enumerate(tensors):
            stride = np.array(self.cropsize)/np.array(o.shape[2:])
            m = (self.margin/stride).astype('int')
            tensors[i] = o[:,:,m[0]:neg(m[0]), m[1]:neg(m[1]),m[2]:neg(m[2])]
        return tensors

    def forward(self, output, fpn_prob, fpn_coord_prob, fpn_coord_diff, fpn_diff, fpn_connects=None, fnames=None):
        output = self.clipmargin(list(output))

        loss1, weight1, fack_weight1, pred_prob_list = self.focal(output, fpn_prob, fpn_coord_prob, fpn_connects, fnames)
        loss2, weight2 = self.lbbox(output, fpn_coord_diff, fpn_diff)
        loss1[1] = loss1[1]*self.lamb1 # 负样本权重
        loss2[0] = loss2[0]*self.lamb2 # bbox 回归权重
        loss = loss1+loss2
        weight = weight1+weight2
        fack_weight = fack_weight1+weight2
        #print(loss, [l.shape for l in loss])
        #print(weight, [l.shape for l in weight])
        return torch.cat(loss).unsqueeze(0), torch.cat(weight+fack_weight).unsqueeze(0), pred_prob_list


class Loss_comb3(Loss_comb2):
    def __init__(self,config):
        super().__init__(config)
        self.lbbox = Bbox_loss(smooth_l1_loss)

class Loss_comb4(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=3, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb11(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=3, lamb2=0)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb12(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=0, lamb2=1)
        self.lbbox = Bbox_loss(l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb13(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb13_LabelSmooth(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = LabelSmooth_loss(self.ignore_index,nms=True)
        
class Loss_comb13_ignoreneg(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss_ignoreneg(self.ignore_index,nms=True)
        
class Loss_comb13_ignoreneg_max_pos(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1, lamb2=0.2)
        """
        :param lamb1: 负样本在loss中权重，正样本以及和负样本取了平均
        :param lamb2: 框的回归loss
        """
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss_ignoreneg_max_pos(self.ignore_index,nms=True)
        
class Loss_comb13_ignoreneg_max_pos_2(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss_ignoreneg_max_pos(self.ignore_index,alpha=2,nms=True)
        
class Loss_comb13_ignoreneg_max_pos_3(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss_ignoreneg_max_pos_nms_kernel(self.ignore_index,nms=True)

class Loss_comb13_ignoreneg_max_pos_diouloss(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(diou_loss)
        self.focal = Focal_loss_ignoreneg_max_pos(self.ignore_index,nms=True)
        
class Loss_comb13_level(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss_level(self.ignore_index,nms=True)

class Loss_comb13_l1loss(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=0.1)
        self.lbbox = Bbox_loss(l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb13_l1loss2(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss2(l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)
        
class Loss_comb13_ignoreneg_l1loss2(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss2(l1_loss)
        self.focal = Focal_loss_ignoreneg(self.ignore_index,nms=True)

class Loss_comb13_diouloss(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss(diou_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb13_l1loss2_2(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=3, lamb2=1)
        self.lbbox = Bbox_loss2(l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb13_l1loss2_focal2(Loss_comb2_v2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss2(l1_loss)
        self.focal = Focal_loss2(self.ignore_index,nms=True,strides=self.strides)

class Loss_comb13_l1loss2_focal3(Loss_comb2_v2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=1)
        self.lbbox = Bbox_loss2(l1_loss)
        self.focal = Focal_loss3(self.ignore_index,nms=True,strides=self.strides)

class Loss_comb_noreg(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=1.5, lamb2=0)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb6(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=2.5, lamb2=2)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class Loss_comb7(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=9, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = LabelSmooth_loss(self.ignore_index,nms=True)


class Loss_comb10(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=9, lamb2=0.1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = LabelSmooth_loss(self.ignore_index,nms=True)

class Loss_comb8(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=6, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = LabelSmooth_loss(self.ignore_index,nms=True)

class Loss_comb9(Loss_comb2):
    def __init__(self,config):
        super().__init__(config, lamb1=2, lamb2=1)
        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.focal = LabelSmooth_localmax_loss(self.ignore_index,nms=True)




class Loss_comb5(Loss_comb2):
    def __init__(self,config):
        super().__init__(config)
        self.lbbox = Bbox_loss(l2_loss)
        self.focal = Focal_loss(self.ignore_index,nms=True)

class ssd_loss_comb1(nn.Module):
    def __init__(self,config, lamb1=1, lamb2=1, lamb3=1):
        super().__init__()
        self.ignore_index = config.prepare['label_para']['ignore_index']
        self.margin = np.array(config.prepare['margin'])
        self.cropsize = np.array(config.prepare['crop_size'])
        weight_cls = [1] + config.classifier['cls_weight']
        if config.classifier['activation'] == 'softmax':
            N_cls = config.classifier['N_cls']+1
            self.focal = Focal_loss_softmax(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'softmax'
        elif config.classifier['activation'] == 'sigmoid':
            N_cls = config.classifier['N_cls']
            self.focal = Focal_loss_sigmoid(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'sigmoid'

        self.lbbox = Bbox_loss(smooth_l1_loss)
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3

    def clipmargin(self,tensors):
        for i,o in enumerate(tensors):
            stride = np.array(self.cropsize)/np.array(o.shape[2:])
            m = (self.margin/stride).astype('int')
            tensors[i] = o[:,:,m[0]:neg(m[0]), m[1]:neg(m[1]),m[2]:neg(m[2])]
        return tensors

    def forward(self, output, fpn_prob, fpn_coord_prob, fpn_coord_diff, fpn_diff):
        output = self.clipmargin(list(output))
        l_diff = self.lbbox(output, fpn_coord_diff, fpn_diff)
        if self.act == 'softmax':
            l_pos,l_neg = self.focal(output, fpn_prob, fpn_coord_prob)
            return l_pos+l_neg*self.lamb1+l_diff*self.lamb2,[l_pos.detach(), l_neg.detach()*self.lamb1, l_diff.detach()*self.lamb2]
        elif self.act == 'sigmoid':
            l_pos,l_neg,l_other = self.focal(output, fpn_prob, fpn_coord_prob)
            return l_pos + l_neg*self.lamb1 + l_other*self.lamb3 + l_diff*self.lamb2,[l_pos.detach(), l_neg.detach(), l_other.detach(), l_diff.detach()]


class ssd_loss_comb2(ssd_loss_comb1):
    def __init__(self,config):
        super().__init__(config, lamb1=0.5,lamb3=0.01)
        weight_cls = [1] + config.classifier['cls_weight']
        if config.classifier['activation'] == 'softmax':
            N_cls = config.classifier['N_cls']+1
            self.focal = Focal_loss_softmax(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'softmax'
        elif config.classifier['activation'] == 'sigmoid':
            N_cls = config.classifier['N_cls']
            self.focal = Focal_loss_sigmoid2(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'sigmoid'

class ssd_loss_comb3(ssd_loss_comb1):
    def __init__(self,config):
        super().__init__(config,lamb1=0.7, lamb2=1, lamb3=1)
        weight_cls = [1] + config.classifier['cls_weight']
        if config.classifier['activation'] == 'softmax':
            N_cls = config.classifier['N_cls']+1
            self.focal = Focal_loss_softmax(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'softmax'
        elif config.classifier['activation'] == 'sigmoid':
            N_cls = config.classifier['N_cls']
            self.focal = Focal_loss_sigmoid3(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'sigmoid'


class ssd_loss_comb4(ssd_loss_comb1):
    def __init__(self,config):
        super().__init__(config,lamb1=40, lamb2=1, lamb3=1)
        weight_cls = [1] + config.classifier['cls_weight']
        if config.classifier['activation'] == 'softmax':
            N_cls = config.classifier['N_cls']+1
            self.focal = Focal_loss_softmax(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls)
            self.act = 'softmax'
        elif config.classifier['activation'] == 'sigmoid':
            N_cls = config.classifier['N_cls']
            self.focal = Focal_loss_sigmoid4(self.ignore_index,N_cls, nms=True, weight_cls=weight_cls,eps=1e-3)
            self.act = 'sigmoid'


def iou(bbox1, bbox2):
    """
    两个box array的iou
    """
    left = np.max([bbox1[:3], bbox2[:3]], axis=0)
    right = np.min([bbox1[3:6], bbox2[3:6]], axis=0)
    valid = np.all(right>left)
    if not valid:
        return 0
    else:
        intersec = np.prod(right-left)
        area1 = np.prod(bbox1[3:6]-bbox1[:3])
        area2 = np.prod(bbox2[3:6]-bbox2[:3])
        return (intersec.astype('float')/(area1+area2-intersec))


def is_contained(bbox1, bbox2):
    """
    判断bbox2的中心坐标点是否包含于bbox1
    :param bbox1: [z1, y1, x1, z2, y2, x2]
    :param bbox2: [z1, y1, x1, z2, y2, x2]

    """
    bbox2_center = [(bbox2[3] + bbox2[0])/2, (bbox2[4] + bbox2[1])/2, (bbox2[5] + bbox2[2])/2]
    if bbox2_center[0] <= bbox1[3] and bbox2_center[0] >= bbox1[0] and \
        bbox2_center[1] <= bbox1[4] and bbox2_center[1] >= bbox1[1] and \
        bbox2_center[2] <= bbox1[5] and bbox2_center[2] >= bbox1[2]:
        return True
    else:
        return False


def cal_distance(p1, p2):
    """
    计算两个点在三维空间中的距离
    :param p1: [z, y, x]
    """
    p1_center = np.array([(p1[3] + p1[0])/2, (p1[4] + p1[1])/2, (p1[5] + p1[2])/2])
    p2_center = np.array([(p2[3] + p2[0])/2, (p2[4] + p2[1])/2, (p2[5] + p2[2])/2])
    return np.sqrt(np.sum(np.power(p1_center - p2_center, 2) ** 2))



def cent2border(bbox_list):
    """
    box的中心坐标形式，转为边界表示
    :param bbox_list: [z, y, x, dz, dy, dx, cls, 1, 1, 1]
    2 > 1

    return: [z1, y1, x1, z2, y2, x2, 1, cls+1]
    """
    if len(bbox_list)>0:
        return torch.cat([bbox_list[:,:3] - bbox_list[:,3:6]/2, bbox_list[:,:3] + bbox_list[:,3:6]/2, bbox_list[:,7:8], bbox_list[:,6:7]+1], dim=1).float().cpu().numpy()
        # return torch.cat([bbox_list[:,:3]-bbox_list[:,3:4]/2, bbox_list[:,:3]+bbox_list[:,3:4]/2, bbox_list[:,5:6], bbox_list[:,4:5]+1], dim=1).float().cpu().numpy()
    else:
        return []

class em(nn.Module):
    def __init__(self,config):
        super().__init__()
        # miccai-2020, iou >= 0.2
        self.iou_thresh = 0.1
        self.sizelim = config.rpn['diam_thresh']
        self.small_size = config.rpn['small_size']
        self.omit_cls = config.classifier['omit_cls']

    def forward(self,bbox_pred,bbox_label):
        """
        更改为适应立方体的计算方式
        根据bbox和gt计算tp，fp，命中规则：pred和label的iou大于iou_thresh
        :param bbox_pred: 预测框 [z1, y1, x1, z2, y2, x2, confidence, cls]
        :param bbox_label: gt [z, y, x, dz, dy, dx, cls, 1, 1, 1]
        
        """

        bbox_label = cent2border(bbox_label)
        if len(bbox_pred)>0:
            bbox_pred = bbox_pred.float().cpu().numpy()

        n_hit = 0
        n_hit_small = 0
        n_fp = 0
        n_fp_small = 0
        n_miss = 0
        n_miss_small = 0
        n_match = 0
        n_pred = len(bbox_pred)
        n_label = len(bbox_label)

        # 是否命中 -1:没有命中
        yes_or_no = np.ones(n_pred, dtype=np.uint16)*-1
        iou_info = np.stack([np.ones(n_pred, dtype=np.float32)*-1, np.zeros(n_pred, dtype=np.float32)], axis=1)
        for i1,b1 in enumerate(bbox_label):
            best_select=None
            best_iou = self.iou_thresh
            for i2,b2 in enumerate(bbox_pred):
                if yes_or_no[i2]>=0:
                    continue
                v_iou = iou(b1, b2)
                # 找到与gt的iou最大的pred
                if v_iou>best_iou:
                    best_iou = v_iou
                    best_select = i2
                # 更新每一个pred和对应的gt最大的iou
                if v_iou >= self.iou_thresh:
                    if iou_info[i2][1] < v_iou:
                        iou_info[i2][0] = i1
                        iou_info[i2][1] = v_iou
            
            if best_select is not None:
                yes_or_no[best_select] = i1
                if (b1[3]-b1[0])>=self.small_size:
                    n_hit += 1
                else:
                    n_hit_small += 1
                if b1[7] == bbox_pred[best_select][7]:
                    n_match += 1
#                     print
            else:
#                 print(b1)
                if b1[6]!=1:
                    continue
                if (b1[3]-b1[0])>=self.small_size and ((b1[7]-1) not in self.omit_cls):
                    n_miss+=1
                elif (b1[3]-b1[0])<self.small_size and ((b1[7]-1) not in self.omit_cls):
                    n_miss_small += 1
#         n_fp_all = np.sum(yes_or_no==-1)

        if len(bbox_pred)>0:
            n_fp_small = np.sum((yes_or_no==-1)&((bbox_pred[:,3]-bbox_pred[:,0])<self.small_size))
            n_fp = np.sum((yes_or_no==-1)&((bbox_pred[:,3]-bbox_pred[:,0])>=self.small_size))
        result = {'N_match':n_match, 'N_hit':n_hit, 'N_hit_small':n_hit_small, "N_fp":n_fp, "N_fp_small":n_fp_small, "N_miss":n_miss,"N_miss_small":n_miss_small}
        
        # print(n_pred)
        # print(n_label)
        # print(yes_or_no.shape)
        # print(np.sum(yes_or_no == -1))
        # exit()
        
        return result, iou_info


class em_point(nn.Module):
    def __init__(self,config):
        super().__init__()
        # self.iou_thresh = 0.1
        self.sizelim = config.rpn['diam_thresh']
        self.small_size = config.rpn['small_size']
        self.omit_cls = config.classifier['omit_cls']

    def forward(self,bbox_pred,bbox_label):
        """
        根据bbox和gt计算tp，fp，命中规则：gt box包含pred的中心位置坐标（忽略框的大小）
        :param bbox_pred: 预测框 [z1, y1, x1, z2, y2, x2, confidence, cls]
        :param bbox_label: gt [z, y, x, d, cls, 1, 1, 1]
        
        """
        bbox_label = cent2border(bbox_label)
        if len(bbox_pred)>0:
            bbox_pred = bbox_pred.float().cpu().numpy()

        n_hit = 0
        n_hit_small = 0
        n_fp = 0
        n_fp_small = 0
        n_miss = 0
        n_miss_small = 0
        n_match = 0
        n_pred = len(bbox_pred)
        n_label = len(bbox_label)

        # 是否命中 -1:没有命中
        yes_or_no = np.ones(n_pred, dtype=np.uint16)*-1
        iou_info = np.stack([np.ones(n_pred, dtype=np.float32)*-1, np.zeros(n_pred, dtype=np.float32)], axis=1)
        for i1,b1 in enumerate(bbox_label):
            best_select = None
            best_dist = float('inf')
            for i2,b2 in enumerate(bbox_pred):
                if yes_or_no[i2]>=0:
                    continue
                
                v_dist = cal_distance(b1[:6], b2[:6])

                # v_iou = iou(b1, b2)
                # 找到与gt的iou最大的pred
                if v_dist < best_dist:
                    best_iou = v_dist
                    best_select = i2

                # 更新每一个pred和对应的gt最大的iou
                # if v_iou >= self.iou_thresh:
                if is_contained(b1[:6], b2[:6]):
                    # if iou_info[i2][1] > -v_dist:
                    iou_info[i2][0] = i1
                    iou_info[i2][1] = v_dist
            
            if best_select is not None:
                yes_or_no[best_select] = i1
                if (b1[3]-b1[0])>=self.small_size:
                    n_hit += 1
                else:
                    n_hit_small += 1
                if b1[7] == bbox_pred[best_select][7]:
                    n_match += 1
#                     print
            else:
#                 print(b1)
                if b1[6]!=1:
                    continue
                if (b1[3]-b1[0])>=self.small_size and ((b1[7]-1) not in self.omit_cls):
                    n_miss+=1
                elif (b1[3]-b1[0])<self.small_size and ((b1[7]-1) not in self.omit_cls):
                    n_miss_small += 1
#         n_fp_all = np.sum(yes_or_no==-1)

        if len(bbox_pred)>0:
            n_fp_small = np.sum((yes_or_no==-1)&((bbox_pred[:,3]-bbox_pred[:,0])<self.small_size))
            n_fp = np.sum((yes_or_no==-1)&((bbox_pred[:,3]-bbox_pred[:,0])>=self.small_size))
        result = {'N_match':n_match, 'N_hit':n_hit, 'N_hit_small':n_hit_small, "N_fp":n_fp, "N_fp_small":n_fp_small, "N_miss":n_miss,"N_miss_small":n_miss_small}
        
        # print(n_pred)
        # print(n_label)
        # print(yes_or_no.shape)
        # print(np.sum(yes_or_no == -1))
        # exit()
        
        return result, iou_info

class seg_em(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.iou_thresh = 0.1
        self.sizelim = config.rpn['diam_thresh']
        self.small_size = config.rpn['small_size']
        self.omit_cls = config.classifier['omit_cls']

    def forward(self,seg_pred, seg_label, bbox_label):
        pos_hit_map = ((seg_pred > 0)*(seg_label>0)*(seg_label<255)).float()
        pos_miss_map = ((seg_pred <= 0)*(seg_label>0)*(seg_label<255)).float()
        neg_hit_map = ((seg_pred > 0)*(seg_label==0)).float()
        pos_hit, pos_miss, neg_hit = torch.sum(pos_hit_map), torch.sum(pos_miss_map), torch.sum(neg_hit_map)
        bbox_mask = torch.zeros(seg_pred.shape).cuda()
        for bbox in bbox_label:
            if bbox[4] not in [3, 4] and bbox[3] > 4.5:
                z, x, y = bbox[:3]
                d = bbox[3]
                sz, ez, sx, ex, sy, ey = int(z - d/2), int(z + d/2), int(x - d/2), int(x + d/2), int(y - d/2), int(y + d/2)
                bbox_mask[:, sz:ez, sx:ex, sy:ey] = 1
        pos_hit2, pos_miss2, neg_hit2 = torch.sum(pos_hit_map*bbox_mask), torch.sum(pos_miss_map*bbox_mask), torch.sum(neg_hit_map*bbox_mask)
        result = {'true_pos': pos_hit, 'false_neg': pos_miss, 'false_pos': neg_hit, \
                  'true_pos2': pos_hit2, 'false_neg2': pos_miss2, 'false_pos2': neg_hit2}
        return result


