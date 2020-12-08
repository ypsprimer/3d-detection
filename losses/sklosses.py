# -*- coding: utf-8 -*-
#
#  lossfunc.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss,NLLLoss


class BinaryDiceLoss(nn.Module):
    def __init__(self, ignore_index = None):
        super().__init__()
        self.ignore_index = ignore_index
    def forward(self, logit, labels):
        assert  logit.shape[1] <= 2
        if logit.shape[1] == 1:
            outs = F.sigmoid(logit)
        else:
            outs = F.softmax(logit, dim=1)[:,1:]

        if self.ignore_index is not None:
            validmask = (labels!=self.ignore_index).float()
            outs = outs.mul(validmask)# can not use inplace for bp
            labels = labels.float().mul(validmask)
        loss = -(2.0 * torch.sum(outs * labels) + 1e-4) / (
            torch.sum(outs) + torch.sum(labels) + 1e-4)
        return loss


class BDC_BCE(nn.Module):
    def __init__(self, ignore_index = None):
        super().__init__()
        self.ignore_index = ignore_index
        if ignore_index is not None:
            self.bce = nn.BCELoss(reduction=None)
        else:
            self.bce = nn.BCELoss(reductoin='mean')

    def forward(self, logit, labels):
        assert  logit.shape[1] <= 2
        batch_size = labels.size(0)
        if logit.shape[1] == 1:
            Sigmoidout = F.sigmoid(logit)
        else:
            Sigmoidout = F.softmax(logit, dim=1)[:,1:]

        if self.ignore_index is not None:
            validmask = (labels!=self.ignore_index).float()
            outs = outs.mul(validmask)# can not use inplace for bp
            labels = labels.float().mul(validmask)
            loss2 = torch.mean(self.bce(Sigmoidout, labels)*validmask)/(torch.mean(validmask)+1e-4)
        else:
            loss2 = self.bce(Sigmoidout, labels)
        loss1 = -(2.0 * torch.sum(outs * labels) + 1e-4) / (
             torch.sum(outs) + torch.sum(labels) + 1e-4)
        meanloss = loss1+loss2
        return meanloss
        #todo 改成并行模式，不要for循环，不要平均

class BinaryDiceScore(nn.Module):
    def __init__(self,ignore_index = None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outs, labs):
        assert  outs.shape[0] ==1
        if self.ignore_index is not None:
            validmask = labs!=self.ignore_index
            out1 = outs.type(torch.cuda.FloatTensor)
            lab1 = labs.type(torch.cuda.FloatTensor)
            validmask = validmask.type(torch.cuda.FloatTensor)

            out1.mul_(validmask) # inplace operation to save memory
            lab1.mul_(validmask)
        
        sampleloss = -(2.0 * torch.sum(out1 * lab1).float() + 1e-2) / (
            torch.sum(out1).float() + torch.sum(lab1).float() + 1e-2)

        return sampleloss


class TverskyLoss(nn.Module):
    def __init__(self, beta = 1.5):
        super(TverskyLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.beta = beta
    def forward(self, output, labels):
        batch_size = labels.size(0)
        nlabels = 1 - labels
        Sigmoidout = self.sigmoid(output)
        loss = 0
        for i in range(batch_size):
            sampleloss = -((1+self.beta*self.beta)*torch.sum(torch.mul(Sigmoidout[i, :], labels[i, :])) + 1e-4) / ((1+self.beta*self.beta)*torch.sum(torch.mul(Sigmoidout[i, :], labels[i, :])) + torch.sum(torch.mul(Sigmoidout[i, :], nlabels[i,:])) + self.beta*self.beta*torch.sum(torch.mul((1-Sigmoidout[i, :]), labels[i, :])) + 1e-4)
            loss += sampleloss
        meanloss = loss / batch_size

        return meanloss




class CELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=False):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce)
        self.ignore_index = ignore_index

    def forward(self, output, labels):
        class_num = output.size(1)
        #        print('output', output)

        label_sum = []
        for i in range(class_num):
            slabel_sum = labels[labels == 0].sum()
            label_sum.append(int(slabel_sum))

        loss_sum = 0.0
        labels = labels.permute(0, 2, 3, 4, 1).contiguous()
        labels = labels.view(-1)
        labels = labels.type(torch.cuda.LongTensor)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(-1, class_num)
        Loss = self.loss(output, labels)

        #        print('Loss:', Loss)

        n_valid = 0
        count = 0
        for i in range(class_num):
            if i != self.ignore_index and label_sum[i] != 0:
                #                print('labels.sum ', labels.sum())

                count += 1

                loss_idx = torch.mean(Loss[labels == i])
                loss_sum += loss_idx
                n_valid += 1

                #                print('loss_sum ', loss_sum)

        if count != 0:
            loss_mean = loss_sum / n_valid
        else:
            loss_mean = torch.mean(Loss)  ### ?

        # print('loss_mean', loss_mean)

        return loss_mean


class CELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=False):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce)
        self.ignore_index = ignore_index

    def forward(self, output, labels):
        class_num = output.size(1)
        loss_sum = 0
        if len(output.size()) == 4:
            labels = labels.permute(0, 2, 3, 1).contiguous()
            labels = labels.view(-1)
            labels = labels.type(torch.cuda.LongTensor)
            output = output.permute(0, 2, 3, 1).contiguous()
            output = output.view(-1, class_num)
        if len(output.size()) == 5:
            labels = labels.permute(0, 2, 3, 4, 1).contiguous()
            labels = labels.view(-1)
            labels = labels.type(torch.cuda.LongTensor)
            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output = output.view(-1, class_num)
        Loss = self.loss(output, labels)
        n_valid = 0
        for i in range(class_num):
            if i != self.ignore_index and len(Loss[labels == i].data.cpu().numpy()) > 0:
                loss_idx = torch.mean(Loss[labels == i])
                loss_sum += loss_idx
                n_valid += 1

        loss_mean = loss_sum / n_valid
        return loss_mean


class FocalLossOri(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLossOri, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    #        print('alpha', self.alpha)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        #        print('inputs', input)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        #        print(class_mask.type)
        #
        ids = targets.view(-1, 1)

        #        print(class_mask.type)
        #        print(ids.data.type)

        #        ids.data = Variable(ids.data)


        #        print(class_mask)
        class_mask.scatter_(1, ids.data, 1)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        class_mask = Variable(class_mask)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=False):
        super(FocalLoss, self).__init__()
        self.loss = FocalLossOri(class_num=2, gamma=0)  # FocalLossOri(weight,size_average,ignore_index,reduce)
        self.ignore_index = ignore_index

    def forward(self, output, labels):
        class_num = output.size(1)
        loss_sum = 0
        if len(output.size()) == 4:
            labels = labels.permute(0, 2, 3, 1).contiguous()
            labels = labels.view(-1)
            labels = labels.type(torch.cuda.LongTensor)
            output = output.permute(0, 2, 3, 1).contiguous()
            output = output.view(-1, class_num)
        if len(output.size()) == 5:
            labels = labels.permute(0, 2, 3, 4, 1).contiguous()
            labels = labels.view(-1)
            labels = labels.type(torch.cuda.LongTensor)
            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output = output.view(-1, class_num)

        Loss = self.loss(output, labels)

        n_valid = 0
        for i in range(class_num):
            if i != self.ignore_index and len(Loss[labels == i].data.cpu().numpy()) > 0:
                loss_idx = torch.mean(Loss[labels == i])
                loss_sum += loss_idx
                n_valid += 1

        loss_mean = loss_sum / n_valid
        return loss_mean


class FocalLoss0(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)

        #        targets = targets.type(torch.cuda.LongTensor)

        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss1(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class DICELossMultiClasswithBG(nn.Module):

    def __init__(self,ignore_index = None):
        super(DICELossMultiClasswithBG, self).__init__()

    def forward(self, output, mask):

        num_classes = output.size(1)

        #add by lxw
        if num_classes > 1:
            output = F.softmax(output, dim=1)

        dice_eso = 0
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :, :], 1).float()
            mask_i = torch.squeeze(mask, 1).float()
            mask_i = (mask_i == i).float()

            num = probs * mask_i
            num = torch.sum(num, 3)
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            # print( num )

            den1 = probs * probs
            # print(den1.size())
            den1 = torch.sum(den1, 3)
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            # print(den1.size())

            den2 = mask_i * mask_i
            # print(den2.size())
            den2 = torch.sum(den2, 3)
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            # print(den2.size())
            eps = 0.0000001
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            # dice_eso = dice[:, 1:]
            #print(dice)
            dice_eso += dice

        loss = - torch.sum(dice_eso) / dice_eso.size(0)
        return loss/(num_classes)


class MultiClassDiceScorewithBG(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output, mask):
        # assert output.shape[1] == 1
        num_classes = int(torch.max(mask)) + 1
        dice_eso = 0
        for i in range(num_classes):
            probs_i = torch.squeeze(output, 1).float()
            mask_i = torch.squeeze(mask, 0).float()
            probs_i = (probs_i == i).float()
            mask_i = (mask_i == i).float()

            num = probs_i * mask_i
            # num = torch.sum(num, 3)
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            #add by lxw
            num = torch.sum(num, 0)

            # print( num )

            den1 = probs_i * probs_i
            # print(den1.size())
            # den1 = torch.sum(den1, 3)
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            #add by lxw
            den1 = torch.sum(den1, 0)

            # print(den1.size())

            den2 = mask_i * mask_i
            # print(den2.size())
            # den2 = torch.sum(den2, 3)
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            #add by lxw
            den2 = torch.sum(den2, 0)

            # print(den2.size())
            eps = 0.0000001
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            # print(dice)
            # dice_eso = dice[:, 1:]
            dice_eso += dice

        # loss = torch.sum(dice_eso) / dice_eso.size(0)
        loss = dice_eso #modify lxw
        return loss / num_classes
