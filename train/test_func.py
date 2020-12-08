#
#  training_func.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from collections import OrderedDict

from wavenet_MTL import SKUNET
from wavenet_MTL import FCResNET


from lossfunc import DiceLoss
from net_io_utils import loadNet
from net_io_utils import saveNetPara
from net_io_utils import writeLossLog
from net_io_utils import writeLossLogWithEM




def prepareNet(net, lossfun, net_weight_file):
    
    if net_weight_file != '':
#        net = loadNet(net, net_weight_file)
        state = torch.load(net_weight_file)
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(state)
    
    use_cuda = torch.cuda.is_available()
    
    print('Use cuda: ' + str(use_cuda))
    
    if use_cuda:
        net = net.cuda()
        lossfun = lossfun.cuda()
        cudnn.benchmark = True
        net = DataParallel(net)
        
    
    
    
    return net, lossfun




    

    
def test(data_loader, net, lossfun, em, epoch, save_dir):
    
    use_cuda = torch.cuda.is_available()
    
    net.eval()

    metrics_main = []
    metrics_em = []
    for i, (data, target, name) in enumerate(data_loader):
        if use_cuda:
            data = Variable(data.cuda(async = True), volatile = True)
            target = Variable(target.cuda(async = True), volatile = True)
        else:
            data = Variable(data, volatile = True)
            target = Variable(target, volatile = True)

        output = net(data)

        prob = torch.nn.functional.softmax(output, dim = 1)
        prob.data.cpu().numpy()[0][0]
        

    writeLossLog(save_dir, epoch, np.mean(metrics_main), 'test_loss')

    return np.mean(metrics_main)
        

    
    
        


def testFunc( train_loader, net_para, train_para, envi_para, output_para):
    
    em = net_para['em']
    net = net_para['net']
    lossfun = net_para['lossfun']
    
    epoch = train_para['epoch']
    weight_decay = train_para['weight_decay']
    learning_rate = train_para['learning_rate']
    
    folder_label = output_para['folder_label']
    save_frequency = output_para['save_frequency']
    save_dir = output_para['save_dir']

    vloss = test(data_loader, net, lossfun, em, epoch, save_dir)
    
    writeLossLog(save_dir, epoch_i, vloss, 'test_loss')
        

