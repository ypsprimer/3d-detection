import os
import time
import numpy as np

import torch
from torch.nn import DataParallel
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable


def forwardSingle(net, test_data, label_num):
    print('test_data 0', test_data.shape)

    test_data = torch.autograd.Variable(torch.from_numpy(test_data[np.newaxis, np.newaxis])).cuda()

    print('test_data 1', test_data.shape)

    output = net(test_data)  ##forward
    prob = torch.nn.functional.softmax(output, dim=1)

    return prob.data.cpu().numpy()[0][0]


def testForward(net, test_data_, train_para):  ##add Predict list

    prelabel_list = []

    ld = test_data_['s_data_list']

    for i in range(len(ld)):
        pre_s_label_list = []

        for j in range(len(ld)):
            #            for k in range(train_para['batch_size']):
            #                print('', test_data_['s_data_list'][i][j].shape)
            prelabel = forwardSingle(net, test_data_['s_data_list'][i][j], 1)

            print('prelabel.sum', prelabel.sum())

            pre_s_label_list.append(prelabel)

        prelabel_list.append(pre_s_label_list)

    test_data_['s_predict_list'] = prelabel_list

    return test_data_


def testForward2(net, test_loader, test_para):
    for i, (data, target, name) in enumerate(data_loader):
        pass
    return
