# -*- coding: utf-8 -*-
#
#  lr_scheme.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#

# 四分之一开始衰减
def base_lr(learning_rate, epoch_i, epoch):
    if epoch_i <= epoch * 0.25:
        lr = learning_rate
    elif epoch_i <= epoch * 0.5:
        lr = 0.1 * learning_rate
    elif epoch_i <= epoch * 0.75:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    return lr

# 一半开始衰减
def base_lr_1(learning_rate, epoch_i, epoch):
    if epoch_i <= epoch * 0.5:
        lr = learning_rate
    elif epoch_i <= epoch * 0.75:
        lr = 0.1 * learning_rate
    elif epoch_i <= epoch * 0.9:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    return lr

# 恒定
def constant(learning_rate, epoch_i, epoch):
    return learning_rate
