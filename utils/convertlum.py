# -*- coding: utf-8 -*-
#
#  converlum.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#

import numpy as np


def convertLum(im_list, win_low, win_high):
    
    im_lum_list = []
    
    for i in range(len(im_list)):
        im = im_list[i]
        
        im_lum = lumTrans(im, win_low, win_high)
        
        im_lum_list.append(im_lum)
        
    
    return im_lum_list

def lumTrans(img, win_low, win_high):
    lungwin = np.array([win_low, win_high])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    
    newimg[newimg<0]=0
    newimg[newimg>1]=1
###test    
#    newimg = newimg.astype('float32')
    return newimg


