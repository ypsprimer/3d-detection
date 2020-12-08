#
#  washdata.py
#  preprocessing
#
#  Created by AthenaX on 27/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#

from skimage import io
import os, sys, time
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import dicom
from scipy.ndimage.interpolation import zoom
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from extract_centerline import find_3D_object_voxel_list
try:
    from cv2 import imread, imwrite, GaussianBlur
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
# TODO: Use scipy instead.




def washYDirectSum(label, del_size):
#find del_pos    
    label_sum_y = label.mean(1)
    label_sum_y[label_sum_y>0] = 255

    label_sum_y = label_sum_y.astype('uint8')
#    print(label_sum_y.shape)
    ret, limsg_bin_y = cv2.threshold(label_sum_y, 120, 255, cv2.THRESH_BINARY)
    limsg_bin_y = limsg_bin_y.astype('uint8')
    _, _contours, hierarchy = cv2.findContours(limsg_bin_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cont in _contours:
        x, y, w, h = cv2.boundingRect(cont)
        if w<del_size or h<del_size:
#            print (x,y, w,h)       
            r = w
            if h>r:
                r=h
            cv2.circle(limsg_bin_y,(x,y), r, (160,0,255), -1)
            
    del_sumy_pos = np.where(limsg_bin_y==160)

    np_del_sumy_pos = np.array(del_sumy_pos)
#    print(np_del_sumy_pos.shape)

#find del_pos in x,y,z
    del_sumy_arr = []
    for i in range(np_del_sumy_pos.shape[1]):
        for j in range(label.shape[1]):
            if label[del_sumy_pos[0][i], j ,del_sumy_pos[1][i]]>0:
                del_sumy_arr.append([del_sumy_pos[0][i], j, del_sumy_pos[1][i]])
#    print(del_sumy_arr)

#del points
    label_output = np.array(label.shape, dtype='uint8')
    label_output = label.copy()

    label_as = np.array(label.shape, dtype='float32')
    label_as = label.copy()

    label_as[label_as>=1]=255

    label_zero = np.array(label.shape, dtype='uint8')
    label_zero = label.copy()
    label_zero[label_zero>0]=0


#del_points
    buf_len = 3
    for i in range(len(del_sumy_arr)):
        curr_pt = del_sumy_arr[i]
        if curr_pt[0]>=buf_len and curr_pt[0]<label_as.shape[0]-buf_len:
            zrange = [curr_pt[0]-buf_len, curr_pt[0]+buf_len]
        
        elif curr_pt[0]<buf_len:
            zrange = [0, buf_len*2]
        elif curr_pt[0]>label_as.shape[0]-buf_len:
            zrange = [label_as.shape[0]-buf_len, label_as.shape[0]]
    
        label_piece = label_as[zrange[0]:zrange[1], :, :].mean(0)
    
        val = label_piece[curr_pt[1]][curr_pt[2]]
    
        label_output[curr_pt[0]-1:curr_pt[0]+1, curr_pt[1]-1:curr_pt[1]+1, curr_pt[2]-1:curr_pt[2]+1] = label_zero[curr_pt[0]-1:curr_pt[0]+1, curr_pt[1]-1:curr_pt[1]+1, curr_pt[2]-1:curr_pt[2]+1]


    return label_output








def washXDirectSum(label, del_size):
#find del_pos     
    label_sum_x = label.mean(2)
    label_sum_x[label_sum_x>0] = 255

    label_sum_x = label_sum_x.astype('uint8')

    print(label_sum_x.shape)
    ret, limsg_bin_x = cv2.threshold(label_sum_x, 120, 255, cv2.THRESH_BINARY)
    limsg_bin_x = limsg_bin_x.astype('uint8')
    _, _contours, hierarchy = cv2.findContours(limsg_bin_x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cont in _contours:
        x, y, w, h = cv2.boundingRect(cont)
        if w<3 or h<3:
#            print (x,y, w,h)       
            r = w
            if h>r:
                r=h
            cv2.circle(limsg_bin_x,(x,y), r, (160,0,255), -1)

#find del_pos in x,y,z
    del_sumx_pos = np.where(limsg_bin_x==160)
#    print(del_pos)
    np_del_sumx_pos = np.array(del_sumx_pos)
#    print(np_del_sumx_pos.shape)

    del_sumx_arr = []
    for i in range(np_del_sumx_pos.shape[1]):
        for j in range(label.shape[1]):
            if label[del_sumx_pos[0][i], j, del_sumx_pos[1][i]]>0:
                del_sumx_arr.append([del_sumx_pos[0][i], j, del_sumx_pos[1][i]])
#    print(del_sumx_arr)


#del points
    label_output = np.array(label.shape, dtype='uint8')
    label_output = label.copy()
    

    label_as = np.array(label.shape, dtype='float32')
    label_as = label.copy()

    label_as[label_as>=1]=255


    label_zero = np.array(label.shape, dtype='float32')
    label_zero = label.copy()
    label_zero[label_zero>0]=0

#    print(label_as.sum())

#del_points
    for i in range(len(del_sumx_arr)):
        curr_pt = del_sumx_arr[i]
        if curr_pt[0]>=3 and curr_pt[0]<label_as.shape[0]-3:
            zrange = [curr_pt[0]-3, curr_pt[0]+3]
        
        elif curr_pt[0]<3:
            zrange = [0, 3+3]
        elif curr_pt[0]>label_as.shape[0]-3:
            zrange = [label_as.shape[0]-3*2, label_as.shape[0]]
    
        label_piece = label_as[zrange[0]:zrange[1], :, :].mean(0)
    
        val = label_piece[curr_pt[1]][curr_pt[2]]
    
#    print(val)

        control_x_val = 200
    
#    print(label_zero.sum())
        if val<control_x_val:
            label_piece[val<control_x_val]=0

#        label_output[curr_pt[0], :, :] = label_piece
#        print(label_output[curr_pt[0], :, :].sum())
            label_output[curr_pt[0]-1:curr_pt[0]+1, curr_pt[1]-1:curr_pt[1]+1, curr_pt[2]-1:curr_pt[2]+1] = label_zero[curr_pt[0]-1:curr_pt[0]+1, curr_pt[1]-1:curr_pt[1]+1, curr_pt[2]-1:curr_pt[2]+1]


    return label_output


def washData(label):

    label_output = washYDirectSum(label, 3)
    
    label_output2 = washYDirectSum(label_output, 3)
    
    
    return label_output2.astype('uint8')





#label = np.load(cur_case_path)
#label_new = washData(label)
