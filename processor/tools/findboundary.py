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

from findboundary_new import findBoundaryX


def findEdgeOfNPYLabel(case_path, edge_buf):
    
    label = np.load(case_path)
    
    rpt1_x = 999
    rpt1_y = 999
    rpt2_x = 0
    rpt2_y = 0
    
    z_max = -1
    z_min = 0
    count = 0
    for i in range(label.shape[0]):
        
        label_s = label[i].copy()
        label_count = label[i].copy()
        
        label_s[label_s>0]=255
        
        ret, label_s_bin = cv2.threshold(label_s, 120, 255, cv2.THRESH_BINARY)
        
        _, label_contours, hierarchy = cv2.findContours(label_s_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in label_contours:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w>1 or h>1:
                if rpt1_x>x and x>edge_buf:
                    rpt1_x = x
                if rpt1_y>y and y>edge_buf:
                    rpt1_y = y
                
                if rpt2_x<x+w and x+w<label.shape[2]-edge_buf:
                    rpt2_x = x+w
                if rpt2_y<y+h and y+h<label.shape[1]-edge_buf:
                    rpt2_y = y+h
    
        #get z min & max
#        print('i', label[i].sum(), i)
        label_count[label_count<=1]=0
        label_count[label_count>1]=1
        
        num = label_count.sum()
        
        if num>0 and z_max==-1:
            z_max = label.shape[0] - i
        
        if num==0 and z_max!=-1 and z_min==0:
            z_min = label.shape[0] - i + 1


    return rpt1_x, rpt1_y, z_min, rpt2_x, rpt2_y, z_max









def findXYEdgeOfNPYLabel(case_path, edge_buf):
    
    label = np.load(case_path)
    
    rpt1_x = 999
    rpt1_y = 999
    rpt2_x = 0
    rpt2_y = 0
    

    count = 0
    for i in range(label.shape[0]):
        
        label_s = label[i].copy()
        
        label_s[label_s>0]=255
        
        ret, label_s_bin = cv2.threshold(label_s, 120, 255, cv2.THRESH_BINARY)
        
        _, label_contours, hierarchy = cv2.findContours(label_s_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in label_contours:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w>1 or h>1:
                if rpt1_x>x and x>edge_buf:
                    rpt1_x = x
                if rpt1_y>y and y>edge_buf:
                    rpt1_y = y
                
                if rpt2_x<x+w and x+w<label.shape[2]-edge_buf:
                    rpt2_x = x+w
                if rpt2_y<y+h and y+h<label.shape[1]-edge_buf:
                    rpt2_y = y+h


    return rpt1_x, rpt1_y, rpt2_x, rpt2_y





def findXZEdgeOfNPYLabel(case_path, edge_buf):
    
    label = np.load(case_path)
    
    rpt1_x = 999
    rpt1_z = 999
    rpt2_x = 0
    rpt2_z = 0
    
    
    count = 0
    for i in range(label.shape[1]):
        
        label_s = label[:,i,:].copy()
        
        label_s[label_s>0]=255
        
        ret, label_s_bin = cv2.threshold(label_s, 120, 255, cv2.THRESH_BINARY)
        
        _, label_contours, hierarchy = cv2.findContours(label_s_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in label_contours:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w>1 or h>1:
                if rpt1_x>x and x>edge_buf:
                    rpt1_x = x
                if rpt1_z>y and y>edge_buf:
                    rpt1_z = y
                
                if rpt2_x<x+w and x+w<label.shape[2]-edge_buf:
                    rpt2_x = x+w
                if rpt2_z<y+h and y+h<label.shape[1]-edge_buf:
                    rpt2_z = y+h


    return rpt1_x, rpt1_z, rpt2_x, rpt2_z






def findEdgeOfNPYLabel(case_path):
    
    label = np.load(case_path)
    
    z1, y1, x1, z2, y2, x2 = findBoundaryX(label)
    
#    if z2==276:
#        print('case', case_path, label.shape)
        
    if z2>label.shape[0]:
        print('z case', case_path)
        z2 = label.shape[0]
    if y2>label.shape[1]:
        print('y case', case_path)
        y2 = label.shape[1]
    if x2>label.shape[2]:
        print('x case', case_path)
        x2 = label.shape[2]
        
    if x1<0:
        print('xn case', case_path)
        x1=0
    if y1<0:
        print('yn case', case_path)
        y1=0
    if z1<0:
        print('zn case', case_path)
        z1=0
        
    print(z2, y2, x2, label.shape, case_path)

    return z1, y1, x1, z2, y2, x2



def rewriteNPY(case_path, z1, y1, x1, z2, y2, x2):

    label = np.load(case_path)

    labeln = np.full(label.shape, 0)

    labeln[z1:z2, y1:y2, x1:x2] = label[z1:z2, y1:y2, x1:x2]

    labeln = labeln.astype('uint8')
    
    np.save(case_path, labeln)



def saveAllPara(prep_path, z1, y1, x1, z2, y2, x2, spac_z, spac_y, spac_x):
    fp = open(prep_path, 'w')
    
    fp.write(str(int(z1)) + '\n' + str(int(y1)) + '\n' + str(int(x1)) + '\n')
    
    fp.write(str(int(z2)) + '\n' + str(int(y2)) + '\n' + str(int(x2)) + '\n')
    
    fp.write(str(spac_z) + '\n' + str(spac_y) + '\n' + str(spac_x) + '\n')
    
    fp.close()
    
    return


def readSpacingPara(case_path):
    f = open(case_path,"r")
    
    lines = f.readlines()
    
    sp_z = float(lines[0])
    sp_y = float(lines[1])
    sp_x = float(lines[2])
    
    return sp_z, sp_y, sp_x


def getBoundaryOfLabel(case_path, case_spacing_path, prep_path):
    
#    x1, y1, z1, x2, y2, z2 = findEdgeOfNPYLabel(case_path, 10)

#    x1, y1, x2, y2 = findXYEdgeOfNPYLabel(case_path, 10)
    
    #    print (x1, y1, x2, y2)
    
    #    x1, z1, x2, z2 = findXZEdgeOfNPYLabel(case_path, 0)
    
    #    print (x1, z1, x2, z2)
    
    
    z1, y1, x1, z2, y2, x2 = findEdgeOfNPYLabel(case_path)
    

    spacing_z, spacing_y, spacing_x = readSpacingPara(case_spacing_path)
    
    saveAllPara(prep_path, z1, y1, x1, z2, y2, x2, spacing_z, spacing_y, spacing_x)
    
    rewriteNPY(case_path, z1, y1, x1, z2, y2, x2)

    
    return
