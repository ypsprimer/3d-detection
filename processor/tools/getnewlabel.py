import numpy as np
import cv2


def getSurroundLabel(label):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    label_sur = cv2.dilate(label, kernel)
    
    return label_sur

    
def calcNewLabel(label_sur, label):
    
    labeln = label_sur - label
    
    labeln[labeln>0]=16
    
    labeln = labeln + label
    
    return labeln


def getNewLabel(cur_case_path, cur_prep_path):
    
    print('cur_case_path', cur_case_path)
    label = np.load(cur_case_path)
    
    label_sur = getSurroundLabel(label)
    
    label_n = calcNewLabel(label_sur, label)
    
    
    np.save(cur_prep_path, label_n)
    
    return