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


LABEL_ENCODING_TABLE = {"AC":1,
              "LAD":2,
              "LCX":4,
              "RCA":8}

def load_scan(path):
#    print('path : %s' % path)
    filelist = os.listdir(path)
    os.system('find ' + path + " -name '*.dcm' > tmp1.lst")
    os.system('sort -n tmp1.lst > tmp.lst')
    num = len(filelist)
    img = list()
    
    num_name = ''
    sample_len = 0
    for line in open('tmp.lst', 'r'):
        sample_len = sample_len + 1
#    print sample_len

    count = 0
    path3 =[]
    for line in open('tmp.lst', 'r'):
        path2 = line[:-1]
        count = count+1
#        print 'len = ' + str(len(line))

        if count>0:
##        if count>=start_count and count<=end_count:
#            print path
            path3.append(path2)
            name = path2.split('/')[-1]
#            print name
            name = name.split('.')
            num_name = name[len(name)-2]

#            print path2

#    print path3
#    slices = [dicom.read_file(s) for s in path3 if s.endswith('.dcm')]
#            slices = [dicom.read_file(path2)]

#    for s in os.listdir(path):
#        if s.endswith('.dcm'):
#            path3 = path + '/' + s
#            print 'path3 ' + str(path3)
#            slices = [dicom.read_file(path3)]
    
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if s.endswith('.dcm')]
    
    
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices



def load_label(case_path, y_dim, x_dim):
    
    filelist = [f.split('.jpg')[0] for f in os.listdir(case_path) if f.endswith('.jpg')]
    filelist = sorted(filelist, reverse = False)
    
    label = np.full([y_dim,x_dim,len(filelist)],0.0)

    
    count=0

    for i in range(len(filelist)):
        labellist = [f.split('.png')[0] for f in os.listdir(case_path) if (f.startswith(filelist[i]) and f.endswith('.png'))]
        img_temp = np.full([y_dim,x_dim],0.0)
    
        count = count+1
        
        for l in labellist:
#                print 'labellist:'
#                print labellist
            mpimg = imread(case_path +'/' + l + '.png', 0)
    
            shortname = l.split('_')[-1]
    
#            if shortname=='AC':
#                mpimg[mpimg>0] = 1
#            if shortname=='LAD':
#                mpimg[mpimg>0] = 2
#            if shortname=='LCX':
#                mpimg[mpimg>0] = 4
#            if shortname=='RCA':
#                mpimg[mpimg>0] = 8
                
            mpimg[mpimg>0] = LABEL_ENCODING_TABLE[shortname]
            
            img_temp = img_temp + mpimg            
            
            label[:,:,len(filelist) - count] = img_temp

    
    label = label.transpose(2,0,1)

    return label


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)




def saveSpacingPara(save_path, spacing_z, spacing_y, spacing_x):
    fp = open(save_path, 'w')
    fp.write(str(spacing_z) + '\n' + str(spacing_y) + '\n' + str(spacing_x) + '\n')
    fp.close()



def getSingleOriNPY(case_path, prep_path_ori, prep_path_label, prep_path_spacing):
    
##load ori data
    case = load_scan(case_path)
    im, spacing = get_pixels_hu(case)
    #     print(im[150])
    #     plt.hist(im[150].reshape([-1]))
    im[np.isnan(im)] = -2000
    
##load label
    label = load_label(case_path, im.shape[1], im.shape[2])
    
    #    im = np.clip(im,-2000,2000)
    
##convert data type
    im3 = im.astype('int16')
    label3 = label.astype('uint8')
    
##save npy file
    np.save(prep_path_ori, im3)
    np.save(prep_path_label, label3)

##save txt file
    saveSpacingPara(prep_path_spacing, spacing[0], spacing[1], spacing[2])


#    np.save(prep_path_spacing, spacing)


