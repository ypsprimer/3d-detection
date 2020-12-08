import os
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



#def getSingleCombinePredict(pt1, pt2, spre_list, spt1_list, spt2_list, crop_size, cut_lenth, cut_stride, cut_flag):
def getSingleCombinePredict(pt1, pt2, spre_list, spt1_list, spt2_list, cut_lenth):

    for i in range(3):
        if pt2[i]<pt1[i]:
            ValueError('wrong pt for combine')
            
    pre_data = np.full([int(pt2[0])-int(pt1[0]), int(pt2[1])-int(pt1[1]), int(pt2[2])-int(pt1[2])], 0.0)

#    print('len:', int(pt2[0])-int(pt1[0]), int(pt2[1])-int(pt1[1]), int(pt2[2])-int(pt1[2]))
    
    for i in range(len(spre_list)):
        spt1 = spt1_list[i] - pt1
        spt2 = spt2_list[i] - pt1

##        print('spt1, spt2, cut_lenth', spt1, spt2, pt1, cut_lenth, spt1_list[i], spt2_list[i], i)
        
#        print('spt:', spt1, spt2, pre_label.shape, spre_list[i].shape)
##        print('range', int(spt1[0]), int(spt1[0])+cut_lenth[0], int(spt1[1]), int(spt1[1])+cut_lenth[1], int(spt1[2]), int(spt1[2])+cut_lenth[2], spre_list[i].shape)
    
#        print('crop_size', crop_size)
        
        pre_data[int(spt1[0]):int(spt1[0])+cut_lenth[0], int(spt1[1]):int(spt1[1])+cut_lenth[1], int(spt1[2]):int(spt1[2])+cut_lenth[2]] = spre_list[i]
        
##        print('sum:', spre_list[i].sum(), pre_data.sum())
#        pre_label[int(spt1[0]):int(spt1[0])+cut_lenth[0], int(spt1[1]):int(spt2[1])+cut_lenth[0], int(spt1[2]):int(spt2[2])+cut_lenth[0]] = spre_list[i]  ###may have prob, need debug.
    
    
    return pre_data


def combinePredictlist(pt1_list, pt2_list, s_pre_list, s_pt1_list, s_pt2_list, crop_size, cut_lenth):

    cpredict_list = []
    
    for i in range(len(pt1_list)):
        cpredict = getSingleCombinePredict(pt1_list[i], pt2_list[i], s_pre_list[i], 
                                           s_pt1_list[i], s_pt2_list[i], 
                                           cut_lenth)

        cpredict_list.append(cpredict)
    
    return cpredict_list


def putPredictResultToImgRegion(cpredict_list, pt1_list, pt2_list, data_list):
    im_list = []
    
    for i in range(len(cpredict_list)):
        imshape = data_list[i].shape
        
        im = np.full(imshape, 0.0)
        
        pt = pt1_list[i]
        pt2 = pt2_list[i]
        im_l = cpredict_list[i].shape
        
        
#        print('imshape, im_l', im.shape, im_l, pt, pt2)
        
        im[int(pt[0]):int(pt[0]+im_l[0]), int(pt[1]):int(pt[1]+im_l[1]), int(pt[2]):int(pt[2]+im_l[2])] = cpredict_list[i]
        

#        print('combine shape', im.shape, imshape)
        im_list.append(im)
    
    return im_list


def savePredictlist(output_para, cpredict_list, data_namelist):
    
    for i in range(len(cpredict_list)):
        save_path = output_para['save_dir'] + '/' + output_para['time_id'] + '/'
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        save_path = save_path + output_para['folder_label'] + '/'
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        save_path = save_path + '/' + data_namelist[i] + '_test.npy'
        
        print(save_path)
        np.save(save_path, cpredict_list[i])
    
    return



#######this func designed for test output
def combinePredictResult(_data_para, output_para):
    cpredict_list = combinePredictlist(_data_para['pos1_list'], 
                                        _data_para['pos2_list'],
                                       _data_para['s_predict_list'], _data_para['s_pos1_list'], 
                                        _data_para['s_pos2_list'], _data_para['crop_size'], 
                                        _data_para['cut_lenth'])

    
    impredict_list = putPredictResultToImgRegion(cpredict_list, _data_para['pos1_list'], _data_para['pos2_list'], _data_para['data_list'])
    
    savePredictlist(output_para, impredict_list, _data_para['data_namelist'])
#    savePredictlist(output_para, cpredict_list, _data_para['data_namelist'])

    
    return



    
    
    
    
    
    
    
    
    
    
    
def savePredictlistForValidation(path, name_sign, cpredict_list, data_namelist):
    
    for i in range(len(cpredict_list)):
        
        path = path + '/' + data_namelist[i] + '_'+name_sign #'_val_predict.npy'
        
        print(path)
        np.save(path, cpredict_list[i])
    
    return  
    
    
    
def getSingleCombinePredictForVal(pt1, pt2, spre_list, spt1_list, spt2_list, cut_lenth, s_cpt1_list, s_cpt2_list):

    for i in range(3):
        if pt2[i]<pt1[i]:
            ValueError('wrong pt for combine')
            
    pre_data = np.full([int(pt2[0])-int(pt1[0]), int(pt2[1])-int(pt1[1]), int(pt2[2])-int(pt1[2])], 0.0)

#    print('len:', int(pt2[0])-int(pt1[0]), int(pt2[1])-int(pt1[1]), int(pt2[2])-int(pt1[2]))
    
    for i in range(len(spre_list)):
        spt1 = spt1_list[i] - pt1
        spt2 = spt2_list[i] - pt1
        
##        print('crop pt: ', spt1_list, s_cpt1_list, '    ', spt2_list, s_cpt2_list)

##        print('spt1, spt2, cut_lenth', spt1, spt2, pt1, cut_lenth, spt1_list[i], spt2_list[i], i)
        
#        print('spt:', spt1, spt2, pre_label.shape, spre_list[i].shape)
##        print('range', int(spt1[0]), int(spt1[0])+cut_lenth[0], int(spt1[1]), int(spt1[1])+cut_lenth[1], int(spt1[2]), int(spt1[2])+cut_lenth[2], spre_list[i].shape)
    
        
        
        shift_ = s_cpt1_list[i] - spt1_list[i]
        
#        print('crop_size', crop_size)
        
        pre_data[int(spt1[0]):int(spt1[0])+cut_lenth[0], int(spt1[1]):int(spt1[1])+cut_lenth[1], int(spt1[2]):int(spt1[2])+cut_lenth[2]] = spre_list[i][int(shift_[0]):int(shift_[0])+cut_lenth[0], int(shift_[1]):int(shift_[1])+cut_lenth[1], int(shift_[2]):int(shift_[2])+cut_lenth[2]]
        
###        print('sum:', spre_list[i].sum(), pre_data.sum())
#        pre_label[int(spt1[0]):int(spt1[0])+cut_lenth[0], int(spt1[1]):int(spt2[1])+cut_lenth[0], int(spt1[2]):int(spt2[2])+cut_lenth[0]] = spre_list[i]  ###may have prob, need debug.
    
    
    return pre_data


def combinePredictlistForVal(pt1_list, pt2_list, s_pre_list, s_pt1_list, s_pt2_list, crop_size, cut_lenth, s_cpt1_list, s_cpt2_list):

    cpredict_list = []
    
    for i in range(len(pt1_list)):
        cpredict = getSingleCombinePredictForVal(pt1_list[i], pt2_list[i], s_pre_list[i], 
                                           s_pt1_list[i], s_pt2_list[i], 
                                           cut_lenth, s_cpt1_list, s_cpt2_list)

        cpredict_list.append(cpredict)
    
    return cpredict_list



#######this func designed for validation output
def combinePredictResultForValidation(_data_para, path, name_sign):
    cpredict_list = combinePredictlistForVal(_data_para['pos1_list'], 
                                        _data_para['pos2_list'],
                                       _data_para['s_predict_list'], _data_para['s_pos1_list'], 
                                        _data_para['s_pos2_list'], _data_para['crop_size'], 
                                        _data_para['cut_lenth'], _data_para['s_crop_pos1_list'], _data_para['s_crop_pos2_list'])

    
    impredict_list = putPredictResultToImgRegion(cpredict_list, _data_para['pos1_list'], _data_para['pos2_list'], _data_para['ori_data_list'])
    
    savePredictlistForValidation(path, name_sign, impredict_list, _data_para['data_namelist'])
    
    return



def makeCombineList(in_data_list, in_data_name, dataloader):

    cdict = dataloader.dataset.name_dict
    spt1 = dataloader.dataset.sample_pt1
    spt2 = dataloader.dataset.sample_pt2
    ori_sample_len = dataloader.dataset.ori_data_num
    sub_sample_len_list = dataloader.dataset.ori_data_sub_num
    
    arr = np.array([0,0,0])
    
#init data list    
    data_list = []
    spt1_list = []
    spt2_list = []
    
##    print ('ori_sample_len', ori_sample_len)


##    print('cid_len', len(cdict), cdict)
##    print('spt1', spt1)
##    print('spt2', spt2)
    
    
    for i in range(ori_sample_len):
        s_data_list = []
        s_pt1_list = []
        s_pt2_list = []
        
##        print ('len:', sub_sample_len_list[i], i)
        
        for j in range(sub_sample_len_list[i]):
            s_data_list.append(arr)
            s_pt1_list.append(arr)
            s_pt2_list.append(arr)
            
        data_list.append(s_data_list)
        spt1_list.append(s_pt1_list)
        spt2_list.append(s_pt2_list)

        
    for i in range(len(in_data_list)):
        
##        print('in_data_list[i].shappe', in_data_list[i].shape, i)
#        print ('in_data_list_len:', len(in_data_list[i]), i )
        
        for j in range(len(in_data_list[i])): #batch
            #in_data_list[i][j][0]
            curr_name = in_data_name[i][j]
            
            cid = cdict[curr_name]
            pt = cdict[cid]
            
##            print('cid', cid, len(spt1))
            
            
#            print('pt:', pt[0], pt[1], i, j, 'data_list_len:', len(data_list), len(data_list[i]))
#            print('pt:', pt[0], pt[1], i, j, 'data_list_len:', len(data_list), len(data_list[i]))
            
            data_list[pt[0]][pt[1]] = in_data_list[i][0].cpu().numpy().copy()
            spt1_list[pt[0]][pt[1]] = spt1[cid-1]
            spt2_list[pt[0]][pt[1]] = spt2[cid-1]
            
##            print('spt1, spt2', spt1[cid-1], spt2[cid-1])
        
##            print ('data_list.shape:', data_list[pt[0]][pt[1]].shape, pt[0], pt[1])
#            curr_name

    _data_para = {}
    
    _data_para['pos1_list'] = dataloader.dataset.pt1_list
    _data_para['pos2_list'] = dataloader.dataset.pt2_list
    _data_para['s_pos1_list'] = spt1_list
    _data_para['s_pos2_list'] = spt2_list
    _data_para['crop_size'] = dataloader.dataset.crop_size
    _data_para['cut_lenth'] = dataloader.dataset.cut_len
    _data_para['s_crop_pos1_list'] = dataloader.dataset.crop_pt1
    _data_para['s_crop_pos2_list'] = dataloader.dataset.crop_pt2
    _data_para['ori_data_list'] = dataloader.dataset.ori_data
    _data_para['data_namelist'] = dataloader.dataset.ori_data_name

    _data_para['s_predict_list'] = data_list
    
##    print('spt1_list, spt2_list', spt1_list, spt2_list)
    
    return _data_para
    


#    savePredictlist(output_para, cpredict_list, _data_para['data_namelist'])
def combineProcess(in_data_list, in_data_name, path, name_sign, dataloader):
    
    _data_para = makeCombineList(in_data_list, in_data_name, dataloader)
    
    pt1_lst = _data_para['pos1_list']
    cut_len = _data_para['cut_lenth']
    s_pos1_list = _data_para['s_pos1_list']
    s_pos2_list = _data_para['s_pos2_list']
    
##    print('pt1, len:', pt1_lst, cut_len)
#    print('s_pos1, s_pos2', s_pos1_list, s_pos2_list)
    
    combinePredictResultForValidation(_data_para, path, name_sign)
    
    return


    
    

