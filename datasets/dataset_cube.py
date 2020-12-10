import os
import time
import random
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from scipy.ndimage import zoom, rotate, binary_dilation

from .split_combine import SplitComb, mypad
# from split_combine import SplitComb, mypad

from skimage.transform import rotate as rotate2d

# from .label_map_cube import Label_mapping_cube as Label_mapping
from .label_map2_2 import Label_mapping2 as Label_mapping
# from label_map2_2 import Label_mapping2 as Label_mapping

import warnings
import numpy as np

from tqdm import tqdm

class myDataset(Dataset):
    def __init__(self, config, phase):

        assert phase in ['train', 'val','valtrain', 'test']
        self.phase = phase
        self.config = config
        # 用于train的文件路径
        if phase == 'train':
            split = config.prepare['train_split']
        elif phase == 'valtrain':
            split = config.prepare['train_split']
            self.split_comb = SplitComb(config)
        elif phase == 'val':
            split = config.prepare['val_split']
            self.split_comb = SplitComb(config)
        elif phase == 'test':
            split = config.prepare['all_split']
            self.split_comb = SplitComb(config)
        with open(split, 'r') as f:
            self.cases = f.readlines()
        black_list = set(config.prepare['blacklist'])
        self.cases = [f.split('\n')[0] for f in self.cases]
        self.cases = [c for c in self.cases if c not in black_list]

        self.cases2idx = {case_name: idx for idx, case_name in enumerate(self.cases)}
        datadir = config.prepare['data_dir']

        self.nodule_info = np.load(config.prepare['nodule_info'])
        
        self.img_files = [os.path.join(datadir, f + config.prepare['img_subfix']) for f in self.cases]
        self.lab_files = [os.path.join(datadir, f + config.prepare['lab_subfix']) for f in self.cases]
        self.left_lung_files = [os.path.join(datadir, f + '_left' + config.prepare['lung_subfix']) for f in self.cases]
        self.right_lung_files = [os.path.join(datadir, f + '_right' + config.prepare['lung_subfix']) for f in self.cases]

        #self.lab_buffers = [np.load(item)[np.newaxis] for item in self.lab_files]
        #self.img_buffers = {item: np.load(item)[np.newaxis] for item in self.img_files[:2000]}
        
        self.label_maper = Label_mapping(config)
        self.crop = Crop_lung(config)
        self.lab_N = config.rpn['N_prob']
        self.size_lim = config.rpn['diam_thresh']
        self.omit_cls = config.classifier['omit_cls']

    @staticmethod
    def mycat(arr_list, axis, dtype = 'float32'):
        shapes = [np.array(arr.shape) for arr in arr_list]
    #     assert np.all(len(s)==)
        newshape = shapes[0].copy()
        for i in range(1, len(shapes)):
            newshape[axis] += shapes[i][axis]
        newx = np.empty(newshape, dtype=dtype)

        start = 0
        for arr in arr_list:
            index = [slice(None)]*len(shapes[0])
            index[axis] = slice(start, start+arr.shape[axis])
            newx[tuple(index)] = arr
            start += arr.shape[axis]
        return newx

    def getraw(self, idx):
        #if self.img_files[idx] in self.img_buffers:
        #    img = self.img_buffers[self.img_files[idx]]
        #else:
        #    img = np.load(self.img_files[idx])[np.newaxis]
        img = np.load(self.img_files[idx])[np.newaxis]
        #lab = np.load(self.lab_files[idx])[np.newaxis]

        lab = self.lab_buffers[idx]
        if self.config.prepare['label_para']['label_buffer']:
                lab = self.__label_buffer__(lab)
        if self.extradata:
            exdata = np.load(self.exdata_files[idx])
            exdata = (exdata>0).astype(img.dtype)*1000
            img = self.mycat([img,exdata], 0,  img.dtype)

        if self.extra:
            exlab = np.load(self.exlab_files[idx])[np.newaxis]
            lab = self.mycat([lab, exlab], 0, 'uint8')
        return img, lab

    def __dtype__(self, x):
        if self.config.half:
            x = x.astype('float16')
        else:
            x = x.astype('float32')
        return x

    def __getitem__(self, idx, debug=False):
        idx = idx % len(self.cases)
        # 加入时间戳的随机seed
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        if debug:
            print(self.img_files[idx])

        if self.phase == 'train':
            img = np.load(self.img_files[idx])
            img = img.astype(np.int16)
            
            if self.config.prepare['noise']==True:
                noise = self.config.prepare['noise_amp']*np.random.randn(img.shape[0],img.shape[1],img.shape[2])
                img = img+noise
            
            lab_raw = np.load(self.lab_files[idx])

            # 左右肺叶取样点
            left_lung_raw = np.load(self.left_lung_files[idx])
            right_lung_raw = np.load(self.right_lung_files[idx])
            # lung_raw = np.concatenate((left_lung_raw, right_lung_raw))

            # 把5去除？
            # lab = np.array([item for item in lab_raw if item[4] not in [6,7]])
            # lab = np.array([item for item in lab_raw if item[4] not in [5,6,7]])
            
            lab = lab_raw

            # 在z上取平均，not used
            if np.random.rand() < 0.3 and False:
                newimg = []
                cp_idx = np.random.randint(2,9)
                for i in range(0,img.shape[0],cp_idx):    
                    if i+cp_idx< img.shape[0]:
                        newimg.append(np.mean(img[i:i+cp_idx],axis = 0, keepdims=True))
                newimg = np.squeeze(np.array(newimg))
                newimg= F.interpolate(torch.tensor(newimg).unsqueeze(0).unsqueeze(0),scale_factor=[cp_idx,1,1], mode='trilinear')[0,0]
                img = newimg.numpy()
            # for i in range(len(lab)):
            #     if lab[i,4]==1 or lab[i,4]==2:
            #         lab[i,5] == 1
            if len(img.shape)==3:
                img = img[np.newaxis]

            # 无masonic aug，crop & flip
            # if np.random.rand() < self.config.prepare['P_random'] or self.config.prepare['mosaic'] == False:
            if self.config.prepare['mosaic'] == False:
                crop_img, crop_lab = self.crop(img, lab, left_lung_raw, right_lung_raw, debug)
                crop_img, crop_lab = augment(crop_img, crop_lab, self.config.augtype)

            # masonic flip
            else:
                split_num = np.random.randint(1,self.config.prepare['mosaic_mode']+1)
                nodule_num = 7*split_num + 1
                choice_p = []
                for l in self.nodule_info:
                    if float(l[1])<7:
                        s_idx = 0
                    elif float(l[1])<14:
                        s_idx = 1
                    else:
                        s_idx = 2
                    choice_p.append((self.config.classifier['sample_factor'][s_idx])*self.config.classifier['mul_factor'][int(float(l[2]))])
                choice_p = np.array(choice_p)
                choice_nodule = self.nodule_info[np.random.choice(len(self.nodule_info), nodule_num, p=choice_p/np.sum(choice_p))] 
                # choice_nodule_sort = choice_nodule[choice_nodule[:,1].astype(np.float32).argsort()[::-1]]
                if split_num == 1:
                    img_list = []
                    lab_list = []
                    # print('0',time.time()-t)
                    for i in range(len(choice_nodule)):
                        # t1 = time.time()
                        if  not choice_nodule[i,0].startswith('DI_'):
                            img_tmp = np.load(os.path.join(self.config.prepare['data_dir'], choice_nodule[i,0].split('_')[0] + self.config.prepare['img_subfix']))
                            lab_tmp = np.load(os.path.join(self.config.prepare['data_dir'], choice_nodule[i,0].split('_')[0] + self.config.prepare['lab_subfix']))  
                        else:
                            img_tmp = np.load(os.path.join(self.config.prepare['data_dir'], 'DI_'+choice_nodule[i,0].split('_')[1] + self.config.prepare['img_subfix']))
                            lab_tmp = np.load(os.path.join(self.config.prepare['data_dir'], 'DI_'+choice_nodule[i,0].split('_')[1] + self.config.prepare['lab_subfix']))                     
                        if len(img_tmp.shape)==3:
                            img_tmp = img_tmp[np.newaxis]
                        # print('load',time.time()-t1)
                        crop_img, crop_lab = self.crop(img_tmp, lab_tmp, fix_size=np.array(self.config.prepare['crop_size'])//2,lab_idx=int(choice_nodule[i,0].split('_')[-1]), debug=False)
                        crop_img, crop_lab = augment(crop_img, crop_lab, self.config.augtype)
                        crop_img = crop_img[:1]
                        ceil_lab = self.config.prepare['crop_size'][0]/2
                        crop_lab = np.array([l for l in crop_lab if (0<l[0]<ceil_lab and 0<l[1]<ceil_lab and 0<l[2]<ceil_lab)])
                        img_list.append(crop_img)
                        lab_list.append(crop_lab)
                    # print('1',time.time()-t)
                    # np.save('/workspace/debug.npy',[img_list,lab_list,seg_list])
                    img_tmp1 = np.concatenate([img_list[0], img_list[1]], axis=1)
                    img_tmp2 = np.concatenate([img_list[2], img_list[3]], axis=1)
                    img_tmp3 = np.concatenate([img_list[4], img_list[5]], axis=1)
                    img_tmp4 = np.concatenate([img_list[6], img_list[7]], axis=1)
                    img_tmp5 = np.concatenate([img_tmp1, img_tmp2], axis=2)
                    img_tmp6 = np.concatenate([img_tmp3, img_tmp4], axis=2)
                    crop_img = np.concatenate([img_tmp5, img_tmp6], axis=3)
                    # print('2',time.time()-t)
                    lab_list[1][:,0] = lab_list[1][:,0] + ceil_lab
                    lab_list[3][:,0] = lab_list[3][:,0] + ceil_lab
                    lab_list[5][:,0] = lab_list[5][:,0] + ceil_lab
                    lab_list[7][:,0] = lab_list[7][:,0] + ceil_lab
                    lab_list[2][:,1] = lab_list[2][:,1] + ceil_lab
                    lab_list[3][:,1] = lab_list[3][:,1] + ceil_lab
                    lab_list[6][:,1] = lab_list[6][:,1] + ceil_lab
                    lab_list[7][:,1] = lab_list[7][:,1] + ceil_lab
                    lab_list[4][:,2] = lab_list[4][:,2] + ceil_lab
                    lab_list[5][:,2] = lab_list[5][:,2] + ceil_lab
                    lab_list[6][:,2] = lab_list[6][:,2] + ceil_lab
                    lab_list[7][:,2] = lab_list[7][:,2] + ceil_lab
                    crop_lab = np.concatenate((lab_list))
                else:
                    choice_nodule = choice_nodule[choice_nodule[:,1].astype(np.float32).argsort()[::-1]]
                    img_list = []
                    lab_list = []
                    for i in range(7):
                        if  not choice_nodule[i,0].startswith('DI_'):
                            img_tmp = (os.path.join(self.config.prepare['data_dir'], choice_nodule[i,0].split('_')[0] + self.config.prepare['img_subfix']))
                            lab_tmp = np.load(os.path.join(self.config.prepare['data_dir'], choice_nodule[i,0].split('_')[0] + self.config.prepare['lab_subfix']))  
                        else:
                            img_tmp = np.load(os.path.join(self.config.prepare['data_dir'], 'DI_'+choice_nodule[i,0].split('_')[1] + self.config.prepare['img_subfix']))
                            lab_tmp = np.load(os.path.join(self.config.prepare['data_dir'], 'DI_'+choice_nodule[i,0].split('_')[1] + self.config.prepare['lab_subfix']))                     
                        if len(img_tmp.shape)==3:
                            img_tmp = img_tmp[np.newaxis]
                        crop_img, crop_lab = self.crop(img_tmp, lab_tmp, fix_size=np.array(self.config.prepare['crop_size'])//2,lab_idx=int(choice_nodule[i,0].split('_')[-1]), debug=False)
                        crop_img, crop_lab = augment(crop_img, crop_lab, self.config.augtype)
                        crop_img = crop_img[:1]
                        ceil_lab = self.config.prepare['crop_size'][0]/2
                        crop_lab = np.array([l for l in crop_lab if (0<l[0]<ceil_lab and 0<l[1]<ceil_lab and 0<l[2]<ceil_lab)])
                        img_list.append(crop_img)
                        lab_list.append(crop_lab)
                    img_small_list = []
                    lab_small_list = []
                    choice_nodule_small = choice_nodule[7:]
                    np.random.shuffle(choice_nodule_small)
                    for i in range(len(choice_nodule_small)):
                        if  not choice_nodule_small[i,0].startswith('DI_'):
                            img_tmp = np.load(os.path.join(self.config.prepare['data_dir'], choice_nodule_small[i,0].split('_')[0] + self.config.prepare['img_subfix']))
                            lab_tmp = np.load(os.path.join(self.config.prepare['data_dir'], choice_nodule_small[i,0].split('_')[0] + self.config.prepare['lab_subfix']))  
                        else:
                            img_tmp = np.load(os.path.join(self.config.prepare['data_dir'], 'DI_'+choice_nodule_small[i,0].split('_')[1] + self.config.prepare['img_subfix']))
                            lab_tmp = np.load(os.path.join(self.config.prepare['data_dir'], 'DI_'+choice_nodule_small[i,0].split('_')[1] + self.config.prepare['lab_subfix']))                     
                        if len(img_tmp.shape)==3:
                            img_tmp = img_tmp[np.newaxis]
                        crop_img, crop_lab = self.crop(img_tmp, lab_tmp, fix_size=np.array(self.config.prepare['crop_size'])//4,lab_idx=int(choice_nodule_small[i,0].split('_')[-1]), debug=False)
                        crop_img, crop_lab = augment(crop_img, crop_lab, self.config.augtype)
                        crop_img = crop_img[:1]
                        ceil_lab_small = self.config.prepare['crop_size'][0]/4
                        crop_lab = np.array([l for l in crop_lab if (0<l[0]<ceil_lab_small and 0<l[1]<ceil_lab_small and 0<l[2]<ceil_lab_small)])
                        img_small_list.append(crop_img)
                        lab_small_list.append(crop_lab)
                    img_small_tmp1 = np.concatenate([img_small_list[0], img_small_list[1]], axis=1)
                    img_small_tmp2 = np.concatenate([img_small_list[2], img_small_list[3]], axis=1)
                    img_small_tmp3 = np.concatenate([img_small_list[4], img_small_list[5]], axis=1)
                    img_small_tmp4 = np.concatenate([img_small_list[6], img_small_list[7]], axis=1)
                    img_small_tmp5 = np.concatenate([img_small_tmp1, img_small_tmp2], axis=2)
                    img_small_tmp6 = np.concatenate([img_small_tmp3, img_small_tmp4], axis=2)
                    crop_img_small = np.concatenate([img_small_tmp5, img_small_tmp6], axis=3)
                    lab_small_list[1][:,0] = lab_small_list[1][:,0] + ceil_lab_small
                    lab_small_list[3][:,0] = lab_small_list[3][:,0] + ceil_lab_small
                    lab_small_list[5][:,0] = lab_small_list[5][:,0] + ceil_lab_small
                    lab_small_list[7][:,0] = lab_small_list[7][:,0] + ceil_lab_small
                    lab_small_list[2][:,1] = lab_small_list[2][:,1] + ceil_lab_small
                    lab_small_list[3][:,1] = lab_small_list[3][:,1] + ceil_lab_small
                    lab_small_list[6][:,1] = lab_small_list[6][:,1] + ceil_lab_small
                    lab_small_list[7][:,1] = lab_small_list[7][:,1] + ceil_lab_small
                    lab_small_list[4][:,2] = lab_small_list[4][:,2] + ceil_lab_small
                    lab_small_list[5][:,2] = lab_small_list[5][:,2] + ceil_lab_small
                    lab_small_list[6][:,2] = lab_small_list[6][:,2] + ceil_lab_small
                    lab_small_list[7][:,2] = lab_small_list[7][:,2] + ceil_lab_small
                    crop_lab_small = np.concatenate((lab_small_list)) 
                    # np.save('/workspace/debug.npy',[img_list,lab_list,seg_list]
                    img_list.append(crop_img_small)
                    lab_list.append(crop_lab_small)
                    rd_idx = np.arange(len(img_list))
                    np.random.shuffle(rd_idx)
                    img_list = np.array(img_list)[rd_idx]
                    lab_list = np.array(lab_list)[rd_idx]
                    img_tmp1 = np.concatenate([img_list[0], img_list[1]], axis=1)
                    img_tmp2 = np.concatenate([img_list[2], img_list[3]], axis=1)
                    img_tmp3 = np.concatenate([img_list[4], img_list[5]], axis=1)
                    img_tmp4 = np.concatenate([img_list[6], img_list[7]], axis=1)
                    img_tmp5 = np.concatenate([img_tmp1, img_tmp2], axis=2)
                    img_tmp6 = np.concatenate([img_tmp3, img_tmp4], axis=2)
                    crop_img = np.concatenate([img_tmp5, img_tmp6], axis=3)
                    lab_list[1][:,0] = lab_list[1][:,0] + ceil_lab
                    lab_list[3][:,0] = lab_list[3][:,0] + ceil_lab
                    lab_list[5][:,0] = lab_list[5][:,0] + ceil_lab
                    lab_list[7][:,0] = lab_list[7][:,0] + ceil_lab
                    lab_list[2][:,1] = lab_list[2][:,1] + ceil_lab
                    lab_list[3][:,1] = lab_list[3][:,1] + ceil_lab
                    lab_list[6][:,1] = lab_list[6][:,1] + ceil_lab
                    lab_list[7][:,1] = lab_list[7][:,1] + ceil_lab
                    lab_list[4][:,2] = lab_list[4][:,2] + ceil_lab
                    lab_list[5][:,2] = lab_list[5][:,2] + ceil_lab
                    lab_list[6][:,2] = lab_list[6][:,2] + ceil_lab
                    lab_list[7][:,2] = lab_list[7][:,2] + ceil_lab
                    crop_lab = np.concatenate((lab_list))    

            # bbox 坐标转换
            gt_prob_fpn, gt_coord_prob_fpn, gt_coord_diff_fpn, gt_diff_fpn, gt_connects_fpn = self.label_maper(crop_lab)

            # window & normalize
            crop_img = self.__lum_trans__(crop_img)
            # print(time.time()-t)
            return self.__dtype__(crop_img), gt_prob_fpn, gt_coord_prob_fpn, gt_coord_diff_fpn, gt_diff_fpn, gt_connects_fpn, self.cases[idx]
            # return self.__dtype__(crop_img), crop_lab


        # validate
        elif (self.phase == 'val') or (self.phase=='valtrain'):
            img = np.load(self.img_files[idx])
            img = img.astype(np.int16)
            if len(img.shape)==3:
                img = img[np.newaxis]
            lab_raw = np.load(self.lab_files[idx])
            # 把5去除？
            # lab = np.array([item for item in lab_raw if item[4] not in [5,6,7]])
            # lab = np.array([item for item in lab_raw if item[4] not in [6,7]])
            lab = lab_raw

            # nswh: 图片pad前后的shape
            crop_img, nswh = self.split_comb.split(img)
            crop_img = self.__lum_trans__(crop_img)

            return self.__dtype__(crop_img), nswh, self.cases[idx], lab

        else:
            img = np.load(self.img_files[idx])
            if len(img.shape)==3:
                img = img[np.newaxis]
            crop_img, nswh = self.split_comb.split(img)
            crop_img = self.__lum_trans__(crop_img)

            return self.__dtype__(crop_img), nswh, self.cases[idx]

    def __lum_trans__(self, x):
        if self.config.prepare['clip']:
            x = np.clip(x, self.config.prepare['lower'], self.config.prepare['higher'])
        if self.config.prepare['normalize']:
            mean = self.config.prepare['sub_value']
            std = self.config.prepare['div_value']
            x = (x.astype('float32') - mean) / std
        return x


    def __len__(self):

        if self.config.debug:
            return 4
        else:
            # return len(self.cases)

            if self.phase == 'train':
                return len(self.cases) * self.config.train['train_repeat']
            else:
                return len(self.cases)


class Crop(object):
    def __init__(self, config):
        self.crop_size = np.array(config.prepare['crop_size']) # 裁剪块的大小
        self.margin = config.prepare['margin']
        self.augtype = config.augtype
        self.ignore = config.prepare['label_para']['ignore_index']
        self.pad_mode = config.prepare['pad_mode'] # pad方法，reflect和constant可选
        self.pad_value = config.prepare['pad_value']
        self.P_random = config.prepare['P_random'] # 随机数，控制取样
        self.size_lim = config.rpn['diam_thresh']  # roi大小限制，只统计位于diam_thresh之间的roi
        self.omit_cls = config.classifier['omit_cls']  # 类别忽略，忽略omit_cls中的roi
        self.mul_factor = config.classifier['mul_factor']
        self.sample_factor = config.classifier['sample_factor']

    def __call__(self, im, lab, fix_size=None, lab_idx=None, debug=False):
        """
        每次随机裁剪出一个小块，

        :param im -> ndarray([1, 160, 160, 160]): 3d图 
        :param lab -> ndarray([n,8]): 标签
        :param lab_idx: 通过id指定lab中的一个roi, 用于debug
        :param fix_size: 
        :param debug: 

        """

        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        
        shape = im.shape[1:]
        margin = self.margin.copy()
        # if lab_idx != None:
        #     self.crop_size = fix_size
            
        #     print(lab)
        
        # step1_size, step2_size == crop_size
        if self.augtype['scale']:
            low, high = self.augtype['scale_lim']
            if lab_idx != None:
                lowlim = np.max([cs / (sh + 2 * b) for cs, sh, b in zip(fix_size, shape, margin)])
            else:
                lowlim = np.max([cs / (sh + 2 * b) for cs, sh, b in zip(self.crop_size, shape, margin)])
            scaleLim = [np.max([low, lowlim]), high]
            scale = np.random.rand() * (scaleLim[1] - scaleLim[0]) + scaleLim[0]
            if lab_idx != None:
                step1_size = (fix_size.astype('float') / scale).astype('int')
                scale = fix_size.astype('float') / step1_size                
            else:
                step1_size = (self.crop_size.astype('float') / scale).astype('int')
                scale = self.crop_size.astype('float') / step1_size
        else:
            if lab_idx != None:
                step1_size = fix_size.copy()
            else:
                step1_size = self.crop_size.copy()

        if self.augtype['rotate']:
            raise NotImplementedError
            angle = (np.random.rand() - 0.5) * self.augtype['rotate_deg_lim']
            # 假如是一个正方形，为了保证旋转后信息不丢失，需要先扩大范围再旋转再切割
            rot_ratio = np.sqrt(2) * np.cos(np.pi / 4 - np.abs(angle) / 180 * np.pi)
            step2_size = np.ceil(step1_size.astype('float') * rot_ratio).astype('int')
            step2_size = np.array([np.min([ss, sh + 2 * b]) for ss, sh, b in zip(step2_size, shape, margin)])
        else:
            step2_size = step1_size.copy()

        # 从区间中随机取一个数
        def start_fn(low, high):
            if low >= high:
                return np.random.random_integers(high, low)
            else:
                return np.random.random_integers(low, high)

        # 计算label中所有的roi
        if lab_idx == None:
            valid_lab = np.array([l for l in lab if ((l[3] >= self.size_lim[0]) and (l[3] <= self.size_lim[1]) and (l[4] not in self.omit_cls) and ((l[5] == 1 ) or (l[5] == 0 and l[4] in [1,2])))])
            if (np.random.rand() < self.P_random) or (len(valid_lab)==0):
                # 随机选取-起始点
                start = [start_fn(-b, s + b - c) for s, c, b in zip(shape, step2_size, margin)]
            else:
                # 含有mask
                diams = []
                for l in valid_lab:
                    if l[3]<7:
                        s_idx = 0
                    elif l[3]<14:
                        s_idx = 1
                    else:
                        s_idx = 2
                    diams.append((self.sample_factor[s_idx])*self.mul_factor[int(l[4])])
                diams = np.array(diams)
                bb = valid_lab[np.random.choice(len(valid_lab),p=diams/np.sum(diams))]
                # print(bb)
                # print(bb)
                low = [np.max([a+bb[3]/2+b-c,-b]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
                high = [np.min([a-bb[3]/2-b,s+b-c]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
                # print(low,high)
                start = [start_fn(l,h) if l<=h else int(a-c/2)   for l,h,a,c in zip(low,high, bb[:3], step2_size)]
        else:
            bb = lab[lab_idx]
            # print(bb)
            # print(shape, step2_size, margin, bb[:3])
            low = [np.max([a+bb[3]/2+b-c,-b]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
            high = [np.min([a-bb[3]/2-b,s+b-c]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
            # print(low,high)
            start = [start_fn(l,h) if l<=h else int(a-c/2)   for l,h,a,c in zip(low,high, bb[:3], step2_size)]
            # print(start)
    #    if debug:
        #    start = [200,100,250]

        # pick 区间[start_point, end_point]
        end_point = [s + c for s, c in zip(start, step2_size)]

        # [[0,0], [-8, 8], [-8, 8], [-8, 8]]
        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + step2_size[i] - shape[i])
            pad.append([leftpad, rightpad])
        pad = np.array(pad)

        # crop img，每次生成一个块
        crop_im = im[:,
               max(start[0], 0):min(start[0] + step2_size[0], shape[0]),
               max(start[1], 0):min(start[1] + step2_size[1], shape[1]),
               max(start[2], 0):min(start[2] + step2_size[2], shape[2])]

        # crop label coords
        if len(lab)>0:
            # crop后可能出现负值，需要pad
            lab[:,0] -= max(start[0], 0)
            lab[:,1] -= max(start[1], 0)
            lab[:,2] -= max(start[2], 0)
            # print([crop.shape, 'crop shape'])
            # print([scale, 'scale'])
        
        # pad
        # constant
        if self.pad_mode == 'constant':
            crop_im = mypad(crop_im, pad, self.pad_value)
        
        # reflect
        else:
            crop_im  = np.pad(crop_im, pad, self.pad_mode)
        
        if len(lab)>0:
            lab[:,0] += pad[1,0]
            lab[:,1] += pad[2,0]
            lab[:,2] += pad[3,0]

        if self.augtype['rotate']:  # 切割回来
            raise NotImplementedError
            # for i in range(crop_im.shape[0]):
            #     crop_im[i] = myrotate(crop_im[i], angle, reshape=False, order=1)
            # for i in range(crop_lab.shape[0]):
            #     crop_lab[i] = myrotate(crop_lab[i], angle, reshape=False, order=0)  # 用 0 阶可以加快速度，也不会有label互相干扰的问题
            # #             crop = np.concatenate([rotate(slice, angle,axes=(1,2))[np.newaxis] for slice in crop], axis=0)
            # start = np.ceil((step2_size - step1_size) / 2).astype('int')
            # crop_im = crop_im[:,
            #        start[0]: start[0] + step1_size[0],
            #        start[1]: start[1] + step1_size[1],
            #        start[2]: start[2] + step1_size[2]]
            # crop_lab = crop_lab[:,
            #        start[0]: start[0] + step1_size[0],
            #        start[1]: start[1] + step1_size[1],
            #        start[2]: start[2] + step1_size[2]]

        if self.augtype['scale']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 用 0 阶可以加快速度，也不会有label互相干扰的问题
                if lab_idx != None:
                    newscale = np.array(fix_size).astype('float')/np.array(crop_im.shape[1:])
                    crop2 = np.zeros([crop_im.shape[0],fix_size[0],fix_size[1],fix_size[2]])
                    for i_slice, crop_slice in enumerate(crop_im):
                        crop2[i_slice] = zoom(crop_slice,[newscale[0],newscale[1],newscale[2]],order=0)
                    crop_im = crop2
                    assert(np.all(np.array(crop_im.shape[1:])==np.array(fix_size)))
                else:
                    newscale = np.array(self.crop_size).astype('float')/np.array(crop_im.shape[1:])
                    crop2 = np.zeros([crop_im.shape[0],self.crop_size[0],self.crop_size[1],self.crop_size[2]])
                    for i_slice, crop_slice in enumerate(crop_im):
                        crop2[i_slice] = zoom(crop_slice,[newscale[0],newscale[1],newscale[2]],order=0)
                    crop_im = crop2
                    assert(np.all(np.array(crop_im.shape[1:])==np.array(self.crop_size)))
            if len(lab)>0:
                for j in range(4):
                    if j<=2:
                        lab[:,j] = lab[:,j]*newscale[j]
                    else:
                        lab[:,j] = lab[:,j]*newscale[2]
        if debug:
            print([shape, 'shape'])
            print([step1_size, 'step1size'])

            print([step2_size, 'step2size'])
            print([margin, 'margin'])
            print([start, 'start'])
            print([pad, 'pad'])
            print([lab,'lab'])
        
        return crop_im, lab


class Crop_lung(object):
    def __init__(self, config):
        self.crop_size = np.array(config.prepare['crop_size']) # 裁剪块的大小
        self.margin = config.prepare['margin']
        self.augtype = config.augtype
        self.ignore = config.prepare['label_para']['ignore_index']
        self.pad_mode = config.prepare['pad_mode'] # pad方法，reflect和constant可选
        self.pad_value = config.prepare['pad_value']
        self.P_random = config.prepare['P_random'] # 随机数，控制取样
        self.size_lim = config.rpn['diam_thresh']  # roi大小限制，只统计位于diam_thresh之间的roi
        self.omit_cls = config.classifier['omit_cls']  # 类别忽略，忽略omit_cls中的roi
        self.mul_factor = config.classifier['mul_factor']
        self.sample_factor = config.classifier['sample_factor']

    def __call__(self, im, lab, left_edge, right_edge, fix_size=None, lab_idx=None, debug=False):
        """
        每次随机裁剪出一个小块，以一定概率在（骨折，肺叶外围，随机）处取样

        :param im -> ndarray([1, 160, 160, 160]): 3d图 
        :param lab -> ndarray([n, 8]): 标签
        :param left_edge -> ndarray([n, 3]): 位于左肺叶外围边缘的点 
        :param right_edge -> ndarray([n, 3]): 位于右肺叶外围边缘的点
        :param lab_idx: 通过id指定lab中的一个roi, 用于debug
        :param fix_size: 
        :param debug: 

        """

        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        
        shape = im.shape[1:]
        margin = self.margin.copy()
        # if lab_idx != None:
        #     self.crop_size = fix_size
            
        #     print(lab)
        
        # step1_size, step2_size == crop_size
        if self.augtype['scale']:
            low, high = self.augtype['scale_lim']
            if lab_idx != None:
                lowlim = np.max([cs / (sh + 2 * b) for cs, sh, b in zip(fix_size, shape, margin)])
            else:
                lowlim = np.max([cs / (sh + 2 * b) for cs, sh, b in zip(self.crop_size, shape, margin)])
            scaleLim = [np.max([low, lowlim]), high]
            scale = np.random.rand() * (scaleLim[1] - scaleLim[0]) + scaleLim[0]
            if lab_idx != None:
                step1_size = (fix_size.astype('float') / scale).astype('int')
                scale = fix_size.astype('float') / step1_size                
            else:
                step1_size = (self.crop_size.astype('float') / scale).astype('int')
                scale = self.crop_size.astype('float') / step1_size
        else:
            if lab_idx != None:
                step1_size = fix_size.copy()
            else:
                step1_size = self.crop_size.copy()

        if self.augtype['rotate']:
            raise NotImplementedError
            angle = (np.random.rand() - 0.5) * self.augtype['rotate_deg_lim']
            # 假如是一个正方形，为了保证旋转后信息不丢失，需要先扩大范围再旋转再切割
            rot_ratio = np.sqrt(2) * np.cos(np.pi / 4 - np.abs(angle) / 180 * np.pi)
            step2_size = np.ceil(step1_size.astype('float') * rot_ratio).astype('int')
            step2_size = np.array([np.min([ss, sh + 2 * b]) for ss, sh, b in zip(step2_size, shape, margin)])
        else:
            step2_size = step1_size.copy()

        # 从区间中随机取一个数
        def start_fn(low, high):
            if low >= high:
                return np.random.random_integers(high, low)
            else:
                return np.random.random_integers(low, high)

        # 计算label中所有的roi
        if lab_idx == None:
            valid_lab = np.array([l for l in lab if ((l[3] >= self.size_lim[0]) and (l[3] <= self.size_lim[1]) and (l[4] not in self.omit_cls) and ((l[5] == 1 ) or (l[5] == 0 and l[4] in [1,2])))])
            a_rand = np.random.rand()
            if (a_rand < self.P_random[0]) or (len(valid_lab)==0):
                # 随机选取-起始点
                start = [start_fn(-b, s + b - c) for s, c, b in zip(shape, step2_size, margin)]
            elif a_rand < self.P_random[1]:
                # 选取骨折
                diams = []
                for l in valid_lab:
                    if l[3]<7:
                        s_idx = 0
                    elif l[3]<14:
                        s_idx = 1
                    else:
                        s_idx = 2
                    diams.append((self.sample_factor[s_idx])*self.mul_factor[int(l[4])])
                diams = np.array(diams)
                bb = valid_lab[np.random.choice(len(valid_lab),p=diams/np.sum(diams))]
                # print(bb)
                # print(bb)
                low = [np.max([a+bb[3]/2+b-c,-b]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
                high = [np.min([a-bb[3]/2-b,s+b-c]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
                # print(low,high)
                start = [start_fn(l,h) if l<=h else int(a-c/2)   for l,h,a,c in zip(low,high, bb[:3], step2_size)]

            else:
                # 肺叶区域采样，等概率左 & 右 (目前肺叶无膨胀操作，需要向不同方向外扩)
                if a_rand <= 0.75:
                    chosen_lung_edge = left_edge
                    cp_stride = -10
                else:
                    chosen_lung_edge = right_edge
                    cp_stride = 10

                chosen_id = random.choice(range(chosen_lung_edge.shape[0]))
                chosen_point = chosen_lung_edge[chosen_id]
                chosen_point[1] += cp_stride
                chosen_point[1], chosen_point[2] = chosen_point[2], chosen_point[1]
                
                start = [b-c//2 for s,c,b in zip(shape, step2_size, chosen_point)]
                
        else:
            bb = lab[lab_idx]
            # print(bb)
            # print(shape, step2_size, margin, bb[:3])
            low = [np.max([a+bb[3]/2+b-c,-b]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
            high = [np.min([a-bb[3]/2-b,s+b-c]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
            # print(low,high)
            start = [start_fn(l,h) if l<=h else int(a-c/2)   for l,h,a,c in zip(low,high, bb[:3], step2_size)]
            # print(start)
    #    if debug:
        #    start = [200,100,250]

        # pick 区间[start_point, end_point]
        end_point = [s + c for s, c in zip(start, step2_size)]

        # [[0,0], [-8, 8], [-8, 8], [-8, 8]]
        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + step2_size[i] - shape[i])
            pad.append([leftpad, rightpad])
        pad = np.array(pad)

        # crop img，每次生成一个块
        crop_im = im[:,
               max(start[0], 0):min(start[0] + step2_size[0], shape[0]),
               max(start[1], 0):min(start[1] + step2_size[1], shape[1]),
               max(start[2], 0):min(start[2] + step2_size[2], shape[2])]

        # crop label coords
        if len(lab)>0:
            # crop后可能出现负值，需要pad
            lab[:,0] -= max(start[0], 0)
            lab[:,1] -= max(start[1], 0)
            lab[:,2] -= max(start[2], 0)
            # print([crop.shape, 'crop shape'])
            # print([scale, 'scale'])
        
        # pad
        # constant
        if self.pad_mode == 'constant':
            crop_im = mypad(crop_im, pad, self.pad_value)
        
        # reflect
        else:
            crop_im  = np.pad(crop_im, pad, self.pad_mode)
        
        if len(lab)>0:
            lab[:,0] += pad[1,0]
            lab[:,1] += pad[2,0]
            lab[:,2] += pad[3,0]

        if self.augtype['rotate']:  # 切割回来
            raise NotImplementedError
            # for i in range(crop_im.shape[0]):
            #     crop_im[i] = myrotate(crop_im[i], angle, reshape=False, order=1)
            # for i in range(crop_lab.shape[0]):
            #     crop_lab[i] = myrotate(crop_lab[i], angle, reshape=False, order=0)  # 用 0 阶可以加快速度，也不会有label互相干扰的问题
            # #             crop = np.concatenate([rotate(slice, angle,axes=(1,2))[np.newaxis] for slice in crop], axis=0)
            # start = np.ceil((step2_size - step1_size) / 2).astype('int')
            # crop_im = crop_im[:,
            #        start[0]: start[0] + step1_size[0],
            #        start[1]: start[1] + step1_size[1],
            #        start[2]: start[2] + step1_size[2]]
            # crop_lab = crop_lab[:,
            #        start[0]: start[0] + step1_size[0],
            #        start[1]: start[1] + step1_size[1],
            #        start[2]: start[2] + step1_size[2]]

        if self.augtype['scale']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 用 0 阶可以加快速度，也不会有label互相干扰的问题
                if lab_idx != None:
                    newscale = np.array(fix_size).astype('float')/np.array(crop_im.shape[1:])
                    crop2 = np.zeros([crop_im.shape[0],fix_size[0],fix_size[1],fix_size[2]])
                    for i_slice, crop_slice in enumerate(crop_im):
                        crop2[i_slice] = zoom(crop_slice,[newscale[0],newscale[1],newscale[2]],order=0)
                    crop_im = crop2
                    assert(np.all(np.array(crop_im.shape[1:])==np.array(fix_size)))
                else:
                    newscale = np.array(self.crop_size).astype('float')/np.array(crop_im.shape[1:])
                    crop2 = np.zeros([crop_im.shape[0],self.crop_size[0],self.crop_size[1],self.crop_size[2]])
                    for i_slice, crop_slice in enumerate(crop_im):
                        crop2[i_slice] = zoom(crop_slice,[newscale[0],newscale[1],newscale[2]],order=0)
                    crop_im = crop2
                    assert(np.all(np.array(crop_im.shape[1:])==np.array(self.crop_size)))
            if len(lab)>0:
                for j in range(4):
                    if j<=2:
                        lab[:,j] = lab[:,j]*newscale[j]
                    else:
                        lab[:,j] = lab[:,j]*newscale[2]
        if debug:
            print([shape, 'shape'])
            print([step1_size, 'step1size'])

            print([step2_size, 'step2size'])
            print([margin, 'margin'])
            print([start, 'start'])
            print([pad, 'pad'])
            print([lab,'lab'])
        
        return crop_im, lab


def augment(im, lab, augtype):
    t = time.time()
    np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
    if augtype['flip']:
        flipid = np.array([1, 1, 1])
        for i in range(3):
            if augtype['flip_axis'][i]:
                flipid[i] = np.random.randint(2) * 2 - 1
                if len(lab)>0 and flipid[i] == -1:
                    lab[:,i] = im.shape[i+1] - 1 - lab[:,i]
        im = np.ascontiguousarray(im[:, ::flipid[0], ::flipid[1], ::flipid[2]])

    if augtype['swap']:
        swap_axis = np.array([i + 1 for (i, v) in enumerate(augtype['swap_axis']) if v == 1])
        shapes = [im.shape[i] for i in swap_axis]
        assert all(x == shapes[0] for x in shapes)
        axisorder = np.random.permutation(len(swap_axis))
        after_swap = swap_axis[axisorder]
        common_order = list(range(4))
        for before, after in zip(swap_axis, after_swap):
            common_order[before] = after
        im = np.transpose(im, common_order)
        if len(lab)>0:
            lab[:,:3] = lab[:,:3][:,np.array(common_order)[1:]-1]
    return im, lab

def myrotate(im, deg, reshape, order):
    newim = []
    for slice in im:
        newim.append(rotate2d(slice, deg, resize=reshape, order=order))
    return np.array(newim)


if __name__ == '__main__':
    from config import Config
    import numpy as np

    Config.load('/yupeng/alg-coronary-seg3d/configs/yp_crop-160_base-lr_stride-4-8_layer-2-3_clip-700-2000_lung.yml')
    # print(vars(Config))
    dataset = myDataset(Config, 'train')
    n_data = len(dataset)
    data = iter(dataset)
    # img, lab = data[0]
    # print(img.shape)
    # print(lab)
    # for idx, d in tqdm(enumerate(data),total=n_data):
    for idx, d in enumerate(data):
        if idx == 50:
            break
        img, label = d
        print('-' * 10)
        print(label.shape)
        img = img[0]
        val_label = []
        for ll in label:
            if ((ll[:3] - ll[3]//2) >= 0).all() and ((ll[:3] - ll[3]//2) < 160).all():
                val_label.append(ll)
        
        np.save('/yupeng/alg-coronary-seg3d/datasets/samples/{}_img.npy'.format(idx), img)
        if val_label == []:
            continue
        else:
            val_label = np.stack(val_label)
            np.save('/yupeng/alg-coronary-seg3d/datasets/samples/{}_label.npy'.format(idx), val_label)
        
        



    # data2 = myDataset(Config, 'val')
    # im, lab, nswh, name = data2[0]
    # print(im.shape)
    # print(lab.shape)
    # print(nswh)

    # comb = data2.split_comb.combine(im,nswh)
    # print(comb.shape)
    # im_orig = np.load(data2.img_files[0])[np.newaxis]
    # print(np.corrcoef(im_orig.reshape(-1), comb.reshape(-1)))
