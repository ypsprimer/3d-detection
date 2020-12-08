import os
import time
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from scipy.ndimage import zoom, rotate, binary_dilation
from .split_combine import SplitComb, mypad
from skimage.transform import rotate as rotate2d
from .label_map2_2 import Label_mapping2 as Label_mapping
import warnings
import numpy as np



class myDataset2(Dataset):
    def __init__(self, config, phase):

        assert phase in ['train', 'val','valtrain', 'test']
        self.phase = phase
        self.config = config
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
        self.img_files = [os.path.join(datadir, f + config.prepare['img_subfix']) for f in self.cases]
        self.lab_files = [os.path.join(datadir, f + config.prepare['lab_subfix']) for f in self.cases]
        self.lab_buffers = {case_name: np.load(lab_file) for case_name, lab_file in zip(self.cases, self.lab_files)}
        #self.img_buffers = {item: np.load(item)[np.newaxis] for item in self.img_files[:2000]}
        self.label_maper = Label_mapping(config)
        self.crop = Crop(config)
        self.lab_N = config.rpn['N_prob']
        self.size_lim = config.rpn['diam_thresh']
        self.omit_cls = config.classifier['omit_cls']
        self.sample_weights = {}
        self.__init_sample_weights__()
        self.samples = [[img_file, lab_file, -1, case_name] for img_file, lab_file, case_name in zip(self.img_files, self.lab_files, self.cases)]

    def __init_sample_weights__(self):
        for idx, case_name in enumerate(self.cases):
            lab = np.load(self.lab_files[idx])
            for idx2, l in enumerate(lab):
                if ((l[3] >= self.size_lim[0]) and (l[3] <= self.size_lim[1]) and (l[4] not in self.omit_cls) and l[5] == 1 ):
                    self.sample_weights[case_name + '___' + str(idx2 + 1)] = [1, l[3], 0]

    def resample(self):
        self.samples = []
        sample_weights_ = [[k,v] for k,v in self.sample_weights.items()]
        print('total samples ', len(sample_weights_))
        sample_weights_.sort(key=lambda x:-x[1][0])
        case_name_hits = []
        for k, v in sample_weights_[:len(sample_weights_)//3]:
            case_name, nodule_idx = k.split('___')
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], int(nodule_idx)-1, case_name, v])
            case_name_hits.append(case_name)
        print('top pos sample ', len(self.samples), ', thresh is ', sample_weights_[len(sample_weights_)//2][1])
        miss_cases = [item for item in self.cases if item not in case_name_hits]
        for idx, case_name in enumerate(miss_cases):
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], -1, case_name, []])
        print('all sample ', len(self.samples), ' cases')

    def resample2(self):
        self.samples = []
        sample_weights_big = [[k,v] for k,v in self.sample_weights.items() if v[1] >= 4]
        sample_weights_small = [[k,v] for k,v in self.sample_weights.items() if v[1] < 4]
        print('total samples: ', len(self.sample_weights), '; big samples: ', len(sample_weights_big), \
              ': small samples: ', len(sample_weights_small))
        sample_weights_big.sort(key=lambda x:-x[1][0])
        sample_weights_small.sort(key=lambda x:-x[1][0])
        case_name_hits = []
        top_big_samples_count = len(sample_weights_big)//3
        for k, v in sample_weights_big[:top_big_samples_count]:
            case_name, nodule_idx = k.split('___')
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], int(nodule_idx)-1, case_name, v])
            case_name_hits.append(case_name)
        print('top big pos sample ', top_big_samples_count, ', thresh is ', sample_weights_big[top_big_samples_count][1][0])
        top_small_samples_count = len(sample_weights_small)//3
        for k, v in sample_weights_small[:top_small_samples_count]:
            case_name, nodule_idx = k.split('___')
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], int(nodule_idx)-1, case_name, v])
            case_name_hits.append(case_name)
        print('top small pos sample ', top_small_samples_count, ', thresh is ', sample_weights_small[top_small_samples_count][1][0])
        miss_cases = [item for item in self.cases if item not in case_name_hits]
        for idx, case_name in enumerate(miss_cases):
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], -1, case_name, []])
        print('all sample ', len(self.samples), ' cases')

    def resample3(self):
        self.samples = []
        sample_weights_ = [[k,v] for k,v in self.sample_weights.items() if v[2] < 3]
        print('total samples ', len(sample_weights_))
        sample_weights_.sort(key=lambda x:-x[1][0])
        case_name_hits = []
        for k, v in sample_weights_[:len(sample_weights_)//3]:
            case_name, nodule_idx = k.split('___')
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], int(nodule_idx)-1, case_name, v])
            case_name_hits.append(case_name)
        print('top pos sample ', len(self.samples), ', thresh is ', sample_weights_[len(sample_weights_)//2][1])
        miss_cases = [item for item in self.cases if item not in case_name_hits]
        for idx, case_name in enumerate(miss_cases):
            case_idx = self.cases2idx[case_name]
            self.samples.append([self.img_files[case_idx], self.lab_files[case_idx], -1, case_name, []])
        print('all sample ', len(self.samples), ' cases')

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
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        if debug:
            print(self.img_files[idx])
        if self.phase == 'train':
            img_file, lab_file, nodule_idx, case_name, _ = self.samples[idx]
            img = np.load(img_file)
            assert case_name in self.lab_buffers
            if case_name in self.lab_buffers:
                lab = self.lab_buffers[case_name].copy()
            else:
                lab = np.load(lab_file)
            if np.random.rand() < 0.3 and False:
                newimg = []
                cp_idx = np.random.randint(2,9)
                for i in range(0,img.shape[0],cp_idx):    
                    if i+cp_idx< img.shape[0]:
                        newimg.append(np.mean(img[i:i+cp_idx],axis = 0, keepdims=True))
                newimg = np.squeeze(np.array(newimg))
                newimg= F.interpolate(torch.tensor(newimg).unsqueeze(0).unsqueeze(0),scale_factor=[cp_idx,1,1], mode='trilinear')[0,0]
                img = newimg.numpy()
                #if len(lab)>0:
                #    lab[lab[:,3]<4.5,5] = 0
            if len(img.shape)==3:
                img = img[np.newaxis]
            crop_img, crop_lab = self.crop(img, lab, nodule_idx=nodule_idx, debug=debug)
            crop_img, crop_lab = augment(crop_img, crop_lab, self.config.augtype)
            gt_prob_fpn, gt_coord_prob_fpn, gt_coord_diff_fpn, gt_diff_fpn, gt_connects_fpn = self.label_maper(crop_lab)

            crop_img = self.__lum_trans__(crop_img)
            return self.__dtype__(crop_img), gt_prob_fpn, gt_coord_prob_fpn, gt_coord_diff_fpn, gt_diff_fpn, gt_connects_fpn, case_name

        elif (self.phase == 'val') or (self.phase=='valtrain'):
            img = np.load(self.img_files[idx])
            if len(img.shape)==3:
                img = img[np.newaxis]
            lab = np.load(self.lab_files[idx])
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
                return len(self.samples) * self.config.train['train_repeat']
            else:
                return len(self.cases)

class Crop(object):
    def __init__(self, config):
        self.crop_size = np.array(config.prepare['crop_size'])
        self.margin = config.prepare['margin']
        self.augtype = config.augtype
        self.ignore = config.prepare['label_para']['ignore_index']
        self.pad_mode = config.prepare['pad_mode']
        self.pad_value = config.prepare['pad_value']
        self.P_random = config.prepare['P_random']
        self.size_lim = config.rpn['diam_thresh']
        self.omit_cls = config.classifier['omit_cls']
        self.mul_factor = config.classifier['mul_factor']
        self.sample_factor = config.classifier['sample_factor']
    def __call__(self, im, lab, nodule_idx=-1, debug=False,):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        shape = im.shape[1:]
        margin = self.margin.copy()

        if self.augtype['scale']:
            low, high = self.augtype['scale_lim']
            lowlim = np.max([cs / (sh + 2 * b) for cs, sh, b in zip(self.crop_size, shape, margin)])
            scaleLim = [np.max([low, lowlim]), high]
            scale = np.random.rand() * (scaleLim[1] - scaleLim[0]) + scaleLim[0]
            step1_size = (self.crop_size.astype('float') / scale).astype('int')
            scale = self.crop_size.astype('float') / step1_size
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

        def start_fn(low, high):
            if low >= high:
                return np.random.random_integers(high, low)
            else:
                return np.random.random_integers(low, high)
        valid_lab = np.array([l for l in lab if ((l[3] >= self.size_lim[0]) and (l[4] not in self.omit_cls) and l[5] == 1 )])
        if (len(valid_lab)==0) or nodule_idx==-1 or (np.random.rand() < 0.2):
            start = [start_fn(-b, s + b - c) for s, c, b in zip(shape, step2_size, margin)]
        else:
            bb = lab[nodule_idx]
            #print(bb)
            #print(bb)
            low = [np.max([a+bb[3]/2+b-c,-b]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
            high = [np.min([a-bb[3]/2-b,s+b-c]) for s,c,b,a in zip(shape, step2_size, margin, bb[:3])]
            #print(low,high)
            start = [start_fn(l,h) if l<=h else int(a-c/2)   for l,h,a,c in zip(low,high, bb[:3], step2_size)]
            #print(start)
       # if debug:
       #     start = [200,100,250]
        end_point = [s + c for s, c in zip(start, step2_size)]

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + step2_size[i] - shape[i])
            pad.append([leftpad, rightpad])
        pad = np.array(pad)
        crop_im = im[:,
               max(start[0], 0):min(start[0] + step2_size[0], shape[0]),
               max(start[1], 0):min(start[1] + step2_size[1], shape[1]),
               max(start[2], 0):min(start[2] + step2_size[2], shape[2])]
        if len(lab)>0:
            lab[:,0] -= max(start[0], 0)
            lab[:,1] -= max(start[1], 0)
            lab[:,2] -= max(start[2], 0)
#             print([crop.shape, 'crop shape'])
#             print([scale, 'scale'])
        if self.pad_mode == 'constant':
            crop_im = mypad(crop_im, pad, self.pad_value)
        else:
            crop_im  = np.pad(crop_im, pad, self.pad_mode)
        if len(lab)>0:
            lab[:,0] += pad[1,0]
            lab[:,1] += pad[2,0]
            lab[:,2] += pad[3,0]

        if self.augtype['rotate']:  # 切割回来
            raise NotImplementedError
#             for i in range(crop_im.shape[0]):
#                 crop_im[i] = myrotate(crop_im[i], angle, reshape=False, order=1)
#             for i in range(crop_lab.shape[0]):
#                 crop_lab[i] = myrotate(crop_lab[i], angle, reshape=False, order=0)  # 用 0 阶可以加快速度，也不会有label互相干扰的问题
#             #             crop = np.concatenate([rotate(slice, angle,axes=(1,2))[np.newaxis] for slice in crop], axis=0)
#             start = np.ceil((step2_size - step1_size) / 2).astype('int')
#             crop_im = crop_im[:,
#                    start[0]: start[0] + step1_size[0],
#                    start[1]: start[1] + step1_size[1],
#                    start[2]: start[2] + step1_size[2]]
#             crop_lab = crop_lab[:,
#                    start[0]: start[0] + step1_size[0],
#                    start[1]: start[1] + step1_size[1],
#                    start[2]: start[2] + step1_size[2]]

        if self.augtype['scale']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 用 0 阶可以加快速度，也不会有label互相干扰的问题
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
    from configs import Config
    import numpy as np

    Config.load('../configs/test.yml')
    crop = Crop(Config)
    # data = np.ones([1, 500,500,500])
    # crop_data = crop(data)
    # crop_data2 = augment(crop_data, Config.augtype)

    data = myDataset(Config, 'train')
    im, lab, name = data[0]

    data2 = myDataset(Config, 'val')
    im, lab, nswh, name = data2[0]
    print(im.shape)
    print(lab.shape)
    print(nswh)

    comb = data2.split_comb.combine(im,nswh)
    print(comb.shape)
    im_orig = np.load(data2.img_files[0])[np.newaxis]
    print(np.corrcoef(im_orig.reshape(-1), comb.reshape(-1)))
