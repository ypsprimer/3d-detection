import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2
from scipy.ndimage import label as lb
from pprint import pprint
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import tqdm
from copy import copy, deepcopy


def cal_boxCoord(mask_arr):
    """
    生成rib box的坐标（6个）
    :param mask_arr: pixel-wise的rib mask

    return:
        ndarray -> [z1, x1, y1, z2, x2, y2]
    """
    mask_pos = np.argwhere(mask_arr != 0)
    z_max, z_min = np.max(mask_pos[:,0]), np.min(mask_pos[:,0])
    y_max, y_min = np.max(mask_pos[:,1]), np.min(mask_pos[:,1])
    x_max, x_min = np.max(mask_pos[:,2]), np.min(mask_pos[:,2])
    
    return np.array([z_max, z_min, y_max, y_min, x_max, x_min])


def get_ribBox(img_arr, thr=300, z_maxIdx=250, y_maxIdx=360, k_size=3):
    """
    获取肋骨所在的大致区域，用box框出

    :param img_arr: case的.npy文件
    :param thr: 肋骨分割的阈值
    :param z_maxIdx: CT最多取slice的数量
    :param y_maxIdx: y的范围最大值
    :param k_size: 膨胀的kernel size

    Return:
        labeled_box: [z1, y1, x1, z2, y2, x2]
    
    """
    
    # img_path = '/yupeng/micca20/ribfrac-train-images/RibFrac{}-image.nii.gz'.format(pid)
    # mask_path = '/yupeng/micca20/ribfrac-train-labels/RibFrac{}-label.nii.gz'.format(pid)

    # img = sitk.ReadImage(img_path)
    # img_arr = sitk.GetArrayFromImage(img)
    # img_arr = np.clip(img_arr, -200, 800) # window 300±1000


    rib_mask = deepcopy(img_arr)
    rib_mask = np.clip(rib_mask, -200, 800)
    rib_mask[rib_mask <= thr] = 0
    rib_mask[rib_mask > thr] = 1
    
    if z_maxIdx == None:
        rib_mask = rib_mask[:,:y_maxIdx,:]
    else:
        rib_mask = rib_mask[:z_maxIdx,:y_maxIdx,:] # 取一定范围内的 rib

    # 膨胀
    kernel = np.ones((k_size,k_size),np.uint8)
    rib_mask = cv2.dilate(rib_mask.astype('uint8'),kernel)

    # 计算连通域
    labeled_array, num_features = lb(rib_mask)
    # print('Num of rib regions: {}'.format(num_features))
    
    # 计算最大连通域的region编号
    max_num_ft = 0
    max_id_ft = 0
    for id in range(1, num_features):
        tmp = np.sum(labeled_array==id)
        if tmp > max_num_ft:
            max_num_ft = tmp
            max_id_ft = id
    # print(max_id_ft)
    
    # 只取最大连通域，其余设为0
    labeled_array[labeled_array != max_id_ft] = 0
    # print(np.sum(labeled_array))
    
    # 计算包围框
    labeled_box = cal_boxCoord(labeled_array) 
    
    # up_left = [labeled_box[5], labeled_box[3]]
    # length = [labeled_box[4]-labeled_box[5], labeled_box[2]-labeled_box[3]]
    return labeled_box



def get_lungEdge(lungseg_root, pid):
    """
    获取肺叶分割的边缘区域部分，pixel-wise

    :param lungseg_root: 分割文件的root dir 
    :param pid: case id
    
    """
    lungseg_dir = os.path.join(lungseg_root, '{}-seg.npz'.format(pid))
    lung_seg = np.load(lungseg_dir)['mask_data']
    
    # 左肺编号：4，5
    # 右肺编号：1，2，3
    right_lung_mask = np.where(lung_seg >= 4, 1, 0)
    left_lung_mask = np.where((lung_seg > 0) & (lung_seg < 4), 1, 0)

    # 获取左肺的左边缘，右肺的右边缘
    left_edge = pick_edge(left_lung_mask, is_left=True)
    right_edge = pick_edge(right_lung_mask, is_left=False)

    return left_edge, right_edge
    

def pick_edge(mask, is_left=True):
    """
    提取一个mask的边缘，分为左or右
    
    :param mask: mask区域
    :param is_left: 是否提取左边缘

    """
    mask = mask.astype(np.uint8)
    n_slice = mask.shape[0]
    # z轴: xy平面上的边缘点
    z2xy = {}

    for i in range(n_slice):
    # for i in tqdm(range(n_slice), total=n_slice):
        # mask: 原图像，cv2.RETR_EXTERNAL: 只检测外轮廓
        # im2: 图像，contours: 轮廓list，hierarchy: 等级（not used）
        im2, contours, hierarchy = cv2.findContours(mask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 存在肺叶区域
        if len(contours):
            contours = np.array(contours[0])
            contours =  np.reshape(contours, (-1, 2))
            # print(contours)
            
            # 取边缘上的最右点或者最左点
            if is_left:
                extreme_pos = np.argmin(contours[:,0])
                extreme_pos = contours[extreme_pos]
                # 当前x轴指针位置
                x_pos = extreme_pos[0] + 1
            else:
                extreme_pos = np.argmax(contours[:,0])
                extreme_pos = contours[extreme_pos]
                x_pos = extreme_pos[0] - 1

            # print(x_pos)
            # 向上/向下搜索是否结束
            up_end = False
            down_end = False
            chosen_pts_up = [[extreme_pos[0],extreme_pos[1]]]
            chosen_pts_down = [[extreme_pos[0],extreme_pos[1]]]
            x_limit = contours[np.argmax(contours[:,0])][0]
            while (x_pos > 0 and x_pos <= x_limit):
                if (up_end & down_end) == True:
                    break
                
                pt_vertical = contours[(contours[:,0] == x_pos)]  
                above_ = (pt_vertical[:,1] > extreme_pos[1]).any()
                below_ = (pt_vertical[:,1] < extreme_pos[1]).any()

                if above_ and below_:
                    y_max = max(pt_vertical[:,1])
                    y_min = min(pt_vertical[:,1])
                    
                    if chosen_pts_up[-1][1] <= y_max and not up_end:
                        chosen_pts_up.append([x_pos, y_max])
                    else:
                        up_end = True
                        
                    if chosen_pts_down[-1][1] >= y_min and not down_end:
                        chosen_pts_down.append([x_pos, y_min])
                    else:
                        down_end = True
                        
                elif above_:
                    y_max = max(pt_vertical[:,1])
                    if chosen_pts_up[-1][1] <= y_max and not up_end:
                        chosen_pts_up.append([x_pos, y_max])
                    else:
                        up_end = True
                        
                elif below_:
                    y_min = min(pt_vertical[:,1])
                    if chosen_pts_down[-1][1] >= y_min and not down_end:
                        chosen_pts_down.append([x_pos, y_min])
                    else:
                        down_end = True
                    
                if is_left:
                    x_pos += 1
                else:
                    x_pos -= 1

            # 合并上游 & 下游分支
            if len(chosen_pts_down) > 1 and len(chosen_pts_up) > 1:
                comb = chosen_pts_down[1:] + chosen_pts_up[1:]
            elif len(chosen_pts_down) > 1:
                comb = chosen_pts_down[1:]
            elif len(chosen_pts_up) > 1:
                comb = chosen_pts_up[1:]
            else:
                comb = contours[contours[:,0] == extreme_pos[0]]
                z2xy[i] = comb.tolist()
                continue

            # comb = chosen_pts_down[1:] + chosen_pts_up[1:]
            comb = np.array(comb)
            # 补全初始点
            source_pos = contours[contours[:,0] == extreme_pos[0]]
            comb = np.concatenate((comb, source_pos))
            z2xy[i] = comb.tolist()

    return z2xy



def bbox_size_stat(train_txt, val_txt, file_dir):

    # 读取train, val
    train_list = []
    with open(train_txt, 'r') as f:
        for line in f:
            line = line.strip()
            train_list.append(line)
    
    val_list = []
    with open(val_txt, 'r') as f:
        for line in f:
            line = line.strip()
            val_list.append(line)

    labels_name = [ii for ii in os.listdir(file_dir) if '_label' in ii]
    train_labels_path = [os.path.join(file_dir, ii) for ii in labels_name if ii.split('_')[0] in train_list]
    val_labels_path = [os.path.join(file_dir, ii) for ii in labels_name if ii.split('_')[0] in val_list]

    # print(len(train_labels_path))
    # print(len(val_labels_path))
    
    train_size_list = []
    for path in train_labels_path:
        label = np.load(path)
        if len(label.shape) == 2 and label.shape[1] == 8:
            size_info = label[:,4]
            train_size_list.append(size_info)

    val_size_list = []
    for path in val_labels_path:
        label = np.load(path)
        if len(label.shape) == 2 and label.shape[1] == 8:
            size_info = label[:,4]
            val_size_list.append(size_info)

    train_size_list = np.concatenate(train_size_list)
    val_size_list = np.concatenate(val_size_list)

    print(np.unique(train_size_list))
    print(np.unique(val_size_list))
    # print(val_size_list)

    # all_size_list = np.concatenate((train_size_list, val_size_list))
    # print(len(train_size_list))
    # print(len(val_size_list))
    # # plt.plot(sorted(train_size_list))
    # print(max(train_size_list))
    # print(min(train_size_list))
    # plt.hist(train_size_list, bins=40)
    # # plt.boxplot(train_size_list, showmeans=True, showcaps=True)
    # plt.savefig('train_stat.jpg')

def convert_dtype(inputs):

    # npy_path = [os.path.join(out_dir, ii) for ii in os.listdir(out_dir) if '_img' in ii]
    # print(npy_path)

    # for npy in tqdm(npy_path, total=len(npy_path)):
    data = np.load(inputs).astype(np.int16)
    if data.dtype != np.int16:
        np.save(inputs, data)

    

def read_sitk(path):
    """
    读取.nii文件
    :param path:

    Return:
        data: ndarray的图片内容
        img.GetSpacing(): CT的spacing大小
    
    """
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data, img.GetSpacing()


def bbox_3d(mask, nodules):
    """
    ndarray

    """
    all_labels_list = []
    n_cands = int(np.max(mask))
    rois = np.array([(mask == ii) * 1 for ii in range(1, n_cands + 1)])

    for rix, r in enumerate(rois):
        if np.sum(r !=0) > 0:
            seg_ixs = np.argwhere(r != 0)
            # z, x, y
            # coord_list = [np.min(seg_ixs[:, 0])-1, 
            #             np.max(seg_ixs[:, 0])+1,
            #             np.min(seg_ixs[:, 1])-1, 
            #             np.max(seg_ixs[:, 1])+1,
            #             np.min(seg_ixs[:, 2])-1,
            #             np.max(seg_ixs[:, 2])+1,]
            z1, z2 = np.min(seg_ixs[:,0]), np.max(seg_ixs[:,0])
            y1, y2 = np.min(seg_ixs[:,1]), np.max(seg_ixs[:,1])
            x1, x2 = np.min(seg_ixs[:,2]), np.max(seg_ixs[:,2])
            z = int((z1 + z2)/2)
            y = int((y1 + y2)/2)
            x = int((x1 + x2)/2)
            d = max(z2 - z1, y2 - y1, x2 - x1)

            if nodules[rix] == -1:
                coord_list = [z, y, x, d, 5, 1, 1, 1]
            else:
                coord_list = [z, y, x, d, nodules[rix], 1, 1, 1]

            all_labels_list.append(coord_list)

    return all_labels_list
    

def mosaic_generator(pid, lab):

    diam_thresh = [0, float('inf')]
    omit_cls = []

    nodule_info = []

    for i in range(len(lab)):
        # if ((lab[i,3] >= diam_thresh[0]) and (lab[i,3] <= diam_thresh[1]) and (lab[i,4] not in omit_cls) and ((lab[i,5] == 1 ) or (lab[i,5] == 0 and lab[i,4] in [1,2]))):
        if ((lab[i,3] >= diam_thresh[0]) and (lab[i,3] <= diam_thresh[1]) and (lab[i,4] not in omit_cls) and ((lab[i,5] == 1 ))):    
            nodule_info.append([pid + '_'+str(i), lab[i,3], lab[i,4]])
    
    nodule_info = np.array(nodule_info)
    
    return nodule_info


def pp_id(pid, with_rib=False, with_lung=True):
    """
    
    :param with_rib: 是否获取rib区域的box，True or False
    :param with_lung: 是否获取肺叶的边缘点，pixel-wise，True or False

    """

    print('Case id: {}'.format(pid))
    img_path = os.path.join(img_subdir, pid + '-image.nii.gz')
    mask_path = os.path.join(label_subdir, pid + '-label.nii.gz')
    
    img_arr, _ = read_sitk(img_path)
    # img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr)
    img_arr = img_arr.astype(np.float16)
    
    if with_rib:
        # 获取肋骨区域
        rib_bbox = get_ribBox(img_arr, z_maxIdx=None, y_maxIdx=400, k_size=5)
        np.save(os.path.join(out_dir, '{}_ribbox.npy'.format(pid)), rib_bbox)

    if with_lung:
        # 获取左肺的左边缘，右肺的右边缘
        left_edge, right_edge = get_lungEdge(lung_dir, pid)
        left_js = json.dumps(left_edge)
        right_js = json.dumps(right_edge)
        with open(os.path.join(out_dir, '{}_left-lung.json'.format(pid)), 'w') as f:
            f.write(left_js)

        with open(os.path.join(out_dir, '{}_right-lung.json'.format(pid)), 'w') as f:
            f.write(right_js)


    mask_arr, _ = read_sitk(mask_path)
    mask_arr = mask_arr.astype(np.uint8)


    nodules = list(map(int, info_df[info_df.public_id == pid].label_code.values))   
   
    if len(nodules) > 1: 
        # 去除0
        nodules = nodules[1:]   
        labels_list = np.array(bbox_3d(mask_arr, nodules))
        nodule_info = mosaic_generator(pid, labels_list)
        # print(nodule_info)
        # img & label
        np.save(os.path.join(out_dir, '{}_img.npy'.format(pid)), img_arr)
        np.save(os.path.join(out_dir, '{}_label.npy'.format(pid)), labels_list)
        np.save(os.path.join(out_dir,'{}_masonic_info.npy'.format(pid)), nodule_info)

    # 全阴性，不生成label
    else:
        np.save(os.path.join(out_dir, '{}_img.npy'.format(pid)), img_arr)
        np.save(os.path.join(out_dir, '{}_label.npy'.format(pid)), np.array([]))


def aggregate_nodule():

    nodule_list = [os.path.join(out_dir, ii) for ii in os.listdir(out_dir) if '_masonic_info' in ii]

    cnt = 0
    agg_nodule = []
    for n in nodule_list:
        one_file = np.load(n)
        if len(one_file.shape) != 2 or one_file.shape[1] != 3:
            continue
            
        agg_nodule.append(one_file)
        cnt += 1

    agg_nodule = np.concatenate(agg_nodule)
    print('Processed file count: {}'.format(cnt))

    return agg_nodule


def write_txt(is_train=False):
    train_split_path = os.path.join(data_dir, 'train.txt')
    val_split_path = os.path.join(data_dir, 'val.txt')

    if is_train:
        file = open(train_split_path, 'w')

    else:
        file = open(val_split_path, 'w')

    for pid in pids:
        file.write(pid + '\n')
    
    file.close()


if __name__ == "__main__":
    
    root_dir = '/ssd/micca20/'
    data_dir = '/ssd/micca20/alg_frame_v3'
    out_dir = '/ssd/micca20/alg_frame_v3/data'
    lung_dir = '/ssd/ribfrac-train-images_lobe2/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    label_subdir = os.path.join(root_dir, 'ribfrac-train-labels')
    img_subdir = os.path.join(root_dir, 'ribfrac-train-images')

    info_file = os.path.join(root_dir, 'ribfrac-train-info.csv')
    info_df = pd.read_csv(info_file)
    pids = set(info_df.public_id.values)
    
    if 'public_id' in pids:
        pids.remove('public_id')
    # pids = ['RibFrac181']
    # pids = ['RibFrac207']
    # print(pids)
    # exit()

    # 把肺叶边缘的json转为npy
    # for i in os.listdir(out_dir):
    #     if '.json' in i:
    #         edges_ = []
    #         with open(os.path.join(out_dir, i), 'r') as f:
    #             data = json.load(f)
    #             for z in data:
    #                 for x, y in data[z]:
    #                     edges_.append([int(z), x, y])

    #         edges_ = np.array(edges_)
    #         np.save(os.path.join(out_dir, i.split('.')[0] + '.npy'), edges_)



    # bbox_size_stat(train_txt=os.path.join(data_dir, 'train.txt'), 
    #                val_txt=os.path.join(data_dir, 'val.txt'), 
    #                file_dir=out_dir)

    # npy_path = [os.path.join(out_dir, ii) for ii in os.listdir(out_dir) if '_img' in ii]
    # for i in npy_path:
    #     convert_dtype(i)
    # pool = Pool(processes=30)
    # p1 = pool.map(convert_dtype, npy_path, chunksize=1)
    # pool.close()
    # pool.join()

    # debug
    # for ix, pid in enumerate(pids):
    #     if ix == 10:
    #         break
    #     pp_id(pid, with_rib=False)
    # pp_id('RibFrac483', with_rib=False)

    # mulitprocess pp_patient
    # pool = Pool(processes=30)
    # p1 = pool.map(pp_id, pids, chunksize=1)
    # pool.close()
    # pool.join()

    agg_nodule = aggregate_nodule()
    np.save(os.path.join(out_dir,'nodule_info.npy'), agg_nodule)

    write_txt(is_train=True)

