import numpy as np

class Label_mapping3():
    # 为 FPN 产生label

    def __init__(self, config):
        self.anchors = config.rpn['anchors'] # anchor 的列表，[[1,2],[3]]表示第一个输出层两个anchor， 第二个输出层一个anchor
        self.stride_levels = config.rpn['strides']   # 各个输出层的stride 列表， 例如 [[1,1,1], [2,2,2]] 代表两个输出层的stride 分别是1，2，各向同性的
        self.iou_pos = config.rpn['iou_pos'] # 与答案的 iou 大于多少就认为某个anchor是正样本
        self.iou_neg = config.rpn['iou_neg'] # 与答案的 iou 小于多少就认为某个anchor是负样本
        self.max_pos = config.rpn['max_pos'] # 是否只安排最大iou的那个anchor做正样本
        self.imsize = np.array(config.prepare['crop_size']) # 图像 crop size，也是stirde=1的大小
        self.spacing_ratio = np.array(config.prepare['spacing_ratio']) # z,x,y 轴 spacing 的比例，会用y轴的spacing来归一化
        self.spacing_ratio = self.spacing_ratio/self.spacing_ratio[-1]
        self.N_regress = config.rpn['N_regress']
        self.N_prob = config.rpn['N_prob']
        self.margin =  np.array(config.prepare['margin'])
        self.diam_thresh = np.array(config.rpn['diam_thresh'])
        self.lab_probpos = config.rpn['lab_probpos']
        self.lab_diffpos = config.rpn['lab_diffpos']
        self.lab_probdiffpos = config.rpn['lab_probdiffpos']
        self.lab_neg = config.rpn['lab_neg']
        self.lab_neut = config.rpn['lab_neut']
        self.cls_type = config.classifier['type']
        self.omit_cls = config.classifier['omit_cls']
        if 'bbox_size_range' in config.rpn:
            self.bbox_size_range = config.rpn['bbox_size_range']
        print('omit cls: ',self.omit_cls)
    @staticmethod
    def one_axis( anc, center, radius, stride, lim):
        # 对一个轴，算一下以center为中心，radius + anc/2 为半径的区域内的overlap 区域，之后会将三个轴累积成overlap 体积

        start_s = int(np.floor((center-radius-anc/2)/stride))
        end_s = int(np.ceil((center+radius + anc/2)/stride))
        if end_s<=0 or start_s>=lim:
            # 如果这个box 的范围超过了crop 图像范围，那就不管了
            return False, None,None,None,None

        start_s = np.clip(start_s,0,lim-1)
        end_s = np.clip(end_s, 0, lim-1)+1
        # 将超出的部分掐掉，end 还要+1 是为了作为闭区间
        start = start_s*stride
        end = end_s*stride
        # 还原到原图坐标

        x = np.arange(start, end, stride)# 生成坐标轴，数值是原图上的坐标
        left1 = x-anc/2 # 每个点所对应的anchor 的左边界
        left2 = center-radius # 答案框的左边界
        left_border = left1.copy()
        left_border[left_border<left2] = left2 # 两者左边界的右边那一个

        right1 = x+anc/2
        right2 = center+radius
        right_border = right1.copy()
        right_border[right_border>right2] = right2# 两者右边界的左边那一个

        overlap = right_border-left_border # anchor 框和答案框的重叠部分
        diff_x = (center-x)/anc # anchor 框中心和答案框中心的偏移
        if np.max(overlap)<=0:
            return False, None,None,None,None
        else:
            # 只留下overlap 大于0的部分
            range_x = x[overlap>0]
            start_s = range_x[0]//stride
            end_s = range_x[-1]//stride+1
            diff_x = diff_x[overlap>0]
            overlap = overlap[overlap>0]

            return True, overlap, diff_x, start_s, end_s

    def quantize(self, iou, omit, train_flag, cls=0, iou_neg_idx = 0, use_pos=False):
        # 将iou 量化
        results_prob = np.ones_like(iou)*self.lab_neut
        results_prob[iou<self.iou_neg[iou_neg_idx]] = self.lab_neg
        results_coord = np.ones_like(iou)*self.lab_neut
        results_coord[iou<self.iou_neg[iou_neg_idx]] = self.lab_neg
        if type(self.iou_pos) == list and len(self.iou_pos) > 1:
            iou_pos = self.iou_pos[iou_neg_idx]
        else:
            iou_pos = self.iou_pos
        if not self.max_pos:
            if (not omit) and use_pos:
                # box[4]从0开始，但是我们的0代表背景，如果只是当rpn来训练，只要1就能表示正样本，
                # 如果当ssd来训练，用不同的数字代表不同的类别
                if len(iou.flatten()) > self.N_prob:
                    dy_iou_pos = max(iou_pos, np.sort(iou.flatten())[- self.N_prob])
                else:
                    dy_iou_pos = iou_pos
                iou_pos_bin = iou>=dy_iou_pos
                results_coord[iou_pos_bin] = train_flag
                results_prob[iou_pos_bin] = cls + 1
                    
                

        else:
            
            if not omit:
                if train_flag == self.lab_probdiffpos or train_flag == self.lab_diffpos:
                    results_coord[iou>= iou_pos] = self.lab_diffpos
                max_idx = np.unravel_index(iou.argmax(), iou.shape)
                max_iou = iou[tuple(max_idx)]
                if max_iou>=iou_pos:
                    results_prob[tuple(max_idx)] = cls + 1
                    results_coord[tuple(max_idx)] = train_flag
        #results[np.logical_and(iou<self.iou_pos, iou>=self.iou_neg)]=0
        
        return results_prob, results_coord

    def generate_one_map(self, bboxs, target_shape, anchor, stride, gt_prob, gt_coord_only, gt_diff, i_anchor, i_level=-1):
        # 给定一个anchor，一个stride，生成一个label_map,包括概率图和bbox回归图
        iou_neg_idx = i_level
        spacing = self.spacing_ratio
        hit_boxs = []
        gt_connects = np.zeros_like(gt_prob)
        for b_idx, box in enumerate(bboxs):

            # box 的格式: 中心点在x,y,z轴上的位置，直径，种类，置信度
            in_x, over_x, diffx, sx, ex = self.one_axis(anchor/spacing[0], box[0], box[3]/2/spacing[0], stride[0], target_shape[0])
            in_y, over_y, diffy, sy, ey = self.one_axis(anchor/spacing[1], box[1], box[3]/2/spacing[1], stride[1], target_shape[1])
            in_z, over_z, diffz, sz, ez = self.one_axis(anchor/spacing[2], box[2], box[3]/2/spacing[2], stride[2], target_shape[2])

            if in_x and in_y and in_z:
                #overlap = over_x[:,None,None]*over_y[None,:,None]*over_z[None,None,:]
                #iou = overlap/(anchor**3+(box[3]**3)-overlap)
                if i_level < len(self.stride_levels) - 1:
                    iou = (1 - np.minimum(np.abs((diffx[:,None,None]*(anchor/spacing[0]))/(box[3]/2/spacing[0])), 1)) \
                        *(1 - np.minimum(np.abs((diffy[None,:,None]*(anchor/spacing[1]))/(box[3]/2/spacing[1])), 1))\
                        *(1 - np.minimum(np.abs((diffz[None,None,:]*(anchor/spacing[2]))/(box[3]/2/spacing[2])), 1))
                else:
                    iou = (1 - np.minimum(np.abs((diffx[:,None,None]*(anchor/spacing[0]))/((stride[0] + box[3])/2/spacing[0])), 1)) \
                        *(1 - np.minimum(np.abs((diffy[None,:,None]*(anchor/spacing[1]))/((stride[1] + box[3])/2/spacing[1])), 1))\
                        *(1 - np.minimum(np.abs((diffz[None,None,:]*(anchor/spacing[2]))/((stride[2] + box[3])/2/spacing[2])), 1))
                if np.max(iou)<= self.iou_neg[iou_neg_idx]:
                    continue

                old_clip = gt_prob[i_anchor,sx:ex,sy:ey,sz:ez]
                old_clip_coord = gt_coord_only[i_anchor,sx:ex,sy:ey,sz:ez]
                # 是否要忽略这个物体，条件有两个，第一是直径要在给定范围内，第二是置信度要高
                # 如果确定要忽略这个物体，这个物体不会参与概率训练，omit=True，但是仍然会参加bbox 回归训练

                omit =  (box[3] < self.diam_thresh[0]) or (box[3] > self.diam_thresh[1]) or (box[5] != 1) or (box[4] in self.omit_cls)
                if not omit:
                    hit_boxs.append(box)
                use_pos = (box[3] >= self.bbox_size_range[i_level][0] and box[3] < self.bbox_size_range[i_level][1])
                if self.cls_type=='None':
                    new_clip, new_clip_coord = self.quantize(iou, omit, box[6], iou_neg_idx=iou_neg_idx, use_pos=use_pos)
                elif self.cls_type=='ssd':
                    new_clip, new_clip_coord = self.quantize(iou, omit, box[6], cls=box[4], iou_neg_idx=iou_neg_idx, use_pos=use_pos)
                    
                overwrite_mask =  old_clip< new_clip
                old_clip[overwrite_mask] = new_clip[overwrite_mask]
                gt_prob[i_anchor,sx:ex,sy:ey,sz:ez] = old_clip
                connect = (new_clip > 0) * (b_idx + 1)
                gt_connects[i_anchor,sx:ex,sy:ey,sz:ez] = connect
                
                overwrite_mask_coord =  old_clip_coord< new_clip_coord
                old_clip_coord[overwrite_mask_coord] = new_clip_coord[overwrite_mask_coord]
                gt_coord_only[i_anchor,sx:ex,sy:ey,sz:ez] = old_clip_coord  
 
                
                newdiff = np.array(np.meshgrid(diffx,diffy,diffz,indexing='ij'))
                old_diff = gt_diff[i_anchor,:,sx:ex,sy:ey,sz:ez]
                old_diff[:3,overwrite_mask_coord] = newdiff[:,overwrite_mask_coord]
                old_diff[3,overwrite_mask_coord] = np.log(box[3]/anchor)
                gt_diff[i_anchor,:,sx:ex,sy:ey,sz:ez] = old_diff

                gt_coord_prob = np.array(np.where((gt_coord_only==self.lab_probpos)|(gt_coord_only==self.lab_probdiffpos))).T
                gt_coord_diff = np.array(np.where((gt_coord_only==self.lab_diffpos)|(gt_coord_only==self.lab_probdiffpos))).T

            else:
                continue

        return gt_prob, gt_diff, hit_boxs, gt_connects

    def pad(self,gt_coord, N_target, other=None):
        if len(gt_coord) <N_target:
            tmp = np.ones([N_target-len(gt_coord),4],dtype=np.long)*-1
            gt_coord = np.concatenate([gt_coord, tmp], axis=0)
            if other is not None:
                other =  np.concatenate([other, tmp.astype('float32')], axis=0)
        elif len(gt_coord)>N_target:
            sampled = np.random.choice(np.arange(len(gt_coord)),N_target,replace=False)
            gt_coord = gt_coord[sampled,:]
            if other is not None:
                other = other[sampled,:]
        return gt_coord, other

    def __call__(self, bboxs):
        gt_prob_fpn = []
        gt_coord_diff_fpn = []
        gt_coord_prob_fpn = []
        gt_diff_fpn = []
        gt_connects_fpn = []
        bboxs = bboxs.copy()
        if len(bboxs)>0:
            bboxs[:,:3] = bboxs[:,:3]-self.margin[None,:]
        for level_idx, stride, anchors in zip(range(len(self.stride_levels)), self.stride_levels, self.anchors):
            stride = np.array(stride)
            assert np.all(np.mod(self.margin,stride)==0)
            assert np.all(np.mod(self.imsize,stride)==0)
            margin = self.margin//stride
#             if stride
            target_shape = self.imsize//stride - margin*2
            if len(bboxs)>0:
                if bboxs[0,7]==0:
                    gt_prob = np.zeros([len(anchors)*1]+list(target_shape))
                else:
                    gt_prob = np.ones([len(anchors)*1]+list(target_shape))*self.lab_neg
            else:
                gt_prob = np.ones([len(anchors)*1]+list(target_shape))*self.lab_neg
            gt_coord_only = np.ones([len(anchors)*1]+list(target_shape))*self.lab_neg
            gt_diff = np.zeros([len(anchors), 4]+list(target_shape))
            for i_anchor, anchor in enumerate(anchors):
                _, _, hit_boxs, gt_connects = self.generate_one_map(bboxs, target_shape, anchor, stride, gt_prob, gt_coord_only, gt_diff, i_anchor, i_level=level_idx)
            gt_coord_prob = np.array(np.where((gt_coord_only==self.lab_probpos)|(gt_coord_only==self.lab_probdiffpos))).T
            gt_coord_diff = np.array(np.where((gt_coord_only==self.lab_diffpos)|(gt_coord_only==self.lab_probdiffpos))).T
            diffs = gt_diff[gt_coord_diff[:,0],:,gt_coord_diff[:,1],gt_coord_diff[:,2], gt_coord_diff[:,3]]
            #if len(hit_boxs) > 0:
            #    print(level_idx, np.array(hit_boxs)[:,3], len(gt_coord_prob))
            #else:
            #    print(level_idx, [], 0)
            gt_coord_prob,_ = self.pad(gt_coord_prob, self.N_prob)
            gt_coord_diff, diffs = self.pad(gt_coord_diff, self.N_regress, diffs)
            gt_prob_fpn.append(gt_prob)
            gt_connects_fpn.append(gt_connects)
            gt_diff_fpn.append(diffs)
            gt_coord_prob_fpn.append(gt_coord_prob)
            gt_coord_diff_fpn.append(gt_coord_diff)
        return gt_prob_fpn, gt_coord_prob_fpn, gt_coord_diff_fpn, gt_diff_fpn, gt_connects_fpn
#                 print(gt_prob.shape, gt_diff.shape)
#                 ipv.figure()
#                 ipv.plot_isosurface(gt_prob[0]>=1, extent=[[0,gt_prob.shape[1]],[0,gt_prob.shape[2]],[0,gt_prob.shape[3]]] )
#                 ipv.show()
