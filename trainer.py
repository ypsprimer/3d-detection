import time, os
import numpy as np
import json

from torch import optim
from torch.autograd import Variable
import torch
from torch import nn
from torch.nn import functional as F
from train import LRLoader, Lookahead
from models.inplace_abn import ABN

from utils import myPrint, Averager
from torch.nn import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from apex.fp16_utils import FP16_Optimizer
from itertools import compress
from torchvision.ops import nms3d
from apex.optimizers import FusedAdam
import shutil
import _pickle as cPickle

def neg(x):
    if x==0:
        return None
    else:
        return -x


def decode_bbox(logits, thresh, idx, config):
    """
    计算box坐标和置信度，改换为立方体
    :param logits:
    :param thresh: not used
    :param idx:
    :param config:

    return:
        bbox_keep: z1 y1, x1, z2, y2, x2, probs, cls - 坐标和类别置信度
        thresh_list:

    """
    all_bboxs = []
    all_probs = []
    all_cls = []
    n_level = len(config.rpn['strides'])

    thresh_list = []
    for level in range(n_level):
        pred_p = logits[n_level*2 - level*2 -2][idx].float()
        pred_d = logits[n_level*2 - level*2 -1][idx].float()
        # pred_d = pred_d.view([4,pred_d.shape[0]//4]+list(pred_d.shape[1:]))
        # cube 6个坐标
        pred_d = pred_d.view([6,pred_d.shape[0]//6]+list(pred_d.shape[1:]))
        stride = torch.tensor(config.rpn['strides'][level]).float().cuda()
        anchors = torch.tensor(config.rpn['anchors'][level])
        if config.classifier['activation'] == 'softmax':
            N_cls = config.classifier['N_cls']+1
            pred_p = pred_p.view([N_cls, pred_p.shape[0]//N_cls] + list(pred_p.shape[1:]))
            pred_p = torch.softmax(pred_p, dim=0)
        else:
            N_cls = config.classifier['N_cls']
            pred_p = pred_p.view([N_cls, pred_p.shape[0]//N_cls] + list(pred_p.shape[1:]))
            top_pred_p, _ = torch.topk(pred_p.reshape([-1]), 1000)
            thresh = -2.0
            thresh_list.append(thresh)

            baseline = torch.ones([1]+list(pred_d.shape[1:])).cuda()*thresh
            pred_p = torch.cat([baseline, pred_p], dim=0)

        maxp_p, cls_p = torch.max(pred_p, dim=0)
        coords = torch.nonzero(cls_p)
        cls_pos = cls_p[tuple(coords.transpose(0,1))]
        probs = maxp_p[tuple(coords.transpose(0,1))]
        diffs = pred_d[:,coords[:,0],coords[:,1],coords[:,2],coords[:,3]].transpose(0,1)
        # 对应于哪种类型的anchor
        ancs = anchors[coords[:,0]].unsqueeze(1).float().cuda()
        # cents = coords[:,1:].float()*stride + diffs[:,:3]*ancs
        cents = coords[:,1:].float()*stride + diffs[:,:3]
        # diams = ancs*torch.exp(diffs[:,3:])
        diams = diffs[:,3:]

        bboxs = torch.cat([cents-diams/2, cents+diams/2], dim=1)
        if len(bboxs)>0:
            all_bboxs.append(bboxs)
            all_probs.append(probs)
            all_cls.append(cls_pos)
    if len(all_bboxs)>0:
        all_bboxs = torch.cat(all_bboxs,dim=0)
        all_probs = torch.cat(all_probs,dim=0)
        all_cls = torch.cat(all_cls,dim=0)
        keep = nms3d(all_bboxs, all_probs, 0.0)
        bbox_keep = all_bboxs[keep]
        prob_keep = all_probs[keep].unsqueeze(1)
        cls_keep = all_cls[keep].unsqueeze(1).float()

        bbox_keep = torch.cat([bbox_keep, prob_keep, cls_keep], dim=1)
    else:
        bbox_keep = []
    # anchors
    # anchors
    return bbox_keep, thresh_list


class Batch_Filler():
    def __init__(self, bs, size, dtype=torch.float):
        """
        bs: batch_size
        size: [1, 160, 160, 160]

        """
        self.bs = bs
        self.size = size
        self.dtype = dtype
        # [8, 1, 160, 160, 160]
        self.batch = torch.zeros([bs]+list(size), dtype = dtype).cuda()
        self.belong = np.zeros(bs)
        self.isFull = False
        self.start = 0
        self.idxlist = []

    def fill(self, idx, x):
        """
        
        :param: idx: CT id
        :param: x -> tensor([n, 1, 160, 160, 160]): 一张CT中的所有小块

        return:
            isFull: batch是否已满
            batch: 当前batch的数据
            belong: 每一个batch所属CT id
            idxlist: 
            x_left: 当前取完后，剩余部分的数据

        """
        # print(x.shape[0],self.start, self.bs)
        
        # 一个batch可以处理完
        if x.shape[0]+self.start <= self.bs:
            self.batch[self.start:self.start+x.shape[0]] = x
            self.belong[self.start:self.start+x.shape[0]] = idx
            self.start += x.shape[0]
            self.idxlist.append(idx)
            x_left = None

        # 分batch处理
        else:
            self.batch[self.start:] = x[:(self.bs-self.start)]
            self.belong[self.start:] = idx
            # 剩余部分
            x_left  =x[(self.bs-self.start):]
            self.start = self.bs

        if self.start==self.bs:
            self.isFull = True
        
        return self.isFull, self.batch, self.belong, self.idxlist, x_left

    def restart(self):
        self.isFull = False
        self.start = 0
        self.idxlist = []

class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))

def restart_logit(x_left,logits_left, logits_left_new, infos_list):
    if x_left is not None:
        if logits_left is not None:
            logits_left = logits_left + logits_left_new
        else:
            logits_left = logits_left_new
        infos_list = [infos_list[-1]]
    else:
#             print('left', x_left)
        logits_left = None
        infos_list = []
    return logits_left, infos_list



class Trainer:
    def __init__(self, warp, args, ioer, emlist):
        self.warp = warp
        self.args = args
        self.ioer = ioer
        self.emlist = emlist
        self.half = args.half
        if self.half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        if isinstance(self.warp, DataParallel):
            self.net = self.warp.module.net
        else:
            self.net = self.warp.net

        self.lrfun = LRLoader.load(args.train['lr_func'])
        self.lr_arg = args.train['lr_arg']
        self.epoch = args.train['epoch']
        self.optimizer = self.__get_opimizer()

        self.save_dir = os.path.join(args.output['result_dir'], args.output['save_dir'])
        self.printf = myPrint(os.path.join(self.save_dir, "log.txt"))
        # print('*' * 10)
        # print(self.save_dir)
        # print('*' * 10)

        self.margin = np.array(args.prepare['margin'])
        self.cropsize = np.array(args.prepare['crop_size'])
        self.small_size = args.rpn['small_size']
        if 'pos_weight_thresh' in args.prepare:
            self.pos_weight_thresh = args.prepare['pos_weight_thresh']
        else:
            self.pos_weight_thresh = 0

    def __get_opimizer(self):
        weight_decay = self.args.train['weight_decay']
        if self.args.train['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.net.parameters(), lr=self.getLR(0), momentum=0.9,
                              weight_decay=weight_decay)
        elif self.args.train['optimizer'] == 'Adam':
            if self.half:
                optimizer = FusedAdam(self.net.parameters(), lr=self.getLR(0))
            else:
                optimizer = optim.Adam(self.net.parameters(), lr=self.getLR(0))
        elif self.args.train['optimizer'] == 'AdamW':
            if self.half:
                optimizer = FusedAdam(self.net.parameters(), lr=self.getLR(0), adam_w_mode=True)
            else:
                optimizer = optim.AdamW(self.net.parameters(), lr=self.getLR(0))
        if self.half:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
                dynamic_loss_args={'scale_factor' : 3})
            optimizer.loss_scale = 512
        if self.args.train['LA'] == True:
            optimizer = Lookahead(optimizer,k=5,alpha=0.5)        
        return optimizer

    def writeLossLog(self, phase, epoch, meanloss, loss_list=[], em_list=[], lr = None, time=None):
        if phase == 'Train':
            st = '%s, Epoch %d, lr %.1e | total loss: %.4f | ' % (phase.ljust(6), epoch, lr, meanloss)
        else:
            st = "%s, Epoch %d | total loss: %.4f | " % (phase.ljust(6), epoch, meanloss)
        for i, l in enumerate(loss_list):
            st += 'loss %d: %.4f  | ' % (i, l)

        for i, e in enumerate(em_list):
            if isinstance(e, dict):
                for k,v in e.items():
                    st += 'em %s: %.4f  | ' % (k, v)
            else:
                em_name = self.args.net['em'][i]
                st += 'em %s: %.4f  | ' % (em_name, e)
        if time is not None:
            st += 'time: %.2f min'%(time)
        self.printf(st)
        return

    def getLR(self, epoch):
        return self.lrfun(self.lr_arg, epoch, self.epoch)

    def clipmargin(self,tensors):
        for i,o in enumerate(tensors):
            stride = np.array(self.cropsize)/np.array(o.shape[2:])
            m = (self.margin/stride).astype('int')
            tensors[i] = o[:,:,m[0]:neg(m[0]), m[1]:neg(m[1]),m[2]:neg(m[2])]
        return tensors

    def train(self, epoch, dataloader):
        use_cuda = torch.cuda.is_available()
        self.net.train()

        for m in self.net.modules():
            if isinstance(m, _BatchNorm) or isinstance(m, ABN):
            # if isinstance(m, _BatchNorm) or isinstance(m, InPlaceABNSync) or isinstance(m,InPlaceABN):
                if self.args.train['freeze']:
                    m.eval()

        lr = self.getLR(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        loss_avg = Averager()
        lls_avg = Averager()

        startt = time.time()
        lastt0 = startt
        for batch_idx, (data, fpn_prob, fpn_coord_prob, fpn_coord_diff, fpn_diff, fpn_connects, names) in enumerate(dataloader):
            t0 = time.time()
            iter_time = t0-lastt0
            lastt0 = t0
            case_idxs = [dataloader.dataset.cases2idx[item] for item in names]
            if use_cuda:
                data = data.cuda()
                fpn_prob = [f.cuda() for f in fpn_prob]
                fpn_connects = [f.cuda() for f in fpn_connects]
                fpn_coord_prob = [f.cuda() for f in fpn_coord_prob]
                fpn_coord_diff = [f.cuda() for f in fpn_coord_diff]
                fpn_diff =  [f.cuda() for f in fpn_diff]
                case_idxs = torch.Tensor(case_idxs).cuda()

            losses, weights, pred_prob_list = self.warp(data, fpn_prob, fpn_coord_prob, fpn_coord_diff, fpn_diff, fpn_connects, case_idxs)
            # print(losses,'losses')
            # print(weights, 'weights')
            if pred_prob_list is not None:
                pred_prob_dict_pos = {}
                pred_prob_dict_neg = {}
                for pred_prob in pred_prob_list.cpu().numpy():
                    if pred_prob[0] == -1:
                        continue
                    case_idx, nodule_idx, n_weight = pred_prob
                    assert nodule_idx != 0
                    nodule_key = dataloader.dataset.cases[int(case_idx)] + '___' + str(abs(int(nodule_idx)))
                    if nodule_idx > 0:
                        if nodule_key not in pred_prob_dict_pos:
                            pred_prob_dict_pos[nodule_key] = n_weight
                        else:
                            pred_prob_dict_pos[nodule_key] = min(n_weight, pred_prob_dict_pos[nodule_key])
                    elif nodule_idx < 0:
                        if nodule_key not in pred_prob_dict_neg:
                            pred_prob_dict_neg[nodule_key] = n_weight
                        else:
                            pred_prob_dict_neg[nodule_key] = max(n_weight, pred_prob_dict_neg[nodule_key])
                for nodule_key, n_weight in pred_prob_dict_pos.items():
                    assert nodule_key in dataloader.dataset.sample_weights
                    dataloader.dataset.sample_weights[nodule_key][0] = n_weight
                    if n_weight > self.pos_weight_thresh:
                        dataloader.dataset.sample_weights[nodule_key][2] += 1
                        if dataloader.dataset.sample_weights[nodule_key][2] >= 3:
                            case_name, nodule_idx = nodule_key.split('___')
                            dataloader.dataset.lab_buffers[case_name][int(nodule_idx)-1][5] = 0

                #for nodule_key, n_weight in pred_prob_dict_neg.items():
                #    assert nodule_key in dataloader.dataset.neg_sample_weights
                #    dataloader.dataset.neg_sample_weights[nodule_key][0] = n_weight

            losses = losses.sum(dim=0)
            weights = weights.sum(dim=0)
            if weights.shape[0] > losses.shape[0]:
                assert weights.shape[0] == losses.shape[0] * 2
                fack_weights = weights[losses.shape[0]:]
                weights = weights[:losses.shape[0]]
            else:
                fack_weights = None
            total_loss = 0
            loss_list = []
            if fack_weights is not None:
                for l, w, fw in zip(losses, weights, fack_weights):
                    l_tmp = (l/ (1e-3+w))
                    total_loss += l_tmp
                    fack_l_tmp = (l/ (1e-3+fw))
                    loss_list.append(fack_l_tmp.detach().cpu().numpy())
            else:
                for l, w in zip(losses, weights):
                    l_tmp = (l/ (1e-3+w))
                    total_loss += l_tmp
                    loss_list.append(l_tmp.detach().cpu().numpy())

            loss_avg.update(total_loss.detach().cpu().numpy())
            info = 'end %d out of %d, '%(batch_idx, len(dataloader))

            

            for lid, l in enumerate(loss_list):
                info += 'loss %d: %.4f, '%(lid, np.mean(l))
            info += 'time: %.2f' %iter_time
            print(info)
            lls_avg.update(tuple(loss_list))
            self.optimizer.zero_grad()
            loss_scalar = total_loss
            if self.half:
                self.optimizer.backward(loss_scalar)
                #self.optimizer.clip_master_grads(1)
            else:
                loss_scalar.backward()
            if self.args.clip:
                torch.nn.utils.clip_grad_value_(self.warp.parameters(),1)
            self.optimizer.step()
        endt = time.time()
        self.writeLossLog('Train', epoch, meanloss = loss_avg.val(), loss_list = lls_avg.val(), lr = lr, time=(endt-startt)/60)

        return lls_avg.val()

    def validate(self, epoch, val_loader, save=False):


        """
        测试每张CT，每一个iter为CT中所有的小块
        :param val_loader -> Dataloader(): 所有测试的CT

        """
        startt = time.time()
        self.net.eval()
        # print(vars(self.ioer))
        if not self.args.val:
            if epoch % self.args.output['save_frequency'] == 0:
                self.ioer.save_file(self.net, epoch, self.args, 0)
            if epoch % self.args.output['val_frequency'] != 0:
                return
        
        loss_avg = Averager()
        lls_avg = Averager()
        em_avg = Averager()
        bs =  self.args.train['batch_size']
        if save:
            savedir = os.path.join(self.ioer.save_dir, '%03d' % epoch)
            if not os.path.exists(savedir):
                #shutil.rmtree(savedir)
                os.mkdir(savedir)

        ap_pred_list = []
        ap_gt_list = []
        with torch.no_grad():

            xbatch_filler = Batch_Filler(bs, size = [self.args.prepare['channel_input']] + self.args.prepare['crop_size'], dtype=self.dtype)
            bbox_left = None
            infos_list = []

            for sample_idx, tmp in enumerate(val_loader):
                # zhw: shape of data
                data, zhw, name, fullab = tmp[0]

                infos_list.append([zhw, name, fullab])
                x_left = torch.from_numpy(data).cuda()

                # 在一张CT中不断取bs大小的数据
                while x_left is not None:
                    isFull, xbatch, belong, idxlist, x_left = xbatch_filler.fill(sample_idx, x_left)

                    if len(val_loader) == sample_idx + 1:  # the last sample force execute test operation.
                        isFull = True
                        if len(idxlist) != bs:
                            fill_length = bs - len(idxlist)
                            for i in range(fill_length):
                                idxlist.append(-1)

                    if isFull:
                        data = xbatch
                        logits = self.warp(data, calc_loss=False)
                        logits = self.clipmargin(list(logits))

                        box_batch = MaskableList()
                        thresh_lists = []
                        for i_batch in range(data.shape[0]):
                            # 坐标&置信度
                            box_iter, thresh_list = decode_bbox(logits, thresh=-2, idx=i_batch, config=self.args)
                            box_batch.append(box_iter)
                            thresh_lists.append(thresh_list)
                    

                        for i_idx, idx in enumerate(idxlist):
                            if idx == -1:
                                break
                            zhw, name, fullab = infos_list[i_idx]
                            fullab = torch.from_numpy(fullab).cuda()
                            zhw = torch.from_numpy(zhw)
                            bbox_pieces = box_batch[belong==idx]

                            if bbox_left is not None:
                                bbox_pieces = bbox_left + bbox_pieces
                                bbox_left = None

                            # 由小块还原到原图的坐标
                            # comb_pred: 所有预测结果框 [n, 8] [z1,y1,x1,z2,y2,x2,confidence, cls]
                            comb_pred = val_loader.dataset.split_comb.combine(bbox_pieces, zhw)
                            # print(comb_pred.shape)
                            # print(fullab.shape)
                            # print(comb_pred)
                            # print(comb_pred.shape)
                            # print(fullab)

                            # 统计个数
                            em_list = []
                            if self.emlist is not None:
                                for em_fun in self.emlist:
                                    # 计算所有预测框的hit情况
                                    # fulllab: 所有gt box [n, 10] [z,y,x,dz,dy,dx,cls,1,1,1]
                                    # iou == 0.2
                                    em_result, iou_info = em_fun(comb_pred, fullab)
                                    # print(em_result)
                                    # exit()

                                    if len(comb_pred) > 0:
                                        comb_pred_tmp = comb_pred.cpu().numpy()
                                        # 预测置信度
                                        pred_probs = comb_pred_tmp[:, 6:7]
                                        # z轴区间为box大小
                                        bbox_size = comb_pred_tmp[:, 3:4] - comb_pred_tmp[:, 0:1]
                                        # [prob, diameter, hit-iou, [coords]]
                                        ap_pred_list.append(np.concatenate([pred_probs, bbox_size, iou_info[:, :1], comb_pred_tmp[:, :6]], axis=1))
                                        # print(ap_pred_list[0][1,2])
                                        # exit()
                                    else:
                                        ap_pred_list.append([])
                                    if fullab.shape[0] > 0:
                                        fullab_tmp = fullab.cpu().numpy()
                                        lab_size = fullab_tmp[:, 3:4]
                                        lab_center = fullab_tmp[:, :3]
                                        ap_gt_list.append(np.concatenate([lab_size, lab_center], axis=1))
                                    else:
                                        ap_gt_list.append([])
                                    em_list.append(em_result)
                                em_avg.update(tuple(em_list))

                            info = 'end %d out of %d, name %s, '%(idx, len(val_loader), name)
                            for lid, l in enumerate(em_list):
                                if isinstance(l,dict):
                                    for k,v in l.items():
                                        info += '%s: %.2f, '%(k, v)
                                else:
                                    info += '%d: %.2f, '%(lid, l)
                            threshs = np.array(thresh_lists).mean(axis=0)
                            info += 'thresh: '
                            for level, thresh in enumerate(threshs):
                                info += 'level %d= %.02f, '%(level, thresh)
                            print(info)
                            if save:
                                if isinstance(comb_pred, torch.Tensor):
                                    comb_pred = comb_pred.cpu().numpy()
                                try:
                                    np.save(os.path.join(savedir, name+'.npy'), np.concatenate([comb_pred, iou_info], axis=1))
                                except:
                                    print(name)
                        bbox_left_new = box_batch[(belong)==belong[-1]]
                        bbox_left, infos_list = restart_logit(x_left, bbox_left, bbox_left_new, infos_list)
                        xbatch_filler.restart()

        cPickle.dump(ap_pred_list, open(os.path.join(self.ioer.save_dir, 'ap_pred_list.pkl'), 'wb'))
        cPickle.dump(ap_gt_list, open(os.path.join(self.ioer.save_dir, 'ap_gt_list.pkl'), 'wb'))
        ap_small_bbox_list = []
        ap_big_bbox_list = []
        ap_small_gt_bbox_count = 0
        ap_big_gt_bbox_count = 0
        
        # 二者数量相同，CT个数
        assert len(ap_pred_list) == len(ap_gt_list)
        # 所有ct
        for idx, ap_pred in enumerate(ap_pred_list):
            ap_gt = ap_gt_list[idx]
            
            # 所有box
            for i, ap_pred_x in enumerate(ap_pred):
                # iou_info >= 0
                if ap_pred_x[2] >= 0:
                    bbox_size = ap_gt[int(ap_pred_x[2])][0]
                else:
                    bbox_size = ap_pred_x[1]

                # 小目标, 
                if bbox_size < self.small_size:
                    # [prob, id(ct)_id(lgt), size]
                    ap_small_bbox_list.append([ap_pred_x[0], str(idx) + '_' + str(int(ap_pred_x[2])), bbox_size])
                else:
                    ap_big_bbox_list.append([ap_pred_x[0], str(idx) + '_' + str(int(ap_pred_x[2])), bbox_size])
            
            for i, ap_gt_x in enumerate(ap_gt):
                bbox_size = ap_gt_x[0]
                if bbox_size < self.small_size:
                    ap_small_gt_bbox_count += 1
                else:
                    ap_big_gt_bbox_count += 1

        # import pdb; pdb.set_trace()
        # cal froc
        froc_val = self.froc(bbox_info=ap_big_bbox_list, 
                             gt_count=ap_big_gt_bbox_count, 
                             fps=[0.5, 1, 2, 4, 8], 
                             n_ct=len(val_loader))
        # print('FROC: {}'.format(froc_val))
        self.printf('FROC: ' + str(froc_val))

        rp_list = []
        # do with small & big box
        # for ap_bbox_list, ap_gt_bbox_count in zip([ap_small_bbox_list, ap_big_bbox_list], \
        #                                           [ap_small_gt_bbox_count, ap_big_gt_bbox_count]):
        ap_bbox_list = ap_big_bbox_list
        ap_gt_bbox_count = ap_big_gt_bbox_count
        recall_level = 1
        rp = {}

        # 按照prob排序
        # ap_bbox_list: [prob, id_cls, size]
        ap_bbox_list.sort(key=lambda x: -x[0])
        gt_bbox_hits = []
        pred_bbox_hit_count = 0
        for idx, ap_bbox in enumerate(ap_bbox_list):
            bbox_tag = ap_bbox[1]
            if not bbox_tag.endswith('-1'):
                pred_bbox_hit_count += 1
                # 如果有-1的框，会多记入一个
                if bbox_tag not in gt_bbox_hits:
                    gt_bbox_hits.append(bbox_tag)
            while len(gt_bbox_hits) / ap_gt_bbox_count >= recall_level*0.1 and recall_level <= 10:
                rp[recall_level] = [pred_bbox_hit_count / (idx + 1), ap_bbox[0]]
                recall_level += 1
        rp_list.append(rp)

        if self.emlist is not None:
            em_list = em_avg.val()
        endt = time.time()
        self.writeLossLog('Val', epoch, meanloss = 0, loss_list = [], em_list=em_list, time=(endt-startt)/60)
        # self.printf('small: ' + str(rp_list[0]))
        # self.printf('big: ' + str(rp_list[1]))
        self.printf('big: ' + str(rp_list))

    
    def froc(self, bbox_info, gt_count, fps, n_ct):
        """
        计算froc指标的值
        :param bbox_info: 预测框的信息 [prob, ctid_boxid, size] id: 
        :param fps: fp/n_ct的不同取值
        :param gt_count: gt box总数
        :param n_ct: ct的数量

        """
        
        # box数量
        # if isinstance(bbox_info, list):
        #     n_boxes = len(bbox_info)
        # elif isinstance(bbox_info, np.ndarray):
        #     n_boxes = bbox_info.shape[0]

        # 置信度排序
        bbox_info.sort(key=lambda x: -x[0])

        # 命中gt的框的数量
        pred_bbox_hit_count = 0
        
        # fp框的数量
        pred_bbox_fp_count = 0

        # 被命中的框的id
        gt_bbox_hits = {}

        # recall-precision
        rps = []
        fp_level = 0

        for idx, ap_bbox in enumerate(bbox_info):
            bbox_tag = ap_bbox[1]
            # fp + 1
            if bbox_tag.endswith('-1'):
                pred_bbox_fp_count += 1
            # hit + 1
            else:
                pred_bbox_hit_count += 1
                if bbox_tag not in gt_bbox_hits:
                    gt_bbox_hits[bbox_tag] = True

            # 大于指定fp
            while fp_level < len(fps) and pred_bbox_fp_count / n_ct >= fps[fp_level]:
                rps.append(len(gt_bbox_hits) / gt_count)
                fp_level += 1
        
        # print(rps)
        return sum(rps) / len(rps)


        
class Tester(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.args = args
        self.net = net
        self.half = args.half
        if self.half:
            self.dtype = torch.HalfTensor
        else:
            self.dtype = torch.FloatTensor

        self.save_dir = os.path.join(args.output['result_dir'], args.output['save_dir'])

        testdir = args.output['test_dir']
        if testdir is None:
            self.testdir = os.path.join(self.save_dir, 'testout')
        else:
            self.testdir = os.path.join(self.save_dir, testdir)
        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

    def test(self, test_loader):
        self.net.eval()
        bs =  self.args.train['batch_size']

        with torch.no_grad():
            batch_filler = Batch_Filler(bs, size=[self.args.prepare['channel_input']] + self.args.prepare['crop_size'], dtype=torch.float)
            logits_left = None
            infos_list = []

            for sample_idx, tmp in enumerate(test_loader):
                data, zhw, name = tmp[0]
                infos_list.append([zhw, name])
                x_left = torch.from_numpy(data).cuda()
                while x_left is not None:

                    isFull, batch, belong, idxlist, x_left = batch_filler.fill(sample_idx, x_left)

                    if len(test_loader) == sample_idx + 1: #the last sample force execute test operation.

                        isFull = True

                        if len(idxlist) != bs:

                            fill_length = bs - len(idxlist)

                            for i in range(fill_length):
                                idxlist.append(-1)

                    if isFull:

                        data = batch
                        logits = self.net(data)
                        for i_idx, idx in enumerate(idxlist):

                            if idx == -1:
                                break

                            zhw, name = infos_list[i_idx]
                            zhw = torch.from_numpy(zhw)

                            pred_pieces = logits[torch.from_numpy(belong)==idx]

                            if logits_left is not None:
                                pred_pieces = torch.cat([logits_left, pred_pieces])
                                logits_left = None


                            # pred_for_em = binarize(pred_pieces)
                            # comb_pred = test_loader.dataset.split_comb.combine(pred_for_em, zhw)[0]
                            #modify by lxw
                            comb_pred = test_loader.dataset.split_comb.combine(pred_pieces, zhw)
                            #comb_pred = arg_index_get(comb_pred)

                            savepath = os.path.join(self.testdir, name+'.npy')
                            print(savepath)
                            np.save(savepath, comb_pred.detach().cpu().numpy())

                            # save_npz_path = os.path.join(self.testdir, name+'.npz')
                            # print(save_npz_path)
                            # np.savez(save_npz_path, comb_pred.detach().cpu().numpy())

                        logits_left_new = logits[torch.from_numpy(belong)==belong[-1]]
                        logits_left, infos_list = restart_logit(x_left, logits_left, logits_left_new, infos_list)
                        batch_filler.restart()

    # def test(self, test_loader):
    #
    #     self.net.eval()
    #
    #     with torch.no_grad():
    #         for sample_idx, tmp in enumerate(test_loader):
    #             data, zhw, name = tmp[0]
    #
    #             data = torch.from_numpy(data[:, 0:1]).type(self.dtype).cuda()
    #
    #             logits = self.net(data, deploy=1)
    #
    #             zhw = torch.from_numpy(zhw)
    #
    #             comb_pred = test_loader.dataset.split_comb.combine(logits, zhw)
    #
    #             comb_pred = arg_index_get(comb_pred)
    #
    #             savepath = os.path.join(self.testdir, name + '.npy')
    #
    #             print(savepath)
    #
    #             np.save(savepath, comb_pred.detach().cpu().numpy())

