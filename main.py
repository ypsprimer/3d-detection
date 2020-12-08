import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
from torch import optim
import torch
from torch.nn import DataParallel
from torch.backends import cudnn

from configs import Config
from trainer import Trainer, Tester
from models import ModelLoader
from losses import LossLoader, warpLoss
from torch.utils.data import DataLoader
from utils import constants, env_utils, myPrint, netIOer
from models.inplace_abn import ABN

import argparse
import warnings
import sys
import traceback
import datetime
import numpy as np
import json
from pprint import pprint

# from imshow3d import ImShow3D

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='ShuKun 3D segmentation model')
parser.add_argument('--model', '-m', metavar='MODEL', default=None,
                    help='model file to be used (default: sample)')
parser.add_argument('--config', '-c', metavar='CONFIG',
                    help='configs')
parser.add_argument('-j', '--cpu_num', default=None, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epoch', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=None, type=int,
                    metavar='N', help='mini-batch size ')
parser.add_argument('--gpu', type=str, default='0',
                    metavar='N', help='the gpu used for training, \
                    separated by comma and no space left(default: 0)')
parser.add_argument('--lr', default=None, type=float,
                    metavar='LR', help='Learning rate, if specified, \
                    the default lr shall be replaced with this one')
parser.add_argument('--loss', default=None, type=str,
                    help='the loss function used')
parser.add_argument('--load-weight', default=None, type=str, metavar='PATH',
                    help='path to loaded checkpoint, start from 0 epoch (default: none)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to loaded checkpoint, start from that epoch (default: none)')
parser.add_argument('--save-dir', default=None, type=str, metavar='SAVE',
                    help='directory to store folders (default: none)')
parser.add_argument('--test-dir', default=None, type=str, metavar='SAVE',
                    help='directory to save test results')
parser.add_argument('--save-frequency', default=None, type=int, metavar='SAVE',
                    help='frequency of store snapshots (default: none)')
parser.add_argument('--folder-label', default=None, type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--optimizer', default=None, type=str, metavar='O',
                    help='optimizer used (default: sgd)')
parser.add_argument('--cudnn', default=None,  type=lambda x: (str(x).lower() in ("yes", "true", "t", "1")),
                    help='cudnn benchmark mode')
parser.add_argument('--debug', action = 'store_true',
                    help='debug mode')
parser.add_argument('--freeze', action = 'store_true',
                    help='freeze bn')
parser.add_argument('--test', action = 'store_true',
                    help='test mode')
parser.add_argument('--val', action = 'store_true',
                    help='val mode')
parser.add_argument('--valtrain', action = 'store_true',
                    help='validate all training set')
parser.add_argument('--clip', action = 'store_true',
                    help='clip grad')
parser.add_argument('--jit', action = 'store_true',
                    help='convert weight to jit model')


def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, ABN):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def prepare(config):
    if 'hardmining' in config.prepare and config.prepare['hardmining']:
        from datasets import myDataset2 as myDataset
    else:
        from datasets import myDataset
    from datasets import myDataset as myDataset_val
    env_utils.setEnvironment(config)
    if config.debug:
        model = ModelLoader.load(config.net["model"], config=config,abn=1)
    elif config.jit:
        model = ModelLoader.load(config.net["model"], config=config,abn=0)
    else:
        model = ModelLoader.load(config.net["model"], config=config)

    loss = LossLoader.load(config.net["rpnloss"], config)
    em_names = config.net['em']
    em_list = []
    for ems in em_names:
        em_list.append(LossLoader.load(ems, config))
    netio = netIOer(config)

    if config.half:
        model = model.half()
        model = BN_convert_float(model)

    if config.net["load_weight"] != '':
        model, config = netio.load_file(model, config.net["load_weight"])
    # optimizer = optim.SGD(model.parameters(), lr= config.train['lr_arg'], momentum=0.9,
    #                       weight_decay=config.train['weight_decay'])

    model = model.cuda()
    if config.jit:
        netio.trace(model)
        sys.exit()

    loss = loss.cuda()
    warp = warpLoss(model, loss, config.prepare['margin'])
    if not config.debug:
        warp = DataParallel(warp)

    trainer = Trainer(warp, config, netio, emlist=em_list)

    train_data = myDataset(config, 'train')
    if config.valtrain:
        val_data = myDataset_val(config, 'valtrain')
    else:
        val_data = myDataset_val(config, 'val')

    print(config.augtype)
    print(config.env['cpu_num'])
    train_loader = DataLoader(train_data, batch_size=config.train['batch_size'],
                              shuffle=True, num_workers=config.env['cpu_num'], drop_last=True, pin_memory=True,
                              worker_init_fn=np.random.seed)

    val_loader = DataLoader(val_data, batch_size=1,
                              shuffle=False, num_workers=5, pin_memory=True, collate_fn=lambda x: x)
    return config, model, loss, warp, trainer, train_data, val_data, train_loader, val_loader


def run(config):
    from datasets import myDataset
    config, model, loss, warp, trainer, train_data, val_data, train_loader, val_loader = prepare(config)
    # print(model)

    # data, gt_prob_fpn, gt_coord_prob_fpn, gt_coord_diff_fpn, gt_diff_fpn, gt_connects_fpn, self.cases[idx] = train_data[0]
    # print(data.shape)
    # exit()

    if config.test:
        print('Start testing')
        #if hasattr(model, 'test'):
        #    model.forward = model.test
        model = DataParallel(model.cuda())

        tester = Tester(model, config)
        val_data = myDataset(config, 'test')
        test_loader = DataLoader(val_data, batch_size=1,
                shuffle=False, num_workers=3, pin_memory=True,  collate_fn=lambda x: x)
        tester.test(test_loader)
        return
    elif config.val:
        print('Start Val')
        start_epoch = config.train['start_epoch']
        trainer.validate(start_epoch, val_loader, save=True)
    else:
        start_epoch = config.train['start_epoch']
        epoch = config.train["epoch"]
        print('Start training from %d-th epoch'%start_epoch)

        epoch2loss = {}
        for i in range(start_epoch, epoch+1):
            try:
                # no hardming
                if 'hardmining' in config.prepare and config.prepare['hardmining']:
                    train_loader.dataset.resample3()
                    json.dump([str(item) for item in train_loader.dataset.samples], open(os.path.join(trainer.save_dir, 'sample_%d.json'%(i)), 'w'), indent=2)
                    json.dump({k: str(v) for k, v in train_loader.dataset.sample_weights.items()}, open(os.path.join(trainer.save_dir, 'sample_weights_%d.json'%(i)), 'w'), indent=2)
                    #json.dump({k: str(v) for k, v in train_loader.dataset.neg_sample_weights.items()}, open(os.path.join(trainer.save_dir, 'neg_sample_weights_%d.json'%(i)), 'w'), indent=2)
                loss_list = trainer.train(i, train_loader)
                epoch2loss[i] = list(loss_list)
                trainer.validate(i, val_loader)
            except KeyboardInterrupt as e:
                traceback.print_exc()
                trainer.ioer.save_file(trainer.net, i, trainer.args, 1e10, isbreak=True)
                sys.exit(0)
        
        print(epoch2loss)
        with open('./epoch_loss.json', 'w') as f:
            f.write(json.dumps(epoch2loss))

def syncKeys(config, args):
    keys = {'net': ['model', 'load_weight', 'loss', 'optimizer'],
            'train': ['start_epoch', 'epoch', 'batch_size', 'freeze', 'cudnn'],
            'output': ['save_frequency','test_dir'],
            'env':['cpu_num']}
    for prop in keys:
        tmp = getattr(config, prop)
        for k in keys[prop]:
            if getattr(args, k) is not None:
                tmp[k] = getattr(args, k)
            setattr(config,prop, tmp)
    if args.resume is not None:
        config.net['load_weight'] = args.resume
        config.net['resume'] = True

    if args.lr is not None: # override the yml setting, use constant lr, useful for hand tuning
        config.train['lr_arg'] = args.lr
        config.train['lr_func'] = 'constant'


    if args.debug:
        config.output['save_dir'] = 'tmp'
    else:
        if config.output['save_dir'] == 0:
            date = datetime.datetime.now()
            date = date.strftime('%Y%m%d')
            config.output['save_dir'] = '_'.join([date, config.net['model'], config.net['rpnloss'], \
                                                  'dropout' + str(config.net['dropout']), \
                                                  args.config.split('/')[-1].split('.yml')[0]])
            if 'region_w_norm' in config.train and config.train['region_w_norm']:
                config.output['save_dir'] += '_region-w-norm'
            if config.rpn['max_pos']:
                config.output['save_dir'] += '_max-pos'
    if args.save_dir is not None:
        config.output['save_dir'] = args.save_dir

    config.debug = args.debug
    config.test = args.test
    config.valtrain = args.valtrain
    config.val = args.val or args.valtrain
    config.clip = args.clip
    config.jit = args.jit
    return config

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    pprint(args)
    
    Config.init()
    Config.update(args.config)
    Config = syncKeys(Config, args)
    if Config.train['cudnn'] and (not Config.debug):
        torch.backends.cudnn.benchmark = True
        print('cudnn mode')
    print(Config.net)
    # pprint(vars(Config))
    # exit()

    run(Config)
    
