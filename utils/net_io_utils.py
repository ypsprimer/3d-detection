#
#  net_io_utils.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#
import os
import time

import torch

from collections import OrderedDict
from torch.nn import DataParallel
import shutil


def my_load(net, state_dict, strict):
    keys = state_dict.keys()
    isparallel = all(['module' in k for k in keys])

    if isinstance(net, DataParallel):
        if isparallel:
            net.load_state_dict(state_dict, strict)
        else:
            net.module.load_state_dict(state_dict, strict)
    else:
        if isparallel:
            new_state_dict = OrderedDict()
            for k, v in model.items():
                name = k[7:]
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict,strict)
        else:
            net.load_state_dict(state_dict,strict)
    return net


class netIOer():
    def __init__(self, config):
        self.config = config
        self.save_dir = os.path.join(config.output['result_dir'], config.output['save_dir'])
        self.save_freq = config.output["save_frequency"]
        self.strict = config.net['strict']
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.save_freq == 0:
            self.best_score = 10000  # generally, loss function, the lower the better

    def load_file(self, net, net_weight_file): # address the issue of DataParallel
        contents = torch.load(net_weight_file,map_location='cpu')  # ('/home/data/dl_processor/net_params.pkl')
        state_dict = contents['state_dict']
        net = my_load(net, state_dict, self.strict)
        if self.config.net['resume'] and self.config.train['start_epoch'] == 1:
            self.config.train['start_epoch'] = contents['epoch'] + 1

        if self.save_freq == 0:
            if os.path.exists(os.path.join(self.save_dir, 'best.pkl')):
                self.best_score = torch.load(os.path.join(self.save_dir, 'best.pkl'))['loss']
            shutil(net_weight_file, os.path.join(self.save_dir,'starter.pkl'))
        return net, self.config

    def save_file(self, net, epoch, config, loss, isbreak=False):

        if isinstance(net, DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        dicts = {'state_dict': state_dict,
                 'epoch': epoch,
                 'config': config,
                 'loss': loss}
        if isbreak:
            save_file = os.path.join(self.save_dir, 'break.pkl')
            print('Manual interrupted, save to %s' %save_file)
            torch.save(dicts, save_file)
        if self.save_freq == 0:
            torch.save(dicts, os.path.join(self.save_dir, 'last.pkl'))
            if loss < self.best_score:
                shutil.copy(os.path.join(self.save_dir, 'last.pkl'), os.path.join(self.save_dir, 'best.pkl'))
                self.best_score = loss
                print('Replace old best.pkl, new best loss: %.4f' % loss)
        elif epoch % self.save_freq ==0 or epoch == self.config.train['epoch']:
            torch.save(dicts, os.path.join(self.save_dir, '%03d.pkl' % epoch))


    def trace(self, model):
        config = self.config
        if config.net["load_weight"] == '':
            raise ValueError('you must load weight before jit')
        else:
            shape = config.prepare['crop_size']
            channel = config.prepare['channel_input']
            sample_data = torch.rand(1,channel, shape[0],shape[1],shape[2]).cuda()
            model = model.eval()
            if config.half:
                sample_data = sample_data.half()
            with torch.no_grad():
                trace = torch.jit.trace(model, sample_data)
                weight_file = config.net["load_weight"]
                trace_file = weight_file.replace('.pkl','.trace')
                torch.jit.save(trace, trace_file)
                print('save model to ', trace_file)



