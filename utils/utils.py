import os
import csv
import sys
import time

import torch
import torch.nn as nn
import torch.nn.init as init

def read_csv(filename):
    with open(filename, 'r') as f:
        content = []
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            content.append(row)
    f.close()
    return content


def read_txt(filename):
    with open(filename) as f:
        content = f.readlines()
    f.close()
    return content


def write_text(data, filename):
    with open(filename, 'w') as f:
        f.writelines(data)
    f.close()


def delete_file(filename):
    if os.path.isfile(filename) is True:
        os.remove(filename)


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def saveargs(args):
    path = args.logs
    if os.path.isdir(path) is False:
        os.makedirs(path)
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg + ' ' + str(getattr(args, arg)) + '\n')


def file_exists(filename):
    return os.path.isfile(filename)


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


term_width = 300

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class myPrint():
    def __init__(self, file):
        self.fname = file
    def __call__(self, str):
        print(str)
        with open(self.fname, 'a') as f:
            f.write(str+'\n')


# 取平均值
class Averager():
    def __init__(self):
        self.n = 0
        self.total = 0
        self.type = 'scalar'

    def update(self, x):
        """
        计算x的每组值的和
        x -> tuple, dict, array, tensor
        """
        if isinstance(x, tuple):
            if self.n == 0:
                self.list = [Averager() for i in range(len(x))]

            assert len(x) == len(self.list)
            for i in range(len(x)):
                self.list[i].update(x[i])
            self.n += 1
            self.type = 'list'
            
        elif isinstance(x, dict):
            if self.n == 0:
                self.dicts = {}
                for key in x.keys():
                    self.dicts[key] = Averager()
                    self.dicts[key].update(x[key])
                self.type = 'dict'
            else:
                for key in x.keys():
                    self.dicts[key].update(x[key])
            self.n += 1
        elif isinstance(x, list):
            assert False, 'only tuple or array or tensor'

        elif hasattr(x, 'shape') and len(x.shape)==1:
            self.n += len(x)
            self.total += x.sum()
        else:
#             assert x.shape==()
            self.n += 1
            self.total += x

    def val(self):
        """
        计算得到平均值 total / n
        """
        if self.type == 'scalar':
            return  self.total/self.n
        elif self.type == 'list':
            return (tuple([l.val() for l in self.list]))
        elif self.type == 'dict':
            result = {}
            for key in self.dicts.keys():
                result[key] = self.dicts[key].val()
            return result


if __name__=='__main__':
    #mp = myPrint('./log')
    #mp('yes')
    import numpy as np

    a = Averager()
    print(a.val())
    a.update((np.array([1,2,3]), np.array([4,5,6,7])))
    print(a.val())
    a.update((np.array([11,22,32]), np.array([40,50,60])))
    print(a.val())
