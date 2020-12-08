import torch
import numpy as np
from models import ModelLoader
from scipy.ndimage import zoom

BUFFER = np.array([16, 16, 16])

class BoundryFinder(object):
    def __init__(self):
        pass

    @staticmethod
    def _resize_npy(input_npy_file):
        data = np.load(input_npy_file)
        ori_size = data.shape
        scale_csize = np.array([64, 128, 128])
        # scale = scale_csize / data.shape
        scale = np.true_divide(scale_csize, data.shape)
        npydata = zoom(data, scale, order=1)
        return npydata, ori_size

    @staticmethod
    def find_by_regression(modelfile, input_npy_file):
        print('running')
        resized_npy, ori_size = BoundryFinder._resize_npy(input_npy_file)

        net_t = ModelLoader.load("FindBoundaryRegressioNet64128128")
        net_t = torch.nn.DataParallel(net_t)
        net_t = net_t.eval()
        state = torch.load(modelfile)
        # state = torch.load('/home/modelout/save_xyt/20180313_21/FindBoundaryRegression-Test/model/400_net_params.pkl')

        # data_sign = 'clean64128128.npy'
        # data_raw_sign = 'clean.npy'

        net_t.load_state_dict(state)
        net_t = torch.nn.DataParallel(net_t).cuda()

        img = resized_npy
        imgc = img.copy()
        imgc[imgc > 500] = 500
        imgc[imgc < -300] = -300
        imgctmp = np.reshape(imgc, (-1, imgc.shape[0]))
        imgc_mean = np.mean(imgctmp)
        imgc_std = np.std(imgctmp)
        newimg = (imgc - imgc_mean) / imgc_std

        newimg = newimg.astype('float32')
        newimgth = torch.autograd.Variable(torch.from_numpy(newimg[np.newaxis, np.newaxis]).cuda(), volatile=True)

        output = net_t(newimgth)
        outputc = output.view(-1)
        outputc_np = outputc.data.cpu().numpy()
        pt1 = outputc_np[0:3] * ori_size
        pt2 = outputc_np[3:6] * ori_size

        pt1 = pt1 - BUFFER
        pt1[pt1 < 0] = 0
        pt2 = pt2 + BUFFER
        index = 0
        for i in pt2:
            if i > ori_size[index]:
                pt2[index] = ori_size[index]
            index += 1

        pt1 = pt1.astype(int)
        pt2 = pt2.astype(int)
        print ('DONE!')

        return pt1, pt2


if __name__ == "__main__":
    BoundryFinder.find_by_regression("/tmp/regression_model.pkl", "/tmp/case.npy")