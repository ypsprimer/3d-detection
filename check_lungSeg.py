import numpy as np
from imshow3d import ImShow3D
import SimpleITK as sitk

def load_seg(pid):
    img_dir = '/yupeng/micca20/ribfrac-train-images/RibFrac{}-image.nii.gz'.format(pid)
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)

    lungseg_dir = '/ssd/ribfrac-val-images_lobe/RibFrac{}-seg.npy'.format(pid)
    lungseg_arr = np.load(lungseg_dir)

    fig = ImShow3D(img_arr)
    fig.add_overlay(lungseg_arr, color='Blues', name='lung')


load_seg('421')