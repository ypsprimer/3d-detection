#
#  getnpy.py
#  testing
#
#  Created by AthenaX on 09/Mar/2018.
#  Copyright 2018 Shukun. All rights reserved.
#

import os
import numpy as np

try:
    import dicom as dcm
except:
    import pydicom as dcm

ORDER_POSITION = 0
ORDER_INSTANCE_NO = 1


class DicomLoader(object):
    def __init__(self, path=None):
        self.slices = None
        self.spacing = None
        if path:
            self.dcm_path = path
            self.slices = self.load_from_folder(self.dcm_path)
            self.spacing = np.array(
                [self.slices[0].SliceThickness] + [self.slices[0].PixelSpacing[0], self.slices[0].PixelSpacing[1]],
                dtype=np.float32)

    def load_from_folder(self, path, order=ORDER_POSITION):
        slices = [dcm.read_file(path + '/' + s, force=True) for s in os.listdir(path) if s.endswith('.dcm')]

        if order == ORDER_POSITION:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)
        elif order == ORDER_INSTANCE_NO:
            slices.sort(key=lambda x: float(x.InstanceNumber))

        return slices

    def get_npy(self):
        if self.slices:
            im = self._get_pixels_hu(self.slices)
            im[np.isnan(im)] = -2000
            im_npy = im.astype('int16')
            return im_npy
        return None

    def save_npy(self, target_file="/tmp/test.npy"):
        if self.slices:
            im_npy = self.get_npy()
            np.save(target_file, im_npy)
            return target_file
        else:
            raise "no dicom loaded"

    def save_spacing(self, target_file="/tmp/test.txt"):
        fp = open(target_file, 'w')
        fp.write(str(self.spacing[0]) + '\n' + str(self.spacing[1]) + '\n' + str(self.spacing[2]) + '\n')
        fp.close()

    def _get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        im = np.array(image, dtype=np.int16)
        return im


if __name__ == '__main__':
    # # load_scan('/Users/zchao/Desktop/JF93075125')
    # # load_scan('/Users/zchao/Desktop/cases/P00428333T20171219/1.2.840.113619.2.416.107222755824736510425617437821827484660/1.2.840.113619.6.80.114374076013742.27144.1513667669687.1')
    # dcm = DicomLoader('/Users/zchao/Desktop/JF93075125')
    # print dcm.save_npy('/tmp/test.npy')
    import sys
    from_folder = sys.argv[1]
    to_file = sys.argv[2]
    dcm = DicomLoader(from_folder)
    dcm.save_npy(to_file)
