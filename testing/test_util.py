import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from numpy import testing as npt
from skimage.metrics import structural_similarity as ssim
from skimage.data import shepp_logan_phantom

from IQT import util


# class NiftiLoaderCase(npt.TestCase):
#     def testStructural(self):
#
#         image, header = util.load_structural("../data/100307/T1w", "T1w_acpc_dc_restore_brain")
#         self.assertEqual(image.shape[:-1], header.get_data_shape())
#
#         image, header = util.load_structural("../data/221319/T1w", "T1w_acpc_dc_restore_brain")
#         self.assertEqual(image.shape[:-1], header.get_data_shape())
#
#         self.assertRaises(FileNotFoundError, util.load_structural, "../data/221319/T1w", "nonExistentT1")


# class NiftiWriterCase(npt.TestCase):
#
#     def setUp(self):
#
#         self.sample_size = util.load_structural("../data/100307/HR", "dt_b1000_2")[0].shape
#
#     def testSaveDTI(self):
#
#         tensors = np.random.randn(self.sample_size[0], self.sample_size[1], self.sample_size[2], 6)
#
#         mask = np.ones(self.sample_size, dtype=bool) * np.random.randint(0, 1, dtype=bool)
#
#         with patch('nibabel.save') as mock_writer:
#
#             self.assertRaises(FileNotFoundError, util.save_dtis, tensors, "./test_out", "test_header")
#
#             mock_writer.assert_called_with()


class InterpolationCase(npt.TestCase):

    # def setUp(self):
    #
    #     nifti_data: nib.Nifti1Image = nib.nifti1.load("../data/100307/T1w/T1w_acpc_dc_restore_brain.nii.gz")
    #
    #     self.nifti_shape = nifti_data.shape
    #     self.nifti_orig = np.array(nifti_data.dataobj)[..., None]

    # def testGridGen(self):
    #
    #     sel_patch = self.nifti_orig[100:120, 100:120, 100:120]
    #
    #     grid = util.config_grid((100, 100, 100),
    #                             (120, 120, 120),
    #                             (20, 20, 20)).astype(int)
    #
    #     self.assertTrue(np.array_equal(grid, np.mgrid[100:120, 100:120, 100:120].transpose((1, 2, 3, 0))))
    #
    #     grid = grid.reshape((20 ** 3, 3))
    #
    #     selected_voxels = np.reshape(self.nifti_orig[grid[:, 0], grid[:, 1], grid[:, 2]], (20, 20, 20, 1))
    #
    #     self.assertTrue(np.array_equal(sel_patch, selected_voxels), "The two arrays must be equal!")

    def testInterpOnGrid(self):

        img = util.config_grid((40, 50, 80), (60, 70, 100), (20, 20, 20))

        img_orig = util.config_grid((40*2.5, 50*2.5, 80*2.5), (40*2.5 + 50, 50*2.5 + 50, 80*2.5 + 50), (50, 50, 50))

        grid = util.config_grid((0, 0, 0), (20, 20, 20), (20, 20, 20)) * 2.5

        npt.assert_array_equal(img * 2.5, util.gridded_interpolation(img_orig, grid))

    # def testInterp(self):
    #
    #     interped_nii = nib.nifti1.load("../data/100307/T1w/T1w_rescaled_1_25.nii.gz")
    #
    #     interped_shape = interped_nii.shape
    #     interped_img = np.array(interped_nii.dataobj)
    #
    #     grid = util.config_grid((0, 0, 0),
    #                             (interped_nii.shape[0], interped_nii.shape[1], interped_nii.shape[2]),
    #                             (interped_nii.shape[0], interped_nii.shape[1], interped_nii.shape[2]))
    #
    #     grid *= np.array(self.nifti_shape) / np.array(interped_shape)
    #
    #     output_img = util.gridded_interpolation(self.nifti_orig, grid)[..., 0]
    #
    #     self.assertTrue(ssim(interped_img, output_img, data_range=np.max(interped_img)) > 0.99,
    #                     "Images are not similar enough!")


if __name__ == '__main__':
    unittest.main()
