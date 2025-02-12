# import unittest

from unittest.mock import patch

import keras
import numpy as np
import numpy.testing as npt

import tensorflow as tf
from tensorflow import test as tft

from IQT import layers
from scipy.ndimage import zoom


class AugmentationLayerCase(tft.TestCase):
    def testInit(self):
        with self.cached_session():
            augmentation_layer = layers.AugmentationLayer()

            self.assertEqual(0., augmentation_layer.contrast_std)
            self.assertEqual(0., augmentation_layer.brightness_std)
            self.assertEqual(0.1, augmentation_layer.max_noise_std)
            self.assertEqual(0.1, augmentation_layer.gamma_std)

            self.assertRaises(AssertionError, layers.AugmentationLayer, contrast_std=-0.1)
            self.assertRaises(AssertionError, layers.AugmentationLayer, brightness_std=-0.1)
            self.assertRaises(AssertionError, layers.AugmentationLayer, max_noise_std=-0.1)
            self.assertRaises(AssertionError, layers.AugmentationLayer, gamma_std=-0.1)

    def testContrast(self):

        with patch('tensorflow.random.normal') as mock_random:

            mock_random.return_value = tf.constant(0.1, dtype=tf.float32)

            augmentation_layer = layers.AugmentationLayer(contrast_std=0.1,
                                                          brightness_std=0.,
                                                          max_noise_std=0.,
                                                          gamma_std=0.)

            aug_in = tf.ones((1, 10, 10, 10, 1), dtype=tf.float32)

            aug_out = augmentation_layer(aug_in)

            self.assertAllClose(aug_in, aug_out)

            mock_random.return_value = tf.constant(0.1, dtype=tf.float64)  # Test with different floating precisions

            augmentation_layer = layers.AugmentationLayer(contrast_std=0.1,
                                                          brightness_std=0.,
                                                          max_noise_std=0.,
                                                          gamma_std=0.,
                                                          dtype=tf.float64)

            aug_in = tf.ones((1, 10, 10, 10, 1), dtype=tf.float64)

            aug_out = augmentation_layer(aug_in)

            self.assertAllClose(aug_in, aug_out)

    # def testNoise(self):
    #     with patch('tensorflow.random.normal') as mock_random, patch('tf.random.normal') as mock_noise:
    #
    #         mock_shape = (1, 10, 10, 10, 1)
    #
    #         mock_random.return_value = tf.constant(0.1, dtype=tf.float32)
    #         mock_noise.return_value = tf.random.normal(mock_shape, stddev=)
    #
    #         aug_in = tf.ones(mock_shape, dtype=tf.float32)


class SamplingLayerCase(tft.TestCase):

    def testGetterSetter(self):

        sampling_layer = layers.SamplingLayer(dsamp_rate=[2], method='nearest', dtype=tf.float32)

        self.assertItemsEqual(np.array(sampling_layer.dsamp_rate), np.array([2., 2., 2.]))

        sampling_layer.dsamp_rate = np.array([3])

        self.assertItemsEqual(np.array(sampling_layer.dsamp_rate), np.array([3., 3., 3.]))

        sampling_layer.dsamp_rate = np.array([3, 1, 5])

        self.assertItemsEqual(np.array(sampling_layer.dsamp_rate), np.array([3., 1., 5.]))

    def testNearestDownsampling(self):

        sampling_layer = layers.SamplingLayer(dsamp_rate=[2], method='nearest', dtype=tf.float64)

        pre_interp = tf.cast(tf.stack(tf.meshgrid(tf.linspace(0, 19, 20),
                                                  tf.linspace(0, 19, 20),
                                                  tf.linspace(0, 19, 20),
                                                  indexing='ij'), axis=-1), dtype=tf.float64)

        post_interp = sampling_layer(pre_interp[None, ...])[0]

        target_interp = tf.convert_to_tensor(
            zoom(pre_interp, zoom=(0.5, 0.5, 0.5, 1.), order=0, prefilter=False), dtype=tf.float64)

        self.assertTrue(tf.reduce_sum(tf.abs(target_interp - post_interp)) <= 1e-5)

    def testTriLinearDownsampling(self):

        sampling_layer = layers.SamplingLayer(dsamp_rate=[2], method='trilinear', dtype=tf.float64)

        pre_interp = tf.cast(tf.stack(tf.meshgrid(tf.linspace(0, 19, 20),
                                                  tf.linspace(0, 19, 20),
                                                  tf.linspace(0, 19, 20),
                                                  indexing='ij'), axis=-1), dtype=tf.float64)

        post_interp = sampling_layer(pre_interp[None, ...])[0]

        target_interp = tf.convert_to_tensor(
            zoom(pre_interp, zoom=(0.5, 0.5, 0.5, 1.), order=1, prefilter=False), dtype=tf.float64)

        self.assertTrue(tf.reduce_sum(tf.abs(target_interp - post_interp)) <= 1e-5)


class DTIFitLayerCase(tft.TestCase):

    def setUp(self):

        from skimage.io import imread

        img = np.mean(np.array(imread("../data/testing/phantom.png"))[..., :3], axis=-1, dtype=int)

        # Value 2 is a mix of 1 and 5
        # Value 4 is a mix of 1 and 7
        # Value 6 is a mix of 5 and 7
        for i, v in enumerate(np.unique(img)):
            img[img == v] = i

        # img = np.repeat(img, repeats=6, axis=-1).astype(float)

        img_dti = np.zeros(img.shape + (3, 3))

        for i in np.unique(img)[1:]:

            if i in [2, 4, 6]:
                continue

            eigenvalues = np.sort(np.concatenate((np.random.uniform(1.5e-3, 2.5e-3, 1),
                                                  np.random.uniform(2.e-4, 8.e-4, 2))))[::-1]

            # Random rotation matrix to orient the tensor
            random_matrix = np.random.randn(3, 3)
            Q, _ = np.linalg.qr(random_matrix)  # QR decomposition for a random orthogonal matrix

            # Construct the DTI matrix: D = Q @ diag(eigenvalues) @ Q.T
            dti_matrix = Q @ np.diag(eigenvalues) @ Q.T

            img_dti[img == i, :] = dti_matrix

        # Value 2 is a mix of 1 and 5
        # Value 4 is a mix of 1 and 7
        # Value 6 is a mix of 5 and 7

        img_dti[img == 2] = np.exp((np.log(img_dti[img == 1][0]) + np.log(img_dti[img == 5][0])) / 2)
        img_dti[img == 4] = np.exp((np.log(img_dti[img == 1][0]) + np.log(img_dti[img == 7][0])) / 2)
        img_dti[img == 6] = np.exp((np.log(img_dti[img == 5][0]) + np.log(img_dti[img == 7][0])) / 2)

        self.dti_img = img_dti


class SamplerLayerCase(tft.TestCase):

    def setUp(self):

        self.sampler_layer = layers.SamplerLayer(max_hr_downsamp=1.5, max_lr_downsamp=2.0,
                                                 apply_blurring=False,
                                                 augment=False)

    def testBuild(self):

        test_layer = layers.SamplerLayer(max_hr_downsamp=1.5, max_lr_downsamp=2.0, apply_blurring=False)

        d_input_layer = keras.Input((24, 24, 24, 3))

        input_layers = [d_input_layer, d_input_layer, 2.0, 1.5]

        output_layers = test_layer(input_layers)

        self.assertEqual([output_layers[0].shape,
                          output_layers[1].shape], [(None, 16, 16, 16, 3),
                                                    (None, 16, 16, 16, 3)])

        input_layers[2] = 1.6
        input_layers[3] = 1.2  # Show that change in upsamp rate doesn't change output shape

        output_layers = test_layer(input_layers)

        self.assertEqual([output_layers[0].shape,
                          output_layers[1].shape], [(None, 16, 16, 16, 3),
                                                    (None, 16, 16, 16, 3)])

        test_layer = layers.SamplerLayer(max_hr_downsamp=1.5, max_lr_downsamp=2.0, apply_blurring=False)

        input_layers.insert(2, keras.Input((24, 24, 24, 1)))

        output_layers = test_layer(input_layers)

        self.assertEqual([output_layers[0].shape,
                          output_layers[1].shape,
                          output_layers[2].shape], [(None, 16, 16, 16, 3),
                                                    (None, 16, 16, 16, 3),
                                                    (None, 16, 16, 16, 1)])

        test_layer = layers.SamplerLayer(max_hr_downsamp=1.5, max_lr_downsamp=2.0, apply_blurring=False)

        input_layers.insert(3, keras.Input((24, 24, 24, 1)))

        output_layers = test_layer(input_layers)

        self.assertEqual([output_layers[0].shape,
                          output_layers[1].shape,
                          output_layers[2].shape,
                          output_layers[3].shape], [(None, 16, 16, 16, 3),
                                                    (None, 16, 16, 16, 3),
                                                    (None, 16, 16, 16, 1),
                                                    (None, 16, 16, 16, 1)])

    def testOutputShape(self):

        output_shapes = self.sampler_layer.compute_output_shape([(None, 24, 24, 24, 3), 2.0, 1.5])

        self.assertEqual([(None, 16, 16, 16, 3), (None, 16, 16, 16, 3)], output_shapes)

        output_shapes = self.sampler_layer.compute_output_shape([(None, 24, 24, 24, 3),
                                                                 (None, 24, 24, 24, 1),
                                                                 2.0,
                                                                 1.5])

        self.assertEqual([(None, 16, 16, 16, 3), (None, 16, 16, 16, 3), (None, 16, 16, 16, 1)],
                         output_shapes)

        output_shapes = self.sampler_layer.compute_output_shape([(None, 24, 24, 24, 3),
                                                                 (None, 24, 24, 24, 1),
                                                                 (None, 24, 24, 24, 1),
                                                                 2.0,
                                                                 1.5])

        self.assertEqual([(None, 16, 16, 16, 3), (None, 16, 16, 16, 3), (None, 16, 16, 16, 1), (None, 16, 16, 16, 1)],
                         output_shapes)

    def testGridOutput(self):

        from IQT.util import config_grid

        grid_pre_calc = config_grid((0, 0, 0), (23, 23, 23), (24, 24, 24))[None, :]

        grid_target = config_grid((0, 0, 0), (15, 15, 15), (16, 16, 16)) * 23/15

        (grid_hr, grid_lr) = self.sampler_layer([grid_pre_calc, grid_pre_calc, 2.0, 1.5])

        self.assertTrue(np.mean(np.abs(grid_target - grid_hr)) < 1e-5)
        # Linear function mapping means lr and hr should be equal
        self.assertTrue(np.mean(np.abs(grid_target - grid_lr)) < 1e-5)


if __name__ == '__main__':
    # unittest.main()
    tft.main()
