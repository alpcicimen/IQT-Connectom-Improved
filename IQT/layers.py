import keras.backend as K
import keras.src.backend
import tensorflow as tf

from keras.layers import *
from keras import Sequential

import numpy as np
from tqdm import tqdm

from typing import List


class DepthToSpaceLayer(Layer):

    def __init__(self, upsampling_rate=2):
        super().__init__()
        self.upsampling_rate = upsampling_rate

    def call(self, inputs, *args, **kwargs):
        batch_size, dim_i, dim_j, dim_k, c = K.int_shape(inputs)

        assert (c % (self.upsampling_rate ** 3) == 0) and (c > 0)  # Number must be exactly divisible by 8

        if batch_size is None:
            batch_size = -1

        dim_i_r = dim_i * self.upsampling_rate
        dim_j_r = dim_j * self.upsampling_rate
        dim_k_r = dim_k * self.upsampling_rate

        oc = c // (self.upsampling_rate ** 3)

        out = K.reshape(inputs, (
            batch_size, dim_i, dim_j, dim_k, self.upsampling_rate, self.upsampling_rate, self.upsampling_rate, oc))
        out = K.permute_dimensions(out, (0, 1, 4, 2, 5, 3, 6, 7))
        out = K.reshape(out, (batch_size, dim_i_r, dim_j_r, dim_k_r, oc))
        return out


class SpaceToDepthLayer(Layer):

    def __init__(self, upsampling_rate=2):
        super().__init__()
        self.upsampling_rate = upsampling_rate

    def call(self, inputs, *args, **kwargs):
        batch_size, dim_i, dim_j, dim_k, c = K.int_shape(inputs)

        assert (dim_i % self.upsampling_rate == 0)  # Number must be exactly divisible by 8
        assert (dim_j % self.upsampling_rate == 0)  # Number must be exactly divisible by 8
        assert (dim_k % self.upsampling_rate == 0)  # Number must be exactly divisible by 8

        if batch_size is None:
            batch_size = -1

        dim_i_r = dim_i // self.upsampling_rate
        dim_j_r = dim_j // self.upsampling_rate
        dim_k_r = dim_k // self.upsampling_rate

        oc = c * (self.upsampling_rate ** 3)

        out = K.reshape(inputs, (batch_size, dim_i_r, self.upsampling_rate,
                                 dim_j_r, self.upsampling_rate,
                                 dim_k_r, self.upsampling_rate, c))
        out = K.permute_dimensions(out, (0, 1, 3, 5, 2, 4, 6, 7))
        out = K.reshape(out, (batch_size, dim_i_r, dim_j_r, dim_k_r, oc))
        return out


class AugmentationLayer(Layer):

    def __init__(self,
                 contrast_std=0.,
                 brightness_std=0.,
                 max_noise_std=0.1,
                 gamma_std=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.max_noise_std = max_noise_std
        self.gamma_std = gamma_std

    # @tf.function
    def __augment__(self, inputs):

        # contrast = tf.minimum(1.4, tf.maximum(0.6, tf.add(1.0, tf.random.normal((), stddev=self.contrast_std))))
        #
        # brightness = tf.minimum(0.4, tf.maximum(-0.4, tf.random.normal((), stddev=self.brightness_std)))
        #
        # noise_stddev = tf.random.uniform(tf.shape(inputs), maxval=self.max_noise_std)
        #
        # modified_output = tf.add(
        #     tf.add(tf.multiply(tf.subtract(inputs, 0.5), contrast), tf.add(0.5, brightness)),
        #     tf.random.normal(tf.shape(inputs), stddev=noise_stddev)
        # )

        modified_output = inputs

        if self.max_noise_std > 0:
            noise_stddev = tf.random.uniform(tf.shape(inputs), maxval=self.max_noise_std)
            noise = tf.multiply(tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True),
                                tf.random.normal(tf.shape(inputs), stddev=noise_stddev))
            modified_output = modified_output + noise

        modified_output = tf.clip_by_value(modified_output,
                                           tf.reduce_min(inputs, axis=[1, 2, 3], keepdims=True),
                                           tf.reduce_max(inputs, axis=[1, 2, 3], keepdims=True))

        if self.gamma_std > 0:

            gamma_t1 = tf.exp(tf.random.normal((), stddev=self.gamma_std))

            modified_output_min = tf.reduce_min(modified_output, axis=[1, 2, 3], keepdims=True)
            modified_output_max = tf.reduce_max(modified_output, axis=[1, 2, 3], keepdims=True)

            modified_output = tf.pow((modified_output - modified_output_min) /
                                     (modified_output_max - modified_output_min), gamma_t1)

            modified_output = modified_output * (modified_output_max - modified_output_min) + modified_output_min

        return modified_output

    def compute_output_shape(self, input_shape):
        return input_shape  # Output shape is same as input shape

    def call(self, inputs, *args, **kwargs):
        return self.__augment__(inputs)


class MinMaxNormLayer(Layer):

    def __init__(self, predet_min=None, predet_max=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.predet_min = predet_min
        self.predet_max = predet_max

    @staticmethod
    def __normalize__(inputs):

        # min_input = tf.reshape(inputs[2], [-1, 1, 1, 1, 1])
        # max_input = tf.reshape(inputs[3], [-1, 1, 1, 1, 1])

        normed_inputs = tf.divide(tf.subtract(inputs[0], inputs[2]),
                                  tf.subtract(inputs[3], inputs[2]))

        normed_inputs = tf.multiply(normed_inputs, inputs[1])

        normed_inputs = tf.clip_by_value(normed_inputs, 0., 1.)

        return normed_inputs

    def call(self, inputs, *args, **kwargs):

        # func = tf.switch_case(self.clip_mode, branch_fns={'constant': self.norm_constant,
        #                                                   'percentile': self.percentile,
        #                                                   'minmax': self.norm_minmax}, default=self.norm_minmax)

        if self.predet_min is not None:
            min_values = tf.convert_to_tensor(self.predet_min)
        else:
            min_values = tf.reshape(inputs[2][:, 0, None], [-1, 1, 1, 1, 1])
        if self.predet_max is not None:
            max_values = tf.convert_to_tensor(self.predet_max)
        else:
            max_values = tf.reshape(inputs[2][:, 1, None], [-1, 1, 1, 1, 1])

        return self.__normalize__([inputs[0], inputs[1], min_values, max_values])


class DTIFitLayer(Layer):

    def __init__(self, bvals, bvecs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dims = None
        self.reshape1 = None
        self.reshape2 = None
        self.permute1 = None
        self.permute2 = None

        self.bvals = bvals if len(bvals.shape) == 1 else bvals[0]
        self.bvecs = bvecs

        X = np.zeros((bvals.shape[0], 7))

        X[:, 0] = -self.bvals * np.power(self.bvecs[:, 0], 2)
        X[:, 1] = -2 * (self.bvals * tf.multiply(self.bvecs[:, 0], self.bvecs[:, 1]))
        X[:, 2] = -2 * (self.bvals * tf.multiply(self.bvecs[:, 0], self.bvecs[:, 2]))
        X[:, 3] = -self.bvals * np.power(self.bvecs[:, 1], 2)
        X[:, 4] = -2 * (self.bvals * tf.multiply(self.bvecs[:, 1], self.bvecs[:, 2]))
        X[:, 5] = -self.bvals * np.power(self.bvecs[:, 2], 2)
        X[:, 6] = 1

        self.X = tf.convert_to_tensor(X, dtype=self.dtype)

    def build(self, input_shape):

        self.input_dims = input_shape[1:-1]
        self.reshape1 = Reshape(target_shape=(self.input_dims[0] * self.input_dims[1] * self.input_dims[2],
                                              input_shape[-1]))
        self.reshape1.build(input_shape)
        self.reshape2 = Reshape(target_shape=(self.input_dims[0], self.input_dims[1], self.input_dims[2], 6))
        self.reshape2.build(input_shape)

        self.permute1 = Permute((2, 1))
        self.permute2 = Permute((2, 1))

        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.input_dims.concatenate(6)

    @tf.function
    def __fit__(self, inputs):

        y = self.permute1(self.reshape1(tf.math.log(tf.add(inputs, keras.src.backend.epsilon()))))
        b = tf.linalg.lstsq(self.X, y)

        out_tensor = self.reshape2(self.permute2(b)[..., :6])
        out_tensor = tf.where(tf.math.is_nan(out_tensor), tf.zeros_like(out_tensor), out_tensor)

        return out_tensor

    def call(self, inputs, *args, **kwargs):
        return self.__fit__(inputs)


    # def build(self, input_shape):

        # igrid = tf.cast(tf.stack(tf.meshgrid(range(0, round(input_shape[1] / self.dsamp_rate[0])),
        #                                      range(0, round(input_shape[2] / self.dsamp_rate[1])),
        #                                      range(0, round(input_shape[3] / self.dsamp_rate[2])),
        #                                      indexing='ij'), axis=-1), self.dtype)
        #
        # dims = tf.cast(tf.divide(input_shape[1:-1], igrid.shape[:-1]), dtype=self.dtype)
        #
        # self.igrid = tf.cast(tf.multiply(igrid, dims), dtype=self.dtype)

        # self.reshape = Reshape(target_shape=(self.igrid.shape[0],
        #                                      self.igrid.shape[1],
        #                                      self.igrid.shape[2], input_shape[-1]),
        #                        dtype=self.dtype,
        #                        name='ReshapeLayer')
        # self.reshape.build((int(self.igrid.shape[0] * self.igrid.shape[1] * self.igrid.shape[2]), input_shape[-1]))

        # return super().build(input_shape)

class InterpLayer(Layer):

    def __init__(self, dsamp_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dsamp_rate = tf.cast(
            (dsamp_rate[0], dsamp_rate[0], dsamp_rate[0]) if len(dsamp_rate) != 3 else dsamp_rate,
            dtype=self.dtype)
        self.igrid = None

        self.reshape = None

    def set_dsamp_rate(self, dsamp_rate):
        self._dsamp_rate = tf.cast(
            (dsamp_rate[0], dsamp_rate[0], dsamp_rate[0]) if dsamp_rate.shape[0] != 3 else dsamp_rate,
            dtype=self.dtype)

    @staticmethod
    @tf.function
    def __interpolate__(vals, weights):

        c00 = tf.math.add(tf.multiply(vals[0], weights[0][..., 2][..., None]),
                          tf.multiply(vals[1], weights[1][..., 2][..., None]))
        c01 = tf.math.add(tf.multiply(vals[2], weights[2][..., 2][..., None]),
                          tf.multiply(vals[3], weights[3][..., 2][..., None]))
        c10 = tf.math.add(tf.multiply(vals[4], weights[4][..., 2][..., None]),
                          tf.multiply(vals[5], weights[5][..., 2][..., None]))
        c11 = tf.math.add(tf.multiply(vals[6], weights[6][..., 2][..., None]),
                          tf.multiply(vals[7], weights[7][..., 2][..., None]))

        c0 = tf.add(tf.multiply(c00, weights[0][..., 1][..., None]),
                    tf.multiply(c01, weights[2][..., 1][..., None]))
        c1 = tf.add(tf.multiply(c10, weights[4][..., 1][..., None]),
                    tf.multiply(c11, weights[6][..., 1][..., None]))

        result = tf.add(tf.multiply(c0, weights[0][..., 0][..., None]),
                        tf.multiply(c1, weights[4][..., 0][..., None]))

        return result

    @tf.function
    def __config_grid__(self, input_shape):

        dim_0 = tf.cast(tf.round(
            tf.divide(tf.cast(input_shape[1], dtype=self.dtype), self._dsamp_rate[0])),
            dtype=tf.uint32)

        dim_1 = tf.cast(tf.round(
            tf.divide(tf.cast(input_shape[2], dtype=self.dtype), self._dsamp_rate[1])),
            dtype=tf.uint32)

        dim_2 = tf.cast(tf.round(
            tf.divide(tf.cast(input_shape[3], dtype=self.dtype), self._dsamp_rate[2])),
            dtype=tf.uint32)

        igrid = tf.cast(tf.stack(tf.meshgrid(tf.linspace(0, tf.subtract(input_shape[1], 1), dim_0),
                                             tf.linspace(0, tf.subtract(input_shape[2], 1), dim_1),
                                             tf.linspace(0, tf.subtract(input_shape[3], 1), dim_2),
                                             indexing='ij'), axis=-1), self.dtype)

        return igrid

        # igrid = tf.cast(tf.stack(tf.meshgrid(tf.range(0, tf.round(tf.divide(input_shape[2], self._dsamp_rate[1]))),
        #                                      tf.range(0, tf.round(tf.divide(input_shape[2], self._dsamp_rate[1]))),
        #                                      tf.range(0, tf.round(tf.divide(input_shape[3], self._dsamp_rate[2]))),
        #                                      indexing='ij'), axis=-1), self.dtype)

        # dims = tf.cast(tf.divide(input_shape[1:-1], igrid.shape[:-1]), dtype=self.dtype)
        # return tf.cast(tf.multiply(igrid, dims), dtype=self.dtype)

        # return tf.cast(tf.multiply(igrid, tf.convert_to_tensor([self._dsamp_rate[0],
        #                                                         self._dsamp_rate[1],
        #                                                         self._dsamp_rate[2]])), dtype=self.dtype)

    def call(self, inputs, *args, **kwargs):

        self.igrid = self.__config_grid__(inputs.shape)

        batch_len = tf.split(tf.shape(inputs), [1, -1])[0]

        # if inputs.shape[0] is None:
        #     output = inputs
        #     output.set_shape(inputs.shape[0],
        #                      self.igrid.shape[0],
        #                      self.igrid.shape[1],
        #                      self.igrid.shape[2],
        #                      inputs.shape[3])
        #     return output

        lgrid = tf.math.floor(self.igrid)
        ugrids = [tf.math.add(lgrid, [int(i & 4 > 0), int(i & 2 > 0), int(i & 1 > 0)]) for i in range(8)]

        udiffs = [tf.subtract(1., tf.abs(tf.subtract(ugrids[i], self.igrid))) for i in range(8)]

        for i in range(8):
            ugrids[i] = tf.cast(ugrids[i], tf.int32)

        flat_grids = [
            tf.repeat(tf.reshape(ugrids[i], shape=(ugrids[i].shape[0] *
                                                   ugrids[i].shape[1] *
                                                   ugrids[i].shape[2], 3))[None, :],
                      batch_len, axis=0) for i in range(8)]

        ugrids_out = [tf.reshape(tf.gather_nd(inputs, flat_grids[i], batch_dims=1),
                                 shape=(-1,
                                        self.igrid.shape[0],
                                        self.igrid.shape[1],
                                        self.igrid.shape[2],
                                        inputs.shape[-1])) for i in range(8)]

        # ugrids_out = []
        #
        # for i in range(8):
        #
        #     flat_grid = tf.reshape(ugrids[i], shape=(ugrids[i].shape[0] * ugrids[i].shape[1] * ugrids[i].shape[2], 3))
        #
        #     ugrid_out = tf.gather_nd(inputs, tf.repeat(flat_grid[None, :], batch_len, axis=0), batch_dims=1)
        #
        #     ugrids_out.append(tf.reshape(
        #         ugrid_out,
        #         shape=(inputs.shape[0],
        #                self.igrid.shape[0],
        #                self.igrid.shape[1],
        #                self.igrid.shape[2],
        #                inputs.shape[-1])))

        return self.__interpolate__(ugrids_out, udiffs)

    # def call(self, inputs, *args, **kwargs):
    #
    #     self.igrid = self.__config_grid__(inputs.shape)
    #
    #     lgrid = tf.math.floor(self.igrid)
    #     ugrids = [tf.math.add(lgrid, [int(i & 4 > 0), int(i & 2 > 0), int(i & 1 > 0)]) for i in range(8)]
    #
    #     udiffs = [tf.subtract(1., tf.abs(tf.subtract(ugrids[i], self.igrid))) for i in range(8)]
    #
    #     for i in range(8):
    #         ugrids[i] = tf.cast(ugrids[i], tf.int32)
    #
    #     ugrids_out = []
    #
    #     for i in range(8):
    #         flat_grid = tf.reshape(ugrids[i], shape=(ugrids[i].shape[0] * ugrids[i].shape[1] * ugrids[i].shape[2], 3))
    #
    #         ugrids_out.append(
    #             tf.reshape(
    #                 tf.stack([tf.gather_nd(inputs[b, ...], flat_grid, batch_dims=0) for b in range(inputs.shape[0])]),
    #                 shape=(inputs.shape[0],
    #                        self.igrid.shape[0],
    #                        self.igrid.shape[1],
    #                        self.igrid.shape[2],
    #                        inputs.shape[-1])
    #             )
    #         )
    #
    #     return self.__interpolate__(ugrids_out, udiffs)


class DynamicSamplingLayer(Layer):

    def __init__(self, max_hr_downsamp, max_lr_downsamp, t1_init_downsamp=1.,
                 static=False,
                 apply_blurring=True, *args, **kwargs):
        super().__init__(trainable=False, *args, **kwargs)

        if max_lr_downsamp < max_hr_downsamp:
            raise ValueError("The value for maximum high-resolution downsampling " +
                             "can not be larger than the low-resolution rate.")

        self.t1_init_downsamp = tf.convert_to_tensor(t1_init_downsamp, dtype=self.dtype)
        self.max_hr_downsamp = tf.convert_to_tensor(max_hr_downsamp, dtype=self.dtype)
        self.max_lr_downsamp = tf.convert_to_tensor(max_lr_downsamp, dtype=self.dtype)

        # self.t1_init_downsamp = float(t1_init_downsamp)
        # self.max_hr_downsamp = float(max_hr_downsamp)
        # self.max_lr_downsamp = float(max_lr_downsamp)

        self.static = static
        self.apply_blurring = apply_blurring

        self.blur_layer: List[Conv3D | None] = [None, None, None]
        self.__blur_kernel_max_size = int(np.int32(np.ceil(2.5 * max_lr_downsamp) / 2) * 2 + 1)
        self.__t1_blur_kernel_max_size = int(np.int32(np.ceil(2.5 * t1_init_downsamp * max_hr_downsamp) / 2) * 2 + 1)

        self._hr_output_dims = None
        self._t1_output_dims = None

        self.t1_downsampler = InterpLayer(dsamp_rate=[self.t1_init_downsamp])
        self.hr_downsampler = InterpLayer(dsamp_rate=[1])
        self.lr_downsampler = InterpLayer(dsamp_rate=[1])
        self.lr_upsampler = InterpLayer(dsamp_rate=[1])

    def compute_output_shape(self, input_shape):

        if input_shape is None or input_shape[0] is None:
            return None

        kernel_penalty = 0

        if self.apply_blurring:
            kernel_penalty = 2 * (self.__blur_kernel_max_size // 2)

        output_shape = tuple([None] +
                             [tf.round((input_shape[0][1] - kernel_penalty) / self.max_hr_downsamp),
                              tf.round((input_shape[0][2] - kernel_penalty) / self.max_hr_downsamp),
                              tf.round((input_shape[0][3] - kernel_penalty) / self.max_hr_downsamp)] +
                             [input_shape[0][-1]])

        # output_shape = tf.convert_to_tensor([input_shape[0][0],
        #     tf.round((input_shape[0][1] - 2*(self.__blur_kernel_max_size//2)) / self.max_hr_downsamp),
        #     tf.round((input_shape[0][2] - 2*(self.__blur_kernel_max_size//2)) / self.max_hr_downsamp),
        #     tf.round((input_shape[0][3] - 2*(self.__blur_kernel_max_size//2)) / self.max_hr_downsamp),
        #     input_shape[0][3]
        # ], dtype=tf.uint32)

        if len(input_shape) > 1:
            # output_shape_t1 = tf.convert_to_tensor([
            #     tf.round((input_shape[1][0] - 2*(self.__t1_blur_kernel_max_size//2)) /
            #              (self.max_hr_downsamp * self.t1_init_downsamp)),
            #     tf.round((input_shape[1][1] - 2*(self.__t1_blur_kernel_max_size//2)) /
            #              (self.max_hr_downsamp * self.t1_init_downsamp)),
            #     tf.round((input_shape[1][2] - 2*(self.__t1_blur_kernel_max_size//2)) /
            #              (self.max_hr_downsamp * self.t1_init_downsamp)),
            #     input_shape[1][3]
            # ], dtype=tf.uint32)

            return [output_shape, output_shape]
        else:
            return [output_shape]

    def build(self, input_shape):

        # self.blur_layer[0] = Conv3D(kernel_size=self.__blur_kernel_max_size,
        #                             filters=input_shape[0][-1],
        #                             name='hr_blur',
        #                             trainable=False,
        #                             use_bias=False,
        #                             kernel_initializer='Zeros')
        #
        # self.blur_layer[1] = Conv3D(kernel_size=self.__blur_kernel_max_size,
        #                             filters=input_shape[0][-1],
        #                             name='lr_blur',
        #                             trainable=False,
        #                             use_bias=False,
        #                             kernel_initializer='Zeros')
        #
        # self.blur_layer[2] = Conv3D(kernel_size=self.__t1_blur_kernel_max_size,
        #                             filters=1,  # T1w is always an image of channel size 1
        #                             name='t1_blur',
        #                             trainable=False,
        #                             use_bias=False,
        #                             kernel_initializer='Zeros')
        #
        # self.blur_layer[0].build(input_shape[0])
        # self.blur_layer[1].build(input_shape[0])

        if len(input_shape) == 1:
            self._hr_output_dims = self.compute_output_shape([input_shape[0][1:]])[0]

        else:
            # self.blur_layer[2].build(input_shape[1])  # Build if T1w input is provided

            output_shape = self.compute_output_shape(input_shape)
            self._hr_output_dims = output_shape[0]
            self._t1_output_dims = output_shape[1]

        return super().build(input_shape)

    # @tf.function
    # def __get_blur_kernel__(self, downsample_rate, channels=1, t1=False):
    #
    #     std_value = 2 * np.log(10) / (2 * np.pi) * downsample_rate
    #     kernel_size = np.int32(np.ceil(2.5 * downsample_rate) / 2) * 2 + 1
    #
    #     kernel_grid = np.copy(
    #         np.mgrid[-kernel_size // 2 + 1:np.ceil(kernel_size / 2),
    #         -kernel_size // 2 + 1:np.ceil(kernel_size / 2),
    #         -kernel_size // 2 + 1:np.ceil(kernel_size / 2)]).transpose((1, 2, 3, 0))
    #
    #     gaussian_kernel = 1 / (np.sqrt(2 * np.pi) * std_value) ** 3 * \
    #                       np.exp(-(kernel_grid[..., 0] ** 2 +
    #                                kernel_grid[..., 1] ** 2 +
    #                                kernel_grid[..., 2] ** 2) / (2 * std_value ** 2))
    #
    #     gaussian_kernel /= np.sum(gaussian_kernel)
    #
    #     kernel_pads = np.abs(self.__t1_blur_kernel_max_size - kernel_size) \
    #         if t1 \
    #         else np.abs(self.__blur_kernel_max_size - kernel_size)
    #
    #     blur_kernel = np.pad(gaussian_kernel,
    #                          pad_width=np.array([[kernel_pads // 2, kernel_pads // 2],
    #                                              [kernel_pads // 2, kernel_pads // 2],
    #                                              [kernel_pads // 2, kernel_pads // 2]]),
    #                          mode='constant',
    #                          constant_values=(0, 0))
    #
    #     full_kernel = np.zeros(blur_kernel.shape + (channels, channels))
    #
    #     return blur_kernel

    @tf.function
    def __get_blur_kernel__(self, downsample_rate, channels=1, t1=False):

        std_value = 2 * np.log(10) / (2 * np.pi) * downsample_rate
        kernel_size = tf.cast((tf.math.ceil(2.5 * downsample_rate) / 2), dtype=tf.uint32) * 2 + 1

        kernel_length = tf.range(-tf.floor(tf.divide(kernel_size, 2)), (tf.floor(tf.divide(kernel_size, 2)) + 1))

        kernel_grid = tf.cast(tf.stack(tf.meshgrid(kernel_length, kernel_length, kernel_length,
                                                   indexing='ij'), axis=-1), self.dtype)

        gaussian_kernel = 1 / tf.pow(tf.math.sqrt(2. * np.pi) * std_value, 3.) * \
            tf.math.exp(-(kernel_grid[..., 0] ** 2 +
                          kernel_grid[..., 1] ** 2 +
                          kernel_grid[..., 2] ** 2) / (2 * std_value ** 2))

        gaussian_kernel /= tf.reduce_sum(gaussian_kernel)

        if t1:
            kernel_pads = tf.math.abs(tf.subtract(tf.convert_to_tensor(self.__t1_blur_kernel_max_size, dtype=tf.int32),
                                                  tf.cast(kernel_size, tf.int32)))
        else:
            kernel_pads = tf.math.abs(tf.subtract(tf.convert_to_tensor(self.__blur_kernel_max_size, dtype=tf.int32),
                                                  tf.cast(kernel_size, tf.int32)))

        paddings = tf.convert_to_tensor([[kernel_pads // 2, kernel_pads // 2],
                                         [kernel_pads // 2, kernel_pads // 2],
                                         [kernel_pads // 2, kernel_pads // 2]])

        blur_kernel = tf.pad(gaussian_kernel, paddings=paddings)

        if t1:
            full_kernel = np.zeros((self.__t1_blur_kernel_max_size,
                                    self.__t1_blur_kernel_max_size,
                                    self.__t1_blur_kernel_max_size, channels, channels))
        else:
            full_kernel = np.zeros((self.__blur_kernel_max_size,
                                    self.__blur_kernel_max_size,
                                    self.__blur_kernel_max_size, channels, channels))

        for i in range(channels):
            full_kernel[..., i, i] = 1.

        return tf.multiply(tf.convert_to_tensor(full_kernel, dtype=self.dtype), blur_kernel[..., None, None])

    @tf.function
    def blur(self, input_tensor, blur_rate, t1):

        if t1:
            blur_kernel_size = self.__t1_blur_kernel_max_size // 2
        else:
            blur_kernel_size = self.__blur_kernel_max_size // 2

        return tf.cond(tf.less_equal(blur_rate, 1.5),
                       lambda: input_tensor[:,
                                            blur_kernel_size:-blur_kernel_size,
                                            blur_kernel_size:-blur_kernel_size,
                                            blur_kernel_size:-blur_kernel_size,
                                            :],
                       lambda: tf.nn.convolution(input_tensor,
                                                 self.__get_blur_kernel__(blur_rate, input_tensor.shape[-1], t1)))

    @tf.function
    def __crop__(self, tensor, rate):
        target_img_shape = tf.cast(self._hr_output_dims[1:-1], rate.dtype) * rate

        crop_values = tf.cast(tf.round(tensor.shape[1:-1] - target_img_shape), dtype=tf.int32) / 2

        return tensor[:,
                      crop_values[0]:-crop_values[0],
                      crop_values[1]:-crop_values[1],
                      crop_values[2]:-crop_values[2],
                      :]

    def call(self, inputs, *args, **kwargs):

        static = self.static or inputs[0].shape[0] is None  # If batch is none just return target shape.

        if static:
            hr_downsamp_rate = self.max_hr_downsamp
            lr_downsamp_rate = self.max_lr_downsamp
        else:
            hr_downsamp_rate = tf.random.uniform([], 1., self.max_hr_downsamp, dtype=self.dtype)
            lr_downsamp_rate = tf.random.uniform([], hr_downsamp_rate, self.max_lr_downsamp, dtype=self.dtype)

        if self.apply_blurring:
            inputs_hr = self.blur(inputs[0], hr_downsamp_rate, False)
            inputs_lr = self.blur(inputs[0], lr_downsamp_rate, False)
        else:
            inputs_hr = inputs[0]
            inputs_lr = inputs[0]

        if len(inputs) > 1 and inputs[1] is not None:
            if static:
                t1_downsamp_rate = self.t1_init_downsamp * self.max_hr_downsamp
            else:
                t1_downsamp_rate = self.t1_init_downsamp * hr_downsamp_rate

            if self.apply_blurring:
                inputs_t1 = self.blur(inputs[1], t1_downsamp_rate, True)
            else:
                inputs_t1 = inputs[1]

        else:
            inputs_t1 = None

        self.hr_downsampler.set_dsamp_rate(tf.convert_to_tensor([hr_downsamp_rate]))
        self.lr_downsampler.set_dsamp_rate(tf.convert_to_tensor([lr_downsamp_rate]))

        if not static:
            inputs_hr = self.__crop__(inputs_hr, hr_downsamp_rate)
            inputs_lr = self.__crop__(inputs_lr, lr_downsamp_rate)

        inputs_hr = self.hr_downsampler(inputs_hr)
        inputs_lr = self.lr_downsampler(inputs_lr)

        self.lr_upsampler.set_dsamp_rate(tf.divide(inputs_lr.shape[1:-1], inputs_hr.shape[1:-1]))

        if inputs_t1 is not None:

            self.t1_downsampler.set_dsamp_rate(tf.divide(inputs_t1.shape[1:-1], inputs_hr.shape[1:-1]))

            if not static:
                inputs_t1 = self.__crop__(inputs_t1, tf.divide(inputs_t1.shape[1:-1], inputs_hr.shape[1:-1]))

            inputs_t1 = self.t1_downsampler(inputs_t1)

        inputs_lr = self.lr_upsampler(inputs_lr)

        return [inputs_lr, inputs_hr] if inputs_t1 is None else [inputs_lr, inputs_hr, inputs_t1]

        # self.__set_blur_kernel__(hr_downsamp_rate)
        # self.__set_blur_kernel__(lr_downsamp_rate)

        # inputs_hr = tf.nn.convolution(inputs[0], ) \
        #     if bool_tensor \
        #     else inputs[0][:,
        #                    self.__blur_kernel_max_size//2:-(self.__blur_kernel_max_size//2),
        #                    self.__blur_kernel_max_size//2:-(self.__blur_kernel_max_size//2),
        #                    self.__blur_kernel_max_size//2:-(self.__blur_kernel_max_size//2),
        #                    :]

        # inputs_lr = self.blur_layer[1](inputs[0])


def unet_downsample_layer(prev_layer,
                          filter_size,
                          kernel_size=3,
                          rep_layers=2):

    conv = Sequential([
        Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding="same",
               strides=2),
        ReLU()
    ])

    for _ in range(rep_layers):
        conv.add(Sequential([
            Conv3D(filters=filter_size,
                   kernel_size=3,
                   padding="same"),
            ReLU()
        ]))

    return conv(prev_layer)


def unet_upsample_layer(prev_layer,
                        filter_size,
                        concat_layer=None,
                        kernel_size=3,
                        rep_layers=2):

    # layer = Sequential([Conv3DTranspose(filters=filter_size,
    #                                     kernel_size=kernel_size*2,
    #                                     strides=2,
    #                                     padding="same"),
    #                    ReLU()])

    layer = Sequential([DepthToSpaceLayer(upsampling_rate=2),
                        Conv3D(filters=filter_size,
                               kernel_size=kernel_size*2,
                               padding="same"),
                        ReLU()])

    conv = layer(prev_layer)

    if concat_layer is not None:
        conv = concatenate([conv, concat_layer], 4)

    layer = Sequential()

    for _ in range(rep_layers):
        layer.add(Sequential([
            Conv3D(filters=filter_size,
                   kernel_size=3,
                   padding="same"),
            ReLU()
        ]))

    return layer(conv)


def unet_downsample_layer_v2(prev_layer,
                             filter_size,
                             kernel_size=3,
                             rep_layers=2,
                             last_layer=False):

    conv = Sequential([
        Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding="same"),
        LeakyReLU()
    ])

    for n in range(rep_layers):
        conv.add(Conv3D(filters=filter_size,
                        kernel_size=3,
                        padding="same"))
        if not (last_layer & (n == (rep_layers - 1))):
            conv.add(LeakyReLU())

    pre_act = conv(prev_layer)

    return pre_act


def unet_upsample_layer_v2(prev_layer,
                           filter_size,
                           concat_layer=None,
                           kernel_size=3,
                           rep_layers=2):

    conv = UpSampling3D(size=(2, 2, 2))(prev_layer)

    if concat_layer is not None:
        conv = Concatenate(axis=4)([conv, concat_layer])

    layer = []

    for _ in range(rep_layers):
        layer.append(Conv3D(filters=filter_size,
                     kernel_size=kernel_size,
                     padding="same"))
        layer.append(LeakyReLU())

    layer.append(BatchNormalization())

    return Sequential(layer)(conv)


def unet_downsample_layer_v3(prev_layer,
                             filter_size,
                             kernel_size=3,
                             rep_layers=2,
                             residual=True,
                             final_activation=True):
    dsamp = Sequential([
        Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding="same"),
        ELU(),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding="same")
    ])(prev_layer)

    conv = [ELU()]

    for n in range(rep_layers):
        conv.append(Conv3D(filters=filter_size,
                           kernel_size=3,
                           padding="same"))
        if n < (rep_layers - 1):
            conv.append(ELU())

    if residual:
        pre_act = Add()([Sequential(conv)(dsamp), dsamp])
    else:
        pre_act = Sequential(conv)(dsamp)

    if final_activation:
        return ELU()(pre_act)
    else:
        return pre_act


class VolumeAttentionLayer(Layer):

    def __init__(self, dims):
        super().__init__()

        self.dims = dims[:-1]
        self.channels = dims[-1]

        self.__dense = [Dense(self.channels, use_bias=False),
                        Dense(self.channels, use_bias=False),
                        Dense(self.channels, use_bias=False)]
        self.__attention = Attention()

    def call(self, inputs, *args, **kwargs):

        queries = inputs[0]
        values = inputs[1] if len(inputs) > 1 else queries
        keys = inputs[2] if len(inputs) > 2 else values

        values = Reshape((self.dims[1] * self.dims[2] * self.dims[3], self.channels))(values)
        keys = Reshape((self.dims[1] * self.dims[2] * self.dims[3], self.channels))(keys)
        queries = Reshape((self.dims[1] * self.dims[2] * self.dims[3], self.channels))(queries)

        result = self.__attention([self.__dense[0](queries),
                                   self.__dense[1](values),
                                   self.__dense[2](keys)])

        result = Reshape((self.dims[1], self.dims[2], self.dims[3], self.channels))(result)

        return result


class BasicTransformerBlock(Layer):

    def __init__(self, dims):
        super().__init__()
        self.attn1 = VolumeAttentionLayer(dims)
        self.linear = Sequential([Conv3D(dims[-1], kernel_size=1),
                                  ELU(),
                                  Conv3D(dims[-1], kernel_size=1)])
        self.attn2 = VolumeAttentionLayer(dims)
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()
        self.norm3 = BatchNormalization()

    def call(self, inputs, *args, **kwargs):
        x = inputs[0]
        context = inputs[1]

        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        x = self.linear(self.norm3(x)) + x
        return x


def unet_attention_fusion(prev_layer,
                          filter_size,
                          attention_layer=None,
                          concat_layer=None,
                          rep_layers=2,
                          residual=True):

    conv = UpSampling3D(size=(2, 2, 2))(prev_layer)

    if attention_layer is None and concat_layer is None:
        prev_layer = Conv3D(filters=filter_size, kernel_size=1, padding='same')(conv)

        conv = ELU()(prev_layer)

    if attention_layer is not None:
        prev_layer = BasicTransformerBlock(attention_layer.shape)(
            [ELU()(Conv3D(filters=filter_size, kernel_size=3, padding='same')(conv)),
             attention_layer]
        )

        conv = ELU()(prev_layer)

    if concat_layer is not None:
        prev_layer = Conv3D(filters=filter_size, kernel_size=1, padding='same')(
            Concatenate(axis=4)([conv, concat_layer])
        )

        conv = ELU()(prev_layer)

    layer = []

    for n in range(rep_layers):
        layer.append(Conv3D(filters=filter_size,
                            kernel_size=3,
                            padding="same"))
        if n < (rep_layers - 1):
            layer.append(ELU())

    if residual:
        return BatchNormalization()(ELU()(Add()([Sequential(layer)(conv), prev_layer])))
    else:
        return BatchNormalization()(ELU()(Sequential(layer)(conv)))

