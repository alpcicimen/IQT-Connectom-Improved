import random

import keras.backend as K
import keras.src.backend
import tensorflow as tf

from keras.layers import *
from keras import Sequential

import numpy as np
from tqdm import tqdm

from typing import List, Literal


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

        assert (dim_i % self.upsampling_rate, dim_j % self.upsampling_rate, dim_k % self.upsampling_rate) == (0, 0, 0)
        # Number must be exactly divisible by 8

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
                                     (modified_output_max - modified_output_min + 1e-7), gamma_t1)

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

    def __init__(self, acquisition_length, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dims = None
        self.reshape1 = None
        self.reshape2 = None
        self.permute1 = None
        self.permute2 = None

        self.acq_len = acquisition_length

    def build(self, input_shape):

        assert (input_shape[0][-1],
                input_shape[1][1],
                input_shape[2][1]) == (self.acq_len,
                                       self.acq_len,
                                       self.acq_len), "Acquisition length must match the input length!"

        self.input_dims = input_shape[0][1:-1]
        self.reshape1 = Reshape(target_shape=(self.input_dims[0] * self.input_dims[1] * self.input_dims[2],
                                              self.acq_len))
        self.reshape1.build(input_shape)
        self.reshape2 = Reshape(target_shape=(self.input_dims[0], self.input_dims[1], self.input_dims[2], 6))
        self.reshape2.build(input_shape)

        self.permute1 = Permute((2, 1))
        self.permute2 = Permute((2, 1))

        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[:-1] + (6, )])

    @staticmethod
    @tf.function
    def __create_X__(bvals, bvecs):

        c_00 = -tf.multiply(bvals, tf.pow(bvecs[..., 0, None], 2.))
        c_01 = tf.multiply(tf.multiply(bvals, tf.multiply(bvecs[..., 0, None], bvecs[..., 1, None])), -2.)
        c_02 = tf.multiply(tf.multiply(bvals, tf.multiply(bvecs[..., 0, None], bvecs[..., 2, None])), -2.)
        c_11 = -tf.multiply(bvals, tf.pow(bvecs[..., 1, None], 2.))
        c_12 = tf.multiply(tf.multiply(bvals, tf.multiply(bvecs[..., 1, None], bvecs[..., 2, None])), -2.)
        c_22 = -tf.multiply(bvals, tf.pow(bvecs[..., 2, None], 2.))
        c_ones = tf.ones(tf.shape(bvals))

        X = tf.concat([c_00, c_01, c_02, c_11, c_12, c_22, c_ones], axis=-1)

        return X

    @tf.function
    def __fit__(self, dwis, bvals, bvecs):

        X = tf.cast(self.__create_X__(bvals, bvecs), dtype=self.dtype)

        y = self.permute1(self.reshape1(tf.math.log(tf.add(dwis, keras.src.backend.epsilon()))))
        b = tf.linalg.lstsq(X, y)

        out_tensor = self.reshape2(self.permute2(b)[..., :6])
        out_tensor = tf.where(tf.math.is_nan(out_tensor), tf.zeros_like(out_tensor), out_tensor)

        return out_tensor

    def call(self, inputs, *args, **kwargs):
        assert isinstance(inputs, list) or isinstance(inputs, tuple)
        assert len(inputs) == 3

        return self.__fit__(inputs[0], inputs[1], inputs[2])


class MAPMRIFitLayer(Layer):

    __herm_coefs = {0: [[0, 0, 0]],  # 1
                    # 1: [[1, 0, 0],  # Unnecessary as only even hermitian order coefficients are needed
                    #     [0, 1, 0],
                    #     [0, 0, 1]],
                    2: [[2, 0, 0],  # 2
                        [1, 1, 0],  # 3
                        [1, 0, 1],  # 4
                        [0, 2, 0],  # 5
                        [0, 1, 1],  # 6
                        [0, 0, 2]],  # 7
                    # 3: [[3, 0, 0],
                    #     [2, 1, 0],
                    #     [2, 0, 1],
                    #     [1, 1, 1],
                    #     [1, 0, 2],
                    #     [0, 3, 0],
                    #     [0, 2, 1],
                    #     [0, 1, 2],
                    #     [0, 0, 3],],
                    4: [[4, 0, 0],
                        [3, 1, 0],
                        [3, 0, 1],
                        [2, 2, 0],
                        [2, 1, 1],
                        [2, 0, 2],
                        [1, 3, 0],
                        [1, 2, 1],
                        [1, 1, 2],
                        [1, 0, 3],
                        [0, 4, 0],
                        [0, 3, 1],
                        [0, 2, 2],
                        [0, 1, 3],
                        [0, 0, 4]],
                    # 5: [[5, 0, 0],
                    #     [4, 1, 0],
                    #     [4, 0, 1],
                    #     [3, 2, 0],
                    #     [3, 1, 1],
                    #     [3, 0, 2],
                    #     [2, 3, 0],
                    #     [2, 2, 1],
                    #     [2, 1, 2],
                    #     [2, 0, 3],
                    #     [1, 4, 0],
                    #     [1, 3, 1],
                    #     [1, 2, 2],
                    #     [1, 1, 3],
                    #     [1, 0, 4],
                    #     [0, 5, 0],
                    #     [0, 4, 1],
                    #     [0, 3, 2],
                    #     [0, 2, 3],
                    #     [0, 1, 4],
                    #     [0, 0, 5]],
                    6: [[6, 0, 0],
                        [5, 1, 0],
                        [5, 0, 1],
                        [4, 2, 0],
                        [4, 1, 1],
                        [4, 0, 2],
                        [3, 3, 0],
                        [3, 2, 1],
                        [3, 1, 2],
                        [3, 0, 3],
                        [2, 4, 0],
                        [2, 3, 1],
                        [2, 2, 2],
                        [2, 1, 3],
                        [2, 0, 4],
                        [1, 5, 0],
                        [1, 4, 1],
                        [1, 3, 2],
                        [1, 2, 3],
                        [1, 1, 4],
                        [1, 0, 5],
                        [0, 6, 0],
                        [0, 5, 1],
                        [0, 4, 2],
                        [0, 3, 3],
                        [0, 2, 4],
                        [0, 1, 5],
                        [0, 0, 6]]
                    }

    @staticmethod
    # @tf.function
    def hermite_basis(n, u, x):
        # tf.linalg.matmul(q_batch[..., 0, None], pqr_batch[:, None, ..., 0])
        # h = tf.linalg.matmul(n, x) * tf.constant([2.0 * np.pi], dtype=x.dtype) * u
        # h = tf.constant([2.0 * np.pi], dtype=x.dtype) * u
        h = tf.repeat(tf.constant([2.0 * np.pi], dtype=x.dtype) * u * x, tf.shape(n)[1], axis=1)

        hh = tf.ones(tf.shape(h))
        nn = tf.ones(tf.shape(h))

        hh = tf.where(n == 1, 2.0 * h, hh)
        hh = tf.where(n == 2, 4.0 * tf.pow(h, 2.0) - 2.0, hh)
        hh = tf.where(n == 3, 8.0 * tf.pow(h, 3.0) - 12.0 * h, hh)
        hh = tf.where(n == 4, 16.0 * tf.pow(h, 4.0) - 48.0 * tf.pow(h, 2.0) + 12.0, hh)
        hh = tf.where(n == 5, 32.0 * tf.pow(h, 5.0) - 160.0 * tf.pow(h, 3.0) + 120.0 * h, hh)
        hh = tf.where(n == 6, 64.0 * tf.pow(h, 6.0) - 480.0 * tf.pow(h, 4.0) + 720.0 * tf.pow(h, 2.0) - 120.0, hh)

        nn = tf.where(n == 1, tf.sqrt(2.), nn)
        nn = tf.where(n == 2, tf.sqrt(8.), nn)
        nn = tf.where(n == 3, tf.sqrt(48.), nn)
        nn = tf.where(n == 4, tf.sqrt(384.), nn)
        nn = tf.where(n == 5, tf.sqrt(3840.), nn)
        nn = tf.where(n == 6, tf.sqrt(46080.), nn)

        # k = (tf.pow(tf.complex(tf.zeros(h.shape), tf.ones(h.shape)), -tf.complex(n, tf.zeros(n.shape))) /
        #      tf.complex(nn, tf.zeros(nn.shape)))
        #
        # phr = tf.exp(-tf.square(h) / 2.0) * hh
        #
        # phi = k * tf.complex(phr, tf.zeros(phr.shape))

        return tf.math.divide(tf.math.multiply(tf.exp(-tf.square(h)/2.0), hh), nn)

    @staticmethod
    def hermite_basis_complex(n, u, x):
        h = tf.repeat(tf.constant([2.0 * np.pi], dtype=x.dtype) * u * x, tf.shape(n)[1], axis=1)

        hh = tf.ones(tf.shape(h))
        nn = tf.ones(tf.shape(h))

        hh = tf.where(n == 1, 2.0 * h, hh)
        hh = tf.where(n == 2, 4.0 * tf.pow(h, 2.0) - 2.0, hh)
        hh = tf.where(n == 3, 8.0 * tf.pow(h, 3.0) - 12.0 * h, hh)
        hh = tf.where(n == 4, 16.0 * tf.pow(h, 4.0) - 48.0 * tf.pow(h, 2.0) + 12.0, hh)
        hh = tf.where(n == 5, 32.0 * tf.pow(h, 5.0) - 160.0 * tf.pow(h, 3.0) + 120.0 * h, hh)
        hh = tf.where(n == 6, 64.0 * tf.pow(h, 6.0) - 480.0 * tf.pow(h, 4.0) + 720.0 * tf.pow(h, 2.0) - 120.0, hh)

        nn = tf.where(n == 1, tf.sqrt(2.), nn)
        nn = tf.where(n == 2, tf.sqrt(8.), nn)
        nn = tf.where(n == 3, tf.sqrt(48.), nn)
        nn = tf.where(n == 4, tf.sqrt(384.), nn)
        nn = tf.where(n == 5, tf.sqrt(3840.), nn)
        nn = tf.where(n == 6, tf.sqrt(46080.), nn)

        return tf.math.divide(tf.math.multiply(tf.exp(-tf.square(h)/2.0), hh), nn)

    def map_basis(self, u, x):

        batch_len = tf.shape(x)[0]

        return (self.hermite_basis(tf.repeat(self.pqr[None, ..., 0, None],
                                             batch_len, axis=0), u, x[..., 0][:, None, :]) *
                self.hermite_basis(tf.repeat(self.pqr[None, ..., 1, None],
                                             batch_len, axis=0), u, x[..., 1][:, None, :]) *
                self.hermite_basis(tf.repeat(self.pqr[None, ..., 2, None],
                                             batch_len, axis=0), u, x[..., 2][:, None, :]))

    # Difftime (0.024 - 0.007 / 3) for Cardiff Data
    def __init__(self, acquisition_length, herm_order, difftime=(0.0431 - 0.0106/3), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dims = None
        self.reshape1 = None
        self.reshape2 = None
        self.permute = None

        self.acq_len = acquisition_length
        # Determine the maximum (even) hermitian order to use. Odd values are rounded down.
        self.h_order = (herm_order // 2) * 2
        self.difftime = difftime

        pqr = []

        for order in np.arange(0, self.h_order + 1, 2):
            pqr += self.__herm_coefs[order]

        self.__output_dims = len(pqr)

        self.pqr = tf.cast(tf.stack(pqr, axis=0), dtype=self.dtype)

    def build(self, input_shape):
        assert (input_shape[0][-1],
                input_shape[1][1],
                input_shape[2][1]) == (self.acq_len,
                                       self.acq_len,
                                       self.acq_len), "Acquisition length must match the input length!"

        self.input_dims = input_shape[0][1:-1]
        self.reshape1 = Reshape(target_shape=(self.input_dims[0] * self.input_dims[1] * self.input_dims[2],
                                              self.acq_len))
        self.reshape1.build(input_shape)

        self.reshape2 = Reshape(target_shape=(self.input_dims[0], self.input_dims[1], self.input_dims[2],
                                              self.__output_dims))
        self.reshape2.build(input_shape)

        self.permute = Permute((2, 1))

        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + self.__output_dims

    def qmat(self, bvals, bvecs):

        q = tf.sqrt(bvals / self.difftime) * bvecs

        return q

    def call(self, inputs, *args, **kwargs):

        dwis = inputs[0]

        qmat = self.qmat(inputs[1], inputs[2])

        map_basis = self.permute(self.map_basis(1.2e-3, qmat))

        y = self.reshape1(dwis)

        # b0_indices = tf.where(inputs[1] < self.b0_limit)[..., :2]
        #
        # b0_mean = tf.reduce_mean(tf.reshape(tf.gather_nd(self.permute(y), b0_indices),
        #                                     [batch_len[0],
        #                                      tf.cast(b0_indices.shape[0]/batch_len, batch_len.dtype)[0],
        #                                      tf.shape(y)[1]]), axis=1, keepdims=True)
        #
        # y = y / (self.permute(b0_mean) + 1e-7)

        b = self.permute(tf.linalg.lstsq(map_basis, self.permute(y)))

        out_tensor = self.reshape2(b)
        out_tensor = tf.where(tf.math.is_nan(out_tensor), tf.zeros_like(out_tensor), out_tensor)

        return out_tensor


class SamplingLayer(Layer):

    @staticmethod
    @tf.function
    def __interpolate__(vals, weights):

        c00 = tf.math.add(tf.multiply(vals[0], weights[0][..., 2, None]),
                          tf.multiply(vals[1], weights[1][..., 2, None]))
        c01 = tf.math.add(tf.multiply(vals[2], weights[2][..., 2, None]),
                          tf.multiply(vals[3], weights[3][..., 2, None]))
        c10 = tf.math.add(tf.multiply(vals[4], weights[4][..., 2, None]),
                          tf.multiply(vals[5], weights[5][..., 2, None]))
        c11 = tf.math.add(tf.multiply(vals[6], weights[6][..., 2, None]),
                          tf.multiply(vals[7], weights[7][..., 2, None]))

        c0 = tf.add(tf.multiply(c00, weights[0][..., 1, None]),
                    tf.multiply(c01, weights[2][..., 1, None]))
        c1 = tf.add(tf.multiply(c10, weights[4][..., 1, None]),
                    tf.multiply(c11, weights[6][..., 1, None]))

        result = tf.add(tf.multiply(c0, weights[0][..., 0, None]),
                        tf.multiply(c1, weights[4][..., 0, None]))

        return result

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

    # @tf.function(reduce_retracing=True)
    def __resample_interpolation__(self, inputs, igrid):
        # igrid = self.__config_grid__(inputs.shape)

        batch_len = tf.split(tf.shape(inputs), [1, -1])[0]

        max_mask = tf.ones(igrid.shape) * (inputs.shape[1] - 1, inputs.shape[2] - 1, inputs.shape[3] - 1)

        lgrid = tf.math.floor(igrid)
        ugrids = [tf.minimum(tf.math.add(lgrid, [int(i & 4 > 0), int(i & 2 > 0), int(i & 1 > 0)]),
                             max_mask) for i in range(8)]

        udiffs = [tf.subtract(1., tf.abs(tf.subtract(ugrids[i], igrid))) for i in range(8)]

        for i in range(8):
            ugrids[i] = tf.cast(ugrids[i], tf.int32)

        flat_grids = [
            tf.repeat(tf.reshape(ugrids[i], shape=(ugrids[i].shape[0] *
                                                   ugrids[i].shape[1] *
                                                   ugrids[i].shape[2], 3))[None, :],
                      batch_len, axis=0) for i in range(8)]

        ugrids_out = [tf.reshape(tf.gather_nd(inputs, flat_grids[i], batch_dims=1),
                                 shape=(-1,
                                        igrid.shape[0],
                                        igrid.shape[1],
                                        igrid.shape[2],
                                        inputs.shape[-1])) for i in range(8)]

        return self.__interpolate__(ugrids_out, udiffs)

    @staticmethod
    def __resample_nearest__(inputs, igrid):

        igrid = tf.cast(tf.math.round(igrid), tf.int32)

        batch_len = tf.split(tf.shape(inputs), [1, -1])[0]

        flat_grid = tf.repeat(tf.reshape(igrid, shape=(igrid.shape[0] * igrid.shape[1] * igrid.shape[2], 3))[None, :],
                              batch_len, axis=0)

        ugrid_out = tf.reshape(tf.gather_nd(inputs, flat_grid, batch_dims=1),
                               shape=(-1,
                                      igrid.shape[0],
                                      igrid.shape[1],
                                      igrid.shape[2],
                                      inputs.shape[-1]))

        return ugrid_out

    def __init__(self, dsamp_rate,
                 method: Literal["nearest", "trilinear"] = "trilinear",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dsamp_rate = tf.cast(
            (dsamp_rate[0], dsamp_rate[0], dsamp_rate[0]) if len(dsamp_rate) != 3 else dsamp_rate,
            dtype=self.dtype)
        self.igrid = None

        self.__sampling_function__ = self.__resample_nearest__ if method == 'nearest' \
            else self.__resample_interpolation__

        self.method = method
        self.reshape = None

    @property
    def dsamp_rate(self):
        return self._dsamp_rate

    @dsamp_rate.setter
    def dsamp_rate(self, dsamp_rate):
        self._dsamp_rate = tf.cast(
            (dsamp_rate[0], dsamp_rate[0], dsamp_rate[0]) if dsamp_rate.shape[0] != 3 else dsamp_rate,
            dtype=self.dtype)

    def call(self, inputs, *args, **kwargs):

        return self.__sampling_function__(inputs, self.__config_grid__(inputs.shape))


class BlurLayer(Layer):
    """
    Implements a 3D gaussian blurring algorithm as a tensorflow layer.
    The layer accepts an input and a downsampling rate and blurs according to the downsampling rate.

    :param max_kernel_size: The maximum size the blurring kernel can have. (Relevant for the patch edge calculation)
    :param blur_limit: The limit for downsampling rate in which blurring rate will not be applied.
        If the rate provided is less than this value the layer will just crop the image instead.
    """

    def __init__(self,
                 max_kernel_size,
                 blur_limit=1.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__max_kernel_size = max_kernel_size
        self.channels = None
        self.blur_limit = blur_limit

    def build(self, input_shape):

        self.channels = input_shape[0][-1]  # Initialize it here based on input channels

        return super().build(input_shape)

    @tf.function
    def __compute_blur_kernel__(self, downsample_rate):
        """
        Computes a gaussian blurring kernel for the convolution operation.
        The kernel will be of shape [K, K, K, C, C],
        where K is the shape of the kernel determined by max_kernel_size property.

        This kernel is compatible with the convolution operation for tensorflow for inputs of shape [B, H, W, D, C].

        :param downsample_rate: The downsampling rate determined for blurring.
        :return: The estimated blurring kernel
        """

        # Check Billot et al. for more details
        std_value = tf.convert_to_tensor(2 * np.log(10) / (2 * np.pi), dtype=self.dtype) * downsample_rate
        kernel_size = tf.cast((tf.math.ceil(2.5 * downsample_rate) / 2), dtype=tf.uint32) * 2 + 1

        # Create a kernel grid for generating the blurring kernel.
        # Grid should be of shape [K_i, K_i, K_i],
        # with values ranging from [-K_i//2 : K_i//2]
        kernel_length = tf.range(-tf.floor(tf.divide(kernel_size, 2)), (tf.floor(tf.divide(kernel_size, 2)) + 1))

        kernel_grid = tf.cast(tf.stack(tf.meshgrid(kernel_length, kernel_length, kernel_length,
                                                   indexing='ij'), axis=-1), self.dtype)

        # 3D Gaussian distribution
        gaussian_kernel = 1 / tf.pow(tf.math.sqrt(2. * np.pi) * std_value, 3.) * \
            tf.math.exp(-(kernel_grid[..., 0] ** 2 +
                          kernel_grid[..., 1] ** 2 +
                          kernel_grid[..., 2] ** 2) / (2 * std_value ** 2))

        # To ensure that the convolution does not add an intensity bias
        gaussian_kernel /= tf.reduce_sum(gaussian_kernel)

        # Our kernel is of shape K_i^3, but we need it to be the size of max_kernel_size. Therefore, we pad with zeros.
        kernel_pads = tf.math.abs(tf.subtract(tf.convert_to_tensor(self.__max_kernel_size, dtype=tf.int32),
                                              tf.cast(kernel_size, tf.int32)))

        paddings = tf.convert_to_tensor([[kernel_pads // 2, kernel_pads // 2],
                                         [kernel_pads // 2, kernel_pads // 2],
                                         [kernel_pads // 2, kernel_pads // 2]])

        # Expand the dims to [K, K, K, 1, 1]
        blur_kernel = tf.pad(gaussian_kernel, paddings=paddings)[..., None, None]

        # Keras convolution kernels are of shape [X, Y, Z, C_i, C_o], where inputs of channel sizes C_i gets convolved
        # and mapped to channel sizes C_o. For this purpose C_i == C_o == C.
        #
        # Due to the blurring operation we need to apply the kernel to each channel separately.
        # This means that for all channels c in range(0, C) kernel[:, :, :, c_1, c_2] == blur_kernel,
        # and kernel[:, :, :, c, c] = 0 otherwise.
        #
        # The kernel shape has to be [K, K, K, C, C].
        # Therefore, we expand and repeat the identity matrix [C, C] by K for all directions to obtain [K, K, K, C, C]
        # We then broadcast the blur kernel [K, K, K, 1, 1] and multiply element-wise to obtain the final kernel.
        full_kernel = tf.tile(tf.eye(self.channels, self.channels)[None, None, None, :, :],
                              [self.__max_kernel_size, self.__max_kernel_size, self.__max_kernel_size, 1, 1])

        return full_kernel * blur_kernel

    @tf.function
    def blur(self, input_tensor, blur_rate):

        input_tensor_shape = input_tensor.shape

        ret = tf.cond(tf.less_equal(blur_rate, self.blur_limit),  # Do not blur at certain rates
                      lambda: input_tensor[
                              :,
                              self.__max_kernel_size // 2:input_tensor_shape[1] - self.__max_kernel_size // 2,
                              self.__max_kernel_size // 2:input_tensor_shape[2] - self.__max_kernel_size // 2,
                              self.__max_kernel_size // 2:input_tensor_shape[3] - self.__max_kernel_size // 2,
                              :],
                      lambda: tf.nn.convolution(input_tensor, self.__compute_blur_kernel__(blur_rate)))

        return ret

    def call(self, inputs, *args, **kwargs):

        return self.blur(inputs[0], inputs[1])


class SamplerLayer(Layer):
    def __init__(self, max_hr_downsamp, max_lr_downsamp, t1_init_downsamp=1.,
                 min_hr_downsamp=1.,
                 apply_blurring=True,
                 augment=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert min_hr_downsamp >= 1.0, "Values smaller than 1 are not supported."

        assert max_hr_downsamp < max_lr_downsamp, "The value for maximum high-resolution downsampling " + \
                                                  "can not be larger than the low-resolution rate."

        self.apply_blurring = apply_blurring
        self.augment = augment

        self.t1_init_downsamp = tf.convert_to_tensor(t1_init_downsamp, dtype=self.dtype)
        self.max_hr_downsamp = tf.convert_to_tensor(max_hr_downsamp, dtype=self.dtype)
        self.min_hr_downsamp = tf.convert_to_tensor(min_hr_downsamp, dtype=self.dtype)
        self.max_lr_downsamp = tf.convert_to_tensor(max_lr_downsamp, dtype=self.dtype)

        self.__blur_kernel_max_size = int(np.int32(np.ceil(2.5 * max_lr_downsamp) / 2) * 2 + 1)
        self.__t1_blur_kernel_max_size = int(np.int32(np.ceil(2.5 * t1_init_downsamp * max_hr_downsamp) / 2) * 2 + 1)

        self._hr_output_dims = None
        self._t1_output_dims = None

        self.hr_blurrer = BlurLayer(self.__blur_kernel_max_size)
        self.lr_blurrer = BlurLayer(self.__blur_kernel_max_size)
        self.t1_blurrer = BlurLayer(self.__t1_blur_kernel_max_size)

        self.t1_downsampler = SamplingLayer(dsamp_rate=[self.t1_init_downsamp])
        self.mask_downsampler = SamplingLayer(dsamp_rate=[self.t1_init_downsamp], method="nearest")
        self.hr_downsampler = SamplingLayer(dsamp_rate=[1])
        self.lr_downsampler = SamplingLayer(dsamp_rate=[1])
        self.lr_upsampler = SamplingLayer(dsamp_rate=[1])

        self.lr_augmenter = AugmentationLayer(gamma_std=0.)
        self.hr_augmenter = AugmentationLayer(gamma_std=0.)
        self.t1_augmenter = AugmentationLayer()

    def get_config(self):
        config = super().get_config()

        config["t1_init_downsamp"] = self.t1_init_downsamp.numpy()
        config["max_hr_downsamp"] = self.max_hr_downsamp.numpy()
        config["min_hr_downsamp"] = self.min_hr_downsamp.numpy()
        config["max_lr_downsamp"] = self.max_lr_downsamp.numpy()

        config["apply_blurring"] = self.apply_blurring
        config["augment"] = self.augment

        return config

    def compute_output_shape(self, input_shape):

        if input_shape is None or input_shape[0] is None:
            return None

        kernel_penalty = 0

        if self.apply_blurring:
            kernel_penalty = self.__blur_kernel_max_size - 1

        output_shape = tuple([None] +
                             [tf.round((input_shape[0][1] - kernel_penalty) / self.max_hr_downsamp),
                              tf.round((input_shape[0][2] - kernel_penalty) / self.max_hr_downsamp),
                              tf.round((input_shape[0][3] - kernel_penalty) / self.max_hr_downsamp)] +
                             [input_shape[0][-1]])

        end_shape = [output_shape]

        if len(input_shape) > 1:

            kernel_penalty = 0

            if self.apply_blurring:
                kernel_penalty = self.__t1_blur_kernel_max_size - 1

            t1_output_shape = tuple([None] +
                                    [tf.round((input_shape[1][1] - kernel_penalty) /
                                              (self.max_hr_downsamp * self.t1_init_downsamp)),
                                     tf.round((input_shape[1][2] - kernel_penalty) /
                                              (self.max_hr_downsamp * self.t1_init_downsamp)),
                                     tf.round((input_shape[1][3] - kernel_penalty) /
                                              (self.max_hr_downsamp * self.t1_init_downsamp))] +
                                    [input_shape[1][-1]])

            assert (t1_output_shape[1] == output_shape[1] and
                    t1_output_shape[2] == output_shape[2] and
                    t1_output_shape[3] == output_shape[3]), "The output shapes of the dwi and T1w output have to match."

            end_shape.append(t1_output_shape)

        if len(input_shape) > 2:

            mask_output_shape = tuple([None] +
                                      [tf.round(input_shape[2][1] / self.max_hr_downsamp),
                                       tf.round(input_shape[2][2] / self.max_hr_downsamp),
                                       tf.round(input_shape[2][3] / self.max_hr_downsamp)] +
                                      [input_shape[2][-1]])

            assert (mask_output_shape[1] == output_shape[1] and
                    mask_output_shape[2] == output_shape[2] and
                    mask_output_shape[3] == output_shape[3]), "The mask and dwi output shapes do not match."

            end_shape.append(mask_output_shape)

        return end_shape

    def build(self, input_shape):

        if len(input_shape) == 1:
            self._hr_output_dims = self.compute_output_shape(input_shape)[0]

        else:
            output_shape = self.compute_output_shape(input_shape)
            self._hr_output_dims = output_shape[0]
            self._t1_output_dims = output_shape[1]

        return super().build(input_shape)

    @tf.function
    def __crop__(self, tensor, rate):
        target_img_shape = tf.cast(self._hr_output_dims[1:-1], rate.dtype) * rate
        crop_values = tf.cast((tensor.shape[1:-1] - tf.cast(tf.round(target_img_shape), dtype=tf.int32)),
                              dtype=self.dtype) / 2.0

        crop_values_f = tf.cast(tf.math.floor(crop_values), dtype=tf.int32)
        crop_values_c = tf.cast(tf.math.ceil(crop_values), dtype=tf.int32)

        tensor_dims = tf.shape(tensor)

        return tensor[:,
                      crop_values_f[0]:tensor_dims[1] - crop_values_c[0],
                      crop_values_f[1]:tensor_dims[2] - crop_values_c[1],
                      crop_values_f[2]:tensor_dims[3] - crop_values_c[2],
                      :]

    def call(self, inputs, *args, **kwargs):

        inputs_lr = inputs[0]
        inputs_hr = inputs[1]
        inputs_t1 = inputs[2]
        inputs_mask = inputs[3]

        lr_downsamp_rate = inputs[4]
        hr_downsamp_rate = inputs[5]

        if self.apply_blurring:
            inputs_hr = self.hr_blurrer([inputs_hr, hr_downsamp_rate])
            inputs_lr = self.lr_blurrer([inputs_lr, lr_downsamp_rate])

        self.hr_downsampler.dsamp_rate = tf.convert_to_tensor([hr_downsamp_rate])
        self.lr_downsampler.dsamp_rate = tf.convert_to_tensor([lr_downsamp_rate])

        inputs_hr = self.__crop__(inputs_hr, hr_downsamp_rate)
        inputs_lr = self.__crop__(inputs_lr, hr_downsamp_rate)

        inputs_hr = self.hr_downsampler(inputs_hr)
        inputs_lr = self.lr_downsampler(inputs_lr)

        self.lr_upsampler.dsamp_rate = tf.divide(inputs_lr.shape[1:-1], inputs_hr.shape[1:-1])

        if inputs_t1 is not None:

            t1_downsamp_rate = self.t1_init_downsamp * hr_downsamp_rate

            if self.apply_blurring:
                inputs_t1 = self.t1_blurrer([inputs_t1, t1_downsamp_rate])

            inputs_t1 = self.__crop__(inputs_t1, t1_downsamp_rate)
            self.t1_downsampler.dsamp_rate = tf.divide(inputs_t1.shape[1:-1], inputs_hr.shape[1:-1])

            inputs_t1 = self.t1_downsampler(inputs_t1)

            if self.augment:
                inputs_t1 = self.t1_augmenter(inputs_t1)

        if inputs_mask is not None:
            inputs_mask = self.__crop__(inputs_mask, hr_downsamp_rate)
            self.mask_downsampler.dsamp_rate = tf.convert_to_tensor([hr_downsamp_rate])

            inputs_mask = self.mask_downsampler(inputs_mask)

        if self.augment:
            inputs_hr = self.hr_augmenter(inputs_hr)
            inputs_lr = self.lr_augmenter(inputs_lr)  # Do at lower space to simulate noise

        inputs_lr = self.lr_upsampler(inputs_lr)

        outputs = [inputs_lr, inputs_hr]

        if inputs_t1 is not None:
            outputs.append(inputs_t1)

        if inputs_mask is not None:
            outputs.append(inputs_mask)

        return outputs


class RandomSamplerLayer(SamplerLayer):

    def __init__(self, max_hr_downsamp, max_lr_downsamp, t1_init_downsamp=1.,
                 min_hr_downsamp=1.,
                 static_hr=False,
                 static_lr=False,
                 apply_blurring=True,
                 augment=True, *args, **kwargs):

        super().__init__(max_hr_downsamp, max_lr_downsamp, t1_init_downsamp, min_hr_downsamp, apply_blurring, augment,
                         *args, **kwargs)

        self.static_hr = static_hr
        self.static_lr = static_lr

    def get_config(self):
        config = super().get_config()

        config["static_hr"] = self.static_hr
        config["static_lr"] = self.static_lr

        return config

    def call(self, inputs, *args, **kwargs):

        inputs_hr = inputs[0]
        inputs_lr = inputs[0]
        inputs_t1 = None if len(inputs) < 2 else inputs[1]
        inputs_mask = None if len(inputs) < 3 else inputs[2]

        static_hr = self.static_hr or inputs[0].shape[0] is None  # If batch is none just return target shape.
        static_lr = self.static_lr or inputs[0].shape[0] is None  # If batch is none just return target shape.

        if static_hr:
            hr_downsamp_rate = self.max_hr_downsamp
        else:
            hr_downsamp_rate = tf.random.uniform([], self.min_hr_downsamp, self.max_hr_downsamp, dtype=self.dtype)

        if static_lr:
            lr_downsamp_rate = self.max_lr_downsamp
        else:
            lr_downsamp_rate = tf.random.uniform([], hr_downsamp_rate, self.max_lr_downsamp, dtype=self.dtype)

        return super().call([inputs_lr, inputs_hr, inputs_t1, inputs_mask,
                             lr_downsamp_rate, hr_downsamp_rate], *args, **kwargs)


class SameRateSamplerLayer(SamplerLayer):

    def __init__(self, max_hr_downsamp,
                 downsamp_rate,
                 t1_init_downsamp=1.,
                 min_hr_downsamp=1.,
                 static=False,
                 apply_blurring=True,
                 augment=True, *args, **kwargs):

        max_lr_downsamp = max_hr_downsamp * downsamp_rate

        super().__init__(max_hr_downsamp, max_lr_downsamp, t1_init_downsamp, min_hr_downsamp, apply_blurring, augment,
                         *args, **kwargs)

        self.downsamp_rate = downsamp_rate
        self.static = static

    def get_config(self):
        config = super().get_config()

        config["downsamp_rate"] = self.downsamp_rate
        config["static"] = self.static

        return config

    def call(self, inputs, *args, **kwargs):

        inputs_hr = inputs[0]
        inputs_lr = inputs[0]
        inputs_t1 = None if len(inputs) < 2 else inputs[1]
        inputs_mask = None if len(inputs) < 3 else inputs[2]

        static = self.static or inputs[0].shape[0] is None  # If batch is none just return target shape.

        if static:
            hr_downsamp_rate = self.max_hr_downsamp
        else:
            hr_downsamp_rate = tf.random.uniform([], self.min_hr_downsamp, self.max_hr_downsamp, dtype=self.dtype)

        lr_downsamp_rate = hr_downsamp_rate * self.downsamp_rate

        return super().call([inputs_lr, inputs_hr, inputs_t1, inputs_mask,
                             lr_downsamp_rate, hr_downsamp_rate], *args, **kwargs)


class ListSamplerLayer(SamplerLayer):

    def __init__(self,
                 hr_downsamp_rates,
                 lr_downsamp_rates,
                 t1_init_downsamp=1.,
                 apply_blurring=True,
                 augment=True, *args, **kwargs):

        super().__init__(max(hr_downsamp_rates), max(lr_downsamp_rates),
                         t1_init_downsamp, min(hr_downsamp_rates), apply_blurring, augment,
                         *args, **kwargs)

        self.hr_downsamp_rates = hr_downsamp_rates
        self.lr_downsamp_rates = lr_downsamp_rates

    def get_config(self):
        config = super().get_config()

        config["hr_downsamp_rates"] = self.hr_downsamp_rates
        config["lr_downsamp_rates"] = self.lr_downsamp_rates

        return config

    def call(self, inputs, *args, **kwargs):

        inputs_hr = inputs[0]
        inputs_lr = inputs[0]
        inputs_t1 = None if len(inputs) < 2 else inputs[1]
        inputs_mask = None if len(inputs) < 3 else inputs[2]

        if len(self.hr_downsamp_rates) == 1 or inputs[0].shape[0] is None:  # If batch is none just return target shape.
            hr_downsamp_rate = self.max_hr_downsamp
        else:
            hr_downsamp_rate = tf.convert_to_tensor(random.choice(self.hr_downsamp_rates),
                                                    dtype=self.dtype)

        if len(self.lr_downsamp_rates) == 1 or inputs[0].shape[0] is None:  # If batch is none just return target shape.
            lr_downsamp_rate = self.max_hr_downsamp
        else:
            lr_downsamp_rate = tf.convert_to_tensor(random.choice(self.lr_downsamp_rates),
                                                    dtype=self.dtype)

        return super().call([inputs_lr, inputs_hr, inputs_t1, inputs_mask,
                             lr_downsamp_rate, hr_downsamp_rate], *args, **kwargs)


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

