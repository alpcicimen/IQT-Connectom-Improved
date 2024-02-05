import tensorflow as tf
import keras

from keras.layers import *
from typing import Tuple

import keras.backend as K

from IQT.layers import *


class DepthToSpaceLayer(Layer):

    def __init__(self, upsampling_rate=2):
        super().__init__()
        self.upsampling_rate = upsampling_rate

    def call(self, inputs):
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

    def call(self, inputs):
        batch_size, dim_i, dim_j, dim_k, c = K.int_shape(inputs)

        assert (dim_i % (self.upsampling_rate) == 0)  # Number must be exactly divisible by 8
        assert (dim_j % (self.upsampling_rate) == 0)  # Number must be exactly divisible by 8
        assert (dim_k % (self.upsampling_rate) == 0)  # Number must be exactly divisible by 8

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


def simple_generator(input_ch, output_ch, ipatch_size=11, f_num=50, layer_num=1, ds=2):
    input_layer = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input')

    model = tf.keras.Sequential()

    model.add(Conv3D(kernel_size=(3, 3, 3), filters=f_num, padding='valid'))
    model.add(ReLU())

    for n in range(layer_num):

        if n == 0:
            k = 1
        else:
            # fn = 2*f_num
            k = 3

        model.add(Conv3D(kernel_size=(k, k, k), filters=2 * f_num, padding='valid'))
        model.add(ReLU())

    model.add(Conv3D(kernel_size=(3, 3, 3), filters=output_ch, padding='valid'))

    return tf.keras.Model(input_layer, model(input_layer))


def vdsr(ipatch_size: int | Tuple[int, int, int]):
    if type(ipatch_size) is int:
        (x_size, y_size, z_size) = (ipatch_size, ipatch_size, ipatch_size)
    else:
        (x_size, y_size, z_size) = ipatch_size

    input_layer = tf.keras.layers.Input(shape=[x_size, y_size, z_size, 6], name='input')

    model = tf.keras.Sequential(layers=[Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                        ReLU(),
                                        Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                        ReLU(),
                                        Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                        ReLU(),
                                        Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                        ReLU(),
                                        Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                        ReLU(),
                                        Conv3D(kernel_size=(3, 3, 3), filters=6, padding='same')])

    output = input_layer + model(input_layer)

    return keras.Model(input_layer, output)


def calc_output(input_dim, filter_size, padding=0, stride=1):
    return (input_dim - filter_size + 2 * padding) / stride + 1


def vdsr_t1(ipatch_size: int | Tuple[int, int, int],
            t1_patch_size: int | Tuple[int, int, int]):
    if type(ipatch_size) is int:
        (x_size, y_size, z_size) = (ipatch_size, ipatch_size, ipatch_size)
    else:
        (x_size, y_size, z_size) = ipatch_size

    input_layer = keras.layers.Input(shape=[x_size, y_size, z_size, 6], name='input')

    t1_layer = keras.layers.Input(shape=[t1_patch_size, t1_patch_size, t1_patch_size, 1], name='t1_input')

    model = keras.Sequential(layers=[Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                     ReLU(),
                                     Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                     ReLU(),
                                     Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                     ReLU(),
                                     Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                     ReLU(),
                                     Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
                                     ReLU(),
                                     Conv3D(kernel_size=(3, 3, 3), filters=6, padding='same')])

    model_t1 = keras.Sequential()

    ksize = 5

    t1_size = t1_patch_size

    while t1_size > ipatch_size:
        fsize = 64 if calc_output(t1_size, filter_size=ksize) > ipatch_size else 6

        model_t1.add(keras.Sequential([
            Conv3D(kernel_size=ksize, filters=fsize, padding='valid'),
            ReLU()]))
        t1_size = calc_output(t1_size, filter_size=ksize)

    output = input_layer + model(input_layer) + model_t1(t1_layer)

    return tf.keras.Model([input_layer, t1_layer], output)


def unet3d(ipatch_size):
    i_layer = keras.layers.Input(shape=[ipatch_size,
                                        ipatch_size,
                                        ipatch_size, 6], name='input')

    t1_layer = keras.layers.Input(shape=[ipatch_size * 2,
                                         ipatch_size * 2,
                                         ipatch_size * 2, 1], name='input_t1')

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6*4, padding='same'),
                             ReLU(),
                             Conv3D(kernel_size=5, filters=6*4, padding='same'),
                             ReLU()])(i_layer)

    d_layer = unet_downsample_layer(conv_input, kernel_size=5, filter_size=6 * 4 * 4)
    d_layer2 = unet_downsample_layer(d_layer, kernel_size=5, filter_size=6 * 4 * 4 * 4)

    u_layer1 = unet_upsample_layer(d_layer2, concat_layer=d_layer, filter_size=6 * 4 * 4, kernel_size=5)
    u_layer2 = unet_upsample_layer(u_layer1, concat_layer=conv_input, filter_size=6 * 4, kernel_size=5)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same')(u_layer2)

    return keras.Model([i_layer, t1_layer], o_layer)


def unet3d_t1(ipatch_size):
    i_layer = keras.layers.Input(shape=[ipatch_size,
                                        ipatch_size,
                                        ipatch_size, 6], name='input')

    t1_layer = keras.layers.Input(shape=[ipatch_size * 2,
                                         ipatch_size * 2,
                                         ipatch_size * 2, 1], name='input_t1')

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6*4, padding='same'),
                             ReLU(),
                             Conv3D(kernel_size=5, filters=6*4, padding='same'),
                             ReLU()])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=5, strides=2, filters=6*4, padding='same'),
                           ReLU(),
                           Conv3D(kernel_size=5, filters=6*4, padding='same'),
                           ReLU(),
                           Conv3D(kernel_size=5, filters=6*4, padding='same'),
                           ReLU()])(t1_layer)

    t1_d_layer1 = unet_downsample_layer(t1_input, kernel_size=5, filter_size=6 * 4 * 4)
    # t1_d_layer2 = unet_downsample_layer(t1_d_layer1, kernel_size=5, filter_size=6 * 4 * 4 * 4)

    d_layer1 = unet_downsample_layer(conv_input, kernel_size=5, filter_size=6 * 4 * 4)
    d_layer2 = unet_downsample_layer(d_layer1, kernel_size=5, filter_size=6 * 4 * 4 * 4)

    u_layer1 = unet_upsample_layer(d_layer2,
                                   concat_layer=(t1_d_layer1 + d_layer1), filter_size=6 * 4 * 4, kernel_size=5)
    u_layer2 = unet_upsample_layer(u_layer1,
                                   concat_layer=(t1_input+conv_input), filter_size=6 * 4, kernel_size=5)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same')(u_layer2)

    return keras.Model([i_layer, t1_layer], o_layer)