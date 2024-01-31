import tensorflow as tf
# from tensorflow.keras.layers import *
from keras.layers import *

import keras.backend as K


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


def vdsr(ipatch_size=11):

    input_layer = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, 6], name='input')

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

    return tf.keras.Model(input_layer, output)
