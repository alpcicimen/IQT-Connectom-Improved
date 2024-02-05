import keras

import tensorflow as tf

from keras.layers import *
from keras import Sequential


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
                   kernel_size=kernel_size,
                   padding="same"),
            ReLU()
        ]))

    return conv(prev_layer)


def unet_upsample_layer(prev_layer,
                        filter_size,
                        concat_layer=None,
                        kernel_size=3,
                        rep_layers=2):

    layer = Sequential([Conv3DTranspose(filters=filter_size,
                                        kernel_size=kernel_size*2,
                                        strides=2,
                                        padding="same"),
                       ReLU()])

    conv = layer(prev_layer)

    if concat_layer is not None:
        conv = concatenate([conv, concat_layer], 4)

    layer = Sequential()

    for _ in range(rep_layers):
        layer.add(Sequential([
            Conv3D(filters=filter_size,
                   kernel_size=kernel_size,
                   padding="same"),
            ReLU()
        ]))

    return layer(conv)


# if __name__ == '__main__':
#
#     i_layer = keras.layers.Input(shape=[32, 32, 32, 1], name='input')
#
#     d_layer = unet_downsample_layer(i_layer, kernel_size=5, filter_size=4)
#     d_layer2 = unet_downsample_layer(d_layer, kernel_size=5, filter_size=8)
#
#     u_layer1 = unet_upsample_layer(d_layer2, filter_size=4, concat_layer=d_layer, kernel_size=5)
#     f_layer = unet_upsample_layer(u_layer1, filter_size=1, concat_layer=i_layer, kernel_size=5)
#
#     print(keras.Model(i_layer, f_layer).summary())

