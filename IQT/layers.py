import keras.backend as K
import tensorflow as tf

from keras.layers import *
from keras import Sequential


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
                             residual=False,
                             final_activation=True):

    dsamp = Conv3D(filters=filter_size,
                   kernel_size=kernel_size,
                   padding="same",
                   strides=2)(prev_layer)

    conv = []

    for n in range(rep_layers):
        conv.append(BatchNormalization())
        conv.append(ELU())
        conv.append(Conv3D(filters=filter_size,
                           kernel_size=3,
                           padding="same"))

    if residual:
        pre_act = Sequential(conv)(dsamp) + dsamp
    else:
        pre_act = Sequential(conv)(dsamp)

    if final_activation:
        return ELU()(BatchNormalization()(pre_act))
    else:
        return pre_act


class VolumeAttentionLayer3D(Layer):

    def __init__(self, dims):
        super().__init__()

        self.dims = dims[:-1]
        self.channels = dims[-1]

        self.__dense = [Dense(self.channels), Dense(self.channels), Dense(self.channels)]
        self.__attention = Attention()

    def call(self, inputs, *args, **kwargs):

        queries = inputs[0]
        values = inputs[1] if len(inputs) > 1 else queries
        keys = inputs[2] if len(inputs) > 2 else values

        values = Reshape((self.dims[1] * self.dims[2] * self.dims[3], self.channels))(values)
        keys = Reshape((self.dims[1] * self.dims[2] * self.dims[3], self.channels))(keys)
        queries = Reshape((self.dims[1] * self.dims[2] * self.dims[3], self.channels))(queries)

        result = self.__attention([queries, values, keys])

        result = Reshape((self.dims[1], self.dims[2], self.dims[3], self.channels))(result)

        return result


def unet_attention_fusion(prev_layer,
                          filter_size,
                          kernel_size=3,
                          attention_layer=None,
                          concat_layer=None,
                          rep_layers=2,
                          residual=False):

    layer = Sequential([BatchNormalization(),
                        ELU(),
                        Conv3DTranspose(filters=filter_size,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        padding="same")])

    if concat_layer is not None or attention_layer is not None:
        layer.add(BatchNormalization())
        layer.add(ELU())

    conv = layer(prev_layer)

    if attention_layer is not None:
        conv = VolumeAttentionLayer3D(attention_layer.shape)([conv, attention_layer])
        conv = ELU()(LayerNormalization()(conv))

    if concat_layer is not None:
        conv = Concatenate(axis=4)([conv, concat_layer])

        conv = Conv3D(filters=filter_size,
                      kernel_size=1,
                      padding='same')(conv)

    pre_res = conv

    layer = []

    for _ in range(rep_layers):
        layer.append(BatchNormalization())
        layer.append(ELU())
        layer.append(Conv3D(filters=filter_size,
                            kernel_size=3,
                            padding="same"))

    if residual:
        return Sequential(layer)(pre_res) + pre_res
    else:
        return Sequential(layer)(pre_res)

