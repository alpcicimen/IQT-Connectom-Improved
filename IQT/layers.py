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

        result = self.__attention([queries, values, keys])

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

