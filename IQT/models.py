from typing import Tuple

import keras
import keras.layers as KL
import os
from keras.activations import tanh

from layers import *


def config_model(model_type,
                 patch_size=16,
                 t1_patch_size=16,
                 weights_dir: None | str | os.PathLike[str] = None):

    match model_type:

        case "UNet-T1":
            model = unet3d_t1_v2(patch_size, t1_patch_size)

        case "UNet-PreFusion":
            model = unet3d_pre_fusion_v2(patch_size, t1_patch_size)

        case "UNet":
            model = unet3d_not1_v2(patch_size)

        case _:
            raise ValueError(f"No model configuration for \"{model_type}\" found!")

    if weights_dir is not None:
        model.load_weights(weights_dir)

    return model


def simple_generator(input_ch, output_ch, ipatch_size=11, f_num=50, layer_num=1, ds=2):
    input_layer = Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input')

    model = Sequential()

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

    return keras.Model(input_layer, model(input_layer))


def basic_model(ipatch_size: int):

    input_layer = Input(shape=[ipatch_size, ipatch_size, ipatch_size, 6], name='input')
    input_layer_t1 = Input(shape=[ipatch_size, ipatch_size, ipatch_size, 1], name='input_t1')

    output_layer = input_layer

    return keras.Model([input_layer, input_layer_t1], output_layer)


def vdsr(ipatch_size: int | Tuple[int, int, int]):
    if type(ipatch_size) is int:
        (x_size, y_size, z_size) = (ipatch_size, ipatch_size, ipatch_size)
    else:
        (x_size, y_size, z_size) = ipatch_size

    input_layer = Input(shape=[x_size, y_size, z_size, 6], name='input')

    model = Sequential(layers=[Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
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

    input_layer = Input(shape=[x_size, y_size, z_size, 6], name='input')

    t1_layer = Input(shape=[t1_patch_size, t1_patch_size, t1_patch_size, 1], name='t1_input')

    model = Sequential(layers=[Conv3D(kernel_size=(3, 3, 3), filters=64, padding='same'),
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

    seq = []

    ksize = 5

    t1_size = t1_patch_size

    while t1_size > ipatch_size:
        fsize = 64 if calc_output(t1_size, filter_size=ksize) > ipatch_size else 6

        seq.append(Conv3D(kernel_size=ksize, filters=fsize, padding='valid'))
        seq.append(ReLU())
        t1_size = calc_output(t1_size, filter_size=ksize)

    model_t1 = Sequential(seq)

    output = input_layer + tanh(model(input_layer)) + tanh(model_t1(t1_layer))

    return keras.Model([input_layer, t1_layer], output)


def unet3d(ipatch_size):

    i_layer = Input(shape=[ipatch_size,
                           ipatch_size,
                           ipatch_size, 6], name='input')

    t1_layer = Input(shape=[ipatch_size * 2,
                            ipatch_size * 2,
                            ipatch_size * 2, 1], name='input_t1')

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6*4, padding='same'),
                             ReLU(),
                             Conv3D(kernel_size=5, filters=6*4, padding='same'),
                             ReLU()])(i_layer)

    d_layer = unet_downsample_layer(conv_input, kernel_size=5, filter_size=6 * 6 * 6)
    d_layer2 = unet_downsample_layer(d_layer, kernel_size=5, filter_size=6 * 6 * 6 * 6)

    u_layer1 = unet_upsample_layer(d_layer2, concat_layer=d_layer, filter_size=6 * 6 * 6, kernel_size=5)
    u_layer2 = unet_upsample_layer(u_layer1, concat_layer=conv_input, filter_size=6 * 6, kernel_size=5)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same')(u_layer2)

    return keras.Model([i_layer, t1_layer], o_layer)


def unet3d_t1_v1(ipatch_size,
                 tw_patch_size=None):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    __tw_patch_size = 2 * ipatch_size if tw_patch_size is None else tw_patch_size

    t1_layer = KL.Input(shape=[__tw_patch_size,
                               __tw_patch_size,
                               __tw_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6 * 8, padding='same'),
                             ReLU(),
                             Conv3D(kernel_size=5, filters=6 * 8, padding='same'),
                             ReLU()])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=5,
                                  strides=(2 if tw_patch_size is None else 1),  # Do not apply stride on custom T1 sizes
                                  filters=6 * 8, padding='same'),
                           ReLU(),
                           Conv3D(kernel_size=5, filters=6 * 8, padding='same'),
                           ReLU()])(t1_layer)

    t1_d_layer1 = unet_downsample_layer(t1_input, kernel_size=5, filter_size=6 * 8 * 8)
    t1_d_layer2 = unet_downsample_layer(t1_d_layer1, kernel_size=5, filter_size=6 * 8 * 8 * 8)

    d_layer1 = unet_downsample_layer(conv_input, kernel_size=5, filter_size=6 * 8 * 8)
    d_layer2 = unet_downsample_layer(d_layer1, kernel_size=5, filter_size=6 * 8 * 8 * 8)

    d_layer_n = Conv3D(kernel_size=3, filters=6 * 8 * 8 * 8, padding='same')(t1_d_layer2 + d_layer2)

    u_layer1 = unet_upsample_layer(d_layer_n,
                                   concat_layer=Conv3D(kernel_size=3, filters=6 * 8 * 8, padding='same')
                                   (t1_d_layer1 + d_layer1),
                                   filter_size=6 * 8 * 8, kernel_size=5)
    u_layer2 = unet_upsample_layer(u_layer1,
                                   concat_layer=Conv3D(kernel_size=3, filters=6 * 8, padding='same')
                                   (t1_input + conv_input),
                                   filter_size=6 * 8, kernel_size=5)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same', dtype=tf.float32)(u_layer2)

    return keras.Model([i_layer, t1_layer], o_layer)


def unet3d_t1_v2(ipatch_size,
                 tw_patch_size=None):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    __tw_patch_size = 2 * ipatch_size if tw_patch_size is None else tw_patch_size

    t1_layer = KL.Input(shape=[__tw_patch_size,
                               __tw_patch_size,
                               __tw_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6 * 6, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=5, filters=6 * 6, padding='same')])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=5,
                                  strides=(2 if tw_patch_size is None else 1),  # Do not apply stride on custom T1 sizes
                                  filters=1 * 6, padding='same'),
                           LeakyReLU(),
                           Conv3D(kernel_size=5, filters=1 * 6, padding='same')])(t1_layer)

    t1_d_layer1 = unet_downsample_layer_v2(LeakyReLU()(t1_input), kernel_size=5, filter_size=1 * 6 * 6)
    t1_d_layer2 = unet_downsample_layer_v2(LeakyReLU()(t1_d_layer1), kernel_size=5, filter_size=1 * 6 * 6 * 6)

    d_layer1 = unet_downsample_layer_v2(LeakyReLU()(conv_input), kernel_size=5, filter_size=6 * 6 * 6)
    d_layer2 = unet_downsample_layer_v2(LeakyReLU()(d_layer1), kernel_size=5, filter_size=6 * 6 * 6 * 6)

    d_layer_n = Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same')(
        LeakyReLU()(Concatenate(axis=4)([t1_d_layer2, d_layer2])))

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same'),
                            LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same')
                            ])(d_layer_n)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=concatenate([d_layer1, t1_d_layer1], axis=4),
                                      filter_size=6 * 6 * 6, kernel_size=5)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=concatenate([conv_input, t1_input], axis=4),
                                      filter_size=6 * 6, kernel_size=5)

    o_layer = LeakyReLU()(u_layer2)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same', dtype=tf.float32)(o_layer)

    o_layer = tanh(o_layer) + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-T1')


def unet3d_pre_fusion_v2(ipatch_size, t1_patch_size):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    t1_layer = KL.Input(shape=[t1_patch_size,
                               t1_patch_size,
                               t1_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6 * 6, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=5, filters=6 * 6, padding='same')])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=5, filters=1 * 6, padding='same'),
                           LeakyReLU(),
                           Conv3D(kernel_size=5, filters=1 * 6, padding='same')])(t1_layer)

    model_input = Concatenate(axis=4)([conv_input, t1_input])

    d_layer1 = unet_downsample_layer_v2(LeakyReLU()(model_input), kernel_size=5, filter_size=7 * 6 * 6)
    d_layer2 = unet_downsample_layer_v2(LeakyReLU()(d_layer1), kernel_size=5, filter_size=7 * 6 * 6 * 6)

    d_layer_n = Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same')(LeakyReLU()(d_layer2))

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same'),
                            LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same')
                            ])(d_layer_n)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=d_layer1,
                                      filter_size=6 * 6 * 6, kernel_size=5)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=model_input,
                                      filter_size=6 * 6, kernel_size=5)

    o_layer = LeakyReLU()(u_layer2)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same', dtype=tf.float32)(o_layer)

    o_layer = tanh(o_layer) + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-NoT1')


def unet3d_not1_v2(ipatch_size):

    i_layer = KL.Input(shape=[ipatch_size,
                                        ipatch_size,
                                        ipatch_size, 6], name='input', dtype=tf.float32)

    __tw_patch_size = ipatch_size

    t1_layer = KL.Input(shape=[__tw_patch_size,
                               __tw_patch_size,
                               __tw_patch_size, 1], name='input_t1', dtype=tf.float32) # T1 input disconnected

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6 * 6, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=5, filters=6 * 6, padding='same')])(i_layer)

    d_layer1 = unet_downsample_layer_v2(LeakyReLU()(conv_input), kernel_size=5, filter_size=6 * 6 * 6)
    d_layer2 = unet_downsample_layer_v2(LeakyReLU()(d_layer1), kernel_size=5, filter_size=6 * 6 * 6 * 6)

    d_layer_n = Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same')(LeakyReLU()(d_layer2))

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same'),
                            LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 6 * 6 * 6, padding='same')
                            ])(d_layer_n)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=d_layer1,
                                      filter_size=6 * 6 * 6, kernel_size=5)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=conv_input,
                                      filter_size=6 * 6, kernel_size=5)

    o_layer = LeakyReLU()(u_layer2)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same', dtype=tf.float32)(o_layer)

    o_layer = tanh(o_layer) + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-NoT1')


def unet3d_t1_v3(ipatch_size,
                 tw_patch_size=None,
                 num_layers=3,
                 ch_mult_per_layer=4,
                 num_rep_layers=2):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    __tw_patch_size = 2 * ipatch_size if tw_patch_size is None else tw_patch_size

    t1_layer = KL.Input(shape=[__tw_patch_size,
                               __tw_patch_size,
                               __tw_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=3, filters=6 * (ch_mult_per_layer ** 1), padding='same'),
                             ELU(),
                             Conv3D(kernel_size=3, filters=6 * (ch_mult_per_layer ** 1), padding='same')])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=3,
                                  strides=(2 if tw_patch_size is None else 1),  # Do not apply stride on custom T1 sizes
                                  filters=6 * (ch_mult_per_layer ** 1), padding='same'),
                           ELU(),
                           Conv3D(kernel_size=3, filters=6 * ch_mult_per_layer ** 1, padding='same')])(t1_layer)

    t1_layers = []
    d_layers = []

    t1_layers.append(t1_input)
    d_layers.append(conv_input)

    for n in range(1, num_layers+1):
        t1_layers.append(unet_downsample_layer_v3(ELU()(t1_layers[-1]),
                                                  kernel_size=3,
                                                  filter_size=6 * (ch_mult_per_layer ** (n+1)),
                                                  rep_layers=num_rep_layers))
        d_layers.append(unet_downsample_layer_v3(ELU()(d_layers[-1]),
                                                 kernel_size=3,
                                                 filter_size=6 * (ch_mult_per_layer ** (n+1)),
                                                 rep_layers=num_rep_layers))

    d_layer_n = (d_layers[-1] + t1_layers[-1])

    for _ in range(num_rep_layers):
        d_layer_n = Conv3D(kernel_size=3,
                           filters=6 * (ch_mult_per_layer ** (num_layers + 1)),
                           padding='same')(ELU()(d_layer_n))

    for n in range(1, num_layers+1):
        d_layer_n = unet_upsample_layer_v3(d_layer_n,
                                           concat_layer=concatenate([d_layers[-1-n], t1_layers[-1-n]], axis=4),
                                           filter_size=6 * (ch_mult_per_layer ** (num_layers - n + 1)),
                                           kernel_size=3,
                                           rep_layers=num_rep_layers)

    o_layer = ELU()(d_layer_n)

    o_layer = Conv3D(kernel_size=3, filters=6, padding='same', dtype=tf.float32)(o_layer)

    o_layer = tanh(o_layer) + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-T1')
