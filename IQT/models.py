from typing import Tuple

import keras
import keras.layers as KL
import os
from keras.activations import tanh, sigmoid

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

        case "UNet-Attention":
            model = unet3d_t1_attention(patch_size, t1_patch_size)

        case _:
            raise ValueError(f"No model configuration for \"{model_type}\" found!")

    if weights_dir is not None:
        model.load_weights(weights_dir)

    return model


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


def unet3d_t1_v2(ipatch_size,
                 tw_patch_size=None):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    __tw_patch_size = 2 * ipatch_size if tw_patch_size is None else tw_patch_size

    t1_layer = KL.Input(shape=[__tw_patch_size,
                               __tw_patch_size,
                               __tw_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                             LeakyReLU()])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                           LeakyReLU(),
                           Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                           LeakyReLU()])(t1_layer)

    t1_d_layer1 = unet_downsample_layer_v2(t1_input,
                                           kernel_size=3, filter_size=6 * 4 * 4)
    t1_d_layer2 = unet_downsample_layer_v2(t1_d_layer1,
                                           kernel_size=3, filter_size=6 * 4 * 4 * 4,
                                           last_layer=True)

    d_layer1 = unet_downsample_layer_v2(conv_input,
                                        kernel_size=3, filter_size=6 * 4 * 4)
    d_layer2 = unet_downsample_layer_v2(d_layer1,
                                        kernel_size=3, filter_size=6 * 4 * 4 * 4,
                                        last_layer=True)

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 4 * 4 * 4, padding='same'),
                            LeakyReLU(),
                            BatchNormalization()])(
        Average()([t1_d_layer2, d_layer2]))

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=Concatenate(axis=4)([d_layer1, t1_d_layer1]),
                                      filter_size=6 * 4 * 4, kernel_size=3)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=Concatenate(axis=4)([conv_input, t1_input]),
                                      filter_size=6 * 4, kernel_size=3)

    o_layer = Conv3D(kernel_size=3, filters=6 * 4, padding='same')(u_layer2)
    o_layer = LeakyReLU()(o_layer)

    o_layer = Conv3D(kernel_size=1, filters=6, padding='same', activation=sigmoid,
                     dtype=tf.float32)(o_layer)

    # o_layer = o_layer + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-T1')


def unet3d_t1_attention(ipatch_size,
                        tw_patch_size=None):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    __tw_patch_size = 2 * ipatch_size if tw_patch_size is None else tw_patch_size

    t1_layer = KL.Input(shape=[__tw_patch_size,
                               __tw_patch_size,
                               __tw_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=1, filters=6 * 4, padding='same'),
                             BatchNormalization(),
                             ELU(),
                             Conv3D(kernel_size=5, filters=6 * 4, padding='same'),
                             BatchNormalization(),
                             ELU(),])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=1, filters=6 * 4, padding='same'),
                           BatchNormalization(),
                           ELU(),
                           Conv3D(kernel_size=5, filters=6 * 4, padding='same'),
                           BatchNormalization(),
                           ELU(),])(t1_layer)

    t1_d_layer1 = unet_downsample_layer_v3(t1_input,
                                           kernel_size=5, filter_size=6 * 4 ** 2)
    t1_d_layer2 = unet_downsample_layer_v3(t1_d_layer1,
                                           kernel_size=5, filter_size=6 * 4 ** 3)
    t1_d_layer3 = unet_downsample_layer_v3(t1_d_layer2,
                                           kernel_size=5, filter_size=6 * 4 ** 4,
                                           final_activation=False)

    d_layer1 = unet_downsample_layer_v3(conv_input,
                                        kernel_size=5, filter_size=6 * 4 ** 2)
    d_layer2 = unet_downsample_layer_v3(d_layer1,
                                        kernel_size=5, filter_size=6 * 4 ** 3)
    d_layer3 = unet_downsample_layer_v3(d_layer2,
                                        kernel_size=5, filter_size=6 * 4 ** 4,
                                        final_activation=False)

    d_layer_n = Sequential(
        [BatchNormalization(),
         ELU(),
         Conv3D(kernel_size=3, filters=6 * 4 ** 4, padding='same')]
    )(Average()([t1_d_layer3, d_layer3]))

    u_layer1 = unet_attention_fusion(d_layer_n,
                                     # concat_layer=Concatenate()([d_layer2, t1_d_layer2]),
                                     concat_layer=d_layer2,
                                     attention_layer=t1_d_layer2,
                                     filter_size=6 * 4 ** 3, kernel_size=5)

    u_layer2 = unet_attention_fusion(u_layer1,
                                     # concat_layer=Concatenate()([d_layer1, t1_d_layer1]),
                                     concat_layer=d_layer1,
                                     attention_layer=t1_d_layer1,
                                     filter_size=6 * 4 ** 2, kernel_size=5)
    u_layer3 = unet_attention_fusion(u_layer2,
                                     # concat_layer=Concatenate()([conv_input, t1_input]),
                                     concat_layer=conv_input,
                                     attention_layer=t1_input,
                                     filter_size=6 * 4 ** 1, kernel_size=5)

    o_layer = ELU()(u_layer3)

    o_layer = Conv3D(kernel_size=1, filters=6, padding='same', dtype=tf.float32)(o_layer)

    o_layer = tanh(o_layer) + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-T1')


def unet3d_pre_fusion_v2(ipatch_size, t1_patch_size):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    t1_layer = KL.Input(shape=[t1_patch_size,
                               t1_patch_size,
                               t1_patch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=5, filters=6 * 4, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=5, filters=6 * 4, padding='same')])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=5, filters=1 * 4, padding='same'),
                           LeakyReLU(),
                           Conv3D(kernel_size=5, filters=1 * 4, padding='same')])(t1_layer)

    model_input = Concatenate(axis=4)([conv_input, t1_input])

    d_layer1 = unet_downsample_layer_v2(LeakyReLU()(model_input), kernel_size=5, filter_size=7 * 4 * 4)
    d_layer2 = unet_downsample_layer_v2(LeakyReLU()(d_layer1), kernel_size=5, filter_size=7 * 4 * 4 * 4)

    d_layer_n = Conv3D(kernel_size=3, filters=6 * 4 * 4 * 4, padding='same')(LeakyReLU()(d_layer2))

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 4 * 4 * 4, padding='same'),
                            LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 4 * 4 * 4, padding='same')
                            ])(d_layer_n)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=d_layer1,
                                      filter_size=6 * 4 * 4, kernel_size=5)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=model_input,
                                      filter_size=6 * 4, kernel_size=5)

    o_layer = LeakyReLU()(u_layer2)

    o_layer = Conv3D(kernel_size=5, filters=6, padding='same', dtype=tf.float32)(o_layer)

    o_layer = tanh(o_layer) + i_layer

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-NoT1')


def unet3d_not1_v2(ipatch_size):

    i_layer = KL.Input(shape=[ipatch_size,
                              ipatch_size,
                              ipatch_size, 6], name='input', dtype=tf.float32)

    t1_layer = KL.Input(shape=[ipatch_size,
                               ipatch_size,
                               ipatch_size, 1], name='input_t1', dtype=tf.float32)

    conv_input = Sequential([Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                             LeakyReLU()])(i_layer)

    d_layer1 = unet_downsample_layer_v2(conv_input,
                                        kernel_size=3, filter_size=6 * 4 * 4)
    d_layer2 = unet_downsample_layer_v2(d_layer1,
                                        kernel_size=3, filter_size=6 * 4 * 4 * 4,
                                        last_layer=True)

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=6 * 4 * 4 * 4, padding='same'),
                            LeakyReLU(),
                            BatchNormalization()])(d_layer2)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=d_layer1,
                                      filter_size=6 * 4 * 4, kernel_size=3)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=conv_input,
                                      filter_size=6 * 4, kernel_size=3)

    o_layer = Conv3D(kernel_size=3, filters=6 * 4, padding='same')(u_layer2)
    o_layer = LeakyReLU()(o_layer)

    o_layer = Conv3D(kernel_size=1, filters=6, padding='same', activation=sigmoid,
                     dtype=tf.float32)(o_layer)

    return keras.Model([i_layer, t1_layer], o_layer, name='UNet-NoT1')

