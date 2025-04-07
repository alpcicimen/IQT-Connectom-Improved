from typing import Tuple, Literal, List
from collections.abc import Sequence

import keras
import keras.layers as KL
import os
from keras.activations import tanh, sigmoid

from .layers import *


def config_model(model_type,
                 target_patch_size=16,
                 downsamp_rates: Sequence[float, float, float] = (1.6, 3.0, 1.25 / 0.7),
                 diff_channel_size=6,
                 train_preprocessors: List[Literal["dti",
                                                   "map",
                                                   "dynamic_rescale",
                                                   "static_rate_rescale",
                                                   "list_rescale",
                                                   "normalize_map",
                                                   "normalize_dti"]] = [],
                 weights_dir: None | str | os.PathLike[str] = None):

    diff_recon_ch_size = diff_channel_size  # Placeholder

    if len(train_preprocessors) > 0:

        kernel_penalty_diff = np.int32(np.ceil(2.5 * downsamp_rates[1]) / 2) * 2
        kernel_penalty_t1 = np.int32(np.ceil(2.5 * downsamp_rates[2] * downsamp_rates[0]) / 2) * 2

        if "rescale" in "".join(train_preprocessors):
            dwi_patch_size = int(np.ceil(target_patch_size * downsamp_rates[0])) + kernel_penalty_diff
            t1w_patch_size = int(np.round(target_patch_size * downsamp_rates[0] * downsamp_rates[2]) + kernel_penalty_t1)
            mask_patch_size = int(np.ceil(target_patch_size * downsamp_rates[0]))

        else:
            (dwi_patch_size,
             t1w_patch_size,
             mask_patch_size) = (target_patch_size,
                                 target_patch_size,
                                 target_patch_size)

        diff_input = Input(shape=(dwi_patch_size, dwi_patch_size, dwi_patch_size, diff_channel_size),
                           name='dmri_input')

        mask_input = Input(shape=(mask_patch_size, mask_patch_size, mask_patch_size, 1),
                           name='mask_input')

        t1w_input = Input((t1w_patch_size, t1w_patch_size, t1w_patch_size, 1), name='t1_weighted_input')

        model_inputs = [diff_input, t1w_input, mask_input]

        if ("dti" in train_preprocessors) or ("map" in train_preprocessors):
            bvals_input = Input((diff_channel_size, 1), name='b_value_input')
            bvecs_input = Input((diff_channel_size, 3), name='b_vector_input')

            model_inputs += [bvals_input, bvecs_input]

        if "normalize_dti" in train_preprocessors:
            t1_metrics_input = Input(shape=(2,), name='t1_metrics_input')
            model_inputs += [t1_metrics_input]

        if ("normalize_map" in train_preprocessors):
            t1_metrics_input = Input(shape=(2,), name='t1_metrics_input')
            map_metrics_input = Input(shape=(2,), name='map_metrics_input')
            model_inputs += [t1_metrics_input, map_metrics_input]

        preproc_outputs = [diff_input, diff_input, t1w_input, mask_input]

        for preproc_step in train_preprocessors:

            match preproc_step:

                case "dynamic_rescale":

                    sampler_layer = RandomSamplerLayer(max_hr_downsamp=downsamp_rates[0],
                                                       max_lr_downsamp=downsamp_rates[1],
                                                       t1_init_downsamp=downsamp_rates[2])

                    preproc_outputs = sampler_layer([preproc_outputs[0], preproc_outputs[2], preproc_outputs[3]])

                case "static_rate_rescale":

                    sampler_layer = SameRateSamplerLayer(max_hr_downsamp=downsamp_rates[0],
                                                         downsamp_rate=downsamp_rates[1]/downsamp_rates[0],
                                                         t1_init_downsamp=downsamp_rates[2])

                    preproc_outputs = sampler_layer([preproc_outputs[0], preproc_outputs[2], preproc_outputs[3]])

                case "list_rescale":

                    sampler_layer = ListSamplerLayer(hr_downsamp_rates=[downsamp_rates[0]],
                                                     lr_downsamp_rates=[1.25, 1.5, 2.0, 2.5],
                                                     t1_init_downsamp=downsamp_rates[2])

                    preproc_outputs = sampler_layer([preproc_outputs[0], preproc_outputs[2], preproc_outputs[3]])

                case "dti":

                    lr_dti_layer = DTIFitLayer(diff_channel_size)
                    hr_dti_layer = DTIFitLayer(diff_channel_size)

                    preproc_outputs = [lr_dti_layer([preproc_outputs[0], bvals_input, bvecs_input]),
                                       hr_dti_layer([preproc_outputs[1], bvals_input, bvecs_input]),
                                       preproc_outputs[2],
                                       preproc_outputs[3]]

                    diff_recon_ch_size = lr_dti_layer.output_shape[-1]

                case "map":

                    lr_map_layer = MAPMRIFitLayer(diff_channel_size, herm_order=4)
                    hr_map_layer = MAPMRIFitLayer(diff_channel_size, herm_order=4)

                    preproc_outputs = [lr_map_layer([preproc_outputs[0], bvals_input, bvecs_input]),
                                       hr_map_layer([preproc_outputs[1], bvals_input, bvecs_input]),
                                       preproc_outputs[2],
                                       preproc_outputs[3]]

                    diff_recon_ch_size = lr_map_layer.output_shape[-1]

                case "normalize_dti":

                    dti_norm_layer_lr = MinMaxNormLayer(predet_min=[0, -2e-3, -2e-3, 0, -2e-3, 0],
                                                        predet_max=[2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3],
                                                        name='lr_norm_layer')

                    dti_norm_layer_hr = MinMaxNormLayer(predet_min=[0, -2e-3, -2e-3, 0, -2e-3, 0],
                                                        predet_max=[2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3],
                                                        name='hr_norm_layer')

                    t1_norm_layer = MinMaxNormLayer(name='t1_norm_layer')

                    preproc_outputs = [dti_norm_layer_lr([preproc_outputs[0], preproc_outputs[3]]),
                                       dti_norm_layer_hr([preproc_outputs[1], preproc_outputs[3]]),
                                       t1_norm_layer([preproc_outputs[2], preproc_outputs[3], t1_metrics_input]),
                                       preproc_outputs[3]]

                case "normalize_map":

                    map_norm_layer_lr = MinMaxNormLayer(name='lr_norm')
                    map_norm_layer_hr = MinMaxNormLayer(name='hr_norm')
                    t1_norm_layer = MinMaxNormLayer(name='t1_norm')

                    preproc_outputs = [map_norm_layer_lr([preproc_outputs[0], preproc_outputs[3], map_metrics_input]),
                                       map_norm_layer_hr([preproc_outputs[1], preproc_outputs[3], map_metrics_input]),
                                       t1_norm_layer([preproc_outputs[2], preproc_outputs[3], t1_metrics_input]),
                                       preproc_outputs[3]]

                case _:
                    continue

        lr_patch = preproc_outputs[0]
        hr_patch = preproc_outputs[1]
        t1_patch = preproc_outputs[2]

    else:
        lr_patch = Input((target_patch_size,
                          target_patch_size,
                          target_patch_size, diff_channel_size), name='lowres_input')

        t1_patch = Input((target_patch_size,
                          target_patch_size,
                          target_patch_size, 1), name='t1_output')

    match model_type:

        case "UNet-T1":
            model = [unet3d_t1_v2(lr_patch, t1_patch, diff_ch_size=diff_recon_ch_size)]

        case "UNet-PreFusion":
            model = [unet3d_pre_fusion_v2(lr_patch, t1_patch, diff_ch_size=diff_recon_ch_size)]

        case "UNet":
            model = [unet3d_not1_v2(lr_patch, diff_ch_size=diff_recon_ch_size)]

        case "UNet-Attention":
            model = [unet3d_t1_attention(lr_patch, t1_patch, diff_ch_size=diff_recon_ch_size)]

        case "Identity":
            model = [lr_patch]

        case _:
            raise ValueError(f"No model configuration for \"{model_type}\" found!")

    if len(train_preprocessors) > 0:
        model += [lr_patch, hr_patch, t1_patch]

        model = keras.Model(model_inputs, model)

    else:
        model = keras.Model([lr_patch, t1_patch], model[0])

    if weights_dir is not None:
        model.load_weights(weights_dir)

    keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

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


def unet3d_t1_v2(i_layer, t1_layer, diff_ch_size=6):

    conv_input = Sequential([Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                             LeakyReLU()])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                           LeakyReLU(),
                           Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                           LeakyReLU()])(t1_layer)

    t1_d_layer1 = unet_downsample_layer_v2(t1_input,
                                           kernel_size=3, filter_size=diff_ch_size * 4 * 4)
    t1_d_layer2 = unet_downsample_layer_v2(t1_d_layer1,
                                           kernel_size=3, filter_size=diff_ch_size * 4 * 4 * 4,
                                           last_layer=True)

    d_layer1 = unet_downsample_layer_v2(conv_input,
                                        kernel_size=3, filter_size=diff_ch_size * 4 * 4)
    d_layer2 = unet_downsample_layer_v2(d_layer1,
                                        kernel_size=3, filter_size=diff_ch_size * 4 * 4 * 4,
                                        last_layer=True)

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=diff_ch_size * 4 * 4 * 4, padding='same'),
                            LeakyReLU(),
                            BatchNormalization()])(
        Average()([t1_d_layer2, d_layer2]))

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=[d_layer1, t1_d_layer1],
                                      filter_size=diff_ch_size * 4 * 4, kernel_size=3)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=[conv_input, t1_input],
                                      filter_size=diff_ch_size * 4, kernel_size=3)

    o_layer = Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same')(u_layer2)
    o_layer = LeakyReLU()(o_layer)

    o_layer = Conv3D(kernel_size=1, filters=diff_ch_size, padding='same', activation=sigmoid,
                     dtype=tf.float32)(o_layer)

    # o_layer = o_layer + i_layer

    return o_layer


def unet3d_t1_attention(i_layer, t1_layer, diff_ch_size=6):

    conv_input = Sequential([Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                             ELU(),
                             Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                             ELU(),])(i_layer)

    t1_input = Sequential([Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                           ELU(),
                           Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                           ELU(),])(t1_layer)

    t1_d_layer1 = unet_downsample_layer_v3(t1_input,
                                           kernel_size=3, filter_size=diff_ch_size * 4 ** 2)
    t1_d_layer2 = unet_downsample_layer_v3(t1_d_layer1,
                                           kernel_size=3, filter_size=diff_ch_size * 4 ** 3)
    t1_d_layer3 = unet_downsample_layer_v3(t1_d_layer2,
                                           kernel_size=3, filter_size=diff_ch_size * 4 ** 4)

    d_layer1 = unet_downsample_layer_v3(conv_input,
                                        kernel_size=3, filter_size=diff_ch_size * 4 ** 2)
    d_layer2 = unet_downsample_layer_v3(d_layer1,
                                        kernel_size=3, filter_size=diff_ch_size * 4 ** 3)
    d_layer3 = unet_downsample_layer_v3(d_layer2,
                                        kernel_size=3, filter_size=diff_ch_size * 4 ** 4)

    # d_layer_n = Sequential(
    #     [BatchNormalization(),
    #      ELU(),
    #      Conv3D(kernel_size=3, filters=6 * 2 ** 4, padding='same')]
    # )(Average()([t1_d_layer3, d_layer3]))

    d_layer_n = BasicTransformerBlock(d_layer3.shape)([d_layer3, t1_d_layer3])
    d_layer_n = BatchNormalization()(ELU()(d_layer_n))

    u_layer1 = unet_attention_fusion(d_layer_n,
                                     concat_layer=d_layer2,
                                     attention_layer=t1_d_layer2,
                                     filter_size=6 * 4 ** 3)

    u_layer2 = unet_attention_fusion(u_layer1,
                                     concat_layer=d_layer1,
                                     attention_layer=t1_d_layer1,
                                     filter_size=6 * 4 ** 2)
    u_layer3 = unet_attention_fusion(u_layer2,
                                     concat_layer=conv_input,
                                     attention_layer=t1_input,
                                     filter_size=6 * 4 ** 1)

    o_layer = Sequential([Conv3D(kernel_size=3, filters=6 * 4, padding='same'),
                          ELU(),
                          Conv3D(kernel_size=1, filters=6, padding='same',
                                 dtype=tf.float32,
                                 activation=sigmoid)])(u_layer3)

    return o_layer


def unet3d_pre_fusion_v2(i_layer, t1_layer, diff_ch_size=6):

    conv_input = Sequential([Conv3D(kernel_size=3, filters=(diff_ch_size + 1) * 4, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=3, filters=(diff_ch_size + 1) * 4, padding='same'),
                             LeakyReLU()])(Concatenate(axis=4)([i_layer, t1_layer]))

    d_layer1 = unet_downsample_layer_v2(conv_input,
                                        kernel_size=3, filter_size=(diff_ch_size + 1) * 4 * 4)
    d_layer2 = unet_downsample_layer_v2(d_layer1,
                                        kernel_size=3, filter_size=(diff_ch_size + 1) * 4 * 4 * 4,
                                        last_layer=True)

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=(diff_ch_size + 1) * 4 * 4 * 4, padding='same'),
                            LeakyReLU(),
                            BatchNormalization()])(d_layer2)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=[d_layer1],
                                      filter_size=(diff_ch_size + 1) * 4 * 4, kernel_size=3)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=[conv_input],
                                      filter_size=(diff_ch_size + 1) * 4, kernel_size=3)

    o_layer = Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same')(u_layer2)
    o_layer = LeakyReLU()(o_layer)

    o_layer = Conv3D(kernel_size=1, filters=diff_ch_size, padding='same', activation=sigmoid,
                     dtype=tf.float32)(o_layer)

    return o_layer


def unet3d_not1_v2(i_layer, diff_ch_size=6):

    conv_input = Sequential([Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                             LeakyReLU(),
                             Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same'),
                             LeakyReLU()])(i_layer)

    d_layer1 = unet_downsample_layer_v2(conv_input,
                                        kernel_size=3, filter_size=diff_ch_size * 4 * 4)
    d_layer2 = unet_downsample_layer_v2(d_layer1,
                                        kernel_size=3, filter_size=diff_ch_size * 4 * 4 * 4,
                                        last_layer=True)

    d_layer_n = Sequential([LeakyReLU(),
                            Conv3D(kernel_size=3, filters=diff_ch_size * 4 * 4 * 4, padding='same'),
                            LeakyReLU(),
                            BatchNormalization()])(d_layer2)

    u_layer1 = unet_upsample_layer_v2(d_layer_n,
                                      concat_layer=[d_layer1],
                                      filter_size=diff_ch_size * 4 * 4, kernel_size=3)
    u_layer2 = unet_upsample_layer_v2(u_layer1,
                                      concat_layer=[conv_input],
                                      filter_size=diff_ch_size * 4, kernel_size=3)

    o_layer = Conv3D(kernel_size=3, filters=diff_ch_size * 4, padding='same')(u_layer2)
    o_layer = LeakyReLU()(o_layer)

    o_layer = Conv3D(kernel_size=1, filters=diff_ch_size, padding='same', activation=sigmoid,
                     dtype=tf.float32)(o_layer)

    return o_layer

