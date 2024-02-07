import os.path
from math import floor
import argparse

import tensorflow as tf
import keras

from data_loader import *
from models import *

from tqdm import tqdm

NUM_EPOCHS = 25

# model: keras.Model = unet3d(16)
model: keras.Model = unet3d_t1(16, 16)

optim = keras.optimizers.Adam(learning_rate=1e-4)

make_dataset = True


@tf.function
def loss_fn(output: tf.Tensor, target: tf.Tensor):
    return tf.reduce_mean(tf.square(target - output))


@tf.function
def train_step(target_batch, input_batch, t1_batch):
    with tf.GradientTape() as tape:
        model_output = model([input_batch, t1_batch])

        loss = loss_fn(target_batch, model_output)

    grads = tape.gradient(loss, model.trainable_weights)

    optim.apply_gradients(zip(grads, model.trainable_weights))

    return loss


@tf.function
def val_step(target_batch, input_batch, t1_batch):
    model_output = model([input_batch, t1_batch])

    return loss_fn(target_batch, model_output)


def main(args):
    # train_seq = TrainingSequence(data_dir=args.dt_data_dir,
    #                              target_dir=args.hr_filename,
    #                              input_dir=args.lr_filename,
    #                              mode='dti',
    #                              normalization_method='stdscore',
    #                              subject_labels=['100307', '221319'],
    #                              batch_size=12,
    #                              pairs_per_subject=8000,
    #                              ipatch_size=16,
    #                              opatch_size=16)

    if make_dataset:
        print(f"Generating patch triplet library on: {args.scratch_dir}")

        PairLoader('../data',
                   '../data',
                   ['100307', '221319'],
                   args.scratch_dir,
                   lr_filedir=os.path.join('LR', 'dt_b1000_lowres_2_'),
                   upsampling_rate=1.25 / .7,
                   hr_filedir=os.path.join('HR', 'dt_b1000_'),
                   t1_filedir=os.path.join('T1w', 'T1w_acpc_dc_restore_brain'),
                   mode='dti',
                   patch_spacing=8,
                   patch_size=16)

        print("Generated patch triplets.")

    train_seq = PairSequence(pair_dir=args.scratch_dir,
                             subject_labels=['100307', '221319'],
                             batch_size=12,
                             pairs_per_subject=800)

    summary_writer = tf.summary.create_file_writer('../logs/run_results_unet')

    (sample_t, sample_i, sample_t1) = train_seq.sample_slice(0, (60, 60, 60))

    with (summary_writer.as_default()):
        tf.summary.image('Target Slice', sample_t[:, :, 7, :, 0:1], step=0)
        tf.summary.image('Input T1w Slice', -sample_t1[:, :, 7, :, :], step=0)

        for run in range(args.epochs):

            train_loss = 0
            val_loss = 0

            train_size = floor(0.9 * train_seq.__len__())

            val_size = train_seq.__len__() - train_size

            for batch, (target_batch, (input_batch, t1_batch)) \
                    in enumerate(tqdm(train_seq, disable=args.cluster_mode)):

                if batch <= train_size:

                    closs = train_step(target_batch, input_batch, -t1_batch)

                    train_loss += closs

                else:

                    closs = val_step(target_batch, input_batch, -t1_batch)
                    val_loss += closs

            # 2.2, 1.25, 0.7 -> 44, 25, 14
            # 44 Patch size on T1w, 25 PS on HR, 14 PS on LR
            # 25 -> 14 setup

            print(f"Run {run + 1} mean training loss: {train_loss / train_size}")
            print(f"Run {run + 1} mean validation loss: {val_loss / val_size}")

            tf.summary.scalar('Training Epoch Mean Loss', train_loss / train_size, step=run)
            tf.summary.scalar('Validation Epoch Mean Loss', val_loss / val_size, step=run)

            tf.summary.image('Model Output Slice',
                             model([sample_i, -sample_t1])[:, :, 7, :, 0:1],
                             step=run)

            train_seq.on_epoch_end()

            model.save_weights(f"../output/UNet_NoT1/Run{run + 1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='IQT-Training',
                                     description='The main training script for IQT.')

    parser.add_argument('model',
                        help="The neural network model to utilise. Options: [ESPCN, ESPCN-T1, UNet, UNet-T1]")

    parser.add_argument('scratch_dir',
                        help='The scratch directory for temporary file storage.')

    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs to run the model for. Default: 10')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='The learning rate of the model. Default: 1e-4')

    parser.add_argument('--dt_data_dir', default='../data')
    parser.add_argument('--t1_data_dir', default='../data')

    parser.add_argument('--subjects', nargs='+', default=['100307'])

    parser.add_argument('--hr_subdir', default='.')
    parser.add_argument('--hr_file_head', default='dt_b1000_')

    parser.add_argument('--lr_subdir', default='.')
    parser.add_argument('--lr_file_head', default='dt_b1000_lowres_2_')

    args = parser.parse_args()

    main(args)
