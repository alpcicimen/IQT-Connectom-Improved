import os.path
from math import floor

import tensorflow as tf
# import tensorflow.keras as keras
import keras

from data_loader import TrainingSequence
from models import simple_generator, vdsr

from tqdm import tqdm

NUM_EPOCHS = 50

# model: keras.Model = simple_generator(input_ch=6, output_ch=6, layer_num=1, ipatch_size=11)

model: keras.Model = vdsr(15)

optim = keras.optimizers.Adam(learning_rate=1e-4)


def loss_fn(output: tf.Tensor, target: tf.Tensor):
    return tf.reduce_mean(tf.square(target - output))


@tf.function
def train_step(target_batch, input_batch):

    with tf.GradientTape() as tape:
        model_output = model(input_batch)

        loss = loss_fn(target_batch, model_output)

    grads = tape.gradient(loss, model.trainable_weights)

    optim.apply_gradients(zip(grads, model.trainable_weights))

    return loss


@tf.function
def val_step(target_batch, input_batch):
    model_output = model(input_batch)

    return loss_fn(target_batch, model_output)


def main():

    train_seq = TrainingSequence(data_dir='../data',
                                 target_dir='HR',
                                 input_dir='LR',
                                 mode='dti',
                                 normalization_method='stdscore',
                                 subject_labels=['100307', '221319'],
                                 batch_size=4,
                                 pairs_per_subject=8000,
                                 ipatch_size=15,
                                 opatch_size=15)

    summary_writer = tf.summary.create_file_writer('../logs/run_results')

    (sample_t, sample_i, sample_t1) = train_seq.sample_slice(0, (60, 60, 60))

    with summary_writer.as_default():
        tf.summary.image('Target Slice', sample_t[:, 7, :, 0:1][None, ...], step=0)
        tf.summary.image('Input T1w Slice', sample_t1[:, sample_t.shape[1]//2, :, :][None, ...] / 1000, step=0)

    for run in range(NUM_EPOCHS):

        train_loss = 0
        val_loss = 0

        train_size = floor(0.9*train_seq.__len__())

        val_size = train_seq.__len__() - train_size

        for batch, (target_batch, (input_batch, t1_batch)) in enumerate(tqdm(train_seq)):

            if batch <= train_size:

                closs = train_step(target_batch, input_batch)

                train_loss += closs

                if batch % 500 == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('Training Loss', closs, step=run*train_size + batch)

                        tf.summary.image('Model Output Slice', model(sample_i[None, ...])[:, :, 7, :, 0:1],
                                         step=run*train_size + batch)

            else:

                closs = val_step(target_batch, input_batch)

                val_loss += closs

                if batch % 500 == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('Validation Loss', closs, step=run*val_size + batch)


        print(f"Run {run+1} mean training loss: {train_loss/train_size}")
        print(f"Run {run+1} mean validation loss: {val_loss/val_size}")

        model.save_weights(f"../output/Run{run+1}")


if __name__ == '__main__':
    main()
