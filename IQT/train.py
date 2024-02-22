import argparse
import os.path
from math import floor

import keras.optimizers.schedules
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

from data_loader import *
from models import *

global model
global optim

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

    time_start = 0

    global model
    model = config_model(args)

    if make_dataset:
        print(f"Generating patch triplet library on: {args.scratch_dir}")

        PairSequence.generate_data(diff_data_dir=args.dt_data_dir,
                                   t1_data_dir=args.t1_data_dir,
                                   subject_labels=args.subjects,
                                   pairs_dir=args.scratch_dir,
                                   upsampling_rate=1.25 / .7,
                                   hr_filedir=os.path.join(args.hr_subdir, args.hr_file_head),
                                   t1_filedir=os.path.join(args.t1_subdir, args.t1_file_head),
                                   mode='dti',
                                   normalization_method='minmax',
                                   patch_spacing=8,
                                   patch_size=16,
                                   mask_erosion=args.mask_erosion,
                                   cluster_mode=args.cluster_mode)

        print("Generated patch triplets.")

    train_seq = PairSequence(pair_dir=args.scratch_dir,
                             subject_labels=args.subjects,
                             batch_size=args.batch_size,
                             pairs_per_subject=800)

    _lr = ExponentialDecay(initial_learning_rate=1e-4,
                           decay_steps=10*len(train_seq),
                           decay_rate=0.5)

    global optim
    optim = keras.optimizers.Adam(learning_rate=_lr)

    summary_writer = tf.summary.create_file_writer(args.log_dir)

    (sample_t, sample_i, sample_t1) = train_seq.sample_slice(0, (60, 60, 60))

    with (summary_writer.as_default()):
        tf.summary.image('Target Slice', sample_t[:, :, 7, :, 0:1], step=0)
        tf.summary.image('Input T1w Slice', sample_t1[:, :, 7, :, :], step=0)

    for run in range(args.epochs):

        train_loss = 0
        val_loss = 0

        with (summary_writer.as_default()):
            tf.summary.scalar('Loss rate at start', optim.learning_rate, step=run)

        train_size = floor(0.9 * train_seq.__len__())

        val_size = train_seq.__len__() - train_size

        if args.cluster_mode:
            time_start = time.time()

        for batch, (target_batch, (input_batch, t1_batch)) \
                in enumerate(tqdm(train_seq, disable=args.cluster_mode)):

            if batch <= train_size:

                closs = train_step(target_batch, input_batch, t1_batch)

                train_loss += closs

            else:

                closs = val_step(target_batch, input_batch, t1_batch)
                val_loss += closs

        # 2.2, 1.25, 0.7 -> 44, 25, 14
        # 44 Patch size on T1w, 25 PS on HR, 14 PS on LR
        # 25 -> 14 setup

        if args.cluster_mode:
            print("Time taken for run {}: {} seconds.".format((run + 1), time.time() - time_start))

        print(f"Run {run + 1} mean training loss: {train_loss / train_size}")
        print(f"Run {run + 1} mean validation loss: {val_loss / val_size}")

        with (summary_writer.as_default()):
            tf.summary.scalar('Training Epoch Mean Loss', train_loss / train_size, step=run)
            tf.summary.scalar('Validation Epoch Mean Loss', val_loss / val_size, step=run)

            tf.summary.image('Model Output Slice',
                             model([sample_i, sample_t1])[:, :, 7, :, 0:1],
                             step=run)

        train_seq.on_epoch_end()

        model.save_weights(f"/cluster/project9/IQTSuperRes/alp_IQT_Output/Run{run + 1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='IQT-Training',
                                     description='The main training script for IQT.')

    parser.add_argument('model',
                        help="The neural network model to utilise. Options: [ESPCN, ESPCN-T1, UNet, UNet-T1]")

    parser.add_argument('scratch_dir',
                        help='The scratch directory for temporary file storage.')

    parser.add_argument('--cluster_mode', type=bool, default=False,
                        help='Determines whether tqdm will be silent (to reduce file size)')

    parser.add_argument('--log_dir', type=str, default='/home/acicimen/IQT-Connectom-Improved/logs/run_results')

    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs to run the model for. Default: 10')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='The learning rate of the model. Default: 1e-4')

    parser.add_argument('--dt_data_dir', default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc')
    parser.add_argument('--t1_data_dir', default='/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim')

    parser.add_argument('--subjects', nargs='+',
                        default=["100307", "131924", "162733", "210617", "541943", "792564", "100408",
                                 "133625", "163129", "211417", "545345", "826353", "101915", "133827",])
            #"163432", "211720", "547046", "856766", "102816", "133928", "165840",
            #"212318", "559053", "857263", "103414", "214019", "561242", "103515",
            #"134324", "167743", "214221", "570243", "859671", "103818", "135932",
            #"169343", "214423", "861456", "105115", "136833", "172332", "579665",
            #"865363", "105216", "137128", "175439", "217126", "581349", "871964",
            #"106016", "138231", "176542", "217429", "586460", "872158", "106319",
            #"138534", "177746", "221319", "598568", "877168", "110411", "139637",
            #"178950", "224022", "627549", "885975", "111009", "140420", "182739",
            #"239944", "638049", "887373", "111312", "140824", "182840", "245333",
            #"645551", "889579", "111514", "142828", "185139", "246133", "654754",
            #"111716", "143325", "188347", "249947", "894673", "112819", "144226",
            #"189450", "250427", "665254", "896879", "113215", "148032", "190031",
            #"255639", "672756", "899885", "113619", "148335", "191437", "280739", "677968"])
    parser.add_argument('--batch_size', type=int, default=12)

    parser.add_argument('--hr_subdir', default='T1w/Diffusion')
    parser.add_argument('--hr_file_head', default='dt_b1000_')

    parser.add_argument('--t1_subdir', default='T1w')
    parser.add_argument('--t1_file_head', default='T1w_acpc_dc_restore_brain')

    parser.add_argument('--mask_erosion', default=3)

    args = parser.parse_args()

    main(args)
