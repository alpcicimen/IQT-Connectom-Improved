import argparse
import os.path

import keras.optimizers.schedules
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay, LearningRateSchedule

from IQT.data_loader import *
from IQT.models import *

global model
global optim
global loss_fn

make_dataset = True


def create_optim(lr,
                 lr_decay,
                 epochs,
                 dataset_size) -> LearningRateSchedule | float:
    global optim

    match lr_decay:

        case 'exponential':
            _lr = ExponentialDecay(initial_learning_rate=lr,
                                   decay_steps=10 * dataset_size,
                                   decay_rate=0.5)

        case 'constant':
            boundaries = (np.arange(0, epochs // 10) + 1) * 10 * dataset_size
            values = np.power(0.5, range(0, epochs // 10 + 1)) * lr

            _lr = PiecewiseConstantDecay(boundaries=boundaries.tolist(),
                                         values=values.tolist())

        case _:
            _lr = lr

    optim = keras.optimizers.Adam(learning_rate=_lr)

    return _lr


@tf.function
def l1_loss_fn(output: tf.Tensor, target: tf.Tensor):
    return tf.reduce_mean(tf.abs(target - output))


@tf.function
def l2_loss_fn(output: tf.Tensor, target: tf.Tensor):
    return tf.reduce_mean(tf.square(target - output))


# @tf.function
def train_step(dwi_batch, t1_batch, mask_patch, bval_patch, bvec_patch, t1_metric, dwi_metric=None):

    model_input = [dwi_batch, t1_batch, mask_patch, bval_patch, bvec_patch, t1_metric]

    if dwi_metric is not None:
        model_input += [dwi_metric]

    with tf.GradientTape() as tape:
        model_output = model(model_input, training=True)

        loss = loss_fn(model_output[0], model_output[2])

    grads = tape.gradient(loss, model.trainable_weights)

    optim.apply_gradients(zip(grads, model.trainable_weights))

    return loss


# @tf.function
def val_step(dwi_batch, t1_batch, mask_patch, bval_patch, bvec_patch, t1_metric, dwi_metric=None):

    model_input = [dwi_batch, t1_batch, mask_patch, bval_patch, bvec_patch, t1_metric]
    if dwi_metric is not None:
        model_input += [dwi_metric]

    model_output = model(model_input, training=False)

    return loss_fn(model_output[0], model_output[2])


loss_funcs = {"l1": l1_loss_fn, "l2": l2_loss_fn}


def main(model_type,
         diffusion_model: Literal['dti', 'map'],
         patch_size,
         patch_spacing,
         pairs_per_subject,
         loss_type,
         output_dir,
         dwi_data_dir,
         t1_data_dir,
         training_subjects,
         validation_subjects,
         scratch_dir,
         lr_downsampling_max_rate: float,
         hr_downsampling_max_rate: float,
         t1_dwi_rate: float,
         dwi_subdir,
         dwi_file_head,
         t1_subdir,
         t1_file_head,
         map_metric_dir,
         cluster_mode,
         preproc_steps,
         batch_size,
         lr,
         lr_decay,
         epochs,
         log_dir):
    global model
    global loss_fn

    time_start = 0
    model = config_model(model_type,
                         target_patch_size=patch_size,
                         downsamp_rates=[hr_downsampling_max_rate, lr_downsampling_max_rate, t1_dwi_rate],
                         diff_channel_size=
                         108 if diffusion_model == 'dti'
                         else 288,
                         train_preprocessors=preproc_steps)

    loss_best = tf.float32.max

    loss_fn = loss_funcs[str(loss_type).lower()]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if make_dataset:
        print(f"Generating patch triplet library on: {scratch_dir}.\nGenerating training subjects...")

    train_seq = DWISequence(diff_data_dir=dwi_data_dir,
                            t1_data_dir=t1_data_dir,
                            subject_labels=training_subjects,
                            pairs_dir=os.path.join(scratch_dir, "training"),
                            make_dataset=make_dataset,
                            batch_size=batch_size,
                            pairs_per_subject=pairs_per_subject,
                            patch_spacing=patch_spacing,
                            max_target_downsamp=hr_downsampling_max_rate,
                            max_downsamp_rate=lr_downsampling_max_rate,
                            t1_to_diff_ratio=t1_dwi_rate,
                            cluster_mode=cluster_mode,
                            dwis_subdir=dwi_subdir,
                            dwis_filename=dwi_file_head,
                            t1_subdir=t1_subdir,
                            t1_filename=t1_file_head,
                            bval_limit=
                            1200.0 if (diffusion_model == "dti") else 10000.0,  # This will include every acq
                            b0_norm=(diffusion_model != "dti"),
                            map_metric_dir=map_metric_dir)

    if make_dataset:
        print("Generating validation subjects...")

    validation_seq = DWISequence(diff_data_dir=dwi_data_dir,
                                 t1_data_dir=t1_data_dir,
                                 subject_labels=validation_subjects,
                                 pairs_dir=os.path.join(scratch_dir, "validation"),
                                 make_dataset=make_dataset,
                                 batch_size=batch_size,
                                 pairs_per_subject=None,
                                 patch_spacing=patch_size,
                                 max_target_downsamp=hr_downsampling_max_rate,
                                 max_downsamp_rate=lr_downsampling_max_rate,
                                 t1_to_diff_ratio=t1_dwi_rate,
                                 cluster_mode=cluster_mode,
                                 dwis_subdir=dwi_subdir,
                                 dwis_filename=dwi_file_head,
                                 t1_subdir=t1_subdir,
                                 t1_filename=t1_file_head,
                                 bval_limit=
                                 1200.0 if (diffusion_model == "dti") else 10000.0,  # This will include every acq
                                 b0_norm=(diffusion_model != "dti"),
                                 map_metric_dir=map_metric_dir)

    if make_dataset:
        print("Generated patch triplets.")

    _lr = create_optim(lr, lr_decay, epochs, len(train_seq))

    summary_writer = tf.summary.create_file_writer(log_dir)

    sample_patch = validation_seq[0]

    for run in range(epochs):

        train_loss = 0
        val_loss = 0

        with (summary_writer.as_default()):
            tf.summary.scalar('Loss rate at start',
                              _lr if isinstance(_lr, float) else _lr(len(train_seq) * run + 1), step=run)

        if cluster_mode:
            time_start = time.time()

        for train_batch in tqdm(train_seq, disable=cluster_mode):

            (dwi_batch, t1_batch, mask_batch,  # Image data
             bval_batch, bvec_batch,  # dMRI gradient data
             t1_metric_batch) = (train_batch[0], train_batch[1], train_batch[2],  # T1w metric data
                                 train_batch[3], train_batch[4],
                                 train_batch[5])

            dwi_metric_batch = None if diffusion_model == 'dti' else train_batch[6]  # Add dMRI recon model metric data

            closs = train_step(dwi_batch, t1_batch, mask_batch,
                               bval_batch, bvec_batch,
                               t1_metric_batch, dwi_metric_batch)
            train_loss += closs

        for val_batch in tqdm(validation_seq, disable=cluster_mode):

            (dwi_batch, t1_batch, mask_batch,  # Image data
             bval_batch, bvec_batch,  # dMRI gradient data
             t1_metric_batch) = (val_batch[0], val_batch[1], val_batch[2],  # T1w metric data
                                 val_batch[3], val_batch[4],
                                 val_batch[5])

            dwi_metric_batch = None if diffusion_model == 'dti' else val_batch[6]

            val_loss += val_step(dwi_batch, t1_batch, mask_batch,
                                 bval_batch, bvec_batch,
                                 t1_metric_batch, dwi_metric_batch)

        if cluster_mode:
            print("Time taken for run {}: {} seconds.".format((run + 1), time.time() - time_start))

        print(f"Run {run + 1} mean training loss: {train_loss / len(train_seq)}")
        print(f"Run {run + 1} mean validation loss: {val_loss / len(validation_seq)}")

        sample_pred = model(sample_patch, training=False)

        (sample_o, sample_i, sample_t, sample_t1) = (sample_pred[0], sample_pred[1], sample_pred[2], sample_pred[3])

        with (summary_writer.as_default()):
            tf.summary.scalar('Training Epoch Mean Loss', train_loss / len(train_seq), step=run)
            tf.summary.scalar('Validation Epoch Mean Loss', val_loss / len(validation_seq), step=run)

            tf.summary.image('Model Output Slice',
                             (sample_o[None, 0, :, patch_size//2-1, :, 0, None] +
                              sample_o[None, 0, :, patch_size//2-1, :, 3, None] +
                              sample_o[None, 0, :, patch_size//2-1, :, 5, None]) / 3,
                             step=run)

            tf.summary.image('Target DTI Slice',
                             (sample_t[None, 0, :, patch_size//2-1, :, 0, None] +
                              sample_t[None, 0, :, patch_size//2-1, :, 3, None] +
                              sample_t[None, 0, :, patch_size//2-1, :, 5, None]) / 3,
                             step=run)

            tf.summary.image('Input DTI Slice',
                             (sample_i[None, 0, :, patch_size//2-1, :, 0, None] +
                              sample_i[None, 0, :, patch_size//2-1, :, 3, None] +
                              sample_i[None, 0, :, patch_size//2-1, :, 5, None]) / 3,
                             step=run)

            tf.summary.image('Input T1w Slice',
                             sample_t1[None, 0, :, patch_size//2-1, :, 0, None],
                             step=run)

        train_seq.on_epoch_end()  # There's no need to shuffle for validation so shuffle only training

        if val_loss < loss_best:
            loss_best = val_loss
            model.save_weights(os.path.join(output_dir, "Run_Best"))

        model.save_weights(os.path.join(output_dir, "Run_Last"))


if __name__ == '__main__':

    # --------------------------------------------- Mandatory Arguments ------------------------------------------------

    parser = argparse.ArgumentParser(prog='IQT-Training',
                                     description='The main training script for IQT.')

    parser.add_argument('model_type',
                        help="The neural network model to utilise. Options: [ESPCN, ESPCN-T1, UNet, UNet-T1]")

    parser.add_argument('scratch_dir',
                        help='The scratch directory for temporary file storage.')

    parser.add_argument('output_dir',
                        help='The output directory for model weights.')

########################################################################################################################

    # ----------------------------------------------- I/O Arguments ----------------------------------------------------

    parser.add_argument('--diffusion_model', type=str, default='dti', choices=['dti', 'map'],
                        help='The diffusion reconstruction model to use. Options: [dti, map]. Default: dti')

    parser.add_argument('--cluster_mode', type=bool, default=False,
                        help='Determines whether tqdm will be silent (to reduce file size)')

    parser.add_argument('--preproc_steps', type=str, nargs='*',
                        choices=["dynamic_rescale", "static_rate_rescale", "list_rescale",
                                 "dti", "map",
                                 "normalize_dti", "normalize_map"],
                        default=["dynamic_rescale",
                                 "dti",
                                 "normalize_dti"])

    parser.add_argument('--log_dir', type=str,
                        default='/home/acicimen/IQT-Connectom-Improved/logs/run_results')

    parser.add_argument('--dwi_data_dir',
                        default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc')
    parser.add_argument('--t1_data_dir',
                        default='/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim')

    parser.add_argument('--dwi_subdir',
                        default='T1w/Diffusion')
    parser.add_argument('--dwi_file_head',
                        default='data')

    parser.add_argument('--t1_subdir',
                        default='T1w')
    parser.add_argument('--t1_file_head',
                        default='T1w_acpc_dc_restore_brain')

    parser.add_argument('--map_metric_dir', default=None,
                        help='The directory where MAP-MRI metrics are stored. Unused if model is not MAP-MRI.')

########################################################################################################################

    # -------------------------------------------- Training Arguments --------------------------------------------------

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to run the model for. Default: 200')

    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l1',
                        help='The loss type utilised during training. Possible values: [l1, l2]. Default: l1')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='The learning rate of the model. Default: 1e-4')
    parser.add_argument('--lr_decay', choices=[None, 'constant', 'exponential'], default=None,
                        help='The learning decay type. Possible values: [(None), constant, exponential]')

    parser.add_argument('--training_subjects', nargs='+',
                        default=["101915", "102816", "103818", "105115", "105216", "106319", "111312", "111716",
                                 "113215", "113619", "115320", "117122", "118932", "120212", "122317", "123117",
                                 "124422", "125525", "128632", "129028", "130316", "131924", "133827", "133928",
                                 "135932", "137128", "138231", "138534", "139637", "142828", "143325", "144226",
                                 "148032", "148335", "150423", "150524", "151223", "151526", "151627", "153025",
                                 "153429", "154431", "156233", "156637", "158540", "159239", "161731", "162329",
                                 "163129", "167743", "175439", "176542", "185139", "188347", "190031", "191437",
                                 "192439", "195647", "196750", "197550", "198451", "199150", "199655", "201111",
                                 "201414", "205119", "205826", "211417", "212318", "214221", "217126", "239944",
                                 "245333", "246133", "249947", "255639", "280739", "284646", "298051", "329440",
                                 "355239", "397760", "429040", "448347", "497865", "499566", "541943", "545345",
                                 "579665", "581349", "645551", "665254", "677968", "680957", "685058", "688569",
                                 "702133", "713239", "715647", "729557", "734045", "748258", "756055", "761957",
                                 "788876", "826353", "856766", "857263", "859671", "861456", "871964", "889579",
                                 "894673", "896879", "899885", "901139", "904044", "917255", "932554", "937160",
                                 "951457", "414229"])

    parser.add_argument('--validation_subjects', nargs='+',
                        default=["169343", "163432", "390645", "250427", "211720", "704238", "705341", "165840",
                                 "210617", "103414", "792564", "209935", "182840", "205725", "753251", "118730",
                                 "561242"])

    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--patch_spacing', type=int, default=12)

    #  Unfortunately bash does not natively support floating point operations, so a possible workaround would be to
    #  calculate the proper floating point before supplying it as a command-line argument.
    parser.add_argument('--lr_downsampling_max_rate', type=float, default=2.5)
    parser.add_argument('--hr_downsampling_max_rate', type=float, default=1.6)
    parser.add_argument('--t1_dwi_rate', type=float, default=1.25/0.7)

    # parser.add_argument('--clip_strategy', type=str, default='constant')
    # parser.add_argument('--clip_value', type=float, default=2e-3)

    parser.add_argument('--pairs_per_subject', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=10)

########################################################################################################################

    keras.utils.set_random_seed(42)

    args = parser.parse_args()

    main(**vars(args))

# Unused Subjects
# "169343", "214423", "861456", "105115", "136833", "172332", "579665",
    # "865363", "105216", "137128", "175439", "217126", "581349", "871964",
    # "106016", "138231", "176542", "217429", "586460", "872158", "106319",
    # "138534", "177746", "221319", "598568", "877168", "110411", "139637",
    # "182739", "239944", "111312", "140824", "182840", "245333", "645551",
    # "889579", "111514", "142828", "185139", "246133", "111716", "143325",
    # "188347", "249947", "894673", "112819", "144226", "189450", "250427",
    # "665254", "896879", "113215", "148032", "190031", "255639", "672756",
    # "899885", "113619", "148335", "191437", "280739", "677968"])

# Old Training Subjects
# ["100307", "131924", "162733", "210617", "541943", "792564", "100408",
#  "133625", "163129", "211417", "545345", "826353", "101915", "133827",
#  "163432", "211720", "547046", "856766", "102816", "133928", "165840",
#  "212318", "559053", "857263", "103414", "214019", "561242", "103515",
#  "134324", "167743", "214221", "570243", "859671", "103818", "135932"]
