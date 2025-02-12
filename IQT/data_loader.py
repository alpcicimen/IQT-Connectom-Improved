import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf
import keras
import os
import time
import copy

from math import ceil
from random import randint, choices
from typing import List, Tuple, Literal, Any
from numpy.typing import NDArray
from collections.abc import Iterable

from tqdm import tqdm

from IQT import util

from scipy.ndimage import zoom, binary_erosion

from skimage import morphology


class DWISequence(keras.utils.Sequence):

    def __get_indices__(self) -> NDArray[str]:
        """
        Private function that randomly acquires self.pair_dir amount of patch triplets,
            and returns the indices for the epoch.

        Returns:
            The selected patch triplets for utilisation in the current epoch
        """

        run_indices = []

        self.__total_patches = 0

        for (s, subj) in enumerate(self.subject_labels):

            subj_pairs = len(os.listdir(os.path.join(self.pairs_dir, subj, "dwi")))
                # if self.pairs_per_subject is None \
                # else self.pairs_per_subject

            if self.pairs_per_subject is not None:
                subj_pairs = min(subj_pairs, self.pairs_per_subject)

            self.__total_patches += subj_pairs

            patch_indices: NDArray[str] = np.stack((np.repeat(s, subj_pairs),
                                                   choices(os.listdir(os.path.join(self.pairs_dir, subj, "dwi")),
                                                           k=subj_pairs)), dtype=str, axis=-1)

            run_indices.append(patch_indices)

        run_indices = np.concatenate(run_indices, axis=0)

        npr.shuffle(run_indices)

        return run_indices

    def __init__(self,
                 diff_data_dir,
                 t1_data_dir,
                 subject_labels,
                 pairs_dir,
                 make_dataset,
                 batch_size,
                 pairs_per_subject=400,
                 target_patch_size=16,
                 patch_spacing=16,
                 max_target_downsamp=1.6,
                 max_downsamp_rate=3.,
                 t1_to_diff_ratio=1.,
                 shuffle_indices=True,
                 cluster_mode=False,
                 dwis_subdir='Raw',
                 dwis_filename="data",
                 mask_filename="nodif_brain_mask",
                 bvals_file="bvals",
                 bvecs_file="bvecs",
                 bval_limit=1200.,
                 t1_subdir='T1w',
                 t1_filename='T1w_acpc_dc_restore_brain',
                 b0_norm=False,
                 b0_limit=100,
                 map_metric_dir: None | os.PathLike = None):

        self.subject_labels = subject_labels
        self.batch_size = batch_size

        self.pairs_per_subject = pairs_per_subject

        self.pairs_dir = pairs_dir

        if cluster_mode:
            cur_time = time.time()

        self.t1_metrics = []
        self.all_bvals = []
        self.all_bvecs = []

        if map_metric_dir is not None:
            self.map_metrics = []
        else:
            self.map_metrics = None

        for subject_label in subject_labels:

            if cluster_mode:
                print("Current subject: ", subject_label)

            t1w_path = os.path.join(pairs_dir, subject_label, "t1w")
            dwi_path = os.path.join(pairs_dir, subject_label, "dwi")
            mask_path = os.path.join(pairs_dir, subject_label, "mask")
            misc_path = os.path.join(pairs_dir, subject_label, "misc")

            if self.map_metrics is not None:
                map_metric = np.array(pd.read_csv(os.path.join(map_metric_dir, f"{subject_label}.csv"),
                                                  header=None))[0]

                self.map_metrics.append(map_metric)

            if make_dataset:

                if not os.path.exists(t1w_path):
                    os.makedirs(t1w_path)

                if not os.path.exists(dwi_path):
                    os.makedirs(dwi_path)

                if not os.path.exists(mask_path):
                    os.makedirs(mask_path)

                if not os.path.exists(misc_path):
                    os.makedirs(misc_path)

                subject_data_base, hr_header = util.load_dwis(
                    os.path.join(diff_data_dir, subject_label, dwis_subdir),
                    dwis_filename
                )

                subject_data_t1_base, t1_header = util.load_structural(
                    os.path.join(t1_data_dir, subject_label, t1_subdir),
                    t1_filename
                )

                mask = np.array(util.__load_nii__(os.path.join(diff_data_dir, subject_label, dwis_subdir),
                                                  mask_filename).dataobj) > 0

                mask = binary_erosion(mask, morphology.ball(1), iterations=1)

                t1_downsample_rate = np.array(subject_data_t1_base.shape[:-1]) / np.array(subject_data_base.shape[:-1])

                mask_t1 = zoom(mask,
                               zoom=(t1_downsample_rate[0],
                                     t1_downsample_rate[1],
                                     t1_downsample_rate[2]),
                               order=0,
                               prefilter=False) & (subject_data_t1_base[..., 0] > 0)

                t1_metric = np.array([np.percentile(subject_data_t1_base[mask_t1], 2),
                                      np.percentile(subject_data_t1_base[mask_t1], 98)])

                self.t1_metrics.append(t1_metric)

                mask = np.array(zoom(mask_t1,
                                     zoom=(1 / t1_downsample_rate[0],
                                           1 / t1_downsample_rate[1],
                                           1 / t1_downsample_rate[2]),
                                     order=0, prefilter=False), dtype=bool)

                bvals = np.array(pd.read_csv(os.path.join(diff_data_dir, subject_label, dwis_subdir, bvals_file),
                                             delimiter="  ",
                                             engine="python",
                                             header=None)).T

                bvecs = np.array(pd.read_csv(os.path.join(diff_data_dir, subject_label, dwis_subdir, bvecs_file),
                                             delimiter="  ",
                                             engine="python",
                                             header=None)).T

                valid_acqs = np.squeeze(bvals < bval_limit)

                valid_bvals = bvals[valid_acqs]
                valid_bvecs = bvecs[valid_acqs]

                self.all_bvals.append(valid_bvals)
                self.all_bvecs.append(valid_bvecs)

                valid_dwis = subject_data_base[..., valid_acqs]

                del subject_data_base
                del bvals
                del bvecs

                t1_base_patch_size = int(np.round(target_patch_size * t1_to_diff_ratio * max_target_downsamp)) + \
                    np.int32(np.ceil(2.5 * t1_to_diff_ratio * max_target_downsamp) / 2) * 2

                dwi_base_patch_size = int(np.round(target_patch_size * max_target_downsamp)) + \
                    np.int32(np.ceil(2.5 * max_downsamp_rate) / 2) * 2

                mask_patch_size = int(np.round(target_patch_size * max_target_downsamp))

                valid_dwis = np.pad(valid_dwis,
                                    pad_width=np.array([[dwi_base_patch_size // 2, dwi_base_patch_size // 2],
                                                        [dwi_base_patch_size // 2, dwi_base_patch_size // 2],
                                                        [dwi_base_patch_size // 2, dwi_base_patch_size // 2],
                                                        [0, 0]]), mode='edge')

                subject_data_t1_base = np.pad(subject_data_t1_base,
                                              pad_width=np.array([[t1_base_patch_size // 2, t1_base_patch_size // 2],
                                                                  [t1_base_patch_size // 2, t1_base_patch_size // 2],
                                                                  [t1_base_patch_size // 2, t1_base_patch_size // 2],
                                                                  [0, 0]]), mode='edge')

                mask = np.pad(mask, pad_width=np.array([[mask_patch_size // 2, mask_patch_size // 2],
                                                        [mask_patch_size // 2, mask_patch_size // 2],
                                                        [mask_patch_size // 2, mask_patch_size // 2]]), mode='edge')

                if b0_norm:

                    dwis_mean = np.mean(valid_dwis[..., (valid_bvals < b0_limit)[:, 0]], axis=-1, keepdims=True)

                    valid_dwis /= (dwis_mean + 1e-10)

                sel_mask_indices = np.zeros(mask.shape, dtype=bool)

                sel_mask_indices[0::patch_spacing,
                                 0::patch_spacing,
                                 0::patch_spacing] = True

                sel_mask_indices = np.array(np.where(sel_mask_indices & mask)).T

                np.save(os.path.join(misc_path, f"grads"),
                        np.concatenate([valid_bvals, valid_bvecs], axis=-1),
                        allow_pickle=True)
                np.save(os.path.join(misc_path, f"t1_minmax"), t1_metric, allow_pickle=True)

                for p, (i, j, k) in enumerate(tqdm(sel_mask_indices, disable=cluster_mode)):

                    if shuffle_indices:
                        i_orig, j_orig, k_orig = (i, j, k)

                        i = np.clip(i + randint(-patch_spacing // 4, patch_spacing // 4),
                                    a_min=mask_patch_size//2, a_max=mask.shape[0] - mask_patch_size//2)
                        j = np.clip(j + randint(-patch_spacing // 4, patch_spacing // 4),
                                    a_min=mask_patch_size//2, a_max=mask.shape[1] - mask_patch_size//2)
                        k = np.clip(k + randint(-patch_spacing // 4, patch_spacing // 4),
                                    a_min=mask_patch_size//2, a_max=mask.shape[2] - mask_patch_size//2)

                        if ~mask[i, j, k]:
                            i, j, k = (i_orig, j_orig, k_orig)  # Use original values if new centre voxel is unmasked

                    mask_patch = mask[
                        i - mask_patch_size//2:i + int(np.ceil(mask_patch_size/2)),
                        j - mask_patch_size//2:j + int(np.ceil(mask_patch_size/2)),
                        k - mask_patch_size//2:k + int(np.ceil(mask_patch_size/2)),
                    ]

                    i_dwi, j_dwi, k_dwi = np.array(
                        np.round(np.array([i, j, k]) - mask_patch_size//2) + dwi_base_patch_size//2,
                        dtype=int)

                    dwis_patch = valid_dwis[
                        i_dwi - dwi_base_patch_size//2:i_dwi + dwi_base_patch_size//2,
                        j_dwi - dwi_base_patch_size//2:j_dwi + dwi_base_patch_size//2,
                        k_dwi - dwi_base_patch_size//2:k_dwi + dwi_base_patch_size//2,
                        :
                    ]

                    i_t1, j_t1, k_t1 = (np.array([i, j, k]) - mask_patch_size/2) * t1_to_diff_ratio

                    t1_grid = util.config_grid((i_t1, j_t1, k_t1),
                                               (i_t1 + t1_base_patch_size - 1,
                                                j_t1 + t1_base_patch_size - 1,
                                                k_t1 + t1_base_patch_size - 1),
                                               (t1_base_patch_size, t1_base_patch_size, t1_base_patch_size))

                    t1_patch = util.gridded_interpolation(subject_data_t1_base, t1_grid)

                    assert t1_patch.shape == (t1_base_patch_size, t1_base_patch_size, t1_base_patch_size, 1)
                    assert mask_patch.shape == (mask_patch_size, mask_patch_size, mask_patch_size)
                    assert dwis_patch.shape[:-1] == (dwi_base_patch_size, dwi_base_patch_size, dwi_base_patch_size)

                    np.save(os.path.join(dwi_path, f"{p}"), dwis_patch, allow_pickle=False)
                    np.save(os.path.join(t1w_path, f"{p}"), t1_patch, allow_pickle=False)
                    np.save(os.path.join(mask_path, f"{p}"), mask_patch, allow_pickle=False)

            else:

                grads = np.load(os.path.join(misc_path, "grads.npy"), )

                self.all_bvals.append(grads[..., 0])
                self.all_bvecs.append(grads[..., 1:])

                self.t1_metrics.append(np.load(os.path.join(misc_path, "t1_minmax.npy")))

        if cluster_mode:
            print(f"Time taken to generate patches: {time.time() - cur_time} seconds")

        self.__run_indices = self.__get_indices__()
        return

    def __len__(self):
        return ceil(self.__total_patches / self.batch_size)

    def on_epoch_end(self):
        self.__run_indices = self.__get_indices__()

    def __getitem__(self, index):

        dwi_patches = []
        t1_patches = []
        mask_patches = []
        bval_patches = []
        bvec_patches = []
        t1_metrics = []

        if self.map_metrics is not None:
            map_metrics = []

        for (subj, patch_indx) in self.__run_indices[index:min(self.__total_patches, index + self.batch_size)]:
            dwi_patch = np.load(os.path.join(self.pairs_dir, self.subject_labels[int(subj)], "dwi", patch_indx))
            t1_patch = np.load(os.path.join(self.pairs_dir, self.subject_labels[int(subj)], "t1w", patch_indx))
            mask_patch = np.load(os.path.join(self.pairs_dir, self.subject_labels[int(subj)], "mask", patch_indx))

            dwi_patches.append(dwi_patch)
            t1_patches.append(t1_patch)
            mask_patches.append(mask_patch)

            bval_patches.append(self.all_bvals[int(subj)])
            bvec_patches.append(self.all_bvecs[int(subj)])
            t1_metrics.append(self.t1_metrics[int(subj)])

            if self.map_metrics is not None:
                map_metrics.append(self.map_metrics[int(subj)])

        ret = (tf.cast(tf.stack(dwi_patches), dtype=tf.float32),
               tf.cast(tf.stack(t1_patches), dtype=tf.float32),
               tf.cast(tf.stack(mask_patches), dtype=tf.float32),
               tf.cast(tf.stack(bval_patches), dtype=tf.float32),
               tf.cast(tf.stack(bvec_patches), dtype=tf.float32),
               tf.cast(tf.stack(t1_metrics), dtype=tf.float32))

        if self.map_metrics is not None:
            ret += (tf.cast(tf.stack(map_metrics), dtype=tf.float32),)

        return ret


class PairSequence(keras.utils.Sequence):

    @staticmethod
    def generate_data(diff_data_dir,
                      t1_data_dir,
                      subject_labels,
                      pairs_dir,
                      downsampling_rates=[1.25 / .7],
                      hr_downsampling_rate=1.,
                      hr_filedir=os.path.join('HR', 'dt_b1000_'),
                      t1_filedir=os.path.join('T1w', 'T1w_acpc_dc_restore_brain'),
                      mode='dti',
                      normalization_method: Literal["minmax", "std"] = 'minmax',
                      clip_strategy='constant',
                      clip_value=3e-3,
                      patch_spacing=12,
                      patch_size=16,
                      mask_erosion=0,
                      random_shift=True,
                      apply_blurring=True,
                      cluster_mode=False) -> None:
        """
        Static method for patch triplet data generation and storage.
        The files are saved individually under the subject ids, and stored as .npy files.

        Args:
            diff_data_dir: The file directory for the diffusion tensors
            t1_data_dir: The file directory for the T1w images
            subject_labels: The list of subjects to be utilised
            pairs_dir: The directory where the calculated pairs (or triplets with T1) will be stored at
            downsampling_rates: The downsampling rates of the patch preprocessor.
                The HR data is *downsampled* by this ratio for each ratio provided.
            hr_downsampling_rate: The downsampling rate of the target \"high resolution\" image.
            hr_filedir: The file directory for individual diffusion tensors.
            t1_filedir: The file directory for the T1w images.
            mode: The diffusion modality.
                Options are \"dti\" for diffusion tensors, and \"map\" for mean apparent propagators.
            normalization_method: The method utilised to normalize the data.
                See parameter ``normalization_method`` in :func:`util.apply_normalization` for more info
            clip_strategy: The strategy utilised when clipping the tensors.
                Valid strategies can be found in :func:`util.clip_on_predef_mode`
            clip_value: The value(s) for the clipping strategy.
                See :func:`util.clip_on_predef_mode` for more details.
            patch_spacing: The spacing between valid patches. Spacing == patch size guarantees no overlap.
            patch_size: The patch size for the model. Odd numbered patches have a central voxel.
            mask_erosion: The value for which the mask will be eroded for. Default value 0 means no erosion.
            random_shift: Boolean condition that determines whether the patch spacing is randomly shifted one way.
                This is so that the model does not inherently learn the general shape of the selected patches.
                Default = True
            apply_blurring: Condition that determines whether the model has blurring applied pre-downsampling.
                Default - True
            cluster_mode: argument that suppresses tqdm outputs (use if you're running this on the cluster)
        Returns:
            None
        """

        norm_channels = None

        match mode:

            case 'dti':
                loader_func = util.load_dtis
                norm_channels = np.array([[0, 3, 5], [1, 2, 4]])

            case 'map':
                loader_func = util.load_maps

            case _:
                raise TypeError("Unsupported data mode: {}".format(mode))

        if cluster_mode:
            cur_time = time.time()

        if hr_downsampling_rate < 1.:  # Up-sampling is not supported! (A value of 1 is allowed)
            raise ValueError("Only values larger than 1 are supported.")

        for downsampling_rate in downsampling_rates:
            assert downsampling_rate >= 1.  # Up-sampling is not supported!

        for subject_label in subject_labels:

            if cluster_mode:
                print("Current subject: ", subject_label)

            if not os.path.exists(os.path.join(pairs_dir, subject_label)):
                os.makedirs(os.path.join(pairs_dir, subject_label))

            subject_data_base, hr_header = loader_func(
                os.path.join(diff_data_dir, subject_label),
                hr_filedir
            )

            subject_data_t1_base, t1_header = util.load_structural(
                os.path.join(t1_data_dir, subject_label),
                t1_filedir
            )

            t1_downsample_rate = np.array(subject_data_t1_base.shape[:-1]) / np.array(subject_data_base.shape[:-1])

            mask = subject_data_base[..., 0]

            if hr_downsampling_rate > 1.:
                mask = zoom(mask,
                            zoom=(1. / hr_downsampling_rate,
                                  1. / hr_downsampling_rate,
                                  1. / hr_downsampling_rate),
                            order=1,
                            prefilter=False)

            mask_base = np.array(mask >= 0, dtype=bool)

            # The masks may have problematic outlier voxels if they were not fine-tuned.
            # Therefore, we just erode them further.
            if mask_erosion:
                mask_base = binary_erosion(mask_base, np.ones((mask_erosion, mask_erosion, mask_erosion)))

########################################################################################################################

# -------------------------------------------------- Scaling Step ------------------------------------------------------

########################################################################################################################

            for ds, downsampling_rate in enumerate(downsampling_rates):

                subject_data_hr = np.copy(subject_data_base[..., 2:])
                subject_data_lr = np.copy(subject_data_base[..., 2:])
                subject_data_t1 = np.copy(subject_data_t1_base)
                mask = np.copy(mask_base)

                if apply_blurring:

                    if hr_downsampling_rate > 1.:
                        subject_data_hr = util.apply_gaussian_filter(subject_data_hr, hr_downsampling_rate)

                    subject_data_lr = util.apply_gaussian_filter(subject_data_lr,
                                                                 downsampling_rate * hr_downsampling_rate)
                    subject_data_t1 = util.apply_gaussian_filter(subject_data_t1,
                                                                 np.mean(t1_downsample_rate) * hr_downsampling_rate)

                if hr_downsampling_rate > 1.:
                    subject_data_hr = zoom(subject_data_hr,
                                           zoom=(1. / hr_downsampling_rate,
                                                 1. / hr_downsampling_rate,
                                                 1. / hr_downsampling_rate, 1),
                                           order=1,
                                           prefilter=False)

                subject_data_lr = zoom(subject_data_lr,
                                       zoom=(1. / (hr_downsampling_rate * downsampling_rate),
                                             1. / (hr_downsampling_rate * downsampling_rate),
                                             1. / (hr_downsampling_rate * downsampling_rate), 1),
                                       order=1,
                                       prefilter=False)

                target_scales = (np.array(subject_data_hr.shape[:-1] + (1,)) /
                                 np.array(subject_data_t1.shape))

                subject_data_t1 = zoom(subject_data_t1, target_scales, order=1, prefilter=False)

                lr_dims = (np.array(subject_data_hr.shape[:-1] + (1,)) /
                           np.array(subject_data_lr.shape[:-1] + (1,)))

                subject_data_lr = zoom(subject_data_lr, lr_dims, order=1, prefilter=False)

########################################################################################################################

# ---------------------------------------------- Normalization Step ----------------------------------------------------

########################################################################################################################

                t1_values = util.get_clip_values(subject_data_t1, mask,
                                                 data_mode='t1w',
                                                 clip_strategy='percentile',
                                                 value=96)

                # Normalize at low-resolution space before applying linear interpolation to target resolution.
                # This is so that we properly simulate our input.
                util.apply_normalization_combined(subject_data_lr, mask,
                                                  method=normalization_method,
                                                  channels=norm_channels,
                                                  values=util.get_clip_values(
                                                      subject_data_lr, mask, mode, clip_strategy, clip_value
                                                  ))
                util.apply_normalization_combined(subject_data_hr, mask,
                                                  method=normalization_method,
                                                  channels=norm_channels,
                                                  values=util.get_clip_values(
                                                      subject_data_hr, mask, mode, clip_strategy, clip_value
                                                  ))
                util.apply_normalization_combined(subject_data_t1, mask,
                                                  method=normalization_method,
                                                  values=t1_values)

########################################################################################################################

# --------------------------------------------- Further Preprocessing --------------------------------------------------

########################################################################################################################

                subject_data_lr[~mask, :] = 0
                subject_data_hr[~mask, :] = 0
                subject_data_t1[~mask, :] = 0

                subject_data_lr = np.pad(subject_data_lr,  # Pad to ensure that there's no possible misshapen patch size
                                         pad_width=np.array([[patch_size // 2, patch_size // 2],
                                                             [patch_size // 2, patch_size // 2],
                                                             [patch_size // 2, patch_size // 2],
                                                             [0, 0]]), mode='edge')

                subject_data_hr = np.pad(subject_data_hr,
                                         pad_width=np.array([[patch_size // 2, patch_size // 2],
                                                             [patch_size // 2, patch_size // 2],
                                                             [patch_size // 2, patch_size // 2],
                                                             [0, 0]]), mode='edge')

                subject_data_t1 = np.pad(subject_data_t1,
                                         pad_width=np.array([[patch_size // 2, patch_size // 2],
                                                             [patch_size // 2, patch_size // 2],
                                                             [patch_size // 2, patch_size // 2],
                                                             [0, 0]]), mode='edge')

                mask = np.pad(mask, pad_width=np.array([[patch_size // 2, patch_size // 2],
                                                        [patch_size // 2, patch_size // 2],
                                                        [patch_size // 2, patch_size // 2]]), mode='edge')

                sel_mask_indices = np.zeros(mask.shape, dtype=bool)

                randshift = (randint(0, patch_spacing // 4),
                             randint(0, patch_spacing // 4),
                             randint(0, patch_spacing // 4)) if random_shift else (0, 0, 0)

                sel_mask_indices[randshift[0]::patch_spacing,
                                 randshift[1]::patch_spacing,
                                 randshift[2]::patch_spacing] = True

                sel_mask_indices = np.array(np.where(sel_mask_indices & mask)).T

                for s, (i, j, k) in enumerate(tqdm(sel_mask_indices, disable=cluster_mode)):
                    comb_patch = np.concatenate([
                        subject_data_hr[i - patch_size // 2:i + round(patch_size / 2),
                                        j - patch_size // 2:j + round(patch_size / 2),
                                        k - patch_size // 2:k + round(patch_size / 2), :],
                        subject_data_lr[i - patch_size // 2:i + round(patch_size / 2),
                                        j - patch_size // 2:j + round(patch_size / 2),
                                        k - patch_size // 2:k + round(patch_size / 2), :],
                        subject_data_t1[i - patch_size // 2:i + round(patch_size / 2),
                                        j - patch_size // 2:j + round(patch_size / 2),
                                        k - patch_size // 2:k + round(patch_size / 2), :],
                    ], axis=-1)

                    if 0 in comb_patch.shape:
                        raise ValueError(f"Array dimension contains zeroes, at index {(i, j, k)}")

                    with open(os.path.join(pairs_dir, subject_label, f'{s}_ds{ds}.npy'), 'wb') as f:
                        np.save(f, comb_patch, allow_pickle=False)

        if cluster_mode:
            print(f"Total time for patch generation: {time.time() - cur_time} seconds.")

    def __init__(self,
                 pair_dir,
                 subject_labels,
                 pairs_per_subject=800,
                 batch_size=12,
                 t1_postprocess=True,
                 gamma_std=0.1,
                 contrast_std=0.1,
                 brightness_std=0.1,
                 max_noise_std=0.1):

        """
        Creates a custom keras Sequence for iteration. This iterator is suitable for use with :func:`keras.model.fit()`

        :param pair_dir: Directory where the triplet data is stored at
        :param subject_labels: Labels of subjects which will be used for training
        :param pairs_per_subject: Selects a subset of triplets per epoch for training
        :param batch_size: The size of the batches in the NN for training
        :param t1_postprocess: Flag to determine whether to apply post-processing additions such as noise and
            gamma correction. Use if data is normalised with min-max normalisation.
        """

        self.pair_dir = pair_dir
        self.subject_labels = subject_labels
        self.pairs_per_subject = pairs_per_subject
        self.batch_size = batch_size
        self.t1_postprocess = t1_postprocess
        self.gamma_std = gamma_std
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.max_noise_std = max_noise_std

        self.__run_indices = self.__get_indices__()

        pass

    def __get_indices__(self) -> np.ndarray[str]:
        """
        Private function that randomly acquires self.pair_dir amount of patch triplets,
            and returns the indices for the epoch.

        Returns:
            The selected patch triplets for utilisation in the current epoch
        """

        run_indices = []

        self.__total_patches = 0

        for subj in self.subject_labels:

            subj_pairs = len(os.listdir(os.path.join(self.pair_dir, subj))) \
                if self.pairs_per_subject is None else self.pairs_per_subject

            self.__total_patches += subj_pairs

            patch_indices: np.ndarray[str] = np.stack((np.repeat(subj, subj_pairs),
                                                       choices(os.listdir(os.path.join(self.pair_dir, subj)),
                                                               k=subj_pairs)), dtype=str, axis=-1)

            run_indices.append(patch_indices)

        run_indices = np.concatenate(run_indices, axis=0)

        npr.shuffle(run_indices)

        return run_indices

    def __augment_dti__(self, input_dti):

        noise_std = self.max_noise_std * np.random.rand(1)[0]

        noise = noise_std * np.random.randn(*input_dti.shape[:-1])[..., None]

        return np.clip(input_dti + noise, a_min=0, a_max=1)

    def __augment_t1__(self, input_t1):

        gamma_t1 = np.exp(self.gamma_std * np.random.randn(1)[0])

        contrast = np.min((1.4, np.max((0.6, 1.0 + self.contrast_std * np.random.randn(1)[0]))))
        brightness = np.min((0.4, np.max((-0.4, self.brightness_std * np.random.randn(1)[0]))))

        noise_std = self.max_noise_std * np.random.rand(1)[0]

        modified_t1 = ((input_t1 - 0.5) * contrast + (0.5 + brightness)) + noise_std * np.random.randn(*input_t1.shape)

        modified_t1 = np.clip(modified_t1, 0, 1)
        modified_t1 = modified_t1 ** gamma_t1
        return modified_t1

    def __getitem__(self, index):
        """
        Acquires the patch triplet at the selected index of the training batch.
            Due to random sampling and shuffling the patch at same index values will differ between epochs.

        :param index: The index value of the batch to acquire the patch from
        :return: The acquired patch triplet of form `Tuple[tf.Tensor, tf.Tensor, tf.Tensor]`
        """

        target_patches = []
        input_patches = []
        t1_patches = []

        for (subj, patch_indx) in self.__run_indices[index:min(self.__total_patches, index + self.batch_size)]:
            patch = np.load(os.path.join(self.pair_dir, subj, patch_indx))

            hr_lim = (patch.shape[-1] - 1) // 2

            target_patches.append(patch[..., :hr_lim])
            input_patches.append(patch[..., hr_lim:-1])

            t1_patch = self.__augment_t1__(patch[..., -1:]) if self.t1_postprocess else patch[..., -1:]
            t1_patches.append(t1_patch)

        return (tf.cast(tf.stack(target_patches), dtype=tf.float32),
                (tf.cast(tf.stack(input_patches), dtype=tf.float32), tf.cast(tf.stack(t1_patches), dtype=tf.float32)))

    def sample_patch(self, subj, *_):

        patch_indx = '100_ds0.npy'

        patch = np.load(os.path.join(self.pair_dir, self.subject_labels[subj], patch_indx))

        hr_lim = (patch.shape[-1] - 1) // 2

        t_patch = patch[..., :hr_lim][None, ...]
        i_patch = patch[..., hr_lim:-1][None, ...]
        t1_patch = patch[..., -1:][None, ...]

        return tf.convert_to_tensor(t_patch), tf.convert_to_tensor(i_patch), tf.convert_to_tensor(t1_patch)

    def __len__(self):

        return ceil(self.__total_patches / self.batch_size)

    def on_epoch_end(self):

        self.__run_indices = self.__get_indices__()


class DataLoader:

    def __init__(self,
                 data_dir: str | os.PathLike[str],
                 subject_labels: Iterable[str],
                 subdir='.',
                 mode: Literal["dti", "T1w"] = 'dti',
                 file_head='dt_b1000_'):
        """
        The basic dataloader class to parse and store in memory the IQT training data. Currently this class can read,
        parse and store either T1w or DTI data.

        :param data_dir: The root data directory to read the data from
        :param subject_labels: The subject labels to read the data from. We store the
        :param subdir:
        :param mode:
        :param file_head:
        """

        self.data_dir = data_dir
        self.subdir = subdir
        self.subject_labels = subject_labels
        self.mode = mode
        self.file_head = file_head

        self.normalization_metrics = []

        self.__subjects: List[NDArray[float]] = []
        self.__subject_masks: List[NDArray[bool]] = []
        self.__subject_S0_values: List[NDArray[float]] = []

        self.load_data()

    def get_masks(self):
        return self.__subject_masks

    def set_masks(self, masks):
        self.__subject_masks = copy.deepcopy(masks)

    def resample(self,
                 sampling_rate: float | Tuple[float, float, float] = 1.25/0.7):
        """
        Resamples every subject in the

        :param sampling_rate:
        :return:
        """

        sr = sampling_rate \
            if type(sampling_rate) is tuple \
            else (sampling_rate, sampling_rate, sampling_rate)

        for i in range(len(self.__subjects)):

            subj = self.__subjects[i]

            if np.mean(sampling_rate) > 1.:
                subj = util.apply_gaussian_filter(subj, np.mean(sampling_rate))

            subj = zoom(subj,
                        zoom=(1. / sr[0],
                              1. / sr[1],
                              1. / sr[2], 1),
                        order=1,
                        prefilter=False)

            if self.mode == 'dti':
                self.__subject_S0_values[i] = zoom(self.__subject_S0_values[i],
                                                   zoom=(1. / sr[0],
                                                         1. / sr[1],
                                                         1. / sr[2]),
                                                   order=1,
                                                   prefilter=False)

            self.__subject_masks[i] = zoom(self.__subject_masks[i],
                                           zoom=(1. / sr[0],
                                                 1. / sr[1],
                                                 1. / sr[2]),
                                           order=0,
                                           prefilter=False)

            self.__subjects[i] = subj

    def normalize(self,
                  normalization_method: Literal["std", "constant", "percentile"],
                  normalization_value,
                  use_prev_metrics=False):

        match self.mode:
            case "dti":
                norm_channels = [[0, 3, 5], [1, 2, 4]]
            case _:
                norm_channels = None

        if ~use_prev_metrics:
            self.normalization_metrics = []

        for s, subj in enumerate(self.__subjects):

            if ~use_prev_metrics:

                norm_metrics = util.get_clip_values(subj, self.__subject_masks[s],
                                                    data_mode=self.mode,
                                                    clip_strategy=normalization_method,
                                                    value=normalization_value)

                self.normalization_metrics.append(norm_metrics)

            util.apply_normalization_combined(subj, self.__subject_masks[s],
                                              method="minmax" if normalization_method != "std" else "std",
                                              channels=norm_channels,
                                              values=self.normalization_metrics[s])

            subj[~self.__subject_masks[s], :] = 0

    def denormalize(self, normalization_method):

        match self.mode:
            case "dti":
                norm_channels = [[0, 3, 5], [1, 2, 4]]
            case _:
                norm_channels = None

        for s, subj in enumerate(self.__subjects):
            util.revert_normalization_combined(subj,
                                               self.__subject_masks[s],
                                               self.normalization_metrics[s],
                                               method="minmax" if normalization_method != "std" else "std",
                                               channels=norm_channels)
            subj[~self.__subject_masks[s], :] = 0

    def pad(self, pad_length):

        if pad_length > 0:

            for s in range(len(self.__subjects)):

                self.__subjects[s] = np.pad(self.__subjects[s],
                                            pad_width=np.array([[pad_length, pad_length],
                                                                [pad_length, pad_length],
                                                                [pad_length, pad_length],
                                                                [0, 0]]), mode='edge')
                self.__subject_masks[s] = np.pad(self.__subject_masks[s],
                                                 pad_width=np.array([[pad_length, pad_length],
                                                                     [pad_length, pad_length],
                                                                     [pad_length, pad_length]]), mode='edge')

                if self.__subject_S0_values is not None:
                    self.__subject_S0_values[s] = np.pad(self.__subject_S0_values[s],
                                                         pad_width=np.array([[pad_length, pad_length],
                                                                             [pad_length, pad_length],
                                                                             [pad_length, pad_length]]), mode='edge')

        else:
            for s in range(len(self.__subjects)):
                self.__subjects[s] = self.__subjects[s][pad_length:-pad_length,
                                                        pad_length:-pad_length,
                                                        pad_length:-pad_length, :]
                self.__subject_masks[s] = self.__subject_masks[s][pad_length:-pad_length,
                                                                  pad_length:-pad_length,
                                                                  pad_length:-pad_length]
                if self.__subject_S0_values is not None:
                    self.__subject_S0_values[s] = self.__subject_S0_values[s][pad_length:-pad_length,
                                                                              pad_length:-pad_length,
                                                                              pad_length:-pad_length]

    def __getitem__(self, item) -> (Tuple[NDArray[float], NDArray[bool]] |
                                    Tuple[NDArray[float], NDArray[bool], NDArray[float]]):
        if self.mode == 'dti':
            return self.__subjects[item], self.__subject_masks[item], self.__subject_S0_values[item]
        elif self.mode == 'T1w':
            return self.__subjects[item], self.__subject_masks[item]
        else:
            raise NotImplementedError("This modality does not exist!")

    def load_data(self):

        match str.lower(self.mode):

            case "dti":
                for subject in self.subject_labels:

                    subject_data, _ = util.load_dtis(
                        os.path.join(self.data_dir, subject, self.subdir),
                        self.file_head
                    )

                    self.__subject_masks.append(np.array(subject_data[..., 0] >= 0, dtype=bool))
                    self.__subject_S0_values.append(np.copy(subject_data[..., 1]))

                    subject_data = subject_data[..., 2:]

                    self.__subjects.append(subject_data)

                return

            case "map":
                raise NotImplementedError("MAP-MRI implementation is not yet implemented.")

            case "t1w":

                self.__subject_S0_values = None

                for subject in self.subject_labels:

                    subject_data, _ = util.load_structural(
                        os.path.join(self.data_dir, subject, self.subdir),
                        self.file_head
                    )

                    subject_mask = np.array(subject_data[..., 0] > 0, dtype=bool)

                    self.__subject_masks.append(subject_mask)
                    self.__subjects.append(subject_data)

                return

            case _:

                raise TypeError("Unsupported data mode: {}".format(self.mode))

    def shape(self, subj):

        if str.lower(self.mode) == 't1w':
            return self.__subjects[subj].shape

        return self.__subjects[subj].shape[:-1]

    def get_patch(self,
                  subj: int,
                  coords: Tuple[int, int, int],
                  patch_size=16) -> tf.Tensor:
        (i, j, k) = coords

        return tf.convert_to_tensor(self.__subjects[subj][
                                    i - patch_size // 2:i + ceil(patch_size / 2),
                                    j - patch_size // 2:j + ceil(patch_size / 2),
                                    k - patch_size // 2:k + ceil(patch_size / 2),
                                    :], dtype='float')

    def get_mask(self,
                 subj: int,
                 coords: List[int],
                 patch_size=16) -> NDArray[bool]:
        (i, j, k) = coords

        return self.__subject_masks[subj][
               i - patch_size // 2:i + ceil(patch_size / 2),
               j - patch_size // 2:j + ceil(patch_size / 2),
               k - patch_size // 2:k + ceil(patch_size / 2)]

    def get_grid_indices(self,
                         subj: int,
                         ipatch_size=5,
                         opatch_size=3,
                         overlap=0) -> List[Tuple[int, int, int]]:

        subj_img = self.__subjects[subj]

        (xsize, ysize, zsize, _) = subj_img.shape

        recon_indx = [(i, j, k)
                      for k in np.arange(ipatch_size + 1,
                                         zsize - ipatch_size + 1,
                                         2 * opatch_size + 1 - overlap)
                      for j in np.arange(ipatch_size + 1,
                                         ysize - ipatch_size + 1,
                                         2 * opatch_size + 1 - overlap)
                      for i in np.arange(ipatch_size + 1,
                                         xsize - ipatch_size + 1,
                                         2 * opatch_size + 1 - overlap)]

        return recon_indx

    def get_mask_indices(self,
                         subj: int,
                         patch_spacing=12,
                         random_shift=True):
        """
        Returns every patch centroid coordinate with patches containing masked voxels with a certain spacing.

        :param subj: The subject to the mask indices from. This value is the array index and not the subject id.
        :param patch_spacing: The spacing between patch centroids.
            A spacing of 12 for patch sizes of 16 means that 8**3 voxels at the centre do not overlap.
        :param random_shift: A randomizer for shifting the patch centroid grid. This doesn't randomize the spacing, but
            rather the centroid values of the patch grid, meaning that the patches are still equally spaced but have the
            centre indices shifted by the same amount.
        :return: An integer array with valid patch indices of form [N, 3] where N is the number of valid indices.
        """

        mask = self.__subject_masks[subj]

        sel_mask_indices = np.zeros(mask.shape, dtype=bool)

        randshift = (randint(0, patch_spacing // 4),
                     randint(0, patch_spacing // 4),
                     randint(0, patch_spacing // 4)) if random_shift else (0, 0, 0)

        sel_mask_indices[randshift[0]::patch_spacing,
                         randshift[1]::patch_spacing,
                         randshift[2]::patch_spacing] = True

        indices = np.array(np.where(sel_mask_indices & mask)).T

        return indices


class TrainingSequence(keras.utils.Sequence):

    def __init__(self,
                 dti_data_dir,
                 dti_data_subdir,
                 subject_labels,
                 t1_data_dir,
                 t1_data_subdir,
                 mode,
                 hr_downsamp_rate=1.,
                 lr_downsamp_rates=[1.25/0.7],
                 pairs_per_subject: int | None = 8000,
                 patch_size=16,
                 patch_spacing=12,
                 clip_strategy: Literal["std", "constant", "percentile"] = 'constant',
                 clip_value=2e-3,
                 random_shift=False,
                 batch_size=6,
                 augment_t1=True,
                 gamma_std=0.1,
                 contrast_std=0.1,
                 brightness_std=0.1,
                 max_noise_std=0.1):
        """
        The data loader for the IQT pipeline. This function reads, processes and stores in memory the High-Resolution
        DTI images, the (possibly resampled) High-Resolution T1w data and the multiple Low-Resolution DTI data.

        This implementation resembles the `DataLoader` primitive from PyTorch, and requires two functions to be defined
        in order to be used by Keras' `Model.fit()`: `__getitem__` and `__len__`.

        The data loader loads the data from the high-res DTI directories of format
        {dti_data_dir}/{subject_label}/{dti_data_subdir}, and high-res T1w input as
        {t1_data_dir}/{subject_label}/{t1_data_subdir}

        Args:
            dti_data_dir: The root data directory for the DTI input to be read from.
            dti_data_subdir: The data subdirectory of the DTI input (After the subject label directory)
            subject_labels: The list of subject labels for the data to be loaded.
            t1_data_dir: The root data directory for the T1 weighted input to be read from. Set this to dti dir if they
            are located under the same directory.
            t1_data_subdir: The data subdirectory under subjects where the T1w input can be found.
            mode: The data mode of the dataloader for parsing and processing the diffusion input.
                Currently only \"dti\" is supported.
            hr_downsamp_rate: The (single) High-Resolution Target downsampling rate.
                If > 1, the input data will be downsampled by the specified rate. The new resolution will thus be
                R_target = R_input * hr_downsamp_rate
            lr_downsamp_rates: The downsampling rates for the DTI to be scaled down to.
                The resulting image will be of final resolution R_lr = R_input * hr_downsamp_rate, but will contain as
                much information as an image at resolution
                R_lract = R_input * hr_downsamp_rate * lr_downsamp_rate.
                This value is multiplicative with the hr downsampling rate, meaning that the downsampling from input
                is the product of the two rates.
            pairs_per_subject: Number of HR target - LR input pairs selected per subject scan.
                The loader randomly selects this amount from all upsampling rates per each subject.
                If set to None the loader selects every patch for all upsampling rates, but shuffles them.
            patch_size: The length of the patch for our model. The resulting patch will be of volume (patch_size)^3.
            patch_spacing: The spacing between every patch centroids.
            clip_strategy: The clipping strategy employed by the normalisation algorithm.
                Valid options are "std" for clipping based on subjectwise standard deviation values,
                "percentile" for clipping between specific percentiles and "constant" for constant values.
                Currently, T1w is only defined for "percentile",
                while DTI has "constant" and "std" configurations in addition to "percentile".
            clip_value: The value used for the clipping algorithm.

                    For "constant", the values are clipped to either [0, clip_value] for diagonal DTI values,
                    or [-clip_value, clip_value] for non-diagonal DTI values.

                    For "percentile", the values are clipped to [0, percentile] for diagonal DTI values,
                    [-percentile, percentile] for non-diagonal DTI values,
                    where percentile is the higher percentile value corresponding to the clip_value's percentile.
                    For the T1w data, the input is clipped to range [lower_percentile, higher_percentile], such that
                    the difference between the lower and higher percentiles is equal to the clip value.

                    For "std" the values are clipped to specific std range specified by the clip value. For example,
                    clip value of 2 means that the images are clipped to 2 standard deviation ranges.
            random_shift: The boolean value that determines whether the patch grid will be randomly shifted.
                See `DataLoader.get_mask_indices` for more details.
            batch_size: The minibatch size for our training. After the iteration patches are selected,
                we select minibatches and pass it per every training iteration in an epoch.
            augment_t1: Boolean value that determines whether the T1w data will be augmented randomly.
            gamma_std: The maximum std
            contrast_std:
            brightness_std:
            max_noise_std:
        """

########################################################################################################################

# ------------------------------------------- Dataloader Parameter Setup -----------------------------------------------

########################################################################################################################

        self.subject_labels = subject_labels

        self.patch_size = patch_size
        self.patch_spacing = patch_spacing

        self.clip_strategy = clip_strategy
        self.clip_value = clip_value

        self.random_shift = random_shift

        self.batch_size = batch_size
        self.pairs_per_subject = pairs_per_subject

        self.augment_t1 = augment_t1

        self.gamma_std = gamma_std
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.max_noise_std = max_noise_std

        self.hr_downsamp_rate = hr_downsamp_rate
        self.lr_downsamp_rates = lr_downsamp_rates

        self.__input_data: List[DataLoader] = []

########################################################################################################################

# --------------------------------------- DTI Data Target Load and Setup -----------------------------------------------

########################################################################################################################

        orig_data = DataLoader(data_dir=dti_data_dir,
                               subject_labels=subject_labels,
                               subdir=dti_data_subdir,
                               mode=mode)

        self.__target_data: DataLoader = copy.deepcopy(orig_data)  # Copy by value and keep orig as reference

        self.__target_data.resample(hr_downsamp_rate)  # Resample to a specific resolution if necessary

        hr_shape = self.__target_data[0][0].shape  # Keep target DTI shape as reference

        print("Loaded target data.")

########################################################################################################################

# --------------------------------------- DTI Low-Resolution Input Setup -----------------------------------------------

########################################################################################################################

        for lr_downsamp_rate in tqdm(self.lr_downsamp_rates):

            idata: DataLoader = copy.deepcopy(orig_data)

            # Resample to lower space from original DTI input resolution.
            # This is done because the downsampling requires a gaussian blurring to remove aliasing effects, and
            # applying gaussian blurring twice to go from orig -> target -> low-res would strip too much information
            # to be useful.
            idata.resample(hr_downsamp_rate*lr_downsamp_rate)

            lr_shape = idata[0][0].shape

            # Fit to exact dimensions of the DTI target image dimensions.
            # This ensures that the image dimensions will be equal.
            resize_scale = (lr_shape[0] / hr_shape[0],
                            lr_shape[1] / hr_shape[1],
                            lr_shape[2] / hr_shape[2])

            idata.resample(resize_scale)

            # This configuration ensures that
            idata.set_masks(self.__target_data.get_masks())

            # Apply padding to ensure that selected patches are always of correct size.
            idata.pad(self.patch_size // 2)

            idata.normalize(self.clip_strategy, self.clip_value)

            self.__input_data.append(idata)

        print("Loaded low-res input data.")

########################################################################################################################

# --------------------------------------- DTI Low-Resolution Input Setup -----------------------------------------------

########################################################################################################################

        self.__t1_data: DataLoader = DataLoader(data_dir=t1_data_dir,
                                                subject_labels=subject_labels,
                                                subdir=t1_data_subdir,
                                                mode='T1w',
                                                file_head='T1w_acpc_dc_restore_brain')

        # This implementation (rather a bit naively) assumes that every training subject has the same dimensions.
        # While this is true for HCP, if you want to train on different datasets you should ensure every subject has
        # the same dimensions.
        t1_shape = self.__t1_data[0][0].shape

        # Apply similar rescaling as how we have done it for LR->target.
        # However, you must ensure that:
        # a) The images are centered,
        # b) The T1w image contains every region contained within the DTI image, and vice versa.
        resize_scale = (t1_shape[0] / hr_shape[0],
                        t1_shape[1] / hr_shape[1],
                        t1_shape[2] / hr_shape[2])

        self.__t1_data.resample(resize_scale)

        self.__target_data.pad(self.patch_size // 2)
        self.__t1_data.pad(self.patch_size // 2)

        self.__t1_data.set_masks(self.__target_data.get_masks())

        self.__target_data.normalize(self.clip_strategy, self.clip_value)
        self.__t1_data.normalize("percentile", 96)

        print("Loaded structural input data.")

        self.valid_patch_indices: List[np.ndarray] = self.__find_all_patch_indices__()

        self.__sel_epoch_indices__()

        self.__no_batches = ceil(self.__cur_epoch_indices.shape[0] / self.batch_size)

    def sample_patch(self, subj, coords: Tuple[int, int, int]):
        """
        Choose a sample patch
        :param subj:
        :param coords:
        :return:
        """

        i_patch = self.__input_data[0].get_patch(subj, coords, self.patch_size)[None,...]
        t_patch = self.__target_data.get_patch(subj, coords, self.patch_size)[None,...]
        t1_patch = self.__t1_data.get_patch(subj, coords, self.patch_size)[None,...]

        return tf.convert_to_tensor(t_patch), tf.convert_to_tensor(i_patch), tf.convert_to_tensor(t1_patch)

    def __sel_epoch_indices__(self) -> None:

        cur_epoch_indices = []

        for subj_index in self.valid_patch_indices:
            cur_epoch_indices.append(
                subj_index[npr.choice(subj_index.shape[0],
                                      size=subj_index.shape[0]
                                      if self.pairs_per_subject is None
                                      else self.pairs_per_subject,
                                      replace=False)]
            )

        cur_epoch_indices = np.concatenate(cur_epoch_indices, axis=0)

        npr.shuffle(cur_epoch_indices)

        self.__cur_epoch_indices = cur_epoch_indices

        return

    def __find_all_patch_indices__(self) -> List[np.ndarray]:

        valid_patch_indices = []

        for s in range(len(self.subject_labels)):

            subj_valid_indices = []

            for r in range(len(self.lr_downsamp_rates)):
                valid_patch_index = self.__target_data.get_mask_indices(s, self.patch_spacing, self.random_shift)

                subj_valid_indices.append(
                    np.concatenate([
                        np.full((valid_patch_index.shape[0], 1), s),
                        np.full((valid_patch_index.shape[0], 1), r),
                        valid_patch_index], axis=-1)
                )

            valid_patch_indices.append(np.concatenate(subj_valid_indices))

        return valid_patch_indices

    def __augment_t1__(self, input_t1):

        gamma_t1 = np.exp(self.gamma_std * np.random.randn(1)[0])

        contrast = np.min((1.4, np.max((0.6, 1.0 + self.contrast_std * np.random.randn(1)[0]))))
        brightness = np.min((0.4, np.max((-0.4, self.brightness_std * np.random.randn(1)[0]))))

        noise_std = self.max_noise_std * np.random.rand(1)[0]

        modified_t1 = ((input_t1 - 0.5) * contrast + (0.5 + brightness)) + noise_std * np.random.randn(*input_t1.shape)

        modified_t1 = np.clip(modified_t1, 0, 1)
        modified_t1 = modified_t1 ** gamma_t1
        return modified_t1

    def __getitem__(self, index):

        target_patches = []
        input_patches = []
        t1_patches = []

        for (s, r, i, j, k) in self.__cur_epoch_indices[index:min(index + self.batch_size, self.__no_batches)]:
            target_patch = self.__target_data.get_patch(s, (i, j, k), self.patch_size)

            input_patch = self.__input_data[r].get_patch(s, (i, j, k), self.patch_size)

            t1_patch = self.__t1_data.get_patch(s, (i, j, k), self.patch_size)

            if self.augment_t1:
                t1_patch = self.__augment_t1__(t1_patch)

            target_patches.append(target_patch)
            input_patches.append(input_patch)
            t1_patches.append(t1_patch)

        return tf.stack(target_patches), (tf.stack(input_patches), tf.stack(t1_patches))

    def __len__(self):
        return self.__no_batches

    def on_epoch_end(self):

        self.__sel_epoch_indices__()

