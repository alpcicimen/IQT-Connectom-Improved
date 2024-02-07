import numpy as np
import numpy.random as npr
import tensorflow as tf
import keras
import os

from math import ceil
from random import randint, choices
from typing import List, Tuple

from IQT import util

from scipy.ndimage import zoom


class PairLoader:

    def __init__(self,
                 diff_data_dir,
                 t1_data_dir,
                 subject_labels,
                 pairs_dir,
                 lr_filedir=os.path.join('LR', 'dt_b1000_lowres_2_'),
                 upsampling_rate=1.25/.7,
                 hr_filedir=os.path.join('HR', 'dt_b1000_'),
                 t1_filedir=os.path.join('T1w', 'T1w_acpc_dc_restore_brain'),
                 mode='dti',
                 patch_spacing=8,
                 patch_size=16):

        for subject_label in subject_labels:

            if not os.path.exists(os.path.join(pairs_dir, subject_label)):
                os.mkdir(os.path.join(pairs_dir, subject_label))

            # subject_data_lr = util.load_dtis(
            #     os.path.join(diff_data_dir, subject_label),
            #     lr_filedir
            # )[..., 2:]

            subject_data_hr = util.load_dtis(
                os.path.join(diff_data_dir, subject_label),
                hr_filedir
            )

            subject_data_lr = zoom(subject_data_hr[..., 2:],
                                   zoom=(1./upsampling_rate,
                                         1./upsampling_rate,
                                         1./upsampling_rate, 1))

            lr_dims = (np.array(subject_data_hr.shape[:-1] + (1,)) /
                       np.array(subject_data_lr.shape[:-1] + (1,)))

            subject_data_lr = zoom(subject_data_lr, lr_dims)

            subject_data_t1 = util.load_structural(
                os.path.join(t1_data_dir, subject_label),
                t1_filedir
            )

            target_scales = (np.array(subject_data_hr.shape[:-1] + (1,)) /
                             np.array(subject_data_t1.shape))

            mask = np.array(subject_data_hr[..., 0] == 0, dtype=bool)
            subject_data_hr = subject_data_hr[..., 2:]

            subject_data_t1 = zoom(subject_data_t1, target_scales)

            util.apply_normalization(subject_data_lr, mask, method='stdscore')
            util.apply_normalization(subject_data_hr, mask, method='stdscore')
            util.apply_normalization(subject_data_t1, mask, method='stdscore')

            sel_mask_indices = np.zeros(mask.shape, dtype=bool)

            sel_mask_indices[randint(0, patch_spacing // 4)::patch_spacing,
                             randint(0, patch_spacing // 4)::patch_spacing,
                             randint(0, patch_spacing // 4)::patch_spacing] = True

            sel_mask_indices = np.array(np.where(sel_mask_indices & mask)).T

            for s, (i, j, k) in enumerate(sel_mask_indices):

                comb_patch = np.concatenate([
                    subject_data_hr[i-patch_size//2:i+round(patch_size/2),
                                    j-patch_size//2:j+round(patch_size/2),
                                    k-patch_size//2:k+round(patch_size/2), :],
                    subject_data_lr[i-patch_size//2:i+round(patch_size/2),
                                    j-patch_size//2:j+round(patch_size/2),
                                    k-patch_size//2:k+round(patch_size/2), :],
                    subject_data_t1[i-patch_size//2:i+round(patch_size/2),
                                    j-patch_size//2:j+round(patch_size/2),
                                    k-patch_size//2:k+round(patch_size/2), :],
                ], axis=-1)

                with open(os.path.join(pairs_dir, subject_label, f'{s}.npy'), 'wb') as f:
                    np.save(f, comb_patch, allow_pickle=False)


class PairSequence(keras.utils.Sequence):

    def __init__(self,
                 pair_dir,
                 subject_labels,
                 pairs_per_subject=800,
                 batch_size=12):

        self.pair_dir = pair_dir
        self.subject_labels = subject_labels
        self.pairs_per_subject = pairs_per_subject
        self.batch_size = batch_size

        self.__total_patches = len(self.subject_labels) * self.pairs_per_subject

        self.__run_indices = self.__get_indices__()

        pass

    def __get_indices__(self) -> np.ndarray[str]:

        run_indices = []

        for subj in self.subject_labels:

            patch_indices: np.ndarray[str] = np.stack((np.repeat(subj, self.pairs_per_subject),
                                                       choices(os.listdir(os.path.join(self.pair_dir, subj)),
                                                               k=self.pairs_per_subject)), dtype=str, axis=-1)

            run_indices.append(patch_indices)

        run_indices = np.concatenate(run_indices, axis=0)

        npr.shuffle(run_indices)

        return run_indices

    def __getitem__(self, index):

        target_patches = []
        input_patches = []
        t1_patches = []

        for (subj, patch_indx) in self.__run_indices[index:min(self.__total_patches, index+self.batch_size)]:

            patch = np.load(os.path.join(self.pair_dir, subj, patch_indx))

            hr_lim = (patch.shape[-1] - 1) // 2

            target_patches.append(patch[..., :hr_lim])
            input_patches.append(patch[..., hr_lim:-1])
            t1_patches.append(patch[..., -1:])

        return tf.stack(target_patches), (tf.stack(input_patches), tf.stack(t1_patches))

    def sample_slice(self, subj, *_):

        patch_indx = '400.npy'

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
                 data_dir,
                 subject_labels,
                 subdir='.',
                 mode='dti',
                 file_head='dt_b1000_',
                 normalization_method: str | None = 'minmax'):
        self.data_dir = data_dir
        self.subdir = subdir
        self.subject_labels = subject_labels
        self.mode = mode
        self.file_head = file_head

        self.normalization_method = normalization_method
        self.normalization_metrics = []

        self.__subjects = []
        self.__subject_masks = []
        self.__subject_S0_values = []

        self.load_data()

    def load_data(self):

        match self.mode:

            case "dti":
                for subject in self.subject_labels:

                    subject_data = util.load_dtis(
                        os.path.join(self.data_dir, subject, self.subdir),
                        self.file_head
                    )

                    subject_mask = subject_data[..., 0]

                    self.__subject_masks.append(np.copy(subject_mask))
                    self.__subject_S0_values.append(np.copy(subject_data[..., 1]))

                    subject_data = subject_data[..., 2:]

                    if self.normalization_method is not None:
                        self.normalization_metrics.append(
                            util.apply_normalization(subject_data,
                                                     np.array(subject_mask == 0),
                                                     self.normalization_method))

                    self.__subjects.append(subject_data)

                return

            case "map":
                return

            case "T1w":

                self.__subject_S0_values = None

                for subject in self.subject_labels:

                    subject_data = util.load_structural(
                        os.path.join(self.data_dir, subject, self.subdir),
                        self.file_head
                    )

                    subject_mask = np.array(subject_data != 0)[..., 0]

                    self.__subject_masks.append(subject_mask)

                    if self.normalization_method is not None:
                        self.normalization_metrics.append(
                            util.apply_normalization(subject_data,
                                                     subject_mask,
                                                     self.normalization_method))

                    self.__subjects.append(subject_data)

                return

            case _:

                raise TypeError("Unsupported data mode: {}".format(self.mode))

    def shape(self, subj):

        if self.mode == 'T1w':
            return self.__subjects[subj].shape

        return self.__subjects[subj].shape[:-1]

    def get_patch(self,
                  subj: int,
                  coords: Tuple[int, int, int],
                  patch_size=5) -> tf.Tensor:
        (i, j, k) = coords

        return tf.convert_to_tensor(self.__subjects[subj][
                                    i - patch_size // 2:i + ceil(patch_size / 2),
                                    j - patch_size // 2:j + ceil(patch_size / 2),
                                    k - patch_size // 2:k + ceil(patch_size / 2),
                                    :], dtype='float')

    def get_mask(self,
                 subj: int,
                 coords: List[int],
                 patch_size=5) -> np.ndarray:
        (i, j, k) = coords

        return (self.__subject_masks[subj][
                i - patch_size // 2:i + ceil(patch_size / 2),
                j - patch_size // 2:j + ceil(patch_size / 2),
                k - patch_size // 2:k + ceil(patch_size / 2)] == 0)

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
                         patch_size=5):

        mask = (self.__subject_masks[subj] == 0)

        dims = mask.shape

        possible_indices = np.array(np.where(mask)).T

        is_keep: np.ndarray[bool] = np.zeros((possible_indices.shape[0], 6), dtype=bool)

        is_keep[:, 0] = (possible_indices[:, 0] - patch_size) >= 0
        is_keep[:, 1] = (possible_indices[:, 0] + patch_size) < dims[0]
        is_keep[:, 2] = (possible_indices[:, 1] - patch_size) >= 0
        is_keep[:, 3] = (possible_indices[:, 1] + patch_size) < dims[1]
        is_keep[:, 4] = (possible_indices[:, 2] - patch_size) >= 0
        is_keep[:, 5] = (possible_indices[:, 2] + patch_size) < dims[2]

        is_keep = np.all(is_keep, axis=1)
        row_list = np.delete(np.array(range(possible_indices.shape[0])), np.where(is_keep == False), 0)

        indices = possible_indices[row_list, :]

        return indices


class TrainingSequence(keras.utils.Sequence):

    def __init__(self,
                 data_dir,
                 subject_labels,
                 target_dir,
                 input_dir,
                 mode,
                 data_dir_t1=None,
                 t1_dir='T1w',
                 normalization_method='stdscore',
                 pairs_per_subject=8000,
                 ipatch_size=5,
                 opatch_size=3,
                 batch_size=6):

        self.subject_labels = subject_labels
        self.ipatch_size = ipatch_size
        self.opatch_size = opatch_size
        self.batch_size = batch_size
        self.pairs_per_subject = pairs_per_subject

        self.__no_batches = ceil(len(self.subject_labels) * self.pairs_per_subject / self.batch_size)

        self.__target_data: DataLoader = DataLoader(data_dir=data_dir,
                                                    subject_labels=subject_labels,
                                                    subdir=target_dir,
                                                    mode=mode,
                                                    normalization_method=normalization_method)

        print("Loaded target data.")

        self.__input_data: DataLoader = DataLoader(data_dir=data_dir,
                                                   subject_labels=subject_labels,
                                                   subdir=input_dir,
                                                   mode=mode,
                                                   normalization_method=normalization_method,
                                                   file_head='dt_b1000_lowres_4_')

        print("Loaded low-res input data.")

        self.__t1_data: DataLoader = DataLoader(data_dir=(data_dir_t1 if data_dir_t1 is not None else data_dir),
                                                subject_labels=subject_labels,
                                                subdir=t1_dir,
                                                mode='T1w',
                                                normalization_method=normalization_method,
                                                file_head='T1w_acpc_dc_restore_brain')

        print("Loaded structural input data.")

        self.valid_patch_indices: List[np.ndarray] = self.__find_all_patch_indices__()

        self.__cur_epoch_indices = None

        self.__sel_epoch_indices__()

        # npr.shuffle(self.valid_patch_indices)

    def sample_slice(self, subj, coords: Tuple[int, int, int]):

        i_patch = self.__input_data.get_patch(subj, coords, self.ipatch_size)
        t_patch = self.__target_data.get_patch(subj, coords, self.opatch_size)

        t1_coords = (round(coords[0] * 1.25 / 0.7),
                     round(coords[1] * 1.25 / 0.7),
                     round(coords[2] * 1.25 / 0.7))  # Will change, placeholder values from existing HCP data

        t1_patch = self.__t1_data.get_patch(subj, t1_coords,
                                            self.opatch_size * 2)  # round(self.opatch_size * 1.25/0.7))

        return tf.convert_to_tensor(t_patch), tf.convert_to_tensor(i_patch), tf.convert_to_tensor(t1_patch)

    def __sel_epoch_indices__(self) -> None:

        cur_epoch_indices = []

        for subj_index in self.valid_patch_indices:
            cur_epoch_indices.append(
                subj_index[npr.choice(subj_index.shape[0], size=self.pairs_per_subject, replace=False)]
            )

        cur_epoch_indices = np.concatenate(cur_epoch_indices, axis=0)

        npr.shuffle(cur_epoch_indices)

        self.__cur_epoch_indices = cur_epoch_indices

        return

    def __find_all_patch_indices__(self) -> List[np.ndarray]:

        valid_patch_indices = []

        for s in range(len(self.subject_labels)):
            valid_patch_index = self.__target_data.get_mask_indices(s, self.ipatch_size)

            valid_patch_indices.append(
                np.concatenate(
                    [np.full((valid_patch_index.shape[0], 1), s), valid_patch_index],
                    axis=-1
                )
            )

        return valid_patch_indices
        # return np.concatenate(valid_patch_indices, axis=0)

    def __getitem__(self, index):

        target_patches = []
        input_patches = []
        t1_patches = []

        # for (s, i, j, k) in self.valid_patch_indices[index:index + self.batch_size]:
        for (s, i, j, k) in self.__cur_epoch_indices[index:min(index + self.batch_size, self.__no_batches)]:
            target_patch = self.__target_data.get_patch(s, (i, j, k), self.opatch_size)
            input_patch = self.__input_data.get_patch(s, (i, j, k), self.ipatch_size)

            t1_coords = (round(i * 1.25 / 0.7),
                         round(j * 1.25 / 0.7),
                         round(k * 1.25 / 0.7))  # Will change, placeholder values from existing HCP data

            t1_patch = self.__t1_data.get_patch(s, t1_coords,
                                                self.opatch_size * 2)  # round(self.opatch_size * 1.25 / 0.7))

            target_patches.append(target_patch)
            input_patches.append(input_patch)
            t1_patches.append(t1_patch)

        return tf.stack(target_patches), (tf.stack(input_patches), tf.stack(t1_patches))

    def __len__(self):
        # return ceil(self.valid_patch_indices.shape[0] / self.batch_size)

        return self.__no_batches

    def on_epoch_end(self):

        self.__sel_epoch_indices__()

        # npr.shuffle(self.valid_patch_indices)

# if __name__ == '__main__':
#
#     seq = PairSequence(pair_dir='../data/patch_pairs',
#                        subject_labels=['100307', '221319'], pairs_per_subject=800, batch_size=12)
#
#     patch = seq[5]
#
#     pass

