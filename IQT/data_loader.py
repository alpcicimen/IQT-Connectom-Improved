import numpy as np
import numpy.random as npr
import tensorflow as tf
import os
import keras

from math import ceil
from typing import List

import util


class DataLoader:

    def __init__(self,
                 data_dir,
                 subject_labels,
                 subdir='.',
                 mode='dti',
                 file_head='dt_b1000_',
                 normalisation_method='minmax'):
        self.data_dir = data_dir
        self.subdir = subdir
        self.subject_labels = subject_labels
        self.mode = mode
        self.file_head = file_head

        self.normalisation_method = normalisation_method

        self.__subjects = []
        self.__subject_masks = []
        self.__subject_S0_values = []

        self.load_data()

    def load_data(self):
        for subject in self.subject_labels:
            self.__subjects.append(
                util.load_dtis(
                    os.path.join(self.data_dir, subject, self.subdir),
                    self.file_head
                ))

    def get_patch(self,
                  subj: int,
                  coords: List[int],
                  patch_size=5) -> tf.Tensor:
        (i, j, k) = coords

        return tf.convert_to_tensor(self.__subjects[subj][
                                    i - patch_size // 2:i + ceil(patch_size / 2),
                                    j - patch_size // 2:j + ceil(patch_size / 2),
                                    k - patch_size // 2:k + ceil(patch_size / 2),
                                    2:
                                    ])

    def get_mask(self,
                 subj: int,
                 coords: List[int],
                 patch_size=5) -> np.ndarray:
        (i, j, k) = coords

        return (self.__subjects[subj][
                i - patch_size // 2:i + ceil(patch_size / 2),
                j - patch_size // 2:j + ceil(patch_size / 2),
                k - patch_size // 2:k + ceil(patch_size / 2), 0] == 0)

    def get_mask_indices(self,
                         subj: int,
                         patch_size=5):
        mask = (self.__subjects[subj][..., 0] == 0)

        dims = mask.shape

        possible_indices = np.array(np.where(mask))

        is_keep = np.zeros((possible_indices.shape[0], 6), dtype=bool)

        is_keep[:, 0] = (possible_indices[:, 0] - 5) >= 0
        is_keep[:, 1] = (possible_indices[:, 0] + 5) < dims[0]
        is_keep[:, 2] = (possible_indices[:, 1] - 5) >= 0
        is_keep[:, 3] = (possible_indices[:, 1] + 5) < dims[1]
        is_keep[:, 4] = (possible_indices[:, 2] - 5) >= 0
        is_keep[:, 5] = (possible_indices[:, 2] + 5) < dims[2]

        is_keep = np.all(is_keep, axis=1)
        row_list = np.delete(np.array(range(possible_indices.shape[0])), np.where(is_keep is False), 0)

        indices = possible_indices[row_list, :]

        return indices.T


class TrainingSequence(keras.utils.Sequence):

    def __init__(self,
                 data_dir,
                 subject_labels,
                 target_dir,
                 input_dir,
                 mode,
                 patch_size=5,
                 batch_size=6):

        self.subject_labels = subject_labels
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.__target_data: DataLoader = DataLoader(data_dir=data_dir,
                                                    subject_labels=subject_labels,
                                                    subdir=target_dir,
                                                    mode=mode)

        self.__input_data: DataLoader = DataLoader(data_dir=data_dir,
                                                   subject_labels=subject_labels,
                                                   subdir=input_dir,
                                                   mode=mode,
                                                   file_head='dt_b1000_lowres_')

        # valid_indices = self.__find_all_patch_indices__()

        self.valid_patch_indices: np.ndarray = self.__find_all_patch_indices__()

        npr.shuffle(self.valid_patch_indices)

    def __find_all_patch_indices__(self) -> np.ndarray:

        valid_patch_indices = []

        for s in range(len(self.subject_labels)):

            valid_patch_index = self.__target_data.get_mask_indices(s)

            valid_patch_indices.append(
                np.concatenate(
                    [np.full((valid_patch_index.shape[0], 1), s), valid_patch_index],
                    axis=-1
                )
            )

        return np.concatenate(valid_patch_indices, axis=0)

    def __getitem__(self, index):

        target_patches = []
        input_patches = []

        for (s, i, j, k) in self.valid_patch_indices[index:index + self.batch_size]:
            target_patch = self.__target_data.get_patch(s, [i, j, k], self.patch_size)
            input_patch = self.__input_data.get_patch(s, [i, j, k], self.patch_size)

            target_patches.append(target_patch)
            input_patches.append(input_patch)

        return tf.stack(target_patches), tf.stack(input_patches)

    def __len__(self):
        return ceil(self.valid_patch_indices.shape[0] / self.batch_size)

    def on_epoch_end(self):

        npr.shuffle(self.valid_patch_indices)

