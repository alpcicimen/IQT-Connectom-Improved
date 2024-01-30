import numpy as np
import nibabel as nib
import os


def load_dtis(directory: str,
              file_head: str) -> np.ndarray:
    subject_dts = []

    for i in range(1, 9):
        if os.path.exists(os.path.join(directory, f"{file_head}{i}.nii")):
            subj = np.array(nib.load(os.path.join(directory, f"{file_head}{i}.nii")).dataobj)
        elif os.path.exists(os.path.join(directory, f"{file_head}{i}.nii.gz")):
            subj = np.array(nib.load(os.path.join(directory, f"{file_head}{i}.nii.gz")).dataobj)
        else:
            raise FileNotFoundError("No such files in directory.")

        subject_dts.append(subj)

    merged_dts = np.stack(subject_dts, axis=-1)
    return merged_dts


def load_maps(directory: str,
              file_head: str) -> np.ndarray:
    pass


def apply_normalization(tensors, mask, method='minmax') -> np.ndarray:
    """
    Applies either min-max or standard score normalisation on the data.
    The normalisation is applied to the reference.

    :param tensors: The tensor input to apply normalisation to.
    The input variable is modified instead of returning a new variable.
    :param: mask: The tensor mask that determines which voxels will be normalised.
    :param method: Which normalisation function to apply.
    The options are minmax for min-max, or stdscore for standard score.
    :return: The normalisation metrics which can be used to revert the normalisation.
    """

    norm_metrics = np.zeros((6, 2))

    if method == 'minmax':

        for t in range(6):

            tensor = tensors[..., t]
            metric = [np.min(tensor[mask]), np.max(tensor[mask])]
            tensors[..., t] = (tensor - metric[0]) / (metric[1] - metric[0])
            norm_metrics[t, :] = metric

    if method == 'stdscore':

        for t, tensor in enumerate(tensors):

            metric = np.array([np.mean(tensor[mask]), np.std(tensor[mask])])
            tensors[..., t] = (tensor - metric[0]) / (metric[1])

    return norm_metrics

