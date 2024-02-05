from typing import Tuple

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

    raise NotImplementedError("Not implemented yet!")


def load_structural(directory: str,
                    file_head: str) -> np.ndarray:

    subj = None

    if os.path.exists(os.path.join(directory, f"{file_head}.nii")):
        subj = np.array(
            nib.load(os.path.join(directory, f"{file_head}.nii")).dataobj
        )[..., None]  # For channels in NN. Data format is [X, Y, Z, C]
    elif os.path.exists(os.path.join(directory, f"{file_head}.nii.gz")):
        subj = np.array(
            nib.load(os.path.join(directory, f"{file_head}.nii.gz")).dataobj
        )[..., None]  # For channels in NN. Data format is [X, Y, Z, C]
    else:
        raise FileNotFoundError("No such files in directory.")

    return subj


def apply_normalization(tensors, mask: np.ndarray[bool], method='minmax') -> np.ndarray:
    """
    Applies either min-max or standard score normalisation on the data.
    The normalisation is applied to the reference.

    :param tensors: The tensor input to apply normalisation to.
    The input variable is modified instead of returning a new variable.
    :param mask: The tensor mask that determines which voxels will be normalised.
    :param method: Which normalisation function to apply.
    The options are minmax for min-max, or stdscore for standard score.
    :return: The normalisation metrics which can be used to revert the normalisation.
    """

    norm_metrics = np.zeros((tensors.shape[-1], 2))

    for t in range(tensors.shape[-1]):
        tensor = tensors[..., t]

        match method:

            case "minmax":
                metric = np.array([np.min(tensor[mask]), np.max(tensor[mask])])
                tensors[..., t] = (tensor - metric[0]) / (metric[1] - metric[0])

                tensors[~mask, :] = 0

            case "stdscore":
                metric = np.array([np.mean(tensor[mask]), np.std(tensor[mask])])
                tensors[..., t] = (tensor - metric[0]) / (metric[1])

                tensors[~mask, :] = 0

            case _:
                raise ValueError("Only \"minmax\" and \"stdscore\" values are allowed.")

        norm_metrics[t, :] = metric

    tensors[mask == False, :] = 0

    return norm_metrics


def revert_normalization(tensors, mask, norm_metrics, method='minmax') -> None:
    norm_metrics = np.zeros((6, 2))

    for t in range(tensors.shape[-1]):

        tensor = tensors[..., t]
        metric = norm_metrics[t, :]

        match method:

            case "minmax":
                tensors[..., t] = tensor * (metric[1] - metric[0]) + metric[0]

            case "stdscore":
                tensors[..., t] = tensor * metric[1] + metric[0]

            case _:
                raise ValueError("Only \"minmax\" and \"stdscore\" values are allowed.")

    tensors[mask == False, :] = 0


def md_fa_cfa(tensors, mask) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Generate the mean diffusivity (MD), fractional anisotropy (FA) and coloured fractional anisotropy (CFA) images from
    the tensor data. The non-masked regions are not evaluated.

    :param tensors: Tensor values to evaluate MD, FA and CFA from
    :param mask: The mask
    :return:
    """

    raise NotImplementedError("Not implemented yet!")
