from typing import Tuple

import numpy as np
import nibabel as nib
import os

from tqdm import tqdm


def load_dtis(directory: str | os.PathLike[str],
              file_head: str) -> np.ndarray:
    subject_dts = []

    for i in range(1, 9):
        if os.path.exists(os.path.join(directory, f"{file_head}{i}.nii")):
            subj = np.array(nib.load(os.path.join(directory, f"{file_head}{i}.nii")).dataobj)
        elif os.path.exists(os.path.join(directory, f"{file_head}{i}.nii.gz")):
            subj = np.array(nib.load(os.path.join(directory, f"{file_head}{i}.nii.gz")).dataobj)
        else:
            raise FileNotFoundError("No such files in directory: \'{}\'"
                                    .format(os.path.join(directory, f"{file_head}{i}.nii")))

        subject_dts.append(subj)

    merged_dts = np.stack(subject_dts, axis=-1)
    return merged_dts


def load_maps(directory: str | os.PathLike[str],
              file_head: str) -> np.ndarray:

    subject_propagators = []

    for i in range(1, 23):
        if os.path.exists(os.path.join(directory, f"{file_head}{i}.nii")):
            prop = np.array(nib.load(os.path.join(directory, f"{file_head}{i}.nii")).dataobj)
        elif os.path.exists(os.path.join(directory, f"{file_head}{i}.nii.gz")):
            prop = np.array(nib.load(os.path.join(directory, f"{file_head}{i}.nii.gz")).dataobj)
        else:
            raise FileNotFoundError("No such files in directory: \'{}\'"
                                    .format(os.path.join(directory, f"{file_head}{i}.nii")))

        subject_propagators.append(prop)

    merged_maps = np.stack(subject_propagators, axis=-1)
    return merged_maps


def load_structural(directory: str | os.PathLike[str],
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
        raise FileNotFoundError("No such files in directory: \"{}\""
                                .format(os.path.join(directory, f"{file_head}.nii.gz")))

    return subj


def save_dtis(tensors: np.ndarray,
              save_file_loc: str | os.PathLike[str],
              reference_header_dir: str | os.PathLike[str],
              dti_file_start='dt_b1000_recon_') -> None:

    if os.path.exists(f"{reference_header_dir}.nii"):
        reference_file = nib.load(f"{reference_header_dir}.nii")
    elif os.path.exists(f"{reference_header_dir}.nii.gz"):
        reference_file = nib.load(f"{reference_header_dir}.nii.gz")
    else:
        raise FileNotFoundError("No such file in directory: \"{}\"".format(f"{reference_header_dir}.nii.gz"))

    reference_header = reference_file.header

    for t in range(tensors.shape[-1]):
        nib.save(nib.Nifti1Image(tensors[..., t], None, reference_header),
                 os.path.join(save_file_loc, f"{dti_file_start}{t+2}"))


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

    tensors[~mask, :] = 0


def apply_clipped_normalization(tensors, mask, method=None, deviations: int = 2) -> np.ndarray | None:
    """
    Applies standard score normalization, and clips the values to the specific std range.
    Depending on the modality the returned value either has the normalization or returned with the original range.

    :param tensors: Input tensors to normalize
    :param mask: Mask regions to determine valid voxels
    :param method: Which normalized method to return the value as. \"stdscore\" returns the values as std score,
        \"minmax\" returns the values as min-max normalized, and everything else returns the original range.

    :param deviations: Number of standard deviations the values will be clipped to.
        Negative values are treated as absolute.

    :return: Normalization metrics if method is valid, otherwise None
    """

    metrics = apply_normalization(tensors, mask, 'stdscore')
    tensors[...] = np.clip(tensors, -abs(deviations), abs(deviations), dtype=float)

    match method:

        case 'stdscore':
            return metrics

        case 'minmax':
            revert_normalization(tensors, mask, metrics, 'stdscore')
            return apply_normalization(tensors, mask, 'minmax')

        case _:
            revert_normalization(tensors, mask, metrics, 'stdscore')
            return None


def md_fa_cfa(tensors, mask) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Generate the mean diffusivity (MD), fractional anisotropy (FA) and coloured fractional anisotropy (CFA) images from
    the tensor data. The non-masked regions are not evaluated.

    :param tensors: Tensor values to evaluate MD, FA and CFA from
    :param mask: The mask for which voxels have to be calculated for
    :return: The calculated MD, FA and CFA maps as Numpy arrays
    """

    md = np.zeros(tensors.shape[:-1])
    fa = np.zeros(tensors.shape[:-1])
    cfa = np.zeros(tensors.shape[:-1] + (3,))

    (x_shape, y_shape, z_shape, _) = tensors.shape

    for i in tqdm(range(x_shape)):
        for j in range(y_shape):
            for k in range(z_shape):
                if mask[i, j, k]:
                    ldt = tensors[i, j, k, :]
                    ldt = np.array([[ldt[0], ldt[1], ldt[2]],
                                    [ldt[1], ldt[3], ldt[4]],
                                    [ldt[2], ldt[4], ldt[5]]])

                    eig_vals, eig_vecs = np.linalg.eigh(ldt)

                    md[i, j, k] = np.mean(eig_vals)

                    fa[i, j, k] = np.sqrt(1.5 * np.sum((eig_vals - eig_vals.mean()) ** 2) / np.sum(eig_vals ** 2))
                    cfa[i, j, k, :] = fa[i, j, k] * np.abs(eig_vecs[:, eig_vals.argmax()])

    return md, fa, cfa
