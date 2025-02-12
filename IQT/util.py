from typing import Any, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

import nibabel as nib
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nibabel.nifti2 import Nifti2Header, Nifti2Image
import os

from tqdm import tqdm
from scipy.ndimage import convolve


def __load_nii__(directory: str | os.PathLike[str], filename: str) -> Nifti1Image | Nifti2Image:

    if os.path.exists(os.path.join(directory, f"{filename}.nii")):
        return nib.load(os.path.join(directory, f"{filename}.nii"))
    elif os.path.exists(os.path.join(directory, f"{filename}.nii.gz")):
        return nib.load(os.path.join(directory, f"{filename}.nii.gz"))
    else:
        raise FileNotFoundError("No such files in directory: \'{}\'"
                                .format(os.path.join(directory, f"{filename}.nii")))


def load_dwis(directory: str | os.PathLike[str],
              file_head: str) -> Tuple[NDArray[Any], Nifti1Header | Nifti2Header]:

    dwi_file = __load_nii__(directory, file_head)

    return np.array(dwi_file.dataobj), dwi_file.header


def load_dtis(directory: str | os.PathLike[str],
              file_head: str) -> Tuple[NDArray[Any], Nifti1Header | Nifti2Header]:
    """
    Loads the (currently preprocessed) DTI inputs as a Numpy array of shape [H, W, D, 8] including a sample header.
    The files are numbered from 1 to 8, where 1 corresponds to the brain mask, 2 the original image intensity S_0,
    and numbers 3-8 the diffusion tensors D_xx, D_xy, D_xz, D_yy, D_yz, D_zz respectively.

    :param directory: The parent directory to load the dtis from
    :param file_head: The file header. Files should be of format **header_x** where x is the file numbers 1-8.
        Every file from 1-8 has to be present!
    :return: The combined diffusion tensors as a numpy array, together with a sample header
    """

    subject_dts = []

    header = __load_nii__(directory, f"{file_head}2").header

    for i in range(1, 9):
        subject_dts.append(np.array(__load_nii__(directory, f"{file_head}{i}").dataobj))

    merged_dts = np.stack(subject_dts, axis=-1)
    return merged_dts, header


def load_maps(directory: str | os.PathLike[str],
              file_head: str) -> Tuple[NDArray[Any], Nifti1Header | Nifti2Header]:

    subject_propagators = []

    header = __load_nii__(directory, f"{file_head}02").header

    for i in range(1, 25):
        subject_propagators.append(np.array(__load_nii__(directory, f"{file_head}{i:02d}").dataobj))

    merged_maps = np.stack(subject_propagators, axis=-1)
    return merged_maps, header


def load_structural(directory: str | os.PathLike[str],
                    file_head: str) -> Tuple[NDArray[Any], Nifti1Header | Nifti2Header]:

    nii = __load_nii__(directory, file_head)
    return np.array(nii.dataobj)[..., None], nii.header


def save_dtis(tensors: NDArray,
              save_file_loc: str | os.PathLike[str],
              reference_header_dir: str | os.PathLike[str],
              mask: NDArray | None = None,
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
                 os.path.join(save_file_loc, f"{dti_file_start}{t+3}"))

    if mask is not None:
        nib.save(nib.Nifti1Image(mask, None, reference_header),
                 os.path.join(save_file_loc, f"{dti_file_start}1"))


def save_md_fa_cfa(md: NDArray[Any],
                   fa: NDArray[Any],
                   cfa: NDArray[Any],
                   save_file_loc: str | os.PathLike[str],
                   reference_header_dir: str | os.PathLike[str] = None,
                   header=None) -> None:

    if header is None:
        if os.path.exists(f"{reference_header_dir}.nii"):
            reference_file = nib.load(f"{reference_header_dir}.nii")
        elif os.path.exists(f"{reference_header_dir}.nii.gz"):
            reference_file = nib.load(f"{reference_header_dir}.nii.gz")
        else:
            raise FileNotFoundError("No such file in directory: \"{}\"".format(f"{reference_header_dir}.nii.gz"))

        reference_header = reference_file.header

    else:
        reference_header = header

    if not os.path.exists(save_file_loc):
        os.makedirs(save_file_loc)

    nib.save(nib.Nifti1Image(md, None, reference_header),
             os.path.join(save_file_loc, f"md"))

    nib.save(nib.Nifti1Image(fa, None, reference_header),
             os.path.join(save_file_loc, f"fa"))

    nib.save(nib.Nifti1Image(cfa, None, reference_header),
             os.path.join(save_file_loc, f"cfa"))


def apply_normalization(tensors, mask: NDArray[bool], method='minmax') -> NDArray:
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

    tensors[~mask, :] = 0

    return norm_metrics


def apply_normalization_combined(tensors, mask: NDArray[bool],
                                 method='minmax',
                                 channels: NDArray | None = None,
                                 values: NDArray | None = None) -> NDArray:
    """
    Applies either min-max or standard score normalisation on the data.
    The normalisation is applied to the reference.

    :param tensors: The tensor input to apply normalisation to.
    The input variable is modified instead of returning a new variable.
    :param mask: The tensor mask that determines which voxels will be normalised.
    :param method: Which normalisation function to apply.
    :param values:
    The options are minmax for min-max, or stdscore for standard score.
    :param channels:
    :param values:
    :return: The normalisation metrics which can be used to revert the normalisation.
    """

    norm_metrics = np.zeros((1, 2)) if channels is None else np.zeros((len(channels), 2))

    sel_channels = np.array([[i for i in range(tensors.shape[-1])]]) if channels is None else channels

    match method:

        case "minmax":
            for i, c in enumerate(sel_channels):

                norm_metrics[i, :] = np.array([np.min(tensors[..., c][mask]),
                                               np.max(tensors[..., c][mask])])
                if values is not None:
                    norm_metrics[i, :] = values[i, :]
                    # norm_metrics[i, :] = [np.max((values[i, 0], norm_metrics[i, 0])),
                    #                       np.min((values[i, 1], norm_metrics[i, 1]))]

                tensors[..., c] = (tensors[..., c] - norm_metrics[i, 0]) / (norm_metrics[i, 1] - norm_metrics[i, 0])

                # Clip to account for unmasked voxels
                tensors[..., c] = np.clip(tensors[..., c], 0, 1)

        case "stdscore":
            for i, c in sel_channels:
                norm_metrics[i, :] = np.array([np.mean(tensors[..., c][mask]),
                                               np.std(tensors[..., c][mask])])
                tensors[..., c] = (tensors[..., c] - norm_metrics[i, 0]) / (norm_metrics[i, 1])

        case _:
            raise ValueError("Only \"minmax\" and \"stdscore\" values are allowed.")

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


def revert_normalization_combined(tensors, mask, norm_metrics,
                                  method='minmax',
                                  channels: NDArray | None = None) -> None:

    sel_channels = np.array([[i for i in range(tensors.shape[-1])]]) if channels is None else channels

    match method:

        case "minmax":

            tensors[...] = np.clip(tensors, 0, 1)[...]  # Clip just in case (mainly because of NN output)

            for i, c in enumerate(sel_channels):
                tensors[..., c] = (tensors[..., c] * (norm_metrics[i, 1] - norm_metrics[i, 0])) + norm_metrics[i, 0]

        case "stdscore":
            for i, c in enumerate(sel_channels):
                tensors[..., c] = tensors[..., c] * norm_metrics[i, 1] + norm_metrics[i, 0]

        case _:
            raise ValueError("Only \"minmax\" and \"stdscore\" values are allowed.")

    tensors[~mask, :] = 0

    return


def get_clip_values(tensors, mask, data_mode, clip_strategy, value: float = 3e-3) -> NDArray[float]:

    match str.lower(data_mode):

        case 'dti':

            norm_channels = np.array([[0, 3, 5], [1, 2, 4]])

            match clip_strategy:

                case 'std':

                    mean = np.mean(tensors[mask, ...][:, norm_channels[0]])
                    std_n = np.std(tensors[mask, ...][:, norm_channels[0]]) * value

                    return np.array([[0, mean + std_n], [-(mean + std_n), mean + std_n]])

                case 'constant':
                    return np.array([[0, value], [-value, value]])

                case 'percentile':

                    high_range = np.percentile(tensors[mask, ...][:, norm_channels[0]], value)

                    return np.array([[0, high_range], [-high_range, high_range]])

        case 't1w':

            match clip_strategy:

                case 'percentile':

                    low_range = np.percentile(tensors[mask], (100 - value) / 2)
                    high_range = np.percentile(tensors[mask], 50 + value / 2)

                    # clip_on_value(tensors, mask, values=[low_range, high_range])

                    return np.array([[low_range, high_range]])

        case _:
            return np.array([np.min(tensors), np.max(tensors)])


def apply_clipped_normalization(tensors,
                                mask,
                                method=None,
                                deviations: int | Sequence[int] = (None, 2)) -> NDArray | None:
    """
    Applies standard score normalization, and clips the values to the specific std range.
    Depending on the modality the returned value either has the normalization or returned with the original range.

    :param tensors: Input tensors to normalize
    :param mask: Mask regions to determine valid voxels
    :param method: Which normalized method to return the value as. \"stdscore\" returns the values as std score,
        \"minmax\" returns the values as min-max normalized, and everything else returns the original range.

    :param deviations: Number of standard deviations the values will be clipped to.
        Negative values are treated as absolute if int, otherwise h_clip > l_clip.

    :return: Normalization metrics if method is valid, otherwise None
    """

    if isinstance(deviations, int):
        l_clip = -abs(deviations)
        h_clip = abs(deviations)
    else:
        l_clip = deviations[0]
        h_clip = deviations[1]

        # Python evaluates booleans in order, so if any values are none (meaning no limit)
        # the program does not throw this error.
        if l_clip is not None and h_clip is not None and h_clip < l_clip:
            raise ValueError("Second clip value cannot be less than the first!")

    metrics = apply_normalization(tensors, mask, 'stdscore')
    tensors[...] = np.clip(tensors, l_clip, h_clip, dtype=float)

    match method:

        # No need to apply stdscore again, just return metrics as is
        case 'stdscore':
            return metrics

        # Revert to normal range (with clipped values) and return min-max normalisation metrics
        case 'minmax':
            revert_normalization(tensors, mask, metrics, 'stdscore')
            return apply_normalization(tensors, mask, 'minmax')

        # Revert normalisation, and return void
        case _:
            revert_normalization(tensors, mask, metrics, 'stdscore')
            return None


def apply_gaussian_filter(input_img: NDArray[float], downsample_rate) -> NDArray[float]:

    std_value = 2 * np.log(10) / (2 * np.pi) * downsample_rate
    kernel_size = np.int32(np.ceil(2.5 * std_value) / 2) * 2 + 1

    kernel_grid = np.copy(np.mgrid[0:kernel_size,
                                   0:kernel_size,
                                   0:kernel_size]).transpose((1, 2, 3, 0)) - kernel_size//2

    gaussian_kernel = 1/(np.sqrt(2*np.pi)*std_value)**3 * np.exp(-(kernel_grid[..., 0] ** 2 +
                                                                   kernel_grid[..., 1] ** 2 +
                                                                   kernel_grid[..., 2] ** 2) / (2 * std_value ** 2))

    gaussian_kernel /= np.sum(gaussian_kernel)

    if len(input_img.shape) == 3:
        return convolve(input_img, gaussian_kernel, mode='constant')
    else:
        return np.stack([convolve(channel, gaussian_kernel, mode='constant')
                         for channel in np.moveaxis(input_img, -1, 0)], axis=-1)


def config_grid(start, end, length: Tuple[int, int, int]):

    range_x = np.linspace(start[0], end[0], length[0], dtype=np.float32)
    range_y = np.linspace(start[1], end[1], length[1], dtype=np.float32)
    range_z = np.linspace(start[2], end[2], length[2], dtype=np.float32)

    grid = np.stack(np.meshgrid(range_x, range_y, range_z,
                                indexing='ij'), axis=-1)  # mgrid doesn't work very well due to ieee754 inaccuracy

    return grid


def gridded_interpolation(image: NDArray[float], grid: NDArray[float]) -> NDArray[float]:

    max_sizes = np.array(image.shape[:-1]) - 1

    lgrid = np.floor(grid)
    ugrids = [np.clip((lgrid + [int(i & 4 > 0), int(i & 2 > 0), int(i & 1 > 0)]).astype(int),
                      a_min=0, a_max=max_sizes, dtype=int) for i in range(8)]

    udiffs = [1. - np.abs(ugrids[i] - grid) for i in range(8)]

    out_img = [[] for _ in range(3)]

    for i in range(8):
        ugrid = ugrids[i].reshape((grid.shape[0] * grid.shape[1] * grid.shape[2], 3))

        out = image[ugrid[:, 0],
                    ugrid[:, 1],
                    ugrid[:, 2]].reshape((grid.shape[0], grid.shape[1], grid.shape[2], image.shape[-1]))

        out_img[0].append(out * udiffs[i][..., 2, None])

    for i in range(4):
        out_img[1].append((out_img[0][2*i] + out_img[0][2*i + 1]) * udiffs[2*i][..., 1, None])

    for i in range(2):
        out_img[2].append((out_img[1][2*i] + out_img[1][2*i + 1]) * udiffs[4*i][..., 0, None])

    return out_img[2][0] + out_img[2][1]


def md_fa_cfa(tensors, mask,
              cluster_mode=False) -> Tuple[NDArray[float],
                                           NDArray[float],
                                           NDArray[float],
                                           NDArray[float]]:
    """
    Generate the mean diffusivity (MD), fractional anisotropy (FA) and coloured fractional anisotropy (CFA) images from
    the tensor data. The non-masked regions are not evaluated.

    :param tensors: Tensor values to evaluate MD, FA and CFA from
    :param mask: The mask for which voxels have to be calculated for
    :param cluster_mode: Boolean value that determines whether tqdm progress bars will be enabled or not
    :return: The calculated MD, FA and CFA maps as Numpy arrays
    """

    md = np.zeros(tensors.shape[:-1], dtype=float)
    fa = np.zeros(tensors.shape[:-1], dtype=float)
    cfa = np.zeros(tensors.shape[:-1] + (3,), dtype=float)
    peigv = np.zeros(tensors.shape[:-1] + (3,), dtype=float)

    (x_shape, y_shape, z_shape, _) = tensors.shape

    for i in tqdm(range(x_shape), disable=cluster_mode):
        for j in range(y_shape):
            for k in range(z_shape):
                if mask[i, j, k]:
                    ldt = tensors[i, j, k, :]
                    ldt = np.array([[ldt[0], ldt[1], ldt[2]],
                                    [ldt[1], ldt[3], ldt[4]],
                                    [ldt[2], ldt[4], ldt[5]]])

                    eig_vals, eig_vecs = np.linalg.eigh(ldt)

                    md[i, j, k] = np.mean(eig_vals)

                    fa[i, j, k] = 0 if np.sum(eig_vals ** 2) == 0 \
                        else np.sqrt(1.5 * np.sum((eig_vals - eig_vals.mean()) ** 2) / np.sum(eig_vals ** 2))
                    peigv[i, j, k] = eig_vecs[:, eig_vals.argmax()]
                    cfa[i, j, k, :] = fa[i, j, k] * np.abs(eig_vecs[:, eig_vals.argmax()])

    return md, fa, cfa, peigv
