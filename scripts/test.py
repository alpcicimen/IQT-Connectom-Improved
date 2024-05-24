import argparse
import os
from typing import List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.ndimage import binary_erosion, zoom
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from IQT.models import config_model
from IQT import util


def dt_rmse(input, target):
    return np.median(np.sqrt(np.mean(np.square(target - input), axis=-1)))


def get_grid_indices(subj_img, mask, i_patch_size=5, o_patch_size=3, overlap=0) -> List[Tuple[int, int, int]]:
    (xsize, ysize, zsize, _) = subj_img.shape

    recon_indx = [(i, j, k)
                  for k in np.arange(i_patch_size,
                                     zsize - i_patch_size,
                                     2 * o_patch_size - overlap * 2)
                  for j in np.arange(i_patch_size,
                                     ysize - i_patch_size,
                                     2 * o_patch_size - overlap * 2)
                  for i in np.arange(i_patch_size,
                                     xsize - i_patch_size,
                                     2 * o_patch_size - overlap * 2)
                  if np.sum(mask[i-o_patch_size:i+o_patch_size,
                                 j-o_patch_size:j+o_patch_size,
                                 k-o_patch_size:k+o_patch_size,] > 0)]

    return recon_indx


def main(model_type,
         model_weights_dir,
         subjects,
         data_dir,
         data_subdir,
         output_dir,
         upsamp_rate,
         clip_strategy,
         clip_value,
         t1_data_dir,
         t1_subdir,
         cluster_mode,
         patch_size,
         patch_overlap,
         max_noise_std):

    model = config_model(model_type,
                         patch_size,
                         patch_size,
                         model_weights_dir)

    df = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for subj_id in subjects:

        print(f"Current Subject: {subj_id}")

        test_data, _ = util.load_dtis(os.path.join(data_dir, subj_id, data_subdir), "dt_b1000_")

        target_data, _ = util.load_dtis(os.path.join(data_dir, subj_id, data_subdir), "dt_b1000_")

        test_data_t1, _ = util.load_structural(os.path.join(t1_data_dir, subj_id, t1_subdir),
                                               "T1w_acpc_dc_restore_brain")

        mask = np.array(test_data[..., 0] >= 0, dtype=bool)

        t1_rescale_factor = np.array(target_data.shape[:-1] + (1,)) / np.array(test_data_t1.shape)

        test_data = test_data[..., 2:]
        target_data = target_data[..., 2:]

        # The masks may have problematic outlier voxels if they were not fine-tuned.
        # Therefore, we just erode them further.
        mask = binary_erosion(mask, np.ones((5, 5, 5)), iterations=1)

########################################################################################################################

# ----------------------------------------------- Rescaling Step -------------------------------------------------------

########################################################################################################################

        # Handle T1w first as it is more straightforward. Apply gaussian, lower resolution to HR DTI level
        test_data_t1 = util.apply_gaussian_filter(test_data_t1, 1.25/0.7)

        t1_rescaled = zoom(test_data_t1, t1_rescale_factor, order=1)

        t1_rescaled_nii = nib.Nifti1Image(t1_rescaled, None,
                                          nib.load(
                                              os.path.join(data_dir, subj_id, data_subdir, 'dt_b1000_2.nii')
                                          ).header)

        qform = nib.load(os.path.join(t1_data_dir, subj_id, t1_subdir, 'T1w_acpc_dc_restore_brain.nii.gz')).get_qform()

        np.fill_diagonal(qform, [-1.25, 1.25, 1.25, 1])

        t1_rescaled_nii.set_qform(qform)

        nib.save(t1_rescaled_nii, os.path.join(output_dir, f"{subj_id}_T1_resc"))

        test_data_rescaled = util.apply_gaussian_filter(test_data, upsamp_rate)

        test_data_rescaled = zoom(test_data_rescaled,
                                  (1/upsamp_rate, 1/upsamp_rate, 1/upsamp_rate, 1),
                                  order=1,
                                  prefilter=False)

        dti_rescale_factor = np.array(test_data.shape[:-1] + (1,)) / np.array(test_data_rescaled.shape[:-1] + (1,))

        test_data = zoom(test_data_rescaled, dti_rescale_factor, order=1, prefilter=False)

########################################################################################################################

# ---------------------------------------------- Normalisation Step ----------------------------------------------------

########################################################################################################################

        # Clip T1w here
        norm_metrics_t1 = util.get_clip_values(t1_rescaled, mask,
                                               data_mode='t1w',
                                               clip_strategy='percentile',
                                               value=96)

        norm_metrics_input = util.apply_normalization_combined(test_data, mask,
                                                               method='minmax',
                                                               channels=np.array([[0, 3, 5], [1, 2, 4]]),
                                                               values=util.get_clip_values(test_data, mask,
                                                                                           'dti',
                                                                                           clip_strategy=clip_strategy,
                                                                                           value=clip_value))

        util.apply_normalization_combined(t1_rescaled, mask, method='minmax', values=norm_metrics_t1)

        target_data[..., np.array([0, 3, 5])] = np.clip(target_data[..., np.array([0, 3, 5])],
                                                        a_min=0, a_max=2e-3)
        target_data[..., np.array([1, 2, 4])] = np.clip(target_data[..., np.array([1, 2, 4])],
                                                        a_min=-2e-3, a_max=2e-3)

########################################################################################################################

# ----------------------------------------------- Estimation Step ------------------------------------------------------

########################################################################################################################

        noise_std = max_noise_std * np.random.rand(1)[0]
        t1_rescaled += noise_std * np.random.randn(*t1_rescaled.shape)

        test_data[~mask, :] = 0
        t1_rescaled[~mask, :] = 0
        target_data[~mask, :] = 0

        input_tensors = np.copy(test_data)[...]

        run_indices = get_grid_indices(input_tensors, mask, 8, 8, overlap=patch_overlap)

        model_output = np.zeros(test_data.shape[:-1] + (6,))

        for (i, j, k) in tqdm(run_indices, disable=cluster_mode):

            i_patch = input_tensors[i - patch_size//2:i + patch_size//2,
                                    j - patch_size//2:j + patch_size//2,
                                    k - patch_size//2:k + patch_size//2][None, ...]

            t1_patch = t1_rescaled[i - patch_size//2:i + patch_size//2,
                                   j - patch_size//2:j + patch_size//2,
                                   k - patch_size//2:k + patch_size//2][None, ...]

            model_output[i - patch_size//2 + patch_overlap:i + patch_size//2 - patch_overlap,
                         j - patch_size//2 + patch_overlap:j + patch_size//2 - patch_overlap,
                         k - patch_size//2 + patch_overlap:k + patch_size//2 - patch_overlap, :] += \
                model([i_patch, t1_patch], training=False).numpy()[0,
                                                                   patch_overlap:patch_size - patch_overlap,
                                                                   patch_overlap:patch_size - patch_overlap,
                                                                   patch_overlap:patch_size - patch_overlap]

        util.revert_normalization_combined(model_output, mask, norm_metrics_input,
                                           method='minmax', channels=np.array([[0, 3, 5], [1, 2, 4]]))
        util.revert_normalization_combined(input_tensors, mask, norm_metrics_input,
                                           method='minmax', channels=np.array([[0, 3, 5], [1, 2, 4]]))

        md_orig, fa_orig, cfa_orig, eigv_orig = util.md_fa_cfa(target_data, mask, cluster_mode=cluster_mode)
        md_in, fa_in, cfa_in, eigv_in = util.md_fa_cfa(input_tensors, mask, cluster_mode=cluster_mode)
        md_gen, fa_gen, cfa_gen, eigv_gen = util.md_fa_cfa(model_output, mask, cluster_mode=cluster_mode)

        linear_dt_rmse = dt_rmse(target_data[mask], input_tensors[mask])
        model_dt_rmse = dt_rmse(target_data[mask], model_output[mask])

        model_md_rmse = np.sqrt(np.mean(np.square(md_orig - md_gen)[mask]))
        model_fa_rmse = np.sqrt(np.mean(np.square(fa_orig - fa_gen)[mask]))

        # Cosine similarity calculation. The CFA is a directional vector with normalized values.
        # This means that the angular difference is a maximum of 90 degrees, for which the cosine range is between [0,1]
        denom = np.multiply(np.linalg.norm(cfa_gen, axis=-1), np.linalg.norm(cfa_orig, axis=-1))
        model_cosine_sim = np.sum(np.multiply(cfa_gen, cfa_orig), axis=-1) / denom

        # Penalise the cosine difference at max for NaN values.
        # Most likely reason for NaN is the nonmasked areas, which aren't included in the mean,
        # but in case NaN persists to masked we treat as maximum difference
        model_cosine_sim[np.isnan(model_cosine_sim)] = 0

        model_cosine_sim = np.mean(model_cosine_sim[mask])

        linear_md_rmse = np.sqrt(np.mean(np.square(md_orig - md_in)[mask]))
        linear_fa_rmse = np.sqrt(np.mean(np.square(fa_orig - fa_in)[mask]))

        # Cosine similarity calculation. The CFA is a directional vector with normalized values.
        # This means that the angular difference is a maximum of 90 degrees, for which the cosine range is between [0,1]
        denom = np.multiply(np.linalg.norm(cfa_in, axis=-1), np.linalg.norm(cfa_orig, axis=-1))
        linear_cosine_sim = np.sum(np.multiply(cfa_in, cfa_orig), axis=-1) / denom

        # Penalise the cosine difference at max for NaN values.
        # Most likely reason for NaN is the nonmasked areas, which aren't included in the mean,
        # but in case NaN persists to masked we treat as maximum difference
        linear_cosine_sim[np.isnan(linear_cosine_sim)] = 0

        linear_cosine_sim = np.mean(linear_cosine_sim[mask])

        model_md_ssim = ssim(md_gen, md_orig, data_range=np.max(md_orig) - np.min(md_orig))
        linear_md_ssim = ssim(md_in, md_orig, data_range=np.max(md_orig) - np.min(md_orig))

        model_fa_ssim = ssim(fa_gen, fa_orig, data_range=np.max(fa_orig) - np.min(fa_orig))
        linear_fa_ssim = ssim(fa_in, fa_orig, data_range=np.max(fa_orig) - np.min(fa_orig))

        df.append({'Subject': subj_id,
                   'DT-RMSE_Linear': linear_dt_rmse,
                   'RMSE_MD_Linear': linear_md_rmse,
                   'RMSE_FA_Linear': linear_fa_rmse,
                   'CosSim_CFA_Linear': linear_cosine_sim,
                   'SSIM_MD_Linear': linear_md_ssim,
                   'SSIM_FA_Linear': linear_fa_ssim,
                   'DT-RMSE_Model': model_dt_rmse,
                   'RMSE_MD_Model': model_md_rmse,
                   'RMSE_FA_Model': model_fa_rmse,
                   'CosSim_CFA_Model': model_cosine_sim,
                   'SSIM_MD_Model': model_md_ssim,
                   'SSIM_FA_Model': model_fa_ssim})

        if not os.path.exists(os.path.join(output_dir, "HCP_output", subj_id)):
            os.makedirs(os.path.join(output_dir, "HCP_output", subj_id))
        if not os.path.exists(os.path.join(output_dir, "HCP_upsamp", subj_id)):
            os.makedirs(os.path.join(output_dir, "HCP_upsamp", subj_id))
        if not os.path.exists(os.path.join(output_dir, "HCP_target", subj_id)):
            os.makedirs(os.path.join(output_dir, "HCP_target", subj_id))

        util.save_md_fa_cfa(md_gen, fa_gen, eigv_gen,
                            os.path.join(output_dir, "HCP_output", subj_id),
                            os.path.join(output_dir, f"{subj_id}_T1_resc"))

        util.save_md_fa_cfa(md_in, fa_in, eigv_in,
                            os.path.join(output_dir, "HCP_upsamp", subj_id),
                            os.path.join(output_dir, f"{subj_id}_T1_resc"))

        util.save_md_fa_cfa(md_orig, fa_orig, eigv_orig,
                            os.path.join(output_dir, "HCP_target", subj_id),
                            os.path.join(output_dir, f"{subj_id}_T1_resc"))

    pd.DataFrame(df).set_index("Subject").to_csv(os.path.join(output_dir, "metrics.csv"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='IQT-Testing',
                                     description='The main testing script for IQT.')
########################################################################################################################

# --------------------------------------------- Mandatory Arguments ----------------------------------------------------

########################################################################################################################

    parser.add_argument('model_type')
    parser.add_argument('model_weights_dir')
    parser.add_argument('output_dir')

########################################################################################################################

# ------------------------------------------------ I/O Arguments -------------------------------------------------------

########################################################################################################################

    parser.add_argument('--subjects', type=str, nargs='+',
                        default=["221319", "178950", "224022", "627549", "885975", "111009", "140420", "638049",
                                 "887373", "654754", "366446", "182739", "877168", "598568", "214423", "100307",
                                 "160123", "351938", "732243", "193239", "570243", "547046", "586460", "157336",
                                 "127933", "162733", "117324", "159340", "130013", "727654", "200614", "978578",
                                 "992774", "865363"])

    parser.add_argument('--data_dir', default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc')
    parser.add_argument('--data_subdir', default='T1w/Diffusion')

    parser.add_argument('--t1_data_dir', default='/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim')
    parser.add_argument('--t1_subdir', default='T1w')

########################################################################################################################

# --------------------------------------------- Upsampling Arguments ---------------------------------------------------

########################################################################################################################

    parser.add_argument('--patch_size', default=16)
    parser.add_argument('--patch_overlap', default=4)

    parser.add_argument('--upsamp_rate', type=float, default=1.25/0.7)

    parser.add_argument('--clip_strategy', type=str, default='constant')
    parser.add_argument('--clip_value', type=float, default=3e-3)

########################################################################################################################

    parser.add_argument('--max_noise_std', type=float, default=0.1)
    parser.add_argument('--cluster_mode', type=bool, default=False)

    args = parser.parse_args()

    main(**vars(args))
