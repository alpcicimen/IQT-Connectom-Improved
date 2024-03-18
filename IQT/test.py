import argparse
import os
import sys

import numpy as np
import keras

import pandas as pd

from scipy.ndimage import binary_erosion, gaussian_filter, zoom
from skimage.metrics import structural_similarity as ssim
import nibabel as nib

from tqdm import tqdm

from typing import List, Tuple

# from models import unet3d_t1_v2 as unet3d_t1, unet3d_not1_v2 as unet3d
from models import config_model
import util

import tensorflow as tf


def dt_rmse(input, target):
    return np.median(np.sqrt(np.mean(np.square(target - input), axis=0)))


def get_grid_indices(subj_img, i_patch_size=5, o_patch_size=3, overlap=0) -> List[Tuple[int, int, int]]:
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
                                     2 * o_patch_size - overlap * 2)]

    return recon_indx


def main(model_type,
         model_weights_dir,
         subjects,
         data_dir,
         data_subdir,
         output_dir,
         upsamp_rate,
         t1_data_dir,
         t1_subdir,
         patch_size,
         patch_overlap):

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

        mask = test_data[..., 0] > -1

        mask = binary_erosion(mask, np.ones((5, 5, 5)))

        test_data_rescaled = test_data[..., 2:]

        test_data_rescaled = util.apply_gaussian_filter(test_data_rescaled, upsamp_rate)

        test_data_rescaled = zoom(test_data_rescaled,
                                  (1/upsamp_rate, 1/upsamp_rate, 1/upsamp_rate, 1),
                                  order=1,
                                  prefilter=False)

        dti_rescale_factor = np.array(test_data.shape[:-1] + (1,)) / np.array(test_data_rescaled.shape[:-1] + (1,))

        test_data = zoom(test_data_rescaled, dti_rescale_factor, order=1, prefilter=False)

        target_data, _ = util.load_dtis(os.path.join(data_dir, subj_id, data_subdir),
                                        "dt_b1000_")
        test_data_t1, _ = util.load_structural(os.path.join(t1_data_dir, subj_id, t1_subdir),
                                               "T1w_acpc_dc_restore_brain")

        t1_rescale_factor = np.array(target_data.shape[:-1] + (1,)) / np.array(test_data_t1.shape)

        test_data_t1 = util.apply_gaussian_filter(test_data_t1, 1.25/0.7)

        t1_rescaled = zoom(test_data_t1, t1_rescale_factor, order=1)

        target_tensors = np.copy(target_data[..., 2:])
        input_tensors = np.copy(test_data)[...]

        run_indices = get_grid_indices(input_tensors, 8, 8, overlap=patch_overlap)

        model_output = np.zeros(test_data.shape[:-1] + (6,))

        t1_rescaled_nii = nib.Nifti1Image(t1_rescaled, None,
                                          nib.load(
                                              os.path.join(data_dir, subj_id, data_subdir, 'dt_b1000_2.nii')
                                          ).header)

        qform = nib.load(os.path.join(t1_data_dir, subj_id, t1_subdir, 'T1w_acpc_dc_restore_brain.nii.gz')).get_qform()

        np.fill_diagonal(qform, [-1.25, 1.25, 1.25, 1])

        t1_rescaled_nii.set_qform(qform)

        nib.save(t1_rescaled_nii, os.path.join(output_dir, f"{subj_id}_T1_resc"))

        norm_metrics_input = util.apply_clipped_normalization(input_tensors, mask, 'minmax')
        norm_metrics_target = util.apply_clipped_normalization(target_tensors, mask, 'minmax')
        norm_metrics_t1 = util.apply_clipped_normalization(t1_rescaled, mask, 'minmax')

        for (i, j, k) in tqdm(run_indices, disable=False):

            model_output[i - patch_size//2 + patch_overlap:i + patch_size//2 - patch_overlap,
                         j - patch_size//2 + patch_overlap:j + patch_size//2 - patch_overlap,
                         k - patch_size//2 + patch_overlap:k + patch_size//2 - patch_overlap, :] += \
                model([input_tensors[i - patch_size//2:i + patch_size//2,
                                     j - patch_size//2:j + patch_size//2,
                                     k - patch_size//2:k + patch_size//2][None, ...],
                       t1_rescaled[i - patch_size//2:i + patch_size//2,
                                   j - patch_size//2:j + patch_size//2,
                                   k - patch_size//2:k + patch_size//2][None, ...]
                       ]).numpy()[0,
                                  patch_overlap:patch_size - patch_overlap,
                                  patch_overlap:patch_size - patch_overlap,
                                  patch_overlap:patch_size - patch_overlap]

        input_data_copy = np.copy(input_tensors)
        target_data_copy = np.copy(target_tensors)
        model_output_rescaled = np.copy(model_output)

        util.revert_normalization(target_data_copy, mask, norm_metrics_target, method='minmax')
        util.revert_normalization(model_output_rescaled, mask, norm_metrics_target, method='minmax')
        util.revert_normalization(input_data_copy, mask, norm_metrics_input, method='minmax')

        md_orig, fa_orig, cfa_orig, eigv_orig = util.md_fa_cfa(target_data_copy, mask, cluster_mode=True)
        md_in, fa_in, cfa_in, eigv_in = util.md_fa_cfa(input_data_copy, mask, cluster_mode=True)
        md_gen, fa_gen, cfa_gen, eigv_gen = util.md_fa_cfa(model_output_rescaled, mask, cluster_mode=True)

        linear_dt_rmse = dt_rmse(target_data_copy[mask], input_data_copy[mask])
        model_dt_rmse = dt_rmse(target_data_copy[mask], model_output_rescaled[mask])

        model_md_rmse = np.sqrt(np.mean(np.square(md_orig - md_gen)[mask]))
        model_fa_rmse = np.sqrt(np.mean(np.square(fa_orig - fa_gen)[mask]))
        model_cfa_rmse = np.sqrt(np.mean(np.square(cfa_orig - cfa_gen)[mask]))

        linear_md_rmse = np.sqrt(np.mean(np.square(md_orig - md_in)[mask]))
        linear_fa_rmse = np.sqrt(np.mean(np.square(fa_orig - fa_in)[mask]))
        linear_cfa_rmse = np.sqrt(np.mean(np.square(cfa_orig - cfa_in)[mask]))

        model_md_ssim = ssim(md_gen, md_orig, data_range=np.max(md_orig) - np.min(md_orig))
        linear_md_ssim = ssim(md_in, md_orig, data_range=np.max(md_orig) - np.min(md_orig))

        model_fa_ssim = ssim(fa_gen, fa_orig, data_range=np.max(fa_orig) - np.min(fa_orig))
        linear_fa_ssim = ssim(fa_in, fa_orig, data_range=np.max(fa_orig) - np.min(fa_orig))

        df.append({'Subject': subj_id,
                   'DT-RMSE_Linear': linear_dt_rmse,
                   'RMSE_MD_Linear': linear_md_rmse,
                   'RMSE_FA_Linear': linear_fa_rmse,
                   'RMSE_CFA_Linear': linear_cfa_rmse,
                   'SSIM_MD_Linear': linear_md_ssim,
                   'SSIM_FA_Linear': linear_fa_ssim,
                   'DT-RMSE_Model': model_dt_rmse,
                   'RMSE_MD_Model': model_md_rmse,
                   'RMSE_FA_Model': model_fa_rmse,
                   'RMSE_CFA_Model': model_cfa_rmse,
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

    # --------------------------------------------- Mandatory Arguments ------------------------------------------------

    parser.add_argument('model_type')
    parser.add_argument('model_weights_dir')
    parser.add_argument('output_dir')

########################################################################################################################

    # ----------------------------------------------- I/O Arguments ----------------------------------------------------

    parser.add_argument('--subjects', type=str, nargs='+',
                        default=["221319", "178950", "224022", "627549", "885975",
                                 "111009", "140420", "638049", "887373", "654754"]
)

    parser.add_argument('--data_dir', default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc')
    parser.add_argument('--data_subdir', default='T1w/Diffusion')

    parser.add_argument('--t1_data_dir', default='/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim')
    parser.add_argument('--t1_subdir', default='T1w')

########################################################################################################################

    # -------------------------------------------- Upsampling Arguments ------------------------------------------------

    parser.add_argument('--patch_size', default=16)
    parser.add_argument('--patch_overlap', default=4)

    parser.add_argument('--upsamp_rate', type=float, default=1.25/0.7)

########################################################################################################################

    args = parser.parse_args()

    main(**vars(args))
