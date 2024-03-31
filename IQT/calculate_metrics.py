import argparse
import os

import util

import numpy as np
import pandas as pd

from scipy.ndimage import binary_erosion
from tqdm import tqdm


def main(subjects,
         data_dir,
         data_subdir,
         output_dir,
         cluster_mode,
         mask_erosion):

    df = []

    n_sum = 0
    pooled_mean = 0
    pooled_indiv_mean = np.zeros(6)
    pooled_indiv_var = np.zeros(6)
    pooled_var = 0

    total_mins = np.ones(6) * np.finfo(float).max
    total_maxs = np.ones(6) * np.finfo(float).min

    for subject in tqdm(subjects, disable=cluster_mode):

        if cluster_mode:
            print(f"Current subject: {subject}")

        subject_data_hr, _ = util.load_dtis(os.path.join(data_dir, subject), data_subdir)

        mask = subject_data_hr[..., 0] > -1

        dti_channels = subject_data_hr[..., 2:]

        if mask_erosion:
            mask = binary_erosion(mask, np.ones((mask_erosion, mask_erosion, mask_erosion)))

        n = np.sum(mask)
        sample_mean = np.mean(dti_channels[mask])

        indiv_means = np.mean(dti_channels[mask], axis=0)
        indiv_vars = np.var(dti_channels[mask], axis=0)

        indiv_mins = np.min(dti_channels[mask], axis=0)
        indiv_maxs = np.max(dti_channels[mask], axis=0)

        sample_variance = np.var(dti_channels[mask])

        metrics_dict = {"Subject": subject,
                        "num_valid_voxels": n,
                        "mean": sample_mean,
                        "std": np.sqrt(sample_variance)}

        metrics_dict.update({f"ch_{n}_mean": indiv_means[n] for n in range(len(indiv_means))})
        metrics_dict.update({f"ch_{n}_std": np.sqrt(indiv_vars[n]) for n in range(len(indiv_means))})

        metrics_dict.update({f"ch_{n}_min": indiv_mins[n] for n in range(len(indiv_mins))})
        metrics_dict.update({f"ch_{n}_max": indiv_maxs[n] for n in range(len(indiv_maxs))})

        df.append(metrics_dict)

        n_sum += n

        pooled_mean += n * sample_mean
        pooled_indiv_mean += n * indiv_means
        pooled_indiv_var += n * (indiv_vars + indiv_means**2)
        pooled_var += n * (sample_variance + sample_mean**2)

        total_mins = np.min((total_mins, indiv_mins), axis=0)
        total_maxs = np.max((total_maxs, indiv_maxs), axis=0)

    pooled_mean /= n_sum
    pooled_indiv_mean /= n_sum
    pooled_var = pooled_var / n_sum - pooled_mean ** 2
    pooled_indiv_var = pooled_indiv_var / n_sum - pooled_indiv_mean ** 2

    metrics_dict = {"Subject": "Pooled Total",
                    "num_valid_voxels": n_sum,
                    "mean": pooled_mean,
                    "std": np.sqrt(pooled_var)}

    metrics_dict.update({f"ch_{n}_mean": pooled_indiv_mean[n] for n in range(len(pooled_indiv_mean))})
    metrics_dict.update({f"ch_{n}_std": np.sqrt(pooled_indiv_var[n]) for n in range(len(pooled_indiv_var))})

    metrics_dict.update({f"ch_{n}_min": total_mins[n] for n in range(len(total_mins))})
    metrics_dict.update({f"ch_{n}_max": total_maxs[n] for n in range(len(total_maxs))})

    df.append(metrics_dict)

    pd.DataFrame(df).set_index("Subject").to_csv(os.path.join(output_dir, "preprocess_metrics.csv"))

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Metric Calculator',
                                     description='Calculates the metrics necessary for outlier clipping.')

    parser.add_argument('data_dir',
                        help='Data directory of the diffusion scans.')
    parser.add_argument('output_dir',
                        help='Directory to write the file to.')
    parser.add_argument('--subjects', nargs='+', type=str,
                        default=["100307", "131924", "162733", "210617", "541943", "792564", "100408",
                                 "133625", "163129", "211417", "545345", "826353", "101915", "133827",
                                 "163432", "211720", "547046", "856766", "102816", "133928", "165840",
                                 "212318", "559053", "857263", "103414", "214019", "561242", "103515",
                                 "134324", "167743", "214221", "570243", "859671", "103818", "135932"])
    parser.add_argument('--data_subdir', type=str, default='T1w/Diffusion')
    parser.add_argument('--cluster_mode', type=bool, default=False)
    parser.add_argument('--mask_erosion', type=int, default=2)

    args = parser.parse_args()

    main(**vars(args))
