import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy import stats

from constants import LABEL_COLOR_MAP_SMALLER
import seaborn as sn

from visualization.global_analysis_utils import filter_columns_and_save

os.chdir("../")


def plot_box(title, data, method_names, conditions):
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=600)  # Increased figure size
    # fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.35)  # More bottom space for rotated labels

    c = 'k'
    black_dict = {  # 'patch_artist': True,
        # 'boxprops': dict(color=c, facecolor=c),
        # 'capprops': dict(color=c),
        # 'flierprops': dict(color=c, markeredgecolor=c),
        'medianprops': dict(color=c),
        # 'whiskerprops': dict(color=c)
    }

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False, **black_dict)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'{title} for all 95 simulations',
        xlabel='Feature Extraction Method',
        ylabel='Performance',
    )

    # Increase title, axis labels, and tick labels font sizes
    ax1.set_title(f'{title}', fontsize=36, pad=20)
    ax1.set_xlabel('Feature Extraction Method', fontsize=15, labelpad=10)
    ax1.set_ylabel('Performance', fontsize=15, labelpad=10)
    ax1.tick_params(axis='y', labelsize=14)

    # Now fill the boxes with desired colors
    num_boxes = len(data)

    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        med = bp['medians'][i]

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=LABEL_COLOR_MAP_SMALLER[i % len(method_names)]))

        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 1.1
    # bottom = 0
    # ax1.set_ylim(bottom, top)

    # Solution 1: Rotate x-axis labels to prevent overlap
    ax1.set_xticklabels(np.repeat(method_names, len(conditions)), rotation=0, fontsize=11, ha='center')

    # Alternative solution (commented out): Use smaller font or abbreviations
    # ax1.set_xticklabels(np.repeat(method_names, len(conditions)),
    #                     rotation=90, fontsize=10)

    # Alternative solution 2 (commented out): Use abbreviations if method names are too long
    # abbreviated_names = [name[:6] + "..." if len(name) > 6 else name for name in method_names]
    # ax1.set_xticklabels(np.repeat(abbreviated_names, len(conditions)),
    #                     rotation=0, fontsize=12)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(num_boxes) + 1
    # for id, (method, y) in enumerate(zip(METHODS, np.arange(0.01, 0.03 * len(METHODS), 0.03).tolist())):
    #     fig.text(0.90, y, METHODS[id],
    #              backgroundcolor=LABEL_COLOR_MAP2[id],
    #              color='black', weight='roman', size='x-small')

    plt.savefig(f"./figures/global/boxplot_{title}_global_analysis.svg", bbox_inches='tight')
    plt.savefig(f"./figures/global/boxplot_{title}_global_analysis.png", bbox_inches='tight')
    plt.close()


def compute_ttest(data, method_names):
    ttest_matrix = np.zeros((len(method_names), len(method_names)), dtype=float)
    labels = np.zeros((len(method_names), len(method_names)), dtype=object)
    for m1_id, m1 in enumerate(method_names):
        for m2_id, m2 in enumerate(method_names):
            result = stats.ttest_ind(data[m1_id], data[m2_id], equal_var=True)[1] * (
                        len(method_names) * (len(method_names) - 1) / 2)
            if result > 0.05:
                ttest_matrix[m1_id][m2_id] = -1
                labels[m1_id][m2_id] = ""
            elif 0.01 < result < 0.05:
                ttest_matrix[m1_id][m2_id] = 0
                labels[m1_id][m2_id] = "*"
            else:
                ttest_matrix[m1_id][m2_id] = 1
                labels[m1_id][m2_id] = f"**"

    return ttest_matrix, labels


def plot_ttest_matrix(metric_name, method_names, ttest_matrix, labels):
    df_cm = pd.DataFrame(ttest_matrix, index=method_names, columns=method_names)
    plt.figure(figsize=(14, 12), dpi=600)  # Increased figure size for better readability
    pallete = sn.color_palette("magma", as_cmap=True)

    # Create heatmap with larger font sizes
    ax = sn.heatmap(df_cm, annot=False, fmt="", cmap=pallete)
    sn.heatmap(df_cm, annot=labels, annot_kws={'va': 'top', 'size': 16},
               fmt="s", cbar=False, cmap=pallete, linewidths=5e-3, linecolor='gray')

    # Increase font sizes for labels and title
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, labelpad=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, labelpad=10)

    # Solution for overlapping axis labels: rotate and adjust
    ax.tick_params(axis='x', labelsize=12, rotation=0)
    ax.tick_params(axis='y', labelsize=12, rotation=0)

    # Set title with larger font
    # ax.set_title(f'T-test Results for {metric_name}', fontsize=18, pad=20)

    # Alternative solutions for long method names (commented out):
    # Solution 1: Truncate method names
    # truncated_names = [name[:8] + "..." if len(name) > 8 else name for name in method_names]
    # ax.set_xticklabels(truncated_names, rotation=45, ha='right', fontsize=10)
    # ax.set_yticklabels(truncated_names, rotation=0, fontsize=10)

    # Solution 2: Use vertical text for y-axis
    # ax.tick_params(axis='y', labelrotation=90)

    plt.tight_layout()  # Automatically adjust spacing to prevent overlap
    plt.savefig(f'./figures/global/confusion_{metric_name}_global_analysis.svg', bbox_inches='tight')
    plt.savefig(f'./figures/global/confusion_{metric_name}_global_analysis.png', bbox_inches='tight')
    plt.close()


def main(methods_dict):
    for metric_id, metric_name in enumerate(metric_names):
        data = []
        method_names = list(methods_dict.keys())
        for method_name in method_names:
            method_data = methods_dict[method_name]
            data.append(method_data[:, metric_id].tolist())

        # np.savetxt(f"./figures/global/ttest_{metric_name}.csv", np.array(ttest_matrix), delimiter=",")

        # T-TESTING
        ttest_matrix, labels = compute_ttest(data, method_names)
        plot_ttest_matrix(metric_name, method_names, ttest_matrix, labels)

        plot_box(metric_name, data, method_names, [metric_name])


if __name__ == "__main__":
    columns = ["adjusted_rand_score", "adjusted_mutual_info_score", "purity_score", "silhouette_score",
               "calinski_harabasz_score", "davies_bouldin_score"]
    metric_names = ['ARI', 'AMI', 'Purity', 'SS', 'CHS', 'DBS']

    FOLDER = "./results/saved_latest/"
    methods_dict = {
        'PCA':          filter_columns_and_save(f"{FOLDER}/pca_kmeans.csv", columns=columns),
        'ICA':          filter_columns_and_save(f"{FOLDER}/ica_kmeans.csv", columns=columns),
        'Isomap':       filter_columns_and_save(f"{FOLDER}/isomap_kmeans.csv", columns=columns),
        'LLE':          filter_columns_and_save(f"{FOLDER}/lle_kmeans.csv", columns=columns),
        't-SNE':        filter_columns_and_save(f"{FOLDER}/tsne_kmeans.csv", columns=columns),
        'DM':           filter_columns_and_save(f"{FOLDER}/diffusion_map_kmeans.csv", columns=columns),
        "ACeDeC":       filter_columns_and_save(f"{FOLDER}/acedec.csv", columns=columns),
        "AEC":          filter_columns_and_save(f"{FOLDER}/aec.csv", columns=columns),
        "DCN":          filter_columns_and_save(f"{FOLDER}/dcn.csv", columns=columns),
        "DDC":          filter_columns_and_save(f"{FOLDER}/ddc.csv", columns=columns),
        "DEC":          filter_columns_and_save(f"{FOLDER}/dec.csv", columns=columns),
        "DKM":          filter_columns_and_save(f"{FOLDER}/dkm.csv", columns=columns),
        "Deep\nECT":      filter_columns_and_save(f"{FOLDER}/deepect.csv", columns=columns),
        "Dip\nDECK":      filter_columns_and_save(f"{FOLDER}/dipdeck.csv", columns=columns),
        "Dip\nEncoder":   filter_columns_and_save(f"{FOLDER}/dipencoder.csv", columns=columns),
        "IDEC":         filter_columns_and_save(f"{FOLDER}/idec.csv", columns=columns),
        "N2D":          filter_columns_and_save(f"{FOLDER}/n2d.csv", columns=columns),
        "VaDE":         filter_columns_and_save(f"{FOLDER}/vade.csv", columns=columns),
    }
    main(methods_dict)