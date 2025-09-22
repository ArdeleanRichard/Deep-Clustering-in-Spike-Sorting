import itertools
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
import time

from constants import DIR_RESULTS, DIR_FIGURES
from algos import load_algorithms, normalize_dbs
from datasets import load_all_data, load_real_data
from visualization import scatter_plot


def plot_signal_clusters(X, y_true, y_pred, dataset_name, algo_name, save_dir=None):
    """Plot signal comparisons between true and predicted clusters."""
    n_true_clusters = len(np.unique(y_true))
    n_pred_clusters = len(np.unique(y_pred))

    # Create figure with subplots
    fig, axes = plt.subplots(2, max(n_true_clusters, n_pred_clusters),
                             figsize=(4 * max(n_true_clusters, n_pred_clusters), 8))

    if max(n_true_clusters, n_pred_clusters) == 1:
        axes = axes.reshape(-1, 1)

    # Plot true clusters (top row)
    for i, cluster_id in enumerate(np.unique(y_true)):
        cluster_mask = y_true == cluster_id
        cluster_signals = X[cluster_mask]

        # Plot all signals in grey
        for signal in cluster_signals:
            axes[0, i].plot(signal, color='grey', alpha=0.3, linewidth=0.5)

        # Plot mean signal in red
        mean_signal = np.mean(cluster_signals, axis=0)
        axes[0, i].plot(mean_signal, color='red', linewidth=2)
        axes[0, i].set_title(f'True Cluster {int(cluster_id)} (n={np.sum(cluster_mask)})')
        axes[0, i].set_ylabel('Amplitude')
        axes[0, i].grid(True, alpha=0.3)

    # Hide unused subplots in top row
    for i in range(n_true_clusters, max(n_true_clusters, n_pred_clusters)):
        axes[0, i].set_visible(False)

    # Plot predicted clusters (bottom row)
    for i, cluster_id in enumerate(np.unique(y_pred)):
        cluster_mask = y_pred == cluster_id
        cluster_signals = X[cluster_mask]

        # Plot all signals in grey
        for signal in cluster_signals:
            axes[1, i].plot(signal, color='grey', alpha=0.3, linewidth=0.5)

        # Plot mean signal in red
        mean_signal = np.mean(cluster_signals, axis=0)
        axes[1, i].plot(mean_signal, color='red', linewidth=2)
        axes[1, i].set_title(f'Pred Cluster {cluster_id} (n={np.sum(cluster_mask)})')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].set_xlabel('Time')
        axes[1, i].grid(True, alpha=0.3)

    # Hide unused subplots in bottom row
    for i in range(n_pred_clusters, max(n_true_clusters, n_pred_clusters)):
        axes[1, i].set_visible(False)

    # Add overall title with ARI score
    ari_score = adjusted_rand_score(y_true, y_pred)
    fig.suptitle(f'{algo_name} on {dataset_name}\nARI: {ari_score:.3f}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{dataset_name}_{algo_name}_signals.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{dataset_name}_{algo_name}_signals.svg', bbox_inches='tight')

    plt.close()


def compute_per_cell_statistics(X, y_true, y_pred, dataset_name, algo_name):
    """
    Compute per-cell statistics to satisfy reviewer's request for individual cell evaluation.

    Returns detailed statistics showing distribution of evaluations for each cell candidate.
    """
    stats = []

    # Get contingency matrix for cluster mapping
    cont_matrix = contingency_matrix(y_true, y_pred)

    # For each true cluster, find best matching predicted cluster
    best_matches = {}
    for true_cluster in np.unique(y_true):
        true_idx = np.where(np.unique(y_true) == true_cluster)[0][0]
        best_pred_cluster = np.unique(y_pred)[np.argmax(cont_matrix[true_idx, :])]
        best_matches[true_cluster] = best_pred_cluster

    # Compute per-cell statistics
    for i in range(len(X)):
        true_label = y_true[i]
        pred_label = y_pred[i]

        # Check if this cell was correctly clustered
        correctly_clustered = (pred_label == best_matches[true_label])

        # Compute signal characteristics
        signal = X[i]
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_max = np.max(signal)
        signal_min = np.min(signal)
        signal_range = signal_max - signal_min

        # Distance to cluster centroid
        same_true_cluster_mask = y_true == true_label
        true_cluster_centroid = np.mean(X[same_true_cluster_mask], axis=0)
        dist_to_true_centroid = np.linalg.norm(signal - true_cluster_centroid)

        same_pred_cluster_mask = y_pred == pred_label
        pred_cluster_centroid = np.mean(X[same_pred_cluster_mask], axis=0)
        dist_to_pred_centroid = np.linalg.norm(signal - pred_cluster_centroid)

        stats.append({
            'dataset': dataset_name,
            'algorithm': algo_name,
            'cell_id': i,
            'true_cluster': true_label,
            'pred_cluster': pred_label,
            'best_match_cluster': best_matches[true_label],
            'correctly_clustered': correctly_clustered,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'signal_range': signal_range,
            'dist_to_true_centroid': dist_to_true_centroid,
            'dist_to_pred_centroid': dist_to_pred_centroid,
            'centroid_distance_ratio': dist_to_pred_centroid / (dist_to_true_centroid + 1e-8)
        })

    return pd.DataFrame(stats)


def plot_per_cell_statistics(per_cell_stats, save_dir=None):
    """Plot distribution of per-cell evaluation metrics."""
    datasets = per_cell_stats['dataset'].unique()
    algorithms = per_cell_stats['algorithm'].unique()

    for dataset in datasets:
        for algo in algorithms:
            subset = per_cell_stats[(per_cell_stats['dataset'] == dataset) &
                                    (per_cell_stats['algorithm'] == algo)]

            if len(subset) == 0:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Accuracy per true cluster
            accuracy_per_cluster = subset.groupby('true_cluster')['correctly_clustered'].mean()
            axes[0, 0].bar(range(len(accuracy_per_cluster)), accuracy_per_cluster.values)
            axes[0, 0].set_title('Accuracy per True Cluster')
            axes[0, 0].set_xlabel('True Cluster ID')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(range(len(accuracy_per_cluster)))
            axes[0, 0].set_xticklabels(accuracy_per_cluster.index)

            # Distance ratio distribution
            axes[0, 1].hist(subset['centroid_distance_ratio'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Centroid Distance Ratio Distribution')
            axes[0, 1].set_xlabel('Pred Distance / True Distance')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].axvline(x=1, color='red', linestyle='--', label='Equal distances')
            axes[0, 1].legend()

            # Signal characteristics by cluster accuracy
            correct = subset[subset['correctly_clustered'] == True]
            incorrect = subset[subset['correctly_clustered'] == False]

            axes[0, 2].scatter(correct['signal_mean'], correct['signal_std'],
                               alpha=0.6, label='Correctly clustered', color='green')
            axes[0, 2].scatter(incorrect['signal_mean'], incorrect['signal_std'],
                               alpha=0.6, label='Incorrectly clustered', color='red')
            axes[0, 2].set_title('Signal Characteristics vs Clustering Accuracy')
            axes[0, 2].set_xlabel('Signal Mean')
            axes[0, 2].set_ylabel('Signal Std')
            axes[0, 2].legend()

            # Per-cluster sample count
            true_counts = subset['true_cluster'].value_counts().sort_index()
            pred_counts = subset['pred_cluster'].value_counts().sort_index()

            # Get all unique cluster IDs from both true and predicted
            all_cluster_ids = sorted(set(true_counts.index) | set(pred_counts.index))

            # Reindex both to have the same cluster IDs (fill missing with 0)
            true_counts = true_counts.reindex(all_cluster_ids, fill_value=0)
            pred_counts = pred_counts.reindex(all_cluster_ids, fill_value=0)

            x = np.arange(len(all_cluster_ids))
            width = 0.35
            axes[1, 0].bar(x - width / 2, true_counts.values, width, label='True clusters', alpha=0.7)
            axes[1, 0].bar(x + width / 2, pred_counts.values, width, label='Predicted clusters', alpha=0.7)
            axes[1, 0].set_title('Cluster Size Comparison')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('Number of Cells')
            axes[1, 0].legend()
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(all_cluster_ids)

            # Confusion matrix heatmap
            conf_matrix = contingency_matrix(subset['true_cluster'], subset['pred_cluster'])
            im = axes[1, 1].imshow(conf_matrix, cmap='Blues')
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted Cluster')
            axes[1, 1].set_ylabel('True Cluster')

            # Add text annotations
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    axes[1, 1].text(j, i, str(conf_matrix[i, j]),
                                    ha="center", va="center", color="black")

            plt.colorbar(im, ax=axes[1, 1])

            # Overall statistics text
            total_accuracy = subset['correctly_clustered'].mean()
            mean_distance_ratio = subset['centroid_distance_ratio'].mean()

            stats_text = f"""Overall Statistics:
            Accuracy: {total_accuracy:.3f}
            Mean Distance Ratio: {mean_distance_ratio:.3f}
            Total Cells: {len(subset)}
            Correctly Clustered: {subset['correctly_clustered'].sum()}
            """

            axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                            fontsize=12, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')

            plt.suptitle(f'{algo} on {dataset} - Per-Cell Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f'{save_dir}/{dataset}_{algo}_per_cell_analysis.png', dpi=150, bbox_inches='tight')
                plt.savefig(f'{save_dir}/{dataset}_{algo}_per_cell_analysis.svg', bbox_inches='tight')

            plt.close()


def run(datasets, algorithms):
    os.makedirs(DIR_RESULTS + "./grid_search/", exist_ok=True)
    os.makedirs(DIR_FIGURES + "signals/", exist_ok=True)

    all_per_cell_stats = []

    for algo_name, algo_details in algorithms.items():
        results = []

        for dataset_name, (X, y, y_true) in datasets:
            print(algo_name, dataset_name)

            pca_2d = PCA(n_components=2)
            X_2d = pca_2d.fit_transform(X)

            scatter_plot.plot(f'{dataset_name} true labels', X_2d, y_true, marker='o', binary_markers=y_true)
            plt.savefig(DIR_FIGURES + "svgs/" + f'{dataset_name}_gt.svg')
            plt.savefig(DIR_FIGURES + "pngs/" + f'{dataset_name}_gt.png')
            plt.close()

            # Normalize dataset
            X_copy = np.copy(X)
            scaler = preprocessing.MinMaxScaler().fit(X)
            X = scaler.transform(X)
            X = np.clip(X, 0, 1)

            param_names = list(algo_details["param_grid"].keys())

            # Special parameter handling
            for param_name in param_names:
                if param_name == "n_clusters":
                    algo_details["param_grid"]["n_clusters"] = [len(np.unique(y))]
                if param_name == "n_clusters_init":
                    algo_details["param_grid"]["n_clusters_init"] = [len(np.unique(y))]
                if param_name == "min_n_clusters":
                    algo_details["param_grid"]["min_n_clusters"] = [len(np.unique(y)) - 1]
                if param_name == "max_n_clusters":
                    algo_details["param_grid"]["max_n_clusters"] = [len(np.unique(y)) + 1]
                if param_name == "input_dim":
                    algo_details["param_grid"]["input_dim"] = [X.shape[1]]

            print(algo_details["param_grid"]["pretrain_optimizer_params"])

            param_combinations = list(itertools.product(*algo_details["param_grid"].values()))

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                scores_per_repeat = []
                avg_fit_predict_time = -1

                estimator = algo_details["estimator"](**param_dict)

                y_pred = estimator.fit_predict(X)
                try:
                    if len(np.unique(y_pred)) > 1:
                        ari = adjusted_rand_score(y_true, y_pred)
                        ami = adjusted_mutual_info_score(y_true, y_pred)
                        contingency_mat = contingency_matrix(y_true, y_pred)
                        purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                        silhouette = silhouette_score(X_copy, y_pred)
                        calinski_harabasz = calinski_harabasz_score(X_copy, y_pred)
                        davies_bouldin = davies_bouldin_score(X_copy, y_pred)

                        # Generate signal comparison plots
                        plot_signal_clusters(X_copy, y_true, y_pred, dataset_name, algo_name, DIR_FIGURES + "signals/")

                        # Compute per-cell statistics
                        per_cell_stats = compute_per_cell_statistics(X_copy, y_true, y_pred, dataset_name, algo_name)
                        all_per_cell_stats.append(per_cell_stats)

                    else:
                        print(f"[1CLUST] {algo_name}, {params}")
                        ari = ami = purity = silhouette = calinski_harabasz = davies_bouldin = -1

                    scatter_plot.plot(f'{algo_name} on {dataset_name}', X_2d, y_pred, marker='o', binary_markers=y_true)
                    plt.savefig(DIR_FIGURES + "svgs/" + f'{dataset_name}_{algo_name}.svg')
                    plt.savefig(DIR_FIGURES + "pngs/" + f'{dataset_name}_{algo_name}.png')
                    plt.close()

                    scores_per_repeat.append({
                        "dataset": dataset_name,
                        "adjusted_rand_score": ari,
                        "adjusted_mutual_info_score": ami,
                        "purity_score": purity,
                        "silhouette_score": silhouette,
                        "calinski_harabasz_score": calinski_harabasz,
                        "davies_bouldin_score": davies_bouldin,
                        "avg_fit_predict_time": avg_fit_predict_time,
                    })
                except Exception as e:
                    print(f"[ERROR] {algo_name}, {params}, {e}")
                    scores_per_repeat.append({
                        "dataset": dataset_name,
                        "adjusted_rand_score": -1,
                        "adjusted_mutual_info_score": -1,
                        "purity_score": -1,
                        "silhouette_score": -1,
                        "calinski_harabasz_score": -1,
                        "davies_bouldin_score": -1,
                        "avg_fit_predict_time": avg_fit_predict_time,
                    })

                print("RESULTS: ", scores_per_repeat[0])
                results.append(scores_per_repeat[0])

            # Save results for this algorithm
            df = pd.DataFrame(results)
            df = normalize_dbs(df)
            df.to_csv(DIR_RESULTS + f"{algo_name}.csv", index=False)

    # Combine all per-cell statistics and generate plots
    if all_per_cell_stats:
        combined_per_cell_stats = pd.concat(all_per_cell_stats, ignore_index=True)
        combined_per_cell_stats.to_csv(DIR_RESULTS + "per_cell_statistics.csv", index=False)
        plot_per_cell_statistics(combined_per_cell_stats, DIR_FIGURES + "signals/")


if __name__ == "__main__":
    datasets = load_real_data()
    algorithms = load_algorithms()
    run(datasets, algorithms)