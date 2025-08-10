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
from gs_algos import load_algorithms
from gs_datasets import load_all_data, load_real_data
from visualization import scatter_plot


def normalize_dbs(df):
    df['norm_davies_bouldin_score'] = 1 / (1 + df['davies_bouldin_score'])
    return df

def perform_grid_search(datasets, algorithms):
    os.makedirs(DIR_RESULTS + "./grid_search/", exist_ok=True)

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
                    algo_details["param_grid"]["min_n_clusters"] = [len(np.unique(y))-1]
                if param_name == "max_n_clusters":
                    algo_details["param_grid"]["max_n_clusters"] = [len(np.unique(y))+1]
                if param_name == "input_dim":
                    algo_details["param_grid"]["input_dim"] = [X.shape[1]]

            print(algo_details["param_grid"]["pretrain_optimizer_params"])


            param_combinations = list(itertools.product(*algo_details["param_grid"].values()))

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                scores_per_repeat = []

                try:
                    estimator = algo_details["estimator"](**param_dict)
                    y_pred = estimator.fit_predict(X)

                    if len(np.unique(y_pred)) > 1:
                        ari = adjusted_rand_score(y_true, y_pred)
                        ami = adjusted_mutual_info_score(y_true, y_pred)
                        contingency_mat = contingency_matrix(y_true, y_pred)
                        purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                        silhouette = silhouette_score(X_copy, y_pred)
                        calinski_harabasz = calinski_harabasz_score(X_copy, y_pred)
                        davies_bouldin = davies_bouldin_score(X_copy, y_pred)
                    else:
                        print(f"[1CLUST] {algo_name}, {params}")
                        ari = ami = purity = silhouette = calinski_harabasz = davies_bouldin = -1

                    scatter_plot.plot(f'{algo_name} on {dataset_name}', X_2d, y_pred, marker='o', binary_markers=y_true)
                    plt.savefig(DIR_FIGURES + "svgs/" + f'{dataset_name}_{algo_name}.svg')
                    plt.savefig(DIR_FIGURES + "pngs/" + f'{dataset_name}_{algo_name}.png')
                    plt.close()

                    scores_per_repeat.append({
                        "dataset": dataset_name, # Track dataset in results
                        "adjusted_rand_score": ari,
                        "adjusted_mutual_info_score": ami,
                        "purity_score": purity,
                        "silhouette_score": silhouette,
                        "calinski_harabasz_score": calinski_harabasz,
                        "davies_bouldin_score": davies_bouldin,
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
                    })

                print("RESULTS: ", scores_per_repeat[0])
                results.append(scores_per_repeat[0])

            # Save results for this algorithm
            df = pd.DataFrame(results)
            df = normalize_dbs(df)
            df.to_csv(DIR_RESULTS + f"{algo_name}.csv", index=False)

if __name__ == "__main__":
    datasets = load_real_data()
    algorithms = load_algorithms()
    perform_grid_search(datasets, algorithms)
