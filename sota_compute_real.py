import itertools
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
from constants import DIR_RESULTS, DIR_FIGURES
from gs_datasets import load_all_data, load_real_data
from visualization import scatter_plot
import time

import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs available


def load_algorithms_fe():
    algorithms = {
        "pca": {
            "estimator": PCA,
            "param_grid": {
                "n_components": 2,
            },
        },
        "ica": {
            "estimator": FastICA,
            "param_grid": {
                "n_components": 2,
                "fun": "logcosh",
                "max_iter": 200,
                "tol": 1e-3,
            },
        },
        "isomap": {
            "estimator": Isomap,
            "param_grid": {
                "n_neighbors": 25,
                "n_components": 2,
                "eigen_solver": "arpack",
                "path_method": "D",
                "n_jobs": -1,
            },
        },
    }

    return algorithms

def load_algorithms_clust():
    algorithms = {

        "kmeans": {
            "estimator": KMeans,
            "param_grid": {
                "n_clusters": 2,
            },
        },

    }

    return algorithms



def normalize_dbs(df):
    df['norm_davies_bouldin_score'] = 1 / (1 + df['davies_bouldin_score'])
    return df

def perform_grid_search(datasets, featureextraction_algorithms, clustering_algorithms, n_repeats=10):
    os.makedirs(DIR_RESULTS + "./grid_search/", exist_ok=True)

    for fe_name, fe_details in featureextraction_algorithms.items():
        results = []

        for clust_name, clust_details in clustering_algorithms.items():

            for dataset_name, (X, y, y_true) in datasets:
                print(fe_name, clust_name, dataset_name)
                # Normalize dataset
                test = np.copy(X)
                # scaler = preprocessing.MinMaxScaler().fit(X)
                # X = scaler.transform(X)
                # X = np.clip(X, 0, 1)

                pca_2d = PCA(n_components=2)
                X_2d = pca_2d.fit_transform(X)

                fe_param_names = list(fe_details["param_grid"].keys())
                clust_param_names = list(clust_details["param_grid"].keys())


                # Special parameter handling
                for param_name in clust_param_names:
                    if param_name == "n_clusters" or param_name == "n_clusters_init":
                        clust_details["param_grid"]["n_clusters"] = len(np.unique(y))
                    if param_name == "input_dim":
                        clust_details["param_grid"]["input_dim"] = [X.shape[1]]

                fe_params = fe_details["param_grid"]
                clust_params = clust_details["param_grid"]


                scores = None

                try:
                    transformer = fe_details["estimator"](**fe_params)
                    times_fit_transform = []

                    for _ in range(5):
                        start_time = time.time()
                        X_transformed = transformer.fit_transform(X)
                        end_time = time.time()

                        elapsed_time = end_time - start_time
                        times_fit_transform.append(elapsed_time)

                    average_time_fit_transform = np.mean(times_fit_transform)

                    np.savetxt(DIR_RESULTS + f"spaces/{fe_name}/{dataset_name}.csv", X_transformed, delimiter=",")

                    estimator = clust_details["estimator"](**clust_params)


                    times_fit_predict = []

                    for _ in range(5):
                        start_time = time.time()
                        y_pred = estimator.fit_predict(X_transformed)
                        end_time = time.time()

                        elapsed_time = end_time - start_time
                        times_fit_predict.append(elapsed_time)

                    average_time_fit_predict = np.mean(times_fit_predict)

                    if len(np.unique(y_pred)) > 1:
                        ari = adjusted_rand_score(y_true, y_pred)
                        ami = adjusted_mutual_info_score(y_true, y_pred)
                        contingency_mat = contingency_matrix(y_true, y_pred)
                        purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                        silhouette = silhouette_score(test, y_pred)
                        calinski_harabasz = calinski_harabasz_score(test, y_pred)
                        davies_bouldin = davies_bouldin_score(test, y_pred)
                    else:
                        print(f"[1CLUST] {fe_name}, {clust_name}, {fe_params}")
                        ari = ami = purity = silhouette = calinski_harabasz = davies_bouldin = -1

                    scores = {
                        "dataset": dataset_name, # Track dataset in results
                        "adjusted_rand_score": ari,
                        "adjusted_mutual_info_score": ami,
                        "purity_score": purity,
                        "silhouette_score": silhouette,
                        "calinski_harabasz_score": calinski_harabasz,
                        "davies_bouldin_score": davies_bouldin,
                        "average_time_fit_transform": average_time_fit_transform,
                        "average_time_fit_predict": average_time_fit_predict,
                    }
                    scatter_plot.plot(f'{fe_name} + {clust_name} on {dataset_name}', X_2d, y_pred, marker='o', binary_markers=y_true)
                    plt.savefig(DIR_FIGURES + "svgs/" + f'{dataset_name}_{fe_name}_{clust_name}.svg')
                    plt.savefig(DIR_FIGURES + "pngs/" + f'{dataset_name}_{fe_name}_{clust_name}.png')
                    plt.close()


                except Exception as e:
                    print(f"[ERROR] {clust_name}, {fe_params}, {e}")
                    scores = {
                        "dataset": dataset_name,
                        "adjusted_rand_score": -1,
                        "adjusted_mutual_info_score": -1,
                        "purity_score": -1,
                        "silhouette_score": -1,
                        "calinski_harabasz_score": -1,
                        "davies_bouldin_score": -1,
                    }

                results.append(scores)

                # Save results for this algorithm
                df = pd.DataFrame(results)
                df = normalize_dbs(df)
                df.to_csv(DIR_RESULTS + f"{fe_name}_{clust_name}.csv", index=False)

if __name__ == "__main__":
    datasets = load_real_data()
    fes = load_algorithms_fe()
    clusts = load_algorithms_clust()
    perform_grid_search(datasets, fes, clusts)
