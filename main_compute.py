import itertools
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix

from constants import DIR_RESULTS
from gs_algos import load_algorithms
from gs_datasets import load_all_data


def normalize_dbs(df):
    df['norm_davies_bouldin_score'] = 1 / (1 + df['davies_bouldin_score'])
    return df

def perform_grid_search(datasets, algorithms, n_repeats=10):
    os.makedirs(DIR_RESULTS + "./grid_search/", exist_ok=True)

    for algo_name, algo_details in algorithms.items():
        results = []

        for dataset_name, (X, y_true) in datasets:
            print(algo_name, dataset_name)

            # Normalize dataset
            X_copy = np.copy(X)
            scaler = preprocessing.MinMaxScaler().fit(X)
            X = scaler.transform(X)
            X = np.clip(X, 0, 1)

            param_names = list(algo_details["param_grid"].keys())

            # Special parameter handling
            for param_name in param_names:
                if param_name == "n_clusters":
                    algo_details["param_grid"]["n_clusters"] = [len(np.unique(y_true))]
                if param_name == "n_clusters_init":
                    algo_details["param_grid"]["n_clusters_init"] = [len(np.unique(y_true))]
                if param_name == "min_n_clusters":
                    algo_details["param_grid"]["min_n_clusters"] = [len(np.unique(y_true))-1]
                if param_name == "max_n_clusters":
                    algo_details["param_grid"]["max_n_clusters"] = [len(np.unique(y_true))+1]
                if param_name == "input_dim":
                    algo_details["param_grid"]["input_dim"] = [X.shape[1]]

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

                results.append(scores_per_repeat[0])

            # Save results for this algorithm
            df = pd.DataFrame(results)
            df = normalize_dbs(df)
            df.to_csv(DIR_RESULTS + f"{algo_name}.csv", index=False)

if __name__ == "__main__":
    datasets = load_all_data()
    algorithms = load_algorithms()
    perform_grid_search(datasets, algorithms)
