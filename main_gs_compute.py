import itertools

import clustpy.deep.enrc
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

def perform_grid_search(datasets, algorithms, n_repeats=10, add=""):
    for dataset_name, (X, y_true) in datasets:

        test = np.copy(X)
        scaler = preprocessing.MinMaxScaler().fit(X)
        X = scaler.transform(X)
        X = np.clip(X, 0, 1) # error, range given is [0.0, 1.0000000000000002] from floating point precision

        # X = torch.tensor(X, dtype=torch.float32)
        # X = torch.clamp(X, 0, 1)


        for algo_name, algo_details in algorithms.items():
            results = []
            param_names = list(algo_details["param_grid"].keys())

            # -------------
            # SPECIAL PARAMS
            # -------------
            for param_name in param_names:
                if param_name == "n_clusters":
                    algo_details["param_grid"]["n_clusters"] = [len(np.unique(y_true))]
                    if algo_details["estimator"] is clustpy.deep.enrc.ENRC:
                        algo_details["param_grid"]["n_clusters"] = [[len(np.unique(y_true)),len(np.unique(y_true)),len(np.unique(y_true))]]
                if param_name == "n_clusters_init":
                    algo_details["param_grid"]["n_clusters_init"] = [len(np.unique(y_true))]
                if param_name == "min_n_clusters":
                    algo_details["param_grid"]["min_n_clusters"] = [len(np.unique(y_true))-1]
                if param_name == "max_n_clusters":
                    algo_details["param_grid"]["max_n_clusters"] = [len(np.unique(y_true))+1]
                if param_name == "input_dim":
                    algo_details["param_grid"]["input_dim"] = [X.shape[1]]
                # if param_name == "initial_clustering_params":
                #     algo_details["param_grid"]["initial_clustering_params"] = [{"n_clusters": [len(np.unique(y_true))]}]
            # -------------


            param_combinations = list(itertools.product(*algo_details["param_grid"].values()))
            print(dataset_name, algo_name)

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                # is_nondeterministic = any(
                #     key in param_dict for key in ["init", "random_state"]
                # )
                is_nondeterministic = False

                scores_per_repeat = []

                for _ in range(n_repeats if is_nondeterministic else 1):
                    try:
                        estimator = algo_details["estimator"](**param_dict)
                        print(param_dict)
                        y_pred = estimator.fit_predict(X)
                        print(estimator.labels_.shape)

                        if len(np.unique(y_pred)) > 1:  # Ensure more than one cluster
                            ari = adjusted_rand_score(y_true, y_pred)
                            ami = adjusted_mutual_info_score(y_true, y_pred)
                            contingency_mat = contingency_matrix(y_true, y_pred)
                            purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                            silhouette = silhouette_score(test, y_pred)
                            calinski_harabasz = calinski_harabasz_score(test, y_pred)
                            davies_bouldin = davies_bouldin_score(test, y_pred)
                        else:
                            print(f"[1CLUST] {algo_name}, {params}")
                            ari = ami = purity = silhouette = calinski_harabasz = davies_bouldin = -1

                        scores_per_repeat.append({
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
                            "adjusted_rand_score": -1,
                            "adjusted_mutual_info_score": -1,
                            "purity_score": -1,
                            "silhouette_score": -1,
                            "calinski_harabasz_score": -1,
                            "davies_bouldin_score": -1,
                        })

                # Aggregate scores across repeats for nondeterministic algorithms
                if is_nondeterministic:
                    aggregated_scores = {
                        key: np.nanmean([score[key] for score in scores_per_repeat])
                        for key in scores_per_repeat[0]
                    }
                else:
                    aggregated_scores = scores_per_repeat[0]

                results.append({
                    **param_dict,
                    **aggregated_scores,
                })

                # Save results to CSV
                results_df = pd.DataFrame(results)
                results_df = normalize_dbs(results_df)
                results_df.to_csv(DIR_RESULTS + f"./grid_search/{algo_name}_{dataset_name}{add}.csv", index=False)


if __name__ == "__main__":
    # datasets = load_all_data()
    # algorithms = load_algorithms()
    # perform_grid_search(datasets, algorithms)

    for i in range(1):
        datasets = load_all_data()
        algorithms = load_algorithms()
        perform_grid_search(datasets, algorithms, add=f"_{i}")