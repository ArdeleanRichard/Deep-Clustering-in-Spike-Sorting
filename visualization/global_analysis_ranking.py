import numpy as np
import pandas as pd

from validation.rank_aggregation.rank_agg.rankagg import FullListRankAggregator
from visualization.global_analysis_utils import filter_columns_and_save


def main(methods_dict):
    method_names = list(methods_dict.keys())
    data_arrays = [methods_dict[name] for name in method_names]

    metric_to_ranked_methods = {}
    for met_id, met in enumerate(metric_names):
        print("----------------------------------------------------------------")
        print(f"------------------------------{met}-----------------------------")
        print("----------------------------------------------------------------")

        ranks_list = []
        ranks_dict = []
        num_samples = len(data_arrays[0])

        for sample_id in range(num_samples):
            method_results = np.array([data[sample_id][met_id] for data in data_arrays])

            if met == "DBS":
                method_results = -method_results

            ranked_methods = np.array(method_names)[np.argsort(method_results)[::-1]].tolist()
            ranks_list.append(ranked_methods)

            ranks = {method_names[i]: method_results[i] for i in range(len(method_names))}
            ranks_dict.append(ranks)

        FLRA = FullListRankAggregator()
        borda_ranking = FLRA.aggregate_ranks(ranks_dict, method='borda')[1]
        print("Borda: ", borda_ranking)
        print()

        # Sort methods by rank (ascending = best first)
        sorted_methods = sorted(borda_ranking.items(), key=lambda x: x[1])
        metric_to_ranked_methods[met] = [method for method, _ in sorted_methods]

    # Reformat for DataFrame: rows = ranks, columns = metrics
    max_rank = len(method_names)
    df_output = pd.DataFrame(index=range(1, max_rank + 1))  # 1-based index for ranks

    for metric in metric_names:
        df_output[metric] = metric_to_ranked_methods[metric]

    df_output.to_csv("./paper/tables/borda_rankings.csv", index_label="Method")
    print("Saved Borda rankings to './paper/tables/borda_rankings.csv'")


if __name__ == "__main__":
    import os
    os.chdir("../")

    METHODS = ["PCA", "ICA", "Isomap", "AE"]

    columns = ["adjusted_rand_score", "adjusted_mutual_info_score", "purity_score", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
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
        "DeepECT":      filter_columns_and_save(f"{FOLDER}/deepect.csv", columns=columns),
        "DipDECK":      filter_columns_and_save(f"{FOLDER}/dipdeck.csv", columns=columns),
        "DipEncoder":   filter_columns_and_save(f"{FOLDER}/dipencoder.csv", columns=columns),
        "IDEC":         filter_columns_and_save(f"{FOLDER}/idec.csv", columns=columns),
        "N2D":          filter_columns_and_save(f"{FOLDER}/n2d.csv", columns=columns),
        "VaDE":         filter_columns_and_save(f"{FOLDER}/vade.csv", columns=columns),
    }

    main(methods_dict)