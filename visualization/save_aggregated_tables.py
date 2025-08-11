import pandas as pd
import glob
import os

def read_data():
    columns = ["dataset", "adjusted_rand_score", "adjusted_mutual_info_score", "purity_score", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]

    FOLDER = "../results/saved_latest/"

    methods_dict = {
        'PCA':          filter_df(f"{FOLDER}/pca_kmeans.csv", columns=columns),
        'ICA':          filter_df(f"{FOLDER}/ica_kmeans.csv", columns=columns),
        'Isomap':       filter_df(f"{FOLDER}/isomap_kmeans.csv", columns=columns),
        'LLE':          filter_df(f"{FOLDER}/lle_kmeans.csv", columns=columns),
        't-SNE':        filter_df(f"{FOLDER}/tsne_kmeans.csv", columns=columns),
        'DM':           filter_df(f"{FOLDER}/diffusion_map_kmeans.csv", columns=columns),
        "ACeDeC":       filter_df(f"{FOLDER}/acedec.csv", columns=columns),
        "AEC":          filter_df(f"{FOLDER}/aec.csv", columns=columns),
        "DCN":          filter_df(f"{FOLDER}/dcn.csv", columns=columns),
        "DDC":          filter_df(f"{FOLDER}/ddc.csv", columns=columns),
        "DEC":          filter_df(f"{FOLDER}/dec.csv", columns=columns),
        "DKM":          filter_df(f"{FOLDER}/dkm.csv", columns=columns),
        "DeepECT":      filter_df(f"{FOLDER}/deepect.csv", columns=columns),
        "DipDECK":      filter_df(f"{FOLDER}/dipdeck.csv", columns=columns),
        "DipEncoder":   filter_df(f"{FOLDER}/dipencoder.csv", columns=columns),
        "IDEC":         filter_df(f"{FOLDER}/idec.csv", columns=columns),
        "N2D":          filter_df(f"{FOLDER}/n2d.csv", columns=columns),
        "VaDE":         filter_df(f"{FOLDER}/vade.csv", columns=columns),
    }


    dfs = []
    method_names = list(methods_dict.keys())
    for method_name in method_names:
        method_data = methods_dict[method_name]
        dfs.append((method_name, method_data))

    return dfs

def aggregate_by_dataset(target_dataset, dfs):
    rows = []
    for (algorithm, df) in dfs:
        # Filter for the row with the desired dataset
        matching_row = df[df['dataset'] == target_dataset]
        if not matching_row.empty:
            row = matching_row.copy()
            row.insert(0, "algorithm", algorithm)
            rows.append(row)

    # Combine all the rows into a single DataFrame
    if rows:
        result_df = pd.concat(rows, ignore_index=True)

        float_cols = result_df.select_dtypes(include=['float'])
        result_df[float_cols.columns] = float_cols.round(3)

        column_renames = {
            'adjusted_rand_score': 'ARI',
            'adjusted_mutual_info_score': 'AMI',
            'purity_score': 'Purity',
            'silhouette_score': 'SS',
            'calinski_harabasz_score': 'CHS',
            'davies_bouldin_score': 'DBS',
        }
        result_df = result_df.rename(columns=column_renames)

        result_df = result_df.drop(columns=['dataset'])
        result_df.to_csv(f"../paper/tables/{target_dataset}.csv", index=False)

        print(f"Aggregation complete: saved to ../paper/tables/{target_dataset}.csv.csv")
    else:
        print(f"No matching dataset '{target_dataset}' found in the CSVs.")



def filter_df(input_csv, columns):
    df = pd.read_csv(input_csv)

    df_filtered = df[columns]

    return df_filtered

if __name__ == "__main__":
    for simulation_number in ["Sim4", "Sim15", "Sim20", "Sim2"]:
        dfs = read_data()
        aggregate_by_dataset(simulation_number, dfs)

    # for c_number in [28, 37]:
    #     dfs = read_data()
    #     aggregate_by_dataset(target_dataset=f"kampff_c{c_number}", dfs=dfs)