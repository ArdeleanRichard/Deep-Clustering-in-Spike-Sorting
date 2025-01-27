from clustpy.deep import ACeDeC, AEC, DCN, DDC, DEC, DeepECT, DipDECK, DipEncoder, DKM, ENRC, IDEC, VaDE, N2D
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, Birch, OPTICS, HDBSCAN, AffinityPropagation

from clustering_algos.autoclustering_pytorch import AutoClustering


def load_algorithms():
    algorithms = {
        # "kmeans": {
        #     "estimator": KMeans,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "init": ["k-means++", "random"],
        #         "max_iter": [300, 500],
        #     },
        # },
        # "dbscan": {
        #     "estimator": DBSCAN,
        #     "param_grid": {
        #         "eps": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        #         "min_samples": [1, 3, 5, 10, 15],
        #     },
        # },
        # "agglomerative": {
        #     "estimator": AgglomerativeClustering,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "linkage": ["ward", "complete", "average"],
        #     },
        # },
        # "spectral": {
        #     "estimator": SpectralClustering,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "affinity": ["nearest_neighbors", "rbf"],
        #         "random_state": [42],
        #     },
        # },
        # "meanshift": {
        #     "estimator": MeanShift,
        #     "param_grid": {
        #         "bandwidth": [None, 0.1, 0.2, 0.3],
        #         "bin_seeding": [True, False],
        #     },
        # },
        # "birch": {
        #     "estimator": Birch,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "threshold": [0.01, 0.05, 0.1, 0.3, 0.5, 0.7],
        #         "branching_factor": [30, 50, 70],
        #     },
        # },
        # "optics": {
        #     "estimator": OPTICS,
        #     "param_grid": {
        #         "min_samples": [5, 10, 15],
        #         "xi": [0.05, 0.1],
        #         "min_cluster_size": [0.05, 0.1],
        #     },
        # },
        # "hdbscan": {
        #     "estimator": HDBSCAN,
        #     "param_grid": {
        #         "min_cluster_size": [5, 10, 15],
        #         "metric": ["euclidean", "manhattan"],
        #         "leaf_size": [25, 40, 70, 100]
        #     },
        # },
        # "affinity": {
        #     "estimator": AffinityPropagation,
        #     "param_grid": {
        #         "damping": [0.5, 0.7, 0.9],
        #         "preference": [None, -50, -100],
        #     },
        # },




        # DEEP CLUSTERINGS:
        "acedec": {
            "estimator": ACeDeC,
            "param_grid": {
                "n_clusters": [2],
                "init": ["acedec", 'subkmeans', 'random', 'sgd'],
                "embedding_size": [20, 30, 40, 50, 60, 70],
                "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}],
                "clustering_optimizer_params": [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_epochs": [100, 150, 200],
                "clustering_epochs": [100, 150, 200],
                "batch_size": [32, 64, 128],

            },
        },
        # "aec": {
        #     "estimator": AEC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "dcn": {
        #     "estimator": DCN,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "ddc": {
        #     "estimator": DDC,
        #     "param_grid": {
        #         "ratio": [0.01, 0.05, 0.1, 0.2],
        #         # "random_state ": [42]
        #     },
        # },
        # "dec": {
        #     "estimator": DEC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "deepect": {
        #     "estimator": DeepECT,
        #     "param_grid": {
        #         "max_n_leaf_nodes": [1,2,3,4,5, 10, 20, 50, 100],
        #         # "random_state ": [42]
        #     },
        # },
        #
        # "dipdeck": {
        #     "estimator": DipDECK,
        #     "param_grid": {
        #         "n_clusters_init": [2,3,5,10],
        #         "dip_merge_threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
        #         # "random_state ": [42]
        #     },
        # },
        # "dipencoder": {
        #     "estimator": DipEncoder,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "dkm": {
        #     "estimator": DKM,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "enrc": {
        #     "estimator": ENRC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "idec": {
        #     "estimator": IDEC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "n2d": {
        #     "estimator": N2D,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "vade": {
        #     "estimator": VaDE,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         # "random_state ": [42]
        #     },
        # },
        # "autoclustering": {
        #     "estimator": AutoClustering,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "input_dim": [1],
        #         "init": ["random"]
        #     },
        # },

    }
    return algorithms

