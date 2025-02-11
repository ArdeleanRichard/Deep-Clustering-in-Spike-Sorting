from clustpy.deep import ACeDeC, AEC, DCN, DDC, DEC, DeepECT, DipDECK, DipEncoder, DKM, ENRC, IDEC, VaDE, N2D
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, Birch, OPTICS, HDBSCAN, AffinityPropagation
from sklearn.mixture import GaussianMixture

from clustering_algos.autoclustering_pytorch import AutoClustering


def load_algorithms():
    layers = [79, 100, 100, 100, 10]
    dropout = 0.5
    random_state = 42
    batch_norm = True
    # neural_network = FeedforwardAutoencoder(layers=layers) #, dropout=dropout, batch_norm=batch_norm, random_state=random_state)

    algorithms = {




        #
        # # DEEP CLUSTERINGS:
        # "acedec": {
        #     "estimator": ACeDeC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "init": ["acedec"], #, 'subkmeans', 'random', 'sgd'],
        #         "embedding_size": [10], #, 20, 30, 40, 50, 60, 70],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #
        #         # "pretrain_epochs": [100, 150, 200],
        #         # "clustering_epochs": [100, 150, 200],
        #         # "batch_size": [32, 64, 128],
        #
        #     },
        # },
        "aec": {
            "estimator": AEC,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-4}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [42]
            },
        },
        # "dcn": {
        #     "estimator": DCN,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "ddc": {
        #     "estimator": DDC,
        #     "param_grid": {
        #         "ratio": [0.15, 0.2, 0.25, 0.3, 0.5], #[0.01, 0.05, 0.1, 0.2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "dec": {
        #     "estimator": DEC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "deepect": {
        #     "estimator": DeepECT,
        #     "param_grid": {
        #         "max_n_leaf_nodes": [1], #,2,3,4,5, 10, 20, 50, 100],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        #
        # "dipdeck": {
        #     "estimator": DipDECK,
        #     "param_grid": {
        #         "n_clusters_init": [2,3,5,10,20],
        #         "dip_merge_threshold": [0.5], #[0.1, 0.3, 0.5, 0.7, 0.9],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "dipencoder": {
        #     "estimator": DipEncoder,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "dkm": {
        #     "estimator": DKM,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "idec": {
        #     "estimator": IDEC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "n2d": {
        #     "estimator": N2D,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },
        # "vade": {
        #     "estimator": VaDE,
        #     "param_grid": {
        #         "n_clusters": [2],
        #
        #         "embedding_size": [10],
        #         # "neural_network": [(FeedforwardAutoencoder, {"layers": [79, 100, 100, 100, 50], "dropout": 0.5, "batch_norm": True})], # error error no sense
        #         # "pretrain_optimizer_params":    [{"lr": 1e-2}],
        #         # "clustering_optimizer_params":  [{"lr": 1e-3}],
        #         # "batch_size": [256],
        #         # "pretrain_epochs": [10],
        #         # "clustering_epochs": [150],
        #
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         # "batch_size": [256, 32, 64, 128, 512],
        #         # "pretrain_epochs": [10, 50, 100],
        #         # "clustering_epochs": [150, 300],
        #         "random_state": [42],
        #
        #         # "initial_clustering_class": [KMeans],
        #         # "initial_clustering_params": [{}],
        #
        #         # "neural_network": [neural_network],
        #         # "embedding_size": [10, 20, 30, 40, 50, 60, 70],
        #
        #         # "embedding_size": [10],
        #         # "clustering_loss_weight": [1.0],
        #         # "ssl_loss_weight": [0.1],
        #
        #         # "embedding_size": [10],
        #         # "batch_size": [1024],
        #         # "clustering_loss_weight": [0.1],
        #         # "ssl_loss_weight": [1.0],
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




        # "enrc": {
        #     "estimator": ENRC,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "embedding_size": [10],
        #         "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
        #         "random_state": [42]
        #     },
        # },

    }
    return algorithms

