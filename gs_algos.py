import os
import random
import numpy as np
import torch
from clustpy.deep import ACeDeC, AEC, DCN, DDC, DEC, DeepECT, DipDECK, DipEncoder, DKM, ENRC, IDEC, VaDE, N2D
from clustering_algos.autoclustering_pytorch import AutoClustering


CUDA_VISIBLE_DEVICES=""
random_state=42
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)


def load_algorithms():
    algorithms = {
        # # DEEP CLUSTERINGS:
        "acedec": {
            "estimator": ACeDeC,
            "param_grid": {
                "n_clusters": [2],
                "init": ["acedec"], #, 'subkmeans', 'random', 'sgd'],
                "embedding_size": [10], #, 20, 30, 40, 50, 60, 70],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-3}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],

                # "pretrain_epochs": [100, 150, 200],
                # "clustering_epochs": [100, 150, 200],
                # "batch_size": [32, 64, 128],

            },
        },
        "aec": {
            "estimator": AEC,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "pretrain_optimizer_params": [{"lr": 1e-5}],
                "clustering_optimizer_params": [{"lr": 1e-2}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state]
            },
        },
        "dcn": {
            "estimator": DCN,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "clustering_optimizer_params": [{"lr": 1e-3}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state],
                "device": ['cpu'],
            },
        },
        "ddc": {
            "estimator": DDC,
            "param_grid": {
                "embedding_size": [10],
                "ratio": [0.1], #[0.01, 0.05, 0.1, 0.2],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "ratio": [0.01, 0.05, 0.1, 0.2, 0.15], #[0.01, 0.05, 0.1, 0.2],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state]
            },
        },
        "dec": {
            "estimator": DEC,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "alpha": [0.25],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-4}],
                "initial_clustering_params": [{"init": 'k-means++'}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "alpha": [0.5, 0.75, 1.0, 1.5],
                "random_state": [random_state],
            },
        },
        "deepect": {
            "estimator": DeepECT,
            "param_grid": {
                "embedding_size": [10],
                "max_n_leaf_nodes": [20],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-4}],
                # "max_n_leaf_nodes": [1,2,3,4,5, 10, 20, 50, 100],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state]
            },
        },

        "dipdeck": {
            "estimator": DipDECK,
            "param_grid": {
                "n_clusters_init": [2],
                "min_n_clusters": [2],
                "max_n_clusters": [2],
                "embedding_size": [10],
                "dip_merge_threshold": [0.9],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-3}],
                # "dip_merge_threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state]
            },
        },
        "dipencoder": {
            "estimator": DipEncoder,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-4}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state],
                "device": ['cpu'],
            },
        },
        "dkm": {
            "estimator": DKM,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-5}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state],
            },
        },
        "idec": {
            "estimator": IDEC,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "alpha": [0.25],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-4}],
                # "alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "random_state": [random_state]
            },
        },
        "n2d": {
            "estimator": N2D,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "batch_size": [256],
                "pretrain_epochs": [100],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "manifold_params": [{"n_components":2, "perplexity": 35, "random_state":42},],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4},],
                # "manifold_params": [{"perplexity": 15}, {"perplexity": 25}, {"perplexity": 35}, {"perplexity": 50}],
                "random_state": [random_state],
            },
        },
        "vade": {
            "estimator": VaDE,
            "param_grid": {
                "n_clusters": [2],

                "embedding_size": [10],
                # "neural_network": [(FeedforwardAutoencoder, {"layers": [79, 100, 100, 100, 50], "dropout": 0.5, "batch_norm": True})], # error no sense
                "pretrain_optimizer_params":    [{"lr": 1e-2}],
                "clustering_optimizer_params":  [{"lr": 1e-3}],
                "batch_size": [256],
                "pretrain_epochs": [10],
                "clustering_epochs": [150],
                "clustering_loss_weight": [1.0],
                "ssl_loss_weight": [1.0],

                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}],
                # "clustering_optimizer_params": [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "batch_size": [256, 32, 64, 128, 512],
                # "pretrain_epochs": [10, 50, 100],
                # "clustering_epochs": [150, 300],
                "random_state": [random_state],

                # "initial_clustering_class": [KMeans],
                # "initial_clustering_params": [{}],

                # "neural_network": [neural_network],
                # "embedding_size": [10, 20, 30, 40, 50, 60, 70],

                # "embedding_size": [10],


                # "embedding_size": [10],
                # "batch_size": [1024],
                # "clustering_loss_weight": [0.1],
                # "ssl_loss_weight": [1.0],
            },
        },


        # "autoclustering": {
        #     "estimator": AutoClustering,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "input_dim": [1]
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

