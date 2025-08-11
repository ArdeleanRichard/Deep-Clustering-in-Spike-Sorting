import os
import random
import numpy as np
import torch
from clustpy.deep import ACeDeC, AEC, DCN, DDC, DEC, DeepECT, DipDECK, DipEncoder, DKM, ENRC, IDEC, VaDE, N2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from pydiffmap import diffusion_map as dm


CUDA_VISIBLE_DEVICES=""
random_state=42
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)


def load_algorithms():
    algorithms = {
        # # DEEP CLUSTERINGS:

        "dcn": {
            "estimator": DCN,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                # "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "clustering_optimizer_params": [{"lr": 1e-3}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-2}],
                "random_state": [random_state],
                "device": ['cpu'],
            },
        },
        "ddc": {
            "estimator": DDC,
            "param_grid": {
                "embedding_size": [10],
                "ratio": [0.1], #[0.01, 0.05, 0.1, 0.2],
                # "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "ratio": [0.01, 0.05, 0.1, 0.2, 0.15], #[0.01, 0.05, 0.1, 0.2],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "random_state": [random_state]
            },
        },
        "dec": {
            "estimator": DEC,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "alpha": [0.25],
                # "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "clustering_optimizer_params": [{"lr": 1e-4}],
                # "initial_clustering_params": [{"init": 'k-means++'}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-5}],
                # "alpha": [0.5, 0.75, 1.0, 1.5],
                "random_state": [random_state],
            },
        },
        "deepect": {
            "estimator": DeepECT,
            "param_grid": {
                "embedding_size": [10],
                "max_n_leaf_nodes": [20],
                # "pretrain_optimizer_params": [{"lr": 1e-2}],
                # "clustering_optimizer_params": [{"lr": 1e-4}],
                # "max_n_leaf_nodes": [1,2,3,4,5, 10, 20, 50, 100],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-4}],
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
                # "pretrain_optimizer_params": [{"lr": 1e-2}],
                # "clustering_optimizer_params": [{"lr": 1e-3}],
                # "dip_merge_threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-2}],
                "random_state": [random_state]
            },
        },
        "dipencoder": {
            "estimator": DipEncoder,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                # "pretrain_optimizer_params": [{"lr": 1e-2}],
                # "clustering_optimizer_params": [{"lr": 1e-4}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-5}],
                "random_state": [random_state],
                "device": ['cpu'],
            },
        },
        "dkm": {
            "estimator": DKM,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                # "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "clustering_optimizer_params": [{"lr": 1e-5}],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "clustering_optimizer_params": [{"lr": 1e-5}],
                "random_state": [random_state],
            },
        },
        "idec": {
            "estimator": IDEC,
            "param_grid": {
                "n_clusters": [2],
                "embedding_size": [10],
                "alpha": [0.25],
                # "pretrain_optimizer_params": [{"lr": 1e-3}],
                # "clustering_optimizer_params": [{"lr": 1e-4}],
                # "alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                # "clustering_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params": [{"lr": 1e-3}],
                "clustering_optimizer_params": [{"lr": 1e-5}],
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
                # "pretrain_optimizer_params": [{"lr": 1e-2}],
                "manifold_params": [{"n_components":2, "perplexity": 35, "random_state":42},],
                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4},],
                # "manifold_params": [{"perplexity": 15}, {"perplexity": 25}, {"perplexity": 35}, {"perplexity": 50}],
                "pretrain_optimizer_params": [{"lr": 1e-2}],
                "random_state": [random_state],
            },
        },
        "vade": {
            "estimator": VaDE,
            "param_grid": {
                "n_clusters": [2],

                "embedding_size": [10],
                # "neural_network": [(FeedforwardAutoencoder, {"layers": [79, 100, 100, 100, 50], "dropout": 0.5, "batch_norm": True})], # error no sense
                # "pretrain_optimizer_params":    [{"lr": 1e-2}],
                # "clustering_optimizer_params":  [{"lr": 1e-3}],
                "batch_size": [256],
                "pretrain_epochs": [10],
                "clustering_epochs": [150],
                "clustering_loss_weight": [1.0],
                "ssl_loss_weight": [1.0],

                # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}],
                # "clustering_optimizer_params": [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
                "pretrain_optimizer_params":    [{"lr": 1e-4}],
                "clustering_optimizer_params":  [{"lr": 1e-4}],
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

        "acedec": {
            "estimator": ACeDeC,
            "param_grid": {
                "n_clusters": [2],
                "init": ["acedec"],  # , 'subkmeans', 'random', 'sgd'],
                "embedding_size": [10],  # , 20, 30, 40, 50, 60, 70],
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

class DiffusionMapWrapper:
    def __init__(self, n_evecs=2, alpha=0.5, **kwargs):
        self.model = dm.DiffusionMap.from_sklearn(n_evecs=n_evecs, alpha=alpha, **kwargs)

    def fit_transform(self, X):
        return self.model.fit_transform(X)

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

        "lle": {
            "estimator": LocallyLinearEmbedding,
            "param_grid": {
                "n_components": 2,
                "n_neighbors": 70,
                "method": "standard"
            },
        },

        "tsne": {
            "estimator": TSNE,
            "param_grid": {
                "n_components": 2,
                "perplexity": 30,
                "max_iter": 1000
            },
        },

        "diffusion_map": {
            "estimator": DiffusionMapWrapper,
            "param_grid": {
                "n_evecs": 2,
                "alpha": 0.5,
                "k": 50,
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

