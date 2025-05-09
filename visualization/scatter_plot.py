import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA

from visualization.label_map import LABEL_COLOR_MAP


def plot(title, X, labels=None, plot=True, marker='o', alpha=1):
    """
    Plots the dataset with or without labels
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    if plot:
        nrDim = len(X[0])
        fig = plt.figure() #figsize=(16, 12), dpi=400
        plt.title(title)

        if labels is not None:
            try:
                label_color = [LABEL_COLOR_MAP[l] for l in labels]
            except KeyError:
                print('Too many labels! Using default colors...\n')
                label_color = [l for l in labels]
        else:
            label_color = 'gray'


        if nrDim == 2:
            plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k', alpha=alpha)
        if nrDim == 3:
            # ax = fig.add_subplot(projection='3d')
            # if labels is None:
            #     ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, edgecolors='k')
            # else:
            #     try:
            #         label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
            #     except KeyError:
            #         print('Too many labels! Using default colors...\n')
            #         label_color = [l for l in labels]
            #     ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label_color, marker=marker, edgecolors='k', alpha=alpha)
            plt.axis('off')

            ax = Axes3D(fig)
            # ax.set_axis_off()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, edgecolors='k', s=25)



def plot2D(title, X, labels=None, plot=True, marker='o', alpha=1):
    """
    Plots the dataset with or without labels
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    pca_2d = PCA(n_components=2)
    X2D = pca_2d.fit_transform(X)
    if plot:
        fig = plt.figure()  # figsize=(16, 12), dpi=400
        plt.title(title)

        if labels is not None:
            try:
                label_color = [LABEL_COLOR_MAP[l] for l in labels]
            except KeyError:
                print('Too many labels! Using default colors...\n')
                label_color = [l for l in labels]
        else:
            label_color = 'gray'


        plt.scatter(X2D[:, 0], X2D[:, 1], c=label_color, marker=marker, edgecolors='k', alpha=alpha)



def plot_grid(title, X, pn, labels=None, plot=True, marker='o'):
    """
    Plots the dataset with grid
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)
    if plot:
        nrDim = len(X[0])
        label_color = [LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        plt.title(title)
        if nrDim == 2:
            ax = fig.gca()

            ax.set_xticks(np.arange(0, pn, 1))
            ax.set_yticks(np.arange(0, pn, 1))

            plt.scatter(X[:, 0], X[:, 1], marker=marker, c=label_color, s=25, edgecolor='k')
            plt.grid(True)
        if nrDim == 3:
            ax = Axes3D(fig)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.set_xticks(np.arange(0, pn, 1))
            # ax.set_zticks(np.arange(0, pn, 1))
            # ax.set_yticks(np.arange(0, pn, 1))
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, s=25)
            # plt.grid(True)
        # fig.savefig("cevajeg.svg", format='svg', dpi=1200)
