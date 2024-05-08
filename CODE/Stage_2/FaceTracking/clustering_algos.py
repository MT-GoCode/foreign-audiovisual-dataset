import hdbscan
import sklearn.metrics.pairwise
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.metrics.pairwise import cosine_similarity
import math
from numpy import dot
from numpy.linalg import norm
import numpy as np
from sklearn import mixture


def plot(data, labels, clustering_type = ''):
    # Plot the clustered data
    reducer = umap.UMAP(n_components=2)
    reduced_data = reducer.fit_transform(data)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(f'{clustering_type} Clustering with UMAP')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

def cosine_distance(X, Y):
  cos_sim = dot(X, Y)/(norm(X)*norm(Y))
  return cos_sim

def cluster_hdbscan(data, min_clusters=1, min_samples=1, plot_data=False):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_clusters, min_samples=min_samples, allow_single_cluster=True, cluster_selection_epsilon=.1)
    labels = clusterer.fit_predict(data)
    if plot_data:
        plot(data, labels, 'HDBSCAN')

    if (labels == -1).all():
        labels += 1
    return labels

def cluster_optics(data, min_samples, plot_data=False):
    optics_model = OPTICS(min_samples=min_samples)
    optics_model.fit_predict(data)
    labels = optics_model.labels_

    if plot_data:
        plot(data, labels, 'OPTICS')

    return labels

def cluster_dpgmm(data, n_components, plot_data=False):
    dpgmm_model = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
    dpgmm_model.fit(data)
    labels = dpgmm_model.predict(data)

    if plot_data:
        plot(data, labels, 'DPGMM')

    return labels

def cluster_gmm(data, n_components, plot_data=False):
    gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
    gmm_model.fit(data)
    labels = gmm_model.predict(data)

    if plot_data:
        plot(data, labels, 'GMM')

    return labels

