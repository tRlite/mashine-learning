import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    dists = sklearn.metrics.pairwise_distances(x, x)

    uniq_clast, idxs, clust_sizes = np.unique(labels, return_inverse=True, return_counts=True)

    if len(uniq_clast) == 1:
        return 0

    sil_score = 0
    for i in range(len(x)):
        mask = labels == labels[i]
        mask[i] = False
        cluster_dists = dists[i, mask]
        if len(cluster_dists):
            s_i = cluster_dists.mean()
        else:
            s_i = 0

        dists[i, mask] = np.inf

        d_i = (np.bincount(idxs, dists[i])/clust_sizes).min()
        if d_i or s_i:
            sil_score += (d_i - s_i)/max(s_i, d_i)
    return sil_score/len(x)


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    precision, recall = 0, 0
    for i in range(len(true_labels)):
        mask1 = true_labels == true_labels[i]
        mask2 = predicted_labels == predicted_labels[i]
        mask = mask1 & mask2
        precision += np.sum(mask)/np.sum(mask2)
        recall += np.sum(mask)/np.sum(mask1)
    precision /= len(true_labels)
    recall /= len(true_labels)
    score = 2*precision*recall/(precision + recall)
    return score
