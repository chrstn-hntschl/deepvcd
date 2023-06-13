import numpy as np
import warnings
import sys

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence


def top_k_error(y_true, y_pred, k=5):
    """Top k error rate.
    For multiclass classification tasks, this metric returns the
    number of times where the correct class was not among the top `k` classes predicted
    (divided by the total number of samples).

    :math:`e= \frac{1}{n} \cdot \sum_k \min_i d(c_i,C_k) ` where
    :math:`d(c_i,C_k)=0` if :math:`c_i=C_k` and :math:`1` otherwise.

    Parameters
    ----------
    y_true : 1d array-like, or class indicator array / sparse matrix
        shape num_samples or [num_samples, num_classes]
        Ground truth (correct) classes.
    y_pred : array-like, shape [num_samples, num_classes]
        For each sample, each row represents the
        likelihood of each possible class.
        The number of columns must be as large as the set of possible
        classes.
    k : int, optional (default=5) predictions are counted as incorrect if
        the correct class is in not the top k classes predicted.
    Returns
    -------
    error : float
        Returns the proportion of samples where the correct class was not among the top k predicted classes.
        The best performance is 0.0.
    """

    if len(y_true.shape) == 2:
        if not y_true.shape == y_pred.shape:
            raise ValueError("Incompatible shapes for `y_true` and `y_pred` - `y_pred` must be either 1-column vector "
                             "with correct label per row or matrix of same shape as `y_pred` in one-hot form!")
        # multilabel currently not supported - check if one-hot-matrix contains exactly one true label per line:
        # -> min == max == 1
        if not np.all(np.sum(y_true, axis=1) == 1):
            raise ValueError("Multi-label currently not supported!")

        y_true = np.argmax(y_true, axis=1)

    elif len(y_true.shape) == 1:
        if not y_true.shape[0] == y_pred.shape[0]:
            raise ValueError("Number of samples in `y_true` and `y_pred` differs: {0} != {1}".format(y_true.shape[0],
                                                                                                     y_pred.shape[0]))
    else:
        raise ValueError("Invalid shape of `y_true`")

    if np.max(y_true) >= y_pred.shape[1]:
        raise ValueError("Max label in `y_true` ({0}) is larger than max index in `y_pred` ({1})".
                         format(np.max(y_true), y_pred.shape[1]))

    if k > y_pred.shape[1]:
        k = y_pred.shape[1]
    if k == y_pred.shape[1]:
        warnings.warn("You compute the top k={0} error rate - which will 0.0!".format(k))

    top_k_indices = y_pred.argsort()[:, ::-1][:, :k]
    correct_in_top_k = (top_k_indices.T == y_true).any(axis=0)
    error = 1.0-(float(np.sum(correct_in_top_k))/y_true.shape[0])
    return error


def mean_average_precision(y_true, y_pred):
    num_classes = y_true.shape[1]
    average_precisions = []
    relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
    tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
    for index in range(num_classes):
        temp = K.sum(tp_whole[:, :index + 1], axis=1)
        average_precisions.append(temp * (1 / (index + 1)))
    AP = K.add(average_precisions) / relevant
    mAP = K.mean(AP, axis=0)
    return mAP
