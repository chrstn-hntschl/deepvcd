import json
import warnings
import math
import logging

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import metrics

log = logging.getLogger(__name__)


callback_mode_for_metric = {
    getattr(metrics, "Accuracy"): "max",
    getattr(metrics, "BinaryAccuracy"): "max",
    getattr(metrics, "CategoricalAccuracy"): "max",
    getattr(metrics, "TopKCategoricalAccuracy"): "max",
    getattr(metrics, "AUC"): "max",
    getattr(metrics, "Precision"): "max",
    getattr(metrics, "PrecisionAtRecall"): "max",
    getattr(metrics, "Recall"): "max",
    getattr(metrics, "RecallAtPrecision"): "max"
}


class StepLearningRate(Callback):
    def __init__(self, initial_lr, gamma=0.1, steps=100000):
        super(StepLearningRate, self).__init__()
        self.initial_lr = initial_lr
        self.gamma = gamma
        self.steps = steps
        self.seen = 0

    def on_train_begin(self, logs=None):
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        self.seen += 1

    def on_batch_begin(self, batch, logs=None):
        lr = self.initial_lr * math.pow(self.gamma, math.floor(self.seen / self.steps))
        K.set_value(self.model.optimizer.lr, lr)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.isscalar(obj):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


class GetBest(Callback):
    """Get the best model at the end of training.
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    # Example
        callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
        mode.fit(X, y, validation_data=(X_eval, Y_eval),
                    callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1, baseline=None):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.best = np.Inf if monitor == np.less else -np.Inf
        self.verbose = verbose
        self.period = period
        self.baseline = baseline
        self.best_epoch = 0
        self.epochs_since_last_save = 0
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

        if self.baseline is not None:
            self.best = self.baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % self.monitor, RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epoch = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epoch, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


class GetLRSchedule(Callback):
    def __init__(self):
        super(GetLRSchedule, self).__init__()
        self.lr_schedule = list()

    def on_train_begin(self, logs=None):
        self.lr_schedule = list()

    def on_epoch_begin(self, epoch, logs=None):
        self.lr_schedule.append(float(K.get_value(self.model.optimizer.lr)))
