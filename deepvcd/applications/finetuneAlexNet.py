# -*- coding: utf-8 -*-
import logging
import pathlib
import os
import functools

import tensorflow
from tensorflow.keras.utils.data_utils import validate_file, get_file
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import losses

from deepvcd.models.alexnet import AlexNetFlat, preprocess_train, preprocess_predict, WEIGHTS_FLAT_URL, WEIGHTS_FLAT_SHA256
from deepvcd.models.utils import get_layer_index
from deepvcd.dataset.descriptor import YAMLLoader, DirectoryLoader
from deepvcd.helpers.image import read_image, one_hot

log = logging.getLogger(__name__)


BATCH_SIZE = 256  # Krizhevsky2012: 128
IMG_SIZE = 227
AUTOTUNE = tensorflow.data.AUTOTUNE

def create_bottleneck_features(num_classes:int, num_units:int=128, layer_prefix:str="rt_"):
    top_model = Sequential()
    top_model.add(Dense(num_units, activation='relu', name=layer_prefix + 'ext_dense',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                        # Krizhevsky2012: 0.01, Caffe?: 0.005
                        bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe: 0.1
                        kernel_regularizer=regularizers.l2(0.0005),
                        bias_regularizer=regularizers.l2(0.0005)))
    top_model.add(Dense(num_classes, activation='softmax', name=layer_prefix + 'predictions',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                        # Krizhevsky2012: 0.01, Caffe?: 0.1
                        bias_initializer=Constant(value=0.0),
                        kernel_regularizer=regularizers.l2(0.0005),
                        bias_regularizer=regularizers.l2(0.0005)))
    return top_model

def setup_model(top_model, chop_off_layer:str='dense_3', weights:str='imagenet', name:str='AlexNetFlat', norm:str="lrn"):
    if not 'imagenet' == weights and not os.path.isfile(weights):
        raise ValueError('The `weights` argument should be either '
                         '`imagenet` (pre-training on ImageNet) or pointing to a local hdf5 weights file.')

    if 'imagenet' == weights:
        # Initialize layer weights from pre-trained ImageNet model
        weights_path = get_file('alexnetflat_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_FLAT_URL,
                                cache_subdir='models',
                                file_hash=WEIGHTS_FLAT_SHA256,
                                hash_algorithm='sha256'
                                )
    else:
        weights_path = weights

    # create the randomly initialized base model
    basemodel = AlexNetFlat(include_top=True, input_shape=(227, 227, 3), norm=norm, weights=None, classes=1000)

    # get index of layer to be cut off the original model
    chop_off_index = get_layer_index(model=basemodel, layer_name=chop_off_layer)

    # the new output is the output of the layer before chop_off_layer:
    x = basemodel.layers[chop_off_index-1].output

    # add the new layers from top_model:
    for layer in top_model.layers:
        x = layer(x)

    model = Model(inputs=basemodel.input, outputs=x, name=name)

    # load model weights `by_name=True` - skip all layers that have a different name (i.e. which shall be retrained)
    model.load_weights(filepath=weights_path, by_name=True, skip_mismatch=True,
                       # reshape=False
                       )

    return model


def trainAlexNet(dataset_descriptor:str, seed=None):
    if seed is None:
        from datetime import datetime
        seed = datetime.now().microsecond

    deepvcd_ds = None
    if pathlib.Path(dataset_descriptor).is_file():
        log.info("Loading dataset from descriptor file '{filename}'".format(filename=dataset_descriptor))
        deepvcd_ds = YAMLLoader.read(yaml_file=dataset_descriptor)
    if pathlib.Path(dataset_descriptor).is_dir():
        log.info("Loading dataset from directory '{directory}'".format(directory=dataset_descriptor))
        deepvcd_ds = DirectoryLoader.load(dataset_dir=dataset_descriptor)
    num_classes = len(deepvcd_ds.get_labels())
    num_train_samples = len(deepvcd_ds.get_train_images())

    train_ds,_ = deepvcd_ds.get_tfdataset(subset="train", shuffle_files=True, seed=seed)
    train_ds = train_ds.shuffle(num_train_samples, seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(lambda x, y: (read_image(x), y), num_parallel_calls=tensorflow.data.AUTOTUNE)

    val_ds,_ = deepvcd_ds.get_tfdataset(subset="val", shuffle_files=False)
    val_ds = val_ds.map(lambda x, y: (read_image(x), y), num_parallel_calls=tensorflow.data.AUTOTUNE)

    mean = [123.0767, 115.56045, 101.68434]
    log.debug("Computed mean: {mean}".format(mean=mean))
    train_ds = train_ds.map(lambda x, y: (preprocess_train(x, mean=mean), y), num_parallel_calls=tensorflow.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_predict(x, mean=mean, crop_region="center", horizontal_flip=False), y), num_parallel_calls=tensorflow.data.AUTOTUNE)

    # convert class labels into one-hot encodings
    train_ds = train_ds.map(lambda x,y: one_hot(x,y,num_classes))
    val_ds = val_ds.map(lambda x,y: one_hot(x,y,num_classes))

    # Batching the dataset
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    train_ds = train_ds.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    # chop-off layer and number of units in retrain layer <have been optimized in test_ft_strategies - should also be part of
    # the model selection process
    chop_off_layer = 'dense_3'
    num_bottleneck_units = 128
    retrain_layer_prefix="rt_"
    top_model = create_bottleneck_features(num_classes=num_classes, num_units=num_bottleneck_units, layer_prefix=retrain_layer_prefix)
    model = setup_model(top_model=top_model, chop_off_layer=chop_off_layer,  weights='imagenet', name='AlexNetFlat-ft')

    # 1.) retrain only newly added layers from scratch:
    # set all layers except new layers to not trainable:
    for layer in model.layers:
        if layer.name.startswith(retrain_layer_prefix):
            layer.trainable = True
        else:
            layer.trainable = False

    decay = 0.0
    momentum = 0.9
    optimizer = SGD(learning_rate=initial_lr_rt, momentum=momentum, decay=decay, nesterov=False)
    loss = losses.categorical_crossentropy
    metrics = ["CategoricalAccuracy"]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=None,
                  sample_weight_mode=None,
                  weighted_metrics=None,
                  target_tensors=None
                  )
    
    # retraining layers:
    def lr_scheduler(epoch, cur_lr, schedule):
        return schedule[epoch]
    callbacks = [
        LearningRateScheduler(schedule=functools.partial(lr_scheduler, schedule=lr_schedule_rt), verbose=1)
    ]

    log.debug("Retraining all layers with layer name equals '{rt_prefix}*'".format(rt_prefix=retrain_layer_prefix))
    history_rt = model.fit(x=train_ds,
                           epochs=max_epochs_rt,
                           verbose=1,
                           callbacks=callbacks,
                           validation_data=val_ds,
                           class_weight=None,
                           )
    scores, gt = predict(model=model, testset_iter=val_ds, preprocessing_func='default')

    # 2.) fine-tune all/other layers with small learning rate:
    log.debug("Fine-tuning all layers with small learning rate (lr={lr})".format(lr=lr))
    if start_ft_layer is None:
        set_trainable = True
    else:
        set_trainable = False
    for layer in model.layers:
        if layer.name == start_ft_layer:
            set_trainable = True
        if set_trainable:
            log.debug("setting layer '{lname}' to trainable.".format(lname=layer.name))
            layer.trainable = True
        else:
            layer.trainable = False

    decay = 0.0
    momentum = 0.9
    optimizer = SGD(learning_rate=lr_ft, momentum=momentum, decay=decay, nesterov=False)
    loss = losses.categorical_crossentropy
    metrics = ["CategoricalAccuracy"]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=None,
                  sample_weight_mode=None,
                  weighted_metrics=None,
                  target_tensors=None
                  )

    history_ft = model.fit(x=train_ds,
                           epochs=max_epochs_ft,
                           verbose=1,
                           callbacks=[],
                           validation_data=val_ds,
                           class_weight=None
                           )
    
    scores, gt = predict(model=model, testset_iter=val_ds, preprocessing_func='default')

    if callable(metrics):
        metrics_ft = metrics(gt, scores)
    else:
        metrics_ft = dict()
        log.debug("Results for fine-tuning {model} on {dataset}:".format(model=model.name,
                                                                         dataset=os.path.basename(descr_fname)))
        for m in metrics:
            metrics_ft[m] = metrics[m](gt, scores)
            log.debug("\t{metric}={val}".format(metric=m, val=metrics_ft[m]))

    return metrics_rt, metrics_ft



