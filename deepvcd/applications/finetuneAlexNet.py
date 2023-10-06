# -*- coding: utf-8 -*-
import logging
import pathlib
import os
import functools

import tensorflow
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import losses
from tensorflow.keras.metrics import AUC

from keras.utils.data_utils import validate_file, get_file

from deepvcd.models.alexnet.AlexNet  import AlexNetFlat, preprocess_train, preprocess_predict, WEIGHTS_FLAT_URL, WEIGHTS_FLAT_SHA256
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

def train(model, train_dataset, validation_dataset, inital_lr:float, callbacks:list, max_epochs:int):
    decay = 0.0
    momentum = 0.9
    optimizer = SGD(learning_rate=initial_lr, momentum=momentum, weight_decay=decay, nesterov=False)
    loss = losses.categorical_crossentropy
    metrics = ["CategoricalAccuracy", AUC(curve="PR", multi_label=True, name="mAP")]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=None,
                  sample_weight_mode=None,
                  weighted_metrics=None,
                  target_tensors=None
                  )
    history = model.fit(x=train_dataset,
                           epochs=max_epochss,
                           verbose=1,
                           callbacks=callbacks,
                           validation_data=validation_dataset,
                           class_weight=None,
                           )
    return model, history

def retrain_and_finetune(dataset_descriptor:str, norm:str="lrn", model_weights:str="imagenet", seed=None):
    initial_lr_rt=0.01
    max_epochs_rt=90
    lr_ft=0.001
    max_epochs_ft=90
    start_ft_layer="conv_1"
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
    model = setup_model(top_model=top_model, chop_off_layer=chop_off_layer,  weights=model_weights, name='AlexNetFlat-ft', norm=norm)

    # 1.) retrain only newly added layers from scratch:
    # set all layers except new layers to not trainable:
    for layer in model.layers:
        if layer.name.startswith(retrain_layer_prefix):
            layer.trainable = True
        else:
            layer.trainable = False
    
    # retraining layers:
    callbacks =list()
    #def lr_scheduler(epoch, cur_lr, schedule):
    #    return schedule[epoch]
    #lr_schedule_cb = LearningRateScheduler(schedule=functools.partial(lr_scheduler, schedule=lr_schedule_rt), verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_categorical_accuracy",
                                     mode="max",
                                     factor=0.1,  # divide by 10
                                     patience=5,
                                     cooldown=5,
                                     min_delta=0.0,
                                     min_lr=0.00001,  # we reduce lr at max three times
                                     verbose=1)
    callbacks.append(reduce_lr_cb)

    # stop training if no more improvement
    early_stop_cb = EarlyStopping(monitor="val_categorical_accuracy",
                                  mode="max",
                                  min_delta=0.0,
                                  patience=11,
                                  verbose=1,
                                  baseline=None,
                                  restore_best_weights=True  # after training, use model weights of
                                                              # best performing epoch
                                  )
    callbacks.append(early_stop_cb)

    log.debug("Retraining all layers with layer name equals '{rt_prefix}*'".format(rt_prefix=retrain_layer_prefix))
    model, history_rt = train(model=model,
                                train_dataset=train_ds,
                                validation_dataset=val_ds,
                                initial_lr=initial_lr_rt,
                                callbacks=callbacks,
                                max_epochs=max_epochs_rt)

    val_loss_rt, val_metric_rt, val_mAP_rt = model.evaluate(val_ds)
    log.info(f"Validation set results after retraining: val_categorical_accuracy={val_metric_rt:.4f} (val_loss={val_loss_rt:.4f})")
    #scores, gt = predict(model=model, testset_iter=val_ds, preprocessing_func='default')

    # 2.) fine-tune all/other layers with small learning rate:
    log.debug(f"Fine-tuning all layers with small learning rate (lr={lr_ft}")
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

    callbacks =list()
    #def lr_scheduler(epoch, cur_lr, schedule):
    #    return schedule[epoch]
    #lr_schedule_cb = LearningRateScheduler(schedule=functools.partial(lr_scheduler, schedule=lr_schedule_rt), verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_categorical_accuracy",
                                     mode="max",
                                     factor=0.1,  # divide by 10
                                     patience=5,
                                     cooldown=5,
                                     min_delta=0.0,
                                     min_lr=0.00001,  # we reduce lr at max three times
                                     verbose=1)
    callbacks.append(reduce_lr_cb)

    # stop training if no more improvement
    early_stop_cb = EarlyStopping(monitor="val_categorical_accuracy",
                                  mode="max",
                                  min_delta=0.0,
                                  patience=11,
                                  verbose=1,
                                  baseline=None,
                                  restore_best_weights=True  # after training, use model weights of
                                                              # best performing epoch
                                  )
    callbacks.append(early_stop_cb)

    model, history_ft = train(model=model,
                              train_dataset=train_ds,
                              validation_dataset=val_ds,
                              initial_lr=lr_ft,
                              callbacks=callbacks,
                              max_epochs=max_epochs_ft)
    val_loss_ft, val_metric_ft, val_mAP_ft = model.evaluate(val_ds)
    log.info(f"Validation set results after fine-tuning: val_categorical_accuracy={val_metric_ft:.4f} (val_loss={val_loss_ft:.4f})")

    log.info(f"val_categorical_accuracy={val_mAP_ft:.4f} ({val_mAP_rt:.4f})")
    #scores, gt = predict(model=model, testset_iter=val_ds, preprocessing_func='default')

    #if callable(metrics):
    #    metrics_ft = metrics(gt, scores)
    #else:
    #    metrics_ft = dict()
    #    log.debug("Results for fine-tuning {model} on {dataset}:".format(model=model.name,
    #                                                                     dataset=os.path.basename(descr_fname)))
    #    for m in metrics:
    #        metrics_ft[m] = metrics[m](gt, scores)
    #        log.debug("\t{metric}={val}".format(metric=m, val=metrics_ft[m]))

    #return metrics_rt, metrics_ft


if __name__ == "__main__":
    import argparse

    # configure logging
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')                                                                                     # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    log.info("Tensorflow version: {ver}".format(ver=tensorflow.__version__))
    log.info("Keras version: {ver}".format(ver=tensorflow.keras.__version__))

    #FIXME: set parameters: initial_lr_rt, lr_ft, max_epochs_rt, max_epochs_ft
    #FIXME: model hyperparameters (learning rate schedule, finetuning layer, chop-off layer) are hard-coded or optimized using test data!

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--norm",
                        dest="norm",
                        type=str,
                        help="Normalization to be used after first and second convolutional layer ('lrn' (default), 'tflrn', 'batch' or None)",
                        default="lrn",
                        required=False)
    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        help="Path to dataset descriptor yaml or directory following a dataset structure (see documentation).",
                        default=None,
                        required=True)
    parser.add_argument("-w", "--weights",
                        dest="weights",
                        type=str,
                        help="Path to hdf5 file with pretrained model weights. If none given, ImageNet weights are loaded.",
                        default="imagenet",
                        required=False)
    args = parser.parse_args()
    retrain_and_finetune(dataset_descriptor=args.dataset, 
                 norm=None if args.norm.lower()=="none" else args.norm,
                 model_weights=args.weights, 
                 seed=None)
