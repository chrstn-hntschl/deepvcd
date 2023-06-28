# -*- coding: utf-8 -*-
import logging
import os
import pathlib
import json
import ast

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import losses
from tensorflow.keras import metrics

from deepvcd.helpers.image import read_image, one_hot
from deepvcd.callbacks import NumpyEncoder
import deepvcd.models.alexnet
from deepvcd.models.alexnet.AlexNet import preprocess_train, preprocess_predict
from deepvcd.dataset.descriptor import YAMLLoader, DirectoryLoader


log = logging.getLogger(__name__)

BATCH_SIZE = 256  # Krizhevsky2012: 128
IMG_SIZE = 227
AUTOTUNE = tf.data.AUTOTUNE


def train_alexnet(train_ds,
                  num_classes,
                  norm="lrn",
                  metric="CategoricalAccuracy", 
                  input_size=227,
                  val_ds=None,
                  max_epochs=90,
                  callbacks=None,
                  ):

    model_class = getattr(deepvcd.models.alexnet, "AlexNetFlat")
    model = model_class(include_top=True, input_shape=(input_size, input_size, 3), norm=norm, weights=None, classes=num_classes)
    model.summary()

    # Krizhevsky2012: "The learning rate was initialized at 0.01 and reduced three times prior to termination"
    initial_lr = 0.01
    # Krizhevsky2012 use decay=0.0005, this is implemented as kernel_regularizer and bias_regularizer in AlexNet layer (see alexnet.AlexNet.py)
    decay = None
    momentum = 0.9  # following Krizhevsky2012

    optimizer = SGD(learning_rate=initial_lr, momentum=momentum, weight_decay=decay, nesterov=False)
    loss = losses.categorical_crossentropy
    metrics = [metric]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None)

    class_weight = None

    if callbacks:
        if isinstance(callbacks, tf.keras.callbacks.Callback):
            callbacks = list(callbacks)

    model_history = model.fit(x=train_ds,
                              epochs=max_epochs,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=val_ds,
                              class_weight=class_weight,
                              )

    return model, model_history


def _main(dataset_descriptor: str,
          norm: str="lrn",
          input_size: int=227,
          max_epochs: int=90,
          metric: str="CategoricalAccuracy",
          metric_args: dict={},
          checkpoints_dest: str=None,
          seed=None,
          model_weights_fname: str = None
          ):
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
    train_ds = train_ds.map(lambda x, y: (read_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    val_ds,_ = deepvcd_ds.get_tfdataset(subset="val", shuffle_files=False)
    val_ds = val_ds.map(lambda x, y: (read_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # compute per channel mean over random samples
    #log.info("Computing dataset stats")
    #rand_samples_ds = train_ds.map(lambda x, y: preprocess_train(x, mean=None), num_parallel_calls=tf.data.AUTOTUNE)
    #normalizer = MeanNormalization(axis=-1)
    #normalizer.adapt(rand_samples_ds)
    #mean = normalizer.weights[0].numpy()
    
    #mean = [0.48184115, 0.453552, 0.3977624]
    mean = [123.0767, 115.56045, 101.68434]
    log.debug("Computed mean: {mean}".format(mean=mean))
    #mean = get_mean()
    #log.debug("ILSVRC2012.get_mean: {mean}".format(mean=mean))
    # preprocess both subsets
    train_ds = train_ds.map(lambda x, y: (preprocess_train(x, mean=mean), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_predict(x, mean=mean, crop_region="center", horizontal_flip=False), y), num_parallel_calls=tf.data.AUTOTUNE)

    # convert class labels into one-hot encodings
    train_ds = train_ds.map(lambda x,y: one_hot(x,y,num_classes))
    val_ds = val_ds.map(lambda x,y: one_hot(x,y,num_classes))

    # Batching the dataset
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    callbacks = []

    #monitor_name = metric if not val_ds else f"val_{metric}"
    metric_cls = getattr(metrics, metric)
    metric_obj = metric_cls(**metric_args)
    # Krizhevsky2012: "divide the learning rate by 10 when the validation error rate stopped improving"
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_"+metric_obj.name,
                                     mode='auto',
                                     factor=0.1,  # divide by 10
                                     patience=5,
                                     min_delta=0.001,
                                     cooldown=10,
                                     min_lr=0.00001,  # we reduce lr at max three times
                                     verbose=1)
    callbacks.append(reduce_lr_cb)

    # stop training if no more improvement
    early_stop_cb = EarlyStopping(monitor="val_"+metric_obj.name,
                                  mode='auto',
                                  min_delta=0.0,
                                  patience=16,
                                  verbose=1,
                                  baseline=None,
                                  restore_best_weights=True  # after training, use model weights of
                                                              # best performing epoch
                                  )
    callbacks.append(early_stop_cb)

    if checkpoints_dest is not None:
        model_checkpoints_cb = ModelCheckpoint(filepath=os.path.join(checkpoints_dest, "AlexNetFlat.epoch-{{epoch:02d}}_{label}-{{{monitor}:.2f}}.hdf5".format(label=metric_obj.name.replace('_',''), monitor="val_"+metric_obj.name)),
                                               monitor="val_"+metric_obj.name,
                                               verbose=0,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='auto',
                                               save_freq='epoch',
                                               options=None,
                                               initial_value_threshold=None
                                               )
        callbacks.append(model_checkpoints_cb)

    alexnet, history = train_alexnet(train_ds=train_ds,
                                     num_classes=num_classes,
                                     norm=norm,
                                     metric=metric_obj,
                                     input_size=input_size,
                                     val_ds=val_ds,
                                     max_epochs=max_epochs,
                                     callbacks=callbacks,
                                     )
    val_loss, val_metric = alexnet.evaluate(val_ds)
    log.info(f"Final validataion set results: {metric}={val_metric:.4f} (val_loss={val_loss:.4f})")

    if model_weights_fname:
        dest_dir = pathlib.Path(model_weights_fname).parent.absolute()
        if not dest_dir.exists():
            os.makedirs(dest_dir)

        alexnet.save_weights(filepath=model_weights_fname)
        with open(model_weights_fname + '.history', 'w') as file_json:
            json.dump(history.history, file_json, cls=NumpyEncoder)


if __name__ == '__main__':
    import argparse
    import keras
    import tensorflow as tf

    # configure logging
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    log.info("Tensorflow version: {ver}".format(ver=tf.__version__))
    log.info("Keras version: {ver}".format(ver=keras.__version__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--norm",
                        dest="norm",
                        type=str,
                        help="Normalization to be used after first and second convolutional layer ('lrn' (default), 'tflrn', 'batch' or None)",
                        default="lrn",
                        required=False)
    parser.add_argument("-i", "--input_size",
                        dest="input_size",
                        type=int,
                        help="Input image size. 224 and 227 (default) are supported. 224 will result in padding.",
                        default=227,
                        required=False)
    parser.add_argument("-s", "--seed", dest="seed", 
                        type=int, 
                        help="Set the random seed.", 
                        default=None,
                        required=False)
    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        help="Train for x epochs",
                        default=200,
                        required=False)
    parser.add_argument("-m", "--metric",
                        dest="metric",
                        type=str,
                        help="Metric to be used for monitoring validation data improvements during training (default: CategoricalAccuracy).",
                        default="CategoricalAccuracy",
                        required=False)
    parser.add_argument("--metric_args",
                        dest="metric_args",
                        help="",
                        type=str,
                        nargs='*',
                        default=None)
    parser.add_argument("-c", "--checkpoints_dest",
                        dest="checkpoints",
                        type=str,
                        help="Path to checkpoints directory. If set, incremental model improvements are stored during training.",
                        default=None,
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
                        help="Path to hdf5 file for storing optimized model weights after training.",
                        default=None,
                        required=True)
    args = parser.parse_args()

    # parse metric_args string to dict:
    metric_args = {}
    if args.metric_args is not None:
        metric_args = dict([(k,v) for k,v in (pair.split('=') for pair in args.metric_args)])
        #metric_args = dict((k, ast.literal_eval(v)) for k, v in (pair.split('=') for pair in args.metric_args))

    _main(dataset_descriptor=args.dataset,
          norm=None if args.norm.lower()=="none" else args.norm,
          input_size=args.input_size,
          max_epochs=args.epochs,
          metric=args.metric,
          metric_args=metric_args,
          checkpoints_dest=args.checkpoints,
          seed=args.seed,
          model_weights_fname=args.weights)
