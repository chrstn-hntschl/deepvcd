import os
import warnings
import logging
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D, ZeroPadding2D, BatchNormalization, Lambda, Activation
from tensorflow.python.keras.utils import layer_utils
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras import regularizers
import tensorflow as tf

from keras.applications import imagenet_utils

from deepvcd.metrics import top_k_error
from deepvcd.helpers.image import read_image, one_hot
from deepvcd.models.layers import LRN2D

log = logging.getLogger(__name__)

WEIGHTS_FLAT_URL = ""
WEIGHTS_FLAT_SHA256 = ""
WEIGHTS_URL = None
WEIGHTS_SHA256 = None


def preprocess_train(image, mean):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    # resize image to have width/height (whichever is smaller) to be equal to 256 while keeping the image aspect ratio
    ratio = tf.math.divide(tf.constant(256.), tf.math.minimum(height, width))
    ratio = tf.cast(ratio, tf.float32)
    image = tf.image.resize(image, tf.cast([tf.math.maximum(tf.constant(256.), height*ratio), tf.math.maximum(tf.constant(256.), width*ratio)], tf.int32), method="bicubic")

    # crop central 256x256 region:
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    offset_height =  tf.math.maximum(tf.constant(0.), tf.math.subtract(height, tf.constant(256.)))
    offset_height = tf.cast(tf.math.ceil(tf.math.divide(offset_height, tf.constant(2.))), tf.int32)
    offset_width = tf.math.maximum(tf.constant(0.), tf.math.subtract(width, tf.constant(256.)))
    offset_width = tf.cast(tf.math.ceil(tf.math.divide(offset_width, tf.constant(2.))), tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, 256, 256)

    # random crop to target size
    image = tf.image.random_crop(image, size=(227,227,3))

    # mean-normalize
    if mean is not None:
        image = tf.math.subtract(image, mean)

    # random horizontal flipping
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_predict(image, mean, crop_region="center", horizontal_flip=False):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    # resize image to have width/height (whichever is smaller) to be equal to 256 while keeping the image aspect ratio
    ratio = tf.math.divide(tf.constant(256.), tf.math.minimum(height, width))
    image = tf.image.resize(image, tf.cast([tf.math.maximum(tf.constant(256.), height*ratio), tf.math.maximum(tf.constant(256.), width*ratio)], tf.int32), method="bicubic")

    # crop central 256x256 region:
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    offset_height =  tf.math.maximum(tf.constant(0.), tf.math.subtract(height, tf.constant(256.)))
    offset_height = tf.cast(tf.math.ceil(tf.math.divide(offset_height, tf.constant(2.))), tf.int32)
    offset_width = tf.math.maximum(tf.constant(0.), tf.math.subtract(width, tf.constant(256.)))
    offset_width = tf.cast(tf.math.ceil(tf.math.divide(offset_width, tf.constant(2.))), tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, 256, 256)

    crop_coords = {}
    # tuples (offset_height, offset_width):
    crop_coords.update(dict.fromkeys(["center", 'c'], (int((256.-227.)/2.), int((256.-227.)/2.)) ))
    crop_coords.update(dict.fromkeys(["top_left", "tl"], (0, 0) ))
    crop_coords.update(dict.fromkeys(["top_right", "tr"], (0, 256-227) ))
    crop_coords.update(dict.fromkeys(["bottom_left", "bl"], (256-227, 0) ))
    crop_coords.update(dict.fromkeys(["bottom_right", "br"], (256-227, 256-227) ))

    # crop to region 227x227
    image = tf.image.crop_to_bounding_box(image, crop_coords[crop_region][0], crop_coords[crop_region][1], 227, 227)

    # mean normalize image
    image = tf.math.subtract(image, mean)

    if horizontal_flip:
        image = tf.image.flip_left_right(image)
    
    return image


def AlexNetFlat(include_top=True,
                input_shape=None, 
                norm="lrn",
                weights='imagenet',
                classes=1000):
    """
    Instantiates a flat version of the AlexNet architecture - basically this reduces the splitting of Conv2D layers
    which was introduced by Krizhevsky et al. in order to overcome limitations of GPU memory. This however, increases
    the amount of filters in these respective layers and thus the total amount of model weights/trainable parameters.

    Optionally loads weights pre-trained on ImageNet.

    :param include_top: whether or not to include final dense layers in returned model
    :param input_shape: tbd
    :param norm: the normalization to be applied (None, 'batch', 'lrn', 'tflrn')
    :param weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet) or a local path
                    pointing to a compatible hdf5 weights file.
    :param classes: optional number of classes to classify images into, only to be specified if no `weights` argument
                    is specified.
    :return: model instance.
    :raises: ValueError: in case of invalid argument for `weights`.
    """

    data_format = K.image_data_format()

    if weights not in {'imagenet', None} and not os.path.isfile(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet) or pointing to a local hdf5 weights file.')

    if input_shape is not None:
        if data_format == "channels_first":
            if input_shape[1] not in {224, 227} or input_shape[2] not in {224, 227}:
                raise ValueError("AlexNetFlat model currently only supports image sizes of 224x224 or 227x227")
        else:
            if input_shape[0] not in {224, 227} or input_shape[1] not in {224, 227}:
                raise ValueError("AlexNetFlat model currently only supports image sizes of 224x224 or 227x227")

    if norm not in {"tflrn", "lrn", "batch", None}:
        raise ValueError("The normalization layer to be applied after first and second convolutional layer must be one of LocalResponseNormalization ('lrn', default), Tensorflow variant of LocalResponseNormalization ('tflrn'),  BatchNormalization ('batch') or no normalization (None).")

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(input_shape=None,
                                                    default_size=227,
                                                    min_size=224,
                                                    data_format=data_format,
                                                    require_flatten=True,
                                                    weights=weights)
    img_input = Input(shape=input_shape)

    if input_shape[1] == 224:
        padding = "same"
    else:
        padding = "valid"

    x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), name='conv_1',
               padding=padding,
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012
               bias_initializer=Constant(value=0),  # Krizhevsky2012
               kernel_regularizer=regularizers.l2(0.0005),
               bias_regularizer=regularizers.l2(0.0005)
              )(img_input)

    # Krizhevsky2012:
    # "The constants k, n, alpha, and beta are hyper-parameters whose values are determined using a validation set;
    # we used k = 2, n = 5, alpha = 10-4, and beta = 0.75"
    # Response-normalization layers follow the first and second convolutional layers.
    if norm is not None:
        if norm == "lrn":
            x = LRN2D(alpha=0.0001, k=2, beta=0.75, n=5, name="lr_norm_1")(x)
        elif norm == "tflrn":
            #x = Lambda(tf.nn.local_response_normalization, name="tflr_norm_1")(x, depth_radius=5, bias=2, alpha=0.0001, beta=0.75,)
            x = Lambda(tf.nn.local_response_normalization, name="tflr_norm_1")(x)
        elif norm== "batch":
            bn_axis = 3 if data_format == "channels_last" else 1
            x = BatchNormalization(axis=bn_axis, name="batch_norm_1")(x)
    x = Activation('relu', name="relu_1")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool_1")(x)
    # Krizhevsky2012:
    # "Max-pooling layers follow both response-normalization layers as well as the fifth convolutional layer."

    x = ZeroPadding2D(padding=(2, 2), name="pad_1")(x)
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), name='conv_2',
               #padding="same",
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012
               bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe: 0.1
               kernel_regularizer=regularizers.l2(0.0005),
               bias_regularizer=regularizers.l2(0.0005)
              )(x)
    if norm is not None:
        if norm == "lrn":
            x = LRN2D(alpha=0.0001, k=2, beta=0.75, n=5, name="lr_norm_2")(x)
        elif norm == "tflrn":
            #x = Lambda(tf.nn.local_response_normalization, name="tflr_norm_2")(x, depth_radius=5, bias=2, alpha=0.0001, beta=0.75,)
            x = Lambda(tf.nn.local_response_normalization, name="tflr_norm_2")(x)
        elif norm == "batch":
            bn_axis = 3 if data_format == "channels_last" else 1
            x = BatchNormalization(axis=bn_axis, name="batch_norm_2")(x)
    x = Activation("relu", name="relu_2")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool_2")(x)  # Krizhevsky2012

    x = ZeroPadding2D(padding=(1, 1), name="pad_2")(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), name="conv_3",
               #padding="same",
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012
               bias_initializer=Constant(value=0),  # Krizhevsky2012
               kernel_regularizer=regularizers.l2(0.0005),
               bias_regularizer=regularizers.l2(0.0005)
              )(x)
    if norm == "batch":
        bn_axis = 3 if data_format == "channels_last" else 1
        x = BatchNormalization(axis=bn_axis, name="batch_norm_3")(x)
    x = Activation("relu", name="relu_3")(x)

    x = ZeroPadding2D(padding=(1, 1), name="pad_3")(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), name="conv_4",
               #padding="same",
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012
               bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe. 0.1
               kernel_regularizer=regularizers.l2(0.0005),
               bias_regularizer=regularizers.l2(0.0005)
              )(x)
    if norm == "batch":
        bn_axis = 3 if data_format == "channels_last" else 1
        x = BatchNormalization(axis=bn_axis, name="batch_norm_4")(x)
    x = Activation("relu", name="relu_4")(x)

    x = ZeroPadding2D(padding=(1, 1), name="pad_4")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name="conv_5",
               #padding="same",
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012
               bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe: 0.1
               kernel_regularizer=regularizers.l2(0.0005),
               bias_regularizer=regularizers.l2(0.0005)
              )(x)
    if norm == "batch":
        bn_axis = 3 if data_format == "channels_last" else 1
        x = BatchNormalization(axis=bn_axis, name="batch_norm_5")(x)
    x = Activation("relu", name="relu_5")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool_3")(x)  # Krizhevsky2012

    if include_top:
        x = Flatten(name="flatten_1")(x)
        x = Dense(4096, activation="relu", name="dense_1",
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012: 0.01, Caffe?:  0.005
                  bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe: 0.1
                  kernel_regularizer=regularizers.l2(0.0005),
                  bias_regularizer=regularizers.l2(0.0005)
                 )(x)
        x = Dropout(0.5, name="dropout_1")(x)
        # Krizhevsky2012: "We use dropout in the first two fully-connected layers"

        x = Dense(4096, activation="relu", name="dense_2",
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012: 0.01, Caffe?: 0.005
                  bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe: 0.1
                  kernel_regularizer=regularizers.l2(0.0005),
                  bias_regularizer=regularizers.l2(0.0005)
                 )(x)
        x = Dropout(0.5, name="dropout_2")(x)

        x = Dense(classes, activation="softmax", name="dense_3",
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  # Krizhevsky2012: 0.01, Caffe?: 0.1
                  bias_initializer=Constant(value=0.0),  # Krizhevsky2012
                  kernel_regularizer=regularizers.l2(0.0005),
                  bias_regularizer=regularizers.l2(0.0005)
                 )(x)

    # Create model
    model = Model(inputs=img_input, outputs=x, name="AlexNetFlat")

    weights_path = None
    if weights:
        if 'imagenet' == weights:
            # Initialize layer weights from pre-trained ImageNet model
            from keras.utils.data_utils import get_file

            weights_path = get_file('alexnetflat_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_FLAT_URL,
                                    cache_subdir='models',
                                    file_hash=WEIGHTS_FLAT_SHA256,
                                    hash_algorithm='sha256'
                                    )
        else:
            weights_path = weights

    if weights_path:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        if data_format == 'channels_first':
            maxpool = model.get_layer(name='convpool_5')
            shape = maxpool.output_shape[1:]
            dense = model.get_layer(name='dense_1')
            layer_utils.convert_dense_weights_data_format(dense=dense,
                                                          previous_feature_map_shape=shape,
                                                          target_data_format='channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def predict(deepvcd_ds, subset="val", weights="imagenet", input_size=227, norm="lrn", average_results=True):
    num_classes = len(deepvcd_ds.get_labels())
    model_class = globals()["AlexNetFlat"]
    model = model_class(include_top=True, input_shape=(input_size, input_size, 3), norm=norm, weights=weights, classes=num_classes)

    ds,_ = deepvcd_ds.get_tfdataset(subset=subset, shuffle_files=False)
    ds = ds.map(lambda x,y: (read_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
   
    gt = list()
    for _, label in ds:
        gt.append(label.numpy())
    gt = np.asarray(gt)

    log.info("Testing AlexNetFlat on dataset subset '{subset}':".format(subset=subset))
    if average_results:
        regions = ['tl', 'tr', 'c', 'bl', 'br']
    else:
        regions = ['c']

    # the following mean value has been computed using the applications/trainAlexNet.py script
    ilsvrc2012_mean = [123.0767, 115.56045, 101.68434] 
    #ilsvrc2012_mean = [0.48184115, 0.453552, 0.3977624]
    log.debug("Using mean={mean}".format(mean=ilsvrc2012_mean))

    avg_scores = None
    for region in regions:
        for flip in [False, True]:
            ds_ = ds.map(lambda x, y: (preprocess_predict(x, mean=ilsvrc2012_mean, crop_region=region, horizontal_flip=flip), y), num_parallel_calls=tf.data.AUTOTUNE)
            ds_ = ds_.map(lambda x,y: one_hot(x,y,num_classes))
            ds_ = ds_.batch(256)
            ds_ = ds_.prefetch(buffer_size=tf.data.AUTOTUNE)

            log.info("Computing model scores for region '{region} (flipped={flipped})'".format(region=region, flipped=flip))
            scores = model.predict(x=ds_,
                                   verbose=2)
            if avg_scores is None:
                avg_scores = scores
            else:
                avg_scores += scores

    scores = avg_scores / (2 * len(regions))
    return gt, scores


def _main(cli_args):
    if cli_args.weights_files == "imagenet" or cli_args.weights_files == ["imagenet"]:
         log.info("Using pre-trained ILSVRC2012 weights")
         weights_files = ["imagenet"]
    else:
        weights_files=cli_args.weights_files

    from deepvcd.dataset.descriptor import YAMLLoader
    log.info("Loading dataset from descriptor '{0}'".format(cli_args.dataset))
    deepvcd_ds = YAMLLoader.read(cli_args.dataset)

    scores = None
    for weights_file in weights_files:
        log.info("Using weights from '{weights_file}'".format(weights_file=weights_file))
        gt, scores_ = predict(norm=None if cli_args.norm.lower()=="none" else cli_args.norm,
                              input_size=cli_args.input_size,
                              weights=weights_file,
                              deepvcd_ds=deepvcd_ds,
                              average_results=True)
        top5error = top_k_error(y_true=gt, y_pred=scores_, k=5)
        log.info("Top-5-error-rate (single net): {top5error}".format(top5error=top5error))

        if scores is None:
            scores = scores_
        else:
            scores += scores_
    scores /= len(weights_files)

    if len(weights_files)>1:
        top5error = top_k_error(y_true=gt, y_pred=scores, k=5)
        log.info("Top-5-error-rate ({n} nets): {top5error}".format(n=len(weights_files), top5error=top5error))


if __name__ == '__main__':
    import argparse
    import keras
    import tensorflow as tf

    # configure logging
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    console.setFormatter(formatter)
    # add ch to logger
    log.addHandler(console)

    log.info("Tensorflow version: {ver}".format(ver=tf.__version__))
    log.info("Keras version: {ver}".format(ver=keras.__version__))

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Path to dataset descriptor yaml', dest='dataset', type=str, required=True)
    parser.add_argument('-n', '--norm',
                        dest='norm',
                        type=str,
                        help="Normalization to be used after first and second convolutional layer ('lrn' (default), 'tflrn', 'batch' or None)",
                        default="lrn",
                        required=False)
    parser.add_argument('-i', '--input_size',
                        dest='input_size',
                        type=int,
                        help="Input image size. 224 and 227 (default) are supported. 224 will result in padding.",
                        default=227,
                        required=False)
    parser.add_argument('-w', '--weights_file', 
                        dest='weights_files', 
                        type=str,
                        nargs='*',
                        help="File with model weights (hdf5). Needs to be compatible with the selected model type. "
                             "If none selected (default) the pre-trained ILSVRC2012 weights (use `lrn` norm, s.a.) will be used. "
                             "If multiple files are provided, the returned scores are averages over softmax outputs of each file.",
                        default="imagenet"
                        )
    args = parser.parse_args()

    _main(args)
