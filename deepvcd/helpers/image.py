"""
Some image preprocessing tools used in this project.

Some general notes to myself since I keep forgetting these:

## Image data format:
Image data format refers to the representation of batches of images. keras supports NHWC
(channels_last, TensorFlow default) and NCHW (channels_first, theano default).
N refers to the number of images in a batch, H refers to the number of pixels in the vertical dimension,
W refers to the number of pixels in the horizontal dimension,
and C refers to the channels (e.g. 1 for black and white, 3 for RGB, etc.)
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.preprocessing.image import img_to_array

import tensorflow as tf

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
else:
    _PIL_INTERPOLATION_METHODS = {}


def load_img(path, grayscale=False, target_size=None,
             interpolation='bicubic'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size), int (image will
            be scaled according to aspect ratio, smallest of (img_height, img_width)
            will be scaled to target_size) or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            img_width, img_height = img.size  # PIL.Image.size: The size is given as a 2-tuple (width, height).
            # rescale according to aspect ratio
            scale = target_size / min(img_width, img_height)
            width_height_tuple = (max(target_size, int(img_width*scale)), max(target_size, int(img_height*scale)))
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        elif isinstance(target_size, tuple):
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        else:
            ValueError("Invalid type for `target_size`: {0}".format(type(target_size)))
    return img


def pil_image_reader(filepath, target_mode='rgb', target_size=None, interpolation='bicubic',
                     data_format=K.image_data_format()):
    if target_mode == "rgb":
        grayscale = False
    elif target_mode == "grayscale":
        grayscale = True
    else:
        raise ValueError('Invalid color mode:', target_mode,
                         '; expected "rgb" or "grayscale".')
    img = load_img(filepath, grayscale=grayscale, target_size=target_size, interpolation=interpolation)
    x = img_to_array(img, data_format=data_format)
    img.close()
    return x


def read_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_image(image, channels=3, dtype=tf.uint8)
    image.set_shape((None, None, 3))
    return image


def read_image_pil(image_file, target_size):
    def _im_file_to_tensor(image_file):
        path = image_file.numpy().decode()
        image = pil_image_reader(filepath=path, target_size=target_size)
        image.set_shape((None, None, 3))
        return image
    return tf.py_function(_im_file_to_tensor, 
                          inp=[image_file], 
                          Tout=tf.float32)


def one_hot(image, label, num_classes):
    # Casts to an Int and performs one-hot ops
    label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
    # Recasts it to Float32
    label = tf.cast(label, tf.float32)
    return image, label
