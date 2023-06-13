from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class LRN2D(Layer):
    # adapted from 'crosschannelnormalization' in https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/customlayers.py
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def call(self, x):
        if K.image_data_format() == 'channels_first':
            b, ch, r, c = K.int_shape(x)
        else:
            b, r, c, ch = K.int_shape(x)

        half_n = self.n // 2  # half the local region
        input_sqr = K.square(x)  # square the input

        if K.image_data_format() == 'channels_first':
            # spatial_2d_padding allows to pad row and col dimensions - we need to pad channels,
            # so we need to shift channels dim to back first
            extra_channels = K.permute_dimensions(input_sqr, (0, 2, 3, 1))
            # padd channels dimension by half_n in each direction
            extra_channels = K.spatial_2d_padding(extra_channels,
                                                  padding=((0, 0), (half_n, half_n)),
                                                  data_format=K.image_data_format())
            extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        else:
            extra_channels = K.permute_dimensions(input_sqr, (0, 1, 3, 2))
            extra_channels = K.spatial_2d_padding(extra_channels,
                                                  padding=((0, 0), (half_n, half_n)),
                                                  data_format=K.image_data_format())
            extra_channels = K.permute_dimensions(extra_channels, (0, 1, 3, 2))

        scale = self.k
        for i in range(self.n):
            if K.image_data_format() == 'channels_first':
                scale += self.alpha * extra_channels[:, i:i + ch, :, :]
            else:
                scale += self.alpha * extra_channels[:, :, :, i:i + ch]
        scale **= self.beta

        return x / scale

    def get_config(self):
        base_config = super(LRN2D, self).get_config()
        base_config["alpha"] = self.alpha
        base_config["k"] = self.k
        base_config["beta"] = self.beta
        base_config["n"] = self.n

        return base_config

