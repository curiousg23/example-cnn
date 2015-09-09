import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig

class ConvLayer():
    """Perform the convolution and apply the sigmoid activation for a layer"""

    def __init__(self, input, prev_layer_shape, filter_shape):
        """
        :type input: 3-d matrix at the moment
        :param input: the input to the layer--either a 2-D matrix or set of 2-D feature maps

        :type prev_layer_shape: tuple
        :param prev_layer_shape: (num input feature maps, height, width) of the previous layer

        :type filter_shape: tuple
        :param filter_shape: (num filters, num input feature maps, height, width)
        """

        self.input = input

        # instantiate weight matrix
        rng = rand.RandomState(23455)
        W = np.asarray(rng.uniform(low=-1.0/(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    high=1.0/(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    size=filter_shape
        ))

        # instantiate bias. need a bias for each filter
        bias_shape = (filter_shape[0],)
        bias = np.asarray(rng.uniform(low=-0.5, high=-0.5, size=bias_shape))

        # perform the convolution
        sig.convolve2d(W, input)
