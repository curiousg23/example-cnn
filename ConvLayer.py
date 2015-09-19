import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig
import scipy.special as spec

class ConvLayer():
    """Perform the convolution and apply the sigmoid activation for a layer, then perform pooling"""

    def __init__(self, prev_layer_shape, filter_shape):
        """
        :type prev_layer_shape: tuple
        :param prev_layer_shape: (height, width, num feature maps) of the previous layer

        :type filter_shape: tuple
        :param filter_shape: (height, width, num filters)
        """

        self.prev_layer_shape = prev_layer_shape
        self.filter_shape = filter_shape
        # instantiate weight matrix
        rng = rand.RandomState(23455)
        W = np.asarray(rng.uniform(low=-1.0/(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    high=1.0/(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    size=filter_shape
        ))

        # instantiate bias. need a bias for each filter
        bias_shape = (filter_shape[2],)
        bias = np.zeros(bias_shape)
        # bias = np.asarray(rng.uniform(low=-0.5, high=-0.5, size=bias_shape))

        self.W = W
        self.bias = bias
        self.bias_shape = bias_shape
        self.params = [W, bias]

    # perform forward propagation--generate the output
    def forwardprop(self, input):
        """
        :type input: 3-d matrix at the moment
        :param input: the input to the layer--either a 2-D matrix or set of 2-D feature maps
        """
        # perform the convolution, each feature map with its respective layer,
        # sum the resultant convolution layers together
        conv_output = np.zeros((self.prev_layer_shape[0]-self.filter_shape[0]+1, self.prev_layer_shape[1]-self.filter_shape[1]+1, self.filter_shape[2]))
        for i in range(0, self.filter_shape[2]):
            for j in range(0, self.prev_layer_shape[2]):
                #rotation keeps the convolution matrix oriented correctly
                conv_output[:,:,i] += sig.convolve2d(input[:,:,j], np.rot90(self.W[:,:,i], 2), mode='valid')
            #add bias for this filter
            conv_output[:,:,i] = conv_output[:,:,i] + self.bias[i]

        self.input = input
        #apply sigmoid
        output = np.tanh(conv_output)
        self.conv_output = conv_output

        # output is a 3-D matrix, essentially a 'stacking' of the different feature maps produced
        self.output = output

    def backprop(self, layer):
        # previous layer should always be a pooling layer--the layer should have upsample() called
        self.error = np.zeros(self.output.shape)
        for i in range(0, layer.output.shape[2]):
            # for each filter do the following
            self.error[:,:,i] = np.multiply(layer.error[:,:,i], 1.0 - np.multiply(self.output[:,:,i], self.output[:,:,i]))
        # now to calculate gradients
        self.gradient_w = np.zeros((self.filter_shape[0], self.filter_shape[1], self.filter_shape[2]))
        for i in range(0, self.filter_shape[2]):
            for j in range(0, self.prev_layer_shape[2]):
                if j == 0:
                    self.gradient_w[:,:,i] = sig.convolve2d(self.input[:,:,j], np.rot90(self.error[:,:,i], 2), mode='valid')
                else:
                    self.gradient_w[:,:,i] += sig.convolve2d(self.input[:,:,j], np.rot90(self.error[:,:,i], 2), mode='valid')
        # may not be correct form of the bias gradient
        self.gradient_b = np.zeros(self.filter_shape[2])
        for i in range(0, self.filter_shape[2]):
            self.gradient_b[i] = np.sum(self.error[:,:,i])
