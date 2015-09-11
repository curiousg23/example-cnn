import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig

class PoolingLayer():
    """Perform the pooling after a convolution layer"""

    def __init__(self, input, input_shape):
        """
        :type input: 3-D matrix
        :param input: the input to the pooling layer--a set of feature maps (height, width, num feature maps)

        :type input_shape: 3-tuple of ints
        :param input_shape: (height, width, num feature maps)
        """

        self.input = input
        self.input_shape = input_shape
        # max-pooling operation
        output = np.zeros((input_shape[0]/2, input_shape[1]/2, input_shape[2]))
        max_indices[i,j,k] = np.zeros((input_shape[0]/2, input_shape[1]/2, input_shape[2]))
        for k in range(0, input_shape[2]):
            for i in range(0, input_shape[0]/2):
                for j in range(0, input_shape[1]/2):
                    output[i,j,k] = np.amax(input[2*i:2*i+2, 2*j:2*j+2, k])
                    max_indices[i,j,k] = np.argmax(input[2*i:2*i+2, 2*j:2*j+2,k])

        self.output = output
        self.max_indices = max_indices

    def upsample(self, layer, flag):
        if flag == 0:
            # previous layer was a fully connected layer, so output was flattened into a vector
            # first determine the partials for the vector components, then re-map into feature maps
            vec_error = np.transpose(layer.W)*layer.error
            self.error = np.zeros(self.input_shape)
            # map back to feature maps
            for i in range(0, self.output.shape[0]):
                for j in range(0, self.output.shape[1]):
                    for k in range(0, self.output.shape[2]):
                        # generalize this mapping later
                        if max_indices[i,j,k] == 0:
                            self.error[2*i,2*j,k] = vec_error[i+j+k]
                        elif max_indices[i,j,k] == 1:
                            self.error[2*i,2*j+1,k] = vec_error[i+j+k]
                        elif max_indices[i,j,k] == 2:
                            self.error[2*i+1,2*j,k] = vec_error[i+j+k]
                        else:
                            self.error[2*i+1,2*j+1,k] = vec_error[i+j+k]
        else:
            # previous layer was another convolutional layer--no need to re-map into feature maps
            # will need to figure out how to get the necessary gradients and errors, though 
            self.error = np.zeros(self.input_shape)
