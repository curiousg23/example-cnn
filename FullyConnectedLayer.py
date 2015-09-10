import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig

class FullyConnectedLayer():
    """Fully conntected layer that takes in inputs (either from another fully-connected layer
    or the last pooling layer--will flatten the feature maps into one giant vector)
    A sigmoid activation function is used"""

    def __init__(self, input, f_in, f_out):
        """
        :type input: vector(?) or matrix with (length of flattened feature vector, num of inputs)
        :param input: the input to the fully-connected layer--likely 1-D

        :type f_in: int
        :param f_in: the number of inputs--links from the previous layer

        :type f_out: int
        :param f_out: the number of outputs--inputs to the next layer
        """

        self.input = input

        # weight matrix initialization
        rng = rand.RandomState(23455)
        W = np.asarray(rng.uniform(low=-1.0/(np.sqrt(f_in), high=1.0/(np.sqrt(f_in))), size=(f_out,f_in)))

        #bias initialization
        bias = np.zeros((f_out,)) #potentially need the 2nd dimension to be nonzero if multiple batches

        #perform activation
        lin_output = W*input + bias
        output = sigmoid(lin_output)
        self.output = output
        self.W = W
        self.bias = bias
        self.params = [self.W self.bias]
        self.lin_output = lin_output

    def sigmoid(x):
        return 1.0/(1+np.exp(-x))
