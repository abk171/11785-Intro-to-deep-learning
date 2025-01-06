# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p
        self.eps = 1e-8

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        if eval:
          return x
        batch_size, in_channel, input_width, input_height = x.shape
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        self.mask = np.random.binomial(1, 1 - self.p, size=[batch_size, in_channel, 1, 1])
        x = self.mask * x
        # 2) Scale your output accordingly
        self.scale = 1.0 / (1 - self.p + self.eps)
        # 3) During test time, you should not apply any mask or scaling.
        return x * self.scale
        #TODO


    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule

        #TODO
        return self.mask * self.scale * delta


