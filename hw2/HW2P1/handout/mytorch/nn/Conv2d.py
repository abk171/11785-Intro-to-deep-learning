import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
                           out_channels, in_channels, kernel_size, kernel_size
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        output_height = self.A.shape[-2] - self.kernel_size + 1
        output_width = self.A.shape[-1] - self.kernel_size + 1

        Z = np.zeros((self.A.shape[0], self.out_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                mat = self.A[ :, :, i:i + self.kernel_size, j: j + self.kernel_size]
                Z[ :, :, i, j] = np.tensordot(mat, self.W, axes = [[1,2,3],[1,2,3]])
        Z = Z + self.b[:, None, None]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
                              out_channels, in_channels, kernel_size, kernel_size
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        #calculate dlda
        pad_width       = self.kernel_size - 1
        padded_dLdZ     = np.pad(dLdZ, pad_width= [[0,0],[0,0],[pad_width, pad_width], [pad_width, pad_width]])
        flipped_weight  = np.flip(self.W, axis=(-1, -2))
        input_height    = self.A.shape[-2]
        input_width     = self.A.shape[-1]
        dLdA            = np.zeros(self.A.shape)

        for i in range(input_height):
            for j in range(input_width):
                mat = padded_dLdZ[ :, :, i: i + self.kernel_size, j: j + self.kernel_size] 
                dLdA[ :, :, i, j] = np.tensordot(mat, flipped_weight, axes = [[1,2,3],[0,2,3]])
        
        #calculate dldw
        output_height   = dLdZ.shape[-2]
        output_width    = dLdZ.shape[-1] 
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                mat = self.A[ :, :, i: i + output_height, j: j + output_width]
                self.dLdW[ :, :, i, j] = np.tensordot(dLdZ, mat, axes=[[0,2,3],[0,2,3]])
        
        #calculate dldb
        self.dLdb = np.sum(dLdZ, axis= (0, 2, 3))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding= 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.pad = padding
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels,
                                             out_channels,
                                             kernel_size,
                                             weight_init_fn,
                                             bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Padding A
        A = np.pad(A, pad_width = ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)))

        # Call Conv2d_stride1
        # TODO
        A = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(A)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        # TODO
        dLdZ = self.downsample2d.backward(dLdZ)
        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)  # TODO

        if self.pad == 0:
            return dLdA
        else:
            return dLdA[:, :, self.pad : -self.pad, self.pad : -self.pad]
