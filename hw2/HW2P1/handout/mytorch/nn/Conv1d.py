# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        output_size = self.A.shape[-1] - self.kernel_size + 1
        Z = np.zeros((self.A.shape[0], self.out_channels, output_size))  # TODO

        for i in range(output_size):
            mat = self.A[ : , : , i : i + self.kernel_size]
            Z[:,:,i] = np.tensordot(mat, self.W, axes= [[1, 2], [1, 2]]) + self.b

        # pdb.set_trace()
        return Z
        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        
        ## pad dldz
        padded_dLdZ = np.pad(dLdZ, pad_width= [[0,0],[0,0],[self.kernel_size - 1, self.kernel_size - 1]])
        dLdA = np.zeros((self.A.shape[0], self.in_channels, self.A.shape[-1]))
        ## calculate dlda
        for i in range(self.A.shape[-1]):
            mat = padded_dLdZ[:,:, i:i+self.kernel_size]
            dLdA[:,:,i] = np.tensordot(mat, np.flip(self.W, axis = -1), axes = [[1,2],[0,2]])
        
        ## calculate dldw
            output_size = self.A.shape[-1] - self.kernel_size + 1
        
        for i in range(self.kernel_size):
            mat = self.A[ :, :, i: i + output_size]
            self.dLdW[:,:,i] = np.tensordot(dLdZ, mat, axes = [[0,2],[0,2]])

        # pdb.set_trace()
        self.dLdb = np.sum(dLdZ, axis = (-1, 0))


        
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, #padding= 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        # self.padding = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels= in_channels, 
                                             out_channels= out_channels,
                                             kernel_size= kernel_size,
                                             weight_init_fn= weight_init_fn,
                                             bias_init_fn= bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(downsampling_factor= self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        A = self.conv1d_stride1.forward(A)# TODO
        
        # downsample
        Z = self.downsample1d.forward(A)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)  # TODO

        return dLdA
