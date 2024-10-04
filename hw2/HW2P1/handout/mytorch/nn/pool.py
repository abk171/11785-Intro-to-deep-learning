import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1

        
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        self.maxIndex = np.empty(Z.shape, dtype=object)

        for batch in range(batch_size):
            for channel in range(in_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        mat = A[batch, channel, i : i + self.kernel, j : j + self.kernel]
                        # pdb.set_trace()
                        x, y = np.unravel_index(
                            np.argmax(mat),
                            mat.shape
                        )
                        self.maxIndex[batch, channel, i, j] = (i + x, j + y)
                        Z[batch, channel, i, j] = A[batch, channel, i + x, j + y]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape

        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, out_channels, input_height, input_width))

        for batch in range(batch_size):
            for channel in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        x, y = self.maxIndex[batch, channel, i, j]
                        # pdb.set_trace()
                        dLdA[batch, channel, x, y] += dLdZ[batch, channel, i, j]

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape

        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for batch in range(batch_size):
            for channel in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        mat = A[batch, channel, i : i + self.kernel, j : j + self.kernel]
                        Z[batch, channel, i, j] = np.mean(mat)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape

        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, out_channels, input_width, input_height))

        dLdZ /= (self.kernel ** 2) 
        
        for batch in range(batch_size):
            for channel in range(out_channels):
                for i in range(self.kernel):
                    for j in range(self.kernel):
                        dLdA[batch, channel, i : i + output_width, j : j + output_height] += dLdZ[batch, channel, :, :]
        
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A_maxpool = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(A_maxpool)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A_mean = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(A_mean)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ_upsampled)
        
        return dLdA
