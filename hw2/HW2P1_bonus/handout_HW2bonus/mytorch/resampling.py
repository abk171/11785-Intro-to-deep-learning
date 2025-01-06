import numpy as np
import pdb


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        if self.upsampling_factor > 1:
            Z = np.kron(A, [1] + [0] * (self.upsampling_factor - 1))[ : , : , : 1 - self.upsampling_factor]  # TODO
            return Z
        else:
            return A

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        if self.upsampling_factor > 1:
            dLdA = dLdZ[ : , : , ::self.upsampling_factor]  # TODO
            return dLdA
        else:
            return dLdZ


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        if self.downsampling_factor > 1:
            self.w_in = A.shape[-1]
            Z = A[ : , : , ::self.downsampling_factor]  # TODO
            return Z
        else:
            return A

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        if self.downsampling_factor > 1:
            dLdA = np.kron(dLdZ, [1] + [0] * (self.downsampling_factor - 1))  # TODO
            dLdA = dLdA[ : , : , : self.w_in]
            return dLdA
        else:
            return dLdZ


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        if self.upsampling_factor > 1:
            b = [1] + [0] * (self.upsampling_factor - 1)
            c = np.array(b)[:, None] # make it a column vector
            x_scaled = np.kron(A, b)[ : , : , : , :(1 - self.upsampling_factor)]
            Z = np.kron(x_scaled, c)[ : , : , :(1 - self.upsampling_factor), : ]        # TODO
            return Z
        else:
            return A

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        if self.upsampling_factor > 1:
            dLdA = dLdZ[ : , : , ::self.upsampling_factor, ::self.upsampling_factor]  # TODO
            return dLdA
        else:
            return dLdZ


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        if self.downsampling_factor > 1:
            self.h_in = A.shape[-2]
            self.w_in = A.shape[-1]
            Z = A[ : , : , ::self.downsampling_factor, ::self.downsampling_factor]  # TODO
            return Z
        else:
            return A

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        if self.downsampling_factor > 1:
            b = [1] + [0] * (self.downsampling_factor - 1)
            c = np.array(b)[ : , None]
            x_scaled = np.kron(dLdZ, b)[ : , : , : , :self.h_in]
            dLdA = np.kron(x_scaled, c)[ : , : , :self.w_in, : ]  # TODO
            return dLdA
        else:
            return dLdZ