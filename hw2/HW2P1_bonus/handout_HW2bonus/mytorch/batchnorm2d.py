# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # Inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        Forward pass for BatchNorm2d.
        Arguments:
          Z (np.array): Input tensor of shape (batch_size, num_features, height, width)
          eval (bool): Whether the layer is in evaluation mode
        Returns:
          np.array: Normalized and scaled tensor of the same shape as Z
        """
        if eval:
            # Inference mode: Use running mean and variance
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)  # Normalize
            BZ = self.BW * NZ + self.Bb  # Scale and shift
            return BZ

        # Training mode
        self.Z = Z

        # Compute batch mean and variance across (Batch, Height, Width) dimensions
        self.M = np.mean(Z, axis=(0, 2, 3), keepdims=True)  # Mean
        self.V = np.var(Z, axis=(0, 2, 3), keepdims=True)  # Variance

        # Normalize the input
        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)

        # Scale and shift
        self.BZ = self.BW * self.NZ + self.Bb

        # Update running mean and variance
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        return self.BZ

    def backward(self, dLdBZ):
        # Compute gradients w.r.t. scale (BW) and shift (Bb)
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3), keepdims=True)

        # Intermediate gradients
        dLdNZ = dLdBZ * self.BW  # shape matches BZ and NZ

        # Number of elements averaged over: N * H * W
        N, C, H, W = self.Z.shape
        N_eff = N * H * W

        # (Z - M)
        X_minus_M = self.Z - self.M

        # Gradient w.r.t variance
        dLdV = np.sum(dLdNZ * X_minus_M, axis=(0,2,3), keepdims=True) * (-0.5) * ((self.V + self.eps) ** (-1.5))

        # Gradient w.r.t mean
        dLdM = np.sum(dLdNZ * (-1.0 / np.sqrt(self.V + self.eps)), axis=(0,2,3), keepdims=True) \
            + dLdV * np.sum(-2.0 * X_minus_M, axis=(0,2,3), keepdims=True) / N_eff

        # Gradient w.r.t. input Z
        dLdZ = (dLdNZ / np.sqrt(self.V + self.eps)) \
            + (dLdV * 2.0 * X_minus_M / N_eff) \
            + (dLdM / N_eff)

        return dLdZ

