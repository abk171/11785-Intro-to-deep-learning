# import numpy as np

# # Copy your Linear class from HW1P1 here
# class Linear:

#     def __init__(self, in_features, out_features, debug=False):

#         self.W = np.zeros((out_features, in_features), dtype="f")
#         self.b = np.zeros((out_features, 1), dtype="f")
#         self.dLdW = np.zeros((out_features, in_features), dtype="f")
#         self.dLdb = np.zeros((out_features, 1), dtype="f")

#         self.debug = debug

#     def forward(self, A):

#         self.A = A
#         self.N = A.shape[0]
#         self.Ones = np.ones((self.N, 1), dtype="f")
#         Z = None  # TODO

#         return NotImplemented

#     def backward(self, dLdZ):

#         dZdA = None  # TODO
#         dZdW = None  # TODO
#         dZdi = None
#         dZdb = None  # TODO
#         dLdA = None  # TODO
#         dLdW = None  # TODO
#         dLdi = None
#         dLdb = None  # TODO
#         self.dLdW = dLdW / self.N
#         self.dLdb = dLdb / self.N

#         if self.debug:

#             self.dZdA = dZdA
#             self.dZdW = dZdW
#             self.dZdi = dZdi
#             self.dZdb = dZdb
#             self.dLdA = dLdA
#             self.dLdi = dLdi

#         return NotImplemented


import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # TODO
        self.b = np.zeros((out_features, 1))  # TODO

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = self.A @ self.W.T + self.Ones @ self.b.T  # TODO

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W  # TODO
        self.dLdW = dLdZ.T @ self.A  # TODO
        self.dLdb = dLdZ.T @ self.Ones  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
    