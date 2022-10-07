import numpy as np
from scipy.special import factorial
from scipy import sparse
import matplotlib.pyplot as plt

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)

class DifferenceUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.matrix = None
        self.grid = grid
        self.D_op()
        pass

    def D_op(self):
        S = np.zeros((self.convergence_order+self.derivative_order,self.convergence_order+self.derivative_order),dtype=float)
        index = np.arange(self.convergence_order+self.derivative_order)-int((self.convergence_order+self.derivative_order)/2)
        for k in range(self.convergence_order+self.derivative_order):
            for j in range(len(index)):
                S[k][j] = pow((index[j]*self.grid.dx),k)/factorial(k)
        S_inv = np.linalg.inv(S)
        b = np.zeros(self.convergence_order+self.derivative_order,dtype=int)
        b[self.derivative_order] = 1
        stencil = S_inv@b
        if abs(stencil[0]) < pow(10,-10):
            stencil = stencil[1:]
            index = index[1:]
        self.matrix = sparse.diags(stencil, offsets=index, shape = [self.grid.N, self.grid.N])
        self.matrix = self.matrix.tocsr()
        for i in range(index[-1]):
            for j in range(int((len(index)-1)/2-i)):
                self.matrix[i,self.grid.N-index[-1]+j+i] = stencil[j]
                self.matrix[self.grid.N-i-1,j] = stencil[i+j+int((len(stencil)+1)/2)]
        return self.matrix
    def __matmul__(self, other):
        return self.matrix@ other


class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.matrix = None
        self.grid = grid
        self.D_op()
        pass

    def D_op(self):
        D = np.zeros((self.grid.N,self.grid.N),dtype=float)
        S = np.zeros((self.convergence_order+self.derivative_order,self.convergence_order+self.derivative_order),dtype=float)
        index = np.arange(self.convergence_order+self.derivative_order)-int((self.convergence_order+self.derivative_order)/2)
        S_inv = np.zeros((self.convergence_order+self.derivative_order,self.convergence_order+self.derivative_order),dtype=float)
        b = np.zeros(self.convergence_order+self.derivative_order,dtype=int)
        b[self.derivative_order] = 1
        stencil = np.zeros((self.convergence_order+self.derivative_order),dtype=float)
        for i in range(self.grid.N):
            for k in range(self.convergence_order+self.derivative_order):
                for j in range(len(index)):
                    if i < self.grid.N-len(index)//2:
                        S[k][j] = pow(self.grid.values[i]-self.grid.values[i+index[j]],k)/factorial(k)
            S_inv = np.linalg.inv(S)
            stencil = S_inv@b
        for i in range(self.grid.N):
            for j in range(len(index)):
                if i+index[j] < self.grid.N:
                    D[i][i+j-len(index)//2] = pow(-1,self.derivative_order)*stencil[j]
        for i in range(index[-1]):
            for j in range(int((len(index)-1)/2-i)):
                D[self.grid.N-1-i][j] = pow(-1,self.derivative_order)*stencil[i+j+int((len(stencil)+1)/2)]    

        self.matrix = D
        return self.matrix

    def __matmul__(self, other):
        return self.matrix @ other
