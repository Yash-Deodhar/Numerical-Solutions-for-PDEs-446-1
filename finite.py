import numpy as np
from scipy.special import factorial
from scipy import sparse

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
        if (self.convergence_order+self.derivative_order)%2 == 0:
            length = self.convergence_order+self.derivative_order-1
        if (self.convergence_order+self.derivative_order)%2 != 0:
            length = self.convergence_order+self.derivative_order
        S = np.zeros((length,length),dtype=float)
        index = np.arange(length)-int((length)/2)
        S_inv = np.zeros((length,length),dtype=float)
        b = np.zeros(length,dtype=int)
        b[self.derivative_order] = 1
        stencil = np.zeros((self.grid.N,length),dtype=float)
        for i in range(self.grid.N):
            for k in range(length):
                for j in range(length):
                        S[k][j] = pow((self.grid.values[(i+index[j])%self.grid.N]-self.grid.values[i]),k)/factorial(k)
                        if (i+index[j]) < 0:
                            S[k][j] = pow(-(self.grid.length-self.grid.values[(i+index[j])%self.grid.N]+self.grid.values[i]),k)/factorial(k)
                        if (i+index[j]) >= self.grid.N:
                            S[k][j] = pow(self.grid.length+self.grid.values[(i+index[j])%self.grid.N]-self.grid.values[i],k)/factorial(k)
            stencil[i] = np.linalg.inv(S)@b
        for i in range(self.grid.N):
            for j in range(len(index)):
                D[i][(i+j-len(index)//2)%self.grid.N] = stencil[i][j] 
        self.matrix = D
        return self.matrix

    def __matmul__(self, other):
        return self.matrix @ other



class Difference:

    def __matmul__(self, other):
        return self.matrix @ other


class ForwardFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1/h, 1/h])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h
        self.matrix = matrix


class CenteredFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix
