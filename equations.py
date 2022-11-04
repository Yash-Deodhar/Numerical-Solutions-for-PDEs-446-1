from scipy import sparse
from sympy import SparseMatrix
from timesteppers import StateVector, CrankNicolson, RK22, Timestepper
import finite
import numpy as np

class Diffusionx:
    
    def __init__(self, c, D, d2x):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        self.M = sparse.eye(N, N)
        self.L = -D*d2x.matrix


class Diffusiony:
    
    def __init__(self, c, D, d2y):
        self.X = StateVector([c], axis=1)
        N = c.shape[1]
        self.M = sparse.eye(N, N)
        self.L = -D*d2y.matrix

class Reaction:

    def __init__(self,c):
        self.X = StateVector([c])
        self.F = lambda X: X.data*(1-X.data)

class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.c = c
        self.D = D
        self.dx2 = dx2
        self.dy2 = dy2
        self.t = 0
        self.diffy = Diffusiony(self.c,self.D,self.dy2)
        self.diffx = Diffusionx(self.c,self.D,self.dx2) 
        self.react = Reaction(self.c)
        self.ts_y = CrankNicolson(self.diffy, 1)
        self.ts_x = CrankNicolson(self.diffx, 0)
        self.ts_react = RK22(self.react)
        pass

    def step(self, dt):
        self.dt = dt
        self.ts_y.step(dt/4)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/4)
        self.ts_react.step(dt)
        self.ts_y.step(dt/4)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/4)
        self.t = self.t + dt
        return self.c


class Burger_DiffusionX_Eqn:
    
    def __init__(self, u, v, D, dx2):
        self.X = StateVector([u, v], axis=0)
        N = len(u)

        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        diff_operator = -D*dx2.matrix

        L00 = diff_operator
        L01 = I
        L10 = I
        L11 = diff_operator
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

class Burger_DiffusionY_Eqn:
    
    def __init__(self, u, v, D, dy2):
        self.X = StateVector([u, v], axis=1)
        N = len(u)

        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        diff_operator = -D*dy2.matrix

        L00 = diff_operator
        L01 = I
        L10 = I
        L11 = diff_operator
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

class Advection_Eqn:
    
    def __init__(self, u, v, dx, dy):
        self.X = StateVector([u, v])

        def f(X):
            u = X.variables[0]
            v = X.variables[1]
            return np.concatenate((u * (dx.matrix * u), u * (dx.matrix * v))) + np.concatenate((v * (dy.matrix * u), v * (dy.matrix * v)))

        self.F = f
    

class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.iter = 0
        self.t = 0

        dx2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)
        dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)

        DiffX_Eqn = Burger_DiffusionX_Eqn(u, v, nu, dx2)
        DiffY_Eqn = Burger_DiffusionY_Eqn(u, v, nu, dy2)
        self.DiffusionX = CrankNicolson(DiffX_Eqn, 0)
        self.DiffusionY = CrankNicolson(DiffY_Eqn, 1)
        self.Advection = RK22(Advection_Eqn(u, v, dx, dy))

    def step(self, dt):
        self.DiffusionX.step(dt/4)
        self.DiffusionY.step(dt/2)
        self.DiffusionX.step(dt/4)

        self.Advection.step(dt)

        self.DiffusionX.step(dt/4)
        self.DiffusionY.step(dt/2)
        self.DiffusionX.step(dt/4)

        self.iter += 1
        self.t += dt


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M01 = Z
        M10 = Z
        M11 = I
        if type(rho0) is not int:
            rho0 = sparse.diags(rho0, offsets = 0,shape=[N,N]).todense()
        M00 = rho0*I

        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])
        
        if type(gammap0) is not int :
            gammap0 = sparse.diags(gammap0, offsets = 0,shape=[N,N]).todense()

        L00 = Z
        L01 = d.matrix
        L10 = gammap0*d.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2c, c_target, D):
        self.X = StateVector([c])
        N = len(c)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        self.M = I

        self.L = -1*D*d2c.matrix

        
        self.F = lambda X: X.data*(c_target-X.data) 
    
