
import numpy as np
from scipy import sparse
import timesteppers
import finite
from timesteppers import StateVector, CrankNicolson, RK22


class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X*(1-X)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J

class BurgersFI:
    
    def __init__(self, X, nu, spatial_order, grid):
        self.X = timesteppers.StateVector([X])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        d1 = finite.DifferenceUniformGrid(1, spatial_order, grid)

        self.M = sparse.eye(len(X))
        self.L = -nu*d2.matrix

        def F(X):
            return -X * (d1 @ X)
        self.F = F
        
        def J(X):
            du_mat = sparse.diags(d1@X)
            u_mat = sparse.diags(X)
            return -du_mat - (u_mat @ d1.matrix)
        
        self.J = J


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid): # Grid was not provided
        self.X = X
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(X.variables[0])
        I = sparse.eye(self.N)
        Z = sparse.csr_matrix((self.N, self.N))
        
        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00,M01],
                              [M10,M11]])

        L00 = -D*d2.matrix
        L01 = Z
        L10 = Z
        L11 = -D*d2.matrix
        self.L = sparse.bmat([[L00,L01],
                              [L10,L11]])

        def F(X):
            c1 = X[:self.N]
            c2 = X[-self.N:]
            f_c1 = c1-c1*c1-c1*c2
            f_c2 = r*c2*c1-r*c2*c2

            return np.concatenate((f_c1, 
                                   f_c2))

        self.F = F
        
        def J(X):
            c1 = X[:self.N]
            c2 = X[-self.N:]

            # Based more on intuition than logic why this is done
            J00 = sparse.diags(1-2*c1-c2)
            J01 = sparse.diags(-c1)
            J10 = sparse.diags(r*c2)
            J11 = sparse.diags(r*c1-2*r*c2)
            J_individual = sparse.bmat([[J00,J01],
                                        [J10,J11]])

            return J_individual

        self.J = J

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
        L01 = Z
        L10 = Z
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
        L01 = Z

        
        L10 = Z
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

class DiffusionxBC:
    
    def __init__(self, c, D, d2x,dx):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        M = sparse.eye(N, N)        
        M = M.tocsr()
        M[0,:] = 0
        M[-1,:] = 0
        M.eliminate_zeros()
        self.M = M        
        L = -D*d2x.matrix
        L = L.tocsr()
        L[0,:] = 0
        L[-1,:] = dx.matrix[-1,:]
        L[0, 0] = 1
        L.eliminate_zeros()
        self.L = L

class DiffusionyBC:
    
    def __init__(self, c, D, d2y):
        self.X = StateVector([c], axis=1)
        N = c.shape[1]
        M = sparse.eye(N, N)
        self.M = M        
        L = -D*d2y.matrix
        self.L = L

class DiffusionBC:
    def __init__(self,c, D, spatial_order, domain):
        self.c = c
        self.D = D
        self.spatial_order = spatial_order
        self.domain = domain
        self.dx2 = finite.DifferenceUniformGrid(2, self.spatial_order, self.domain.grids[0], 0)
        self.dx = finite.DifferenceUniformGrid(1,self.spatial_order, self.domain.grids[0], 0)
        self.dy2 = finite.DifferenceUniformGrid(2, self.spatial_order, self.domain.grids[1], 1)
        self.diffy = DiffusionyBC(self.c,self.D,self.dy2)
        self.diffx = DiffusionxBC(self.c,self.D,self.dx2,self.dx) 
        self.t = 0
        self.iter = 0
        self.ts_y = CrankNicolson(self.diffy, 1)
        self.ts_x = CrankNicolson(self.diffx, 0)
    
    def step(self,dt):
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/1)
        self.ts_y.step(dt/2)
        self.t += dt
        self.iter += 1
        pass

class Advection_EqnX:
    
    def __init__(self, u, v, dx, dy):
        self.X = StateVector([u, v])
        n = len(u)
        def f(K):
            u1 = K.data[0:n, :]
            v1 = K.data[:-n, :]
            u1out = - u1 * (dx @ u1) - v1 * (dy @ u1)
            u2out = - u1 * (dx @ v1) - v1 * (dy @ v1)
            return np.concatenate((u1out, u2out))    
        self.F = f

class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.iter = 0
        self.t = 0

        dx2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)
        dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)
        dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)

        DiffX_Eqn = Burger_DiffusionX_Eqn(u, v, nu, dx2)
        DiffY_Eqn = Burger_DiffusionY_Eqn(u, v, nu, dy2)
        self.DiffusionX = CrankNicolson(DiffX_Eqn, 0)
        self.DiffusionY = CrankNicolson(DiffY_Eqn, 1)
        self.AdvectionX = RK22(Advection_EqnX(u, v, dx, dy))

    def step(self, dt):
        self.DiffusionX.step(dt/4)
        self.DiffusionY.step(dt/2)
        self.DiffusionX.step(dt/4)

        self.AdvectionX.step(dt)

        self.DiffusionX.step(dt/4)
        self.DiffusionY.step(dt/2)
        self.DiffusionX.step(dt/4)

        self.iter += 1
        self.t += dt

class Wave2DBC:
    def __init__(self, u, v, p, spatial_order, domain):
        self.X = StateVector([u,v,p])
        dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)
        dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)
        self.N = len(u)

        def F(X):
            u = X.data[:self.N, :]
            v = X.data[self.N:2*self.N, :]
            p = X.data[2*self.N:, :]
            out1 = (dx @ p)
            out2 = (dy @ p)
            out3 = (dx@u + dy@v)

            return np.concatenate([out1, out2, out3])
        self.F = F

        def BC(X):
            X.data[0, :] = 0
            X.data[self.N-1, :] = 0
        self.BC = BC






