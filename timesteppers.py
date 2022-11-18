import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque
from farray import axslice, apply_matrix

class StateVector:

    def __init__(self, variables, axis=0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[axslice(self.axis, i*self.N, (i+1)*self.N)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[axslice(self.axis, i*self.N, (i+1)*self.N)])


class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1
    
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.F = eq_set.F
        if hasattr(eq_set, 'BC'):
            self.BC = eq_set.BC
        else:
            self.BC = None

    def step(self, dt):
        super().step(dt)
        if self.BC:
            self.BC(self.X)
            self.X.scatter()


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.X.data + dt*self.F(self.X)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        N = len(X.data)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.X.data + dt*self.F(self.X)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.X_old = np.copy(self.X.data)
            return self.X.data + dt*self.F(self.X)
        else:
            X_temp = self.X_old + 2*dt*self.F(self.X)
            self.X_old = np.copy(self.X)
            return X_temp


class LaxWendroff(ExplicitTimestepper):

    def __init__(self, X, F1, F2):
        self.t = 0
        self.iter = 0
        self.X = X
        self.F1 = F1
        self.F2 = F2

    def _step(self, dt):
        return self.X.data + dt*self.F1(self.X) + dt**2/2*self.F2(self.X)


class Multistage(ExplicitTimestepper):

    def __init__(self, eq_set, stages, a, b):
        super().__init__(eq_set)
        self.stages = stages
        self.a = a
        self.b = b

        self.X_list = []
        self.K_list = []
        for i in range(self.stages):
            self.X_list.append(StateVector([np.copy(var) for var in self.X.variables]))
            self.K_list.append(np.copy(self.X.data))

    def _step(self, dt):
        X = self.X
        X_list = self.X_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(X_list[0].data, X.data)
        for i in range(1, stages):
            K_list[i-1] = self.F(X_list[i-1])

            np.copyto(X_list[i].data, X.data)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                X_list[i].data += self.a[i, j]*dt*K_list[j]
            if self.BC:
                self.BC(X_list[i])

        K_list[-1] = self.F(X_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            X.data += self.b[i]*dt*K_list[i]

        return X.data


def RK22(eq_set):
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, eq_set, steps, dt):
        super().__init__(eq_set)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(X.data))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.F(self.X)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += self.dt*coeff*self.f_list[i].data

        return self.X.data

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


class ImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, axis):
        super().__init__()
        self.axis = axis
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L

    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape)-1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS = self.M - dt/2*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        pass

    def _step(self, dt):
        pass


class IMEXTimestepper:

    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        pass

    def _step(self, dt):
        pass


class NewtonsMethod():
    # You are solving the equation LHS(X) = RHS 
    # (We are taking the form that the LHS has the X_n+1 terms (which are to be guessed)
    # and the RHS only has the X_n terms which are a constant for the present time step)
    
    # You bring it in the following form: LHS(X) - RHS = residual
    
    # Knowing this initial value of the residual based on an initial guess, you try solving for the residual alone:
    # residual + residual_dash @ delta_X = 0
    # residual_dash @ delta_X = residual (Matrix solve)
    
    # Based on the obtained value for delta_X, you recalculate the LHS(X)

    def __init__(self, LHS, RHS, residual_dash, guess, tolerance):
        self.LHS = LHS
        self.RHS = RHS
        self.resdiual_dash = residual_dash
        self.guess = guess
        self.tol = tolerance

    def run(self):
        i_loop = 0
        residual = self.LHS(self.guess) - self.RHS
        while np.max(np.abs(residual)) > self.tol:
            r_dash = self.resdiual_dash(self.guess)

            delta_X = spla.spsolve(r_dash, -residual)

            self.guess[:] += delta_X

            residual = self.LHS(self.guess) - self.RHS

            i_loop += 1
            if i_loop == 40:
                print('error: reached more than 40 iterations')
                break
        return self.guess



class FullyImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, tol=1e-7):
        super().__init__()
        if type(eq_set).__name__ == 'ReactionTwoSpeciesDiffusion':
            self._step = self.BackwardEulerFI_step
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.tol = tol
        self.J = eq_set.J
        
    def step(self, dt, guess=None):
        self.X.gather()
        self.X.data = self._step(dt, guess)
        self.X.scatter()
        self.t += dt
        self.iter += 1

    def BackwardEulerFI_step(self, dt, guess = None):
        if dt != self.dt:
            self.dt = dt
        
        if guess is None:
            guess = self.X.data

        LHS = lambda X: self.M @ X + dt*(self.L @ X - self.F(X))
        RHS = self.M @ self.X.data
        residual_dash = lambda X: self.M + dt*(self.L - self.J(X))

        solver = NewtonsMethod(LHS, RHS, residual_dash, guess, self.tol)

        return solver.run()


class BackwardEulerFI(FullyImplicitTimestepper):

    def _step(self, dt, guess):
        if dt != self.dt:
            self.dt = dt
        
        if guess is None:
            guess = self.X.data

        LHS = lambda X: self.M @ X + dt*(self.L @ X - self.F(X))
        RHS = self.M @ self.X.data
        residual_dash = lambda X: self.M + dt*(self.L - self.J(X))

        solver = NewtonsMethod(LHS, RHS, residual_dash, guess, self.tol)

        return solver.run()

class CrankNicolsonFI(FullyImplicitTimestepper):



    def _step(self, dt, guess = None):
        if dt != self.dt:
            self.dt = dt

        if guess is None:
            guess = self.X.data

        LHS = lambda X: (self.M @ X) + (dt/2) * ((self.L @ X) - self.F(X))
        RHS = (self.M @ self.X.data) - (dt/2) * ((self.L @ self.X.data) - self.F(self.X.data))
        residual_dash = lambda X: self.M + (dt/2)*(self.L - self.J(X)) # Most probably correct
        # residual_dash = lambda X: self.M + dt/2*(self.L - self.J(X))

        solver = NewtonsMethod(LHS, RHS, residual_dash, guess, self.tol)

        return solver.run()
