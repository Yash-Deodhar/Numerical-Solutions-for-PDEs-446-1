import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque


class Timestepper:

    def __init__(self, u, f):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper):

    def _step(self, dt):
        return self.u + dt*self.func(self.u)


class LaxFriedrichs(Timestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.func(self.u)


class Leapfrog(Timestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.func(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.func(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, func1, func2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(Timestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

        self.u_list = []
        self.K_list = []
        for i in range(self.stages):
            self.u_list.append(np.copy(u))
            self.K_list.append(np.copy(u))

    def _step(self, dt):
        u = self.u
        u_list = self.u_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(u_list[0], u)
        for i in range(1, stages):
            K_list[i-1] = self.func(u_list[i-1])

            np.copyto(u_list[i], u)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                u_list[i] += self.a[i, j]*dt*K_list[j]

        K_list[-1] = self.func(u_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            u += self.b[i]*dt*K_list[i]

        return u


class AdamsBashforth(Timestepper):

    def __init__(self, u, L_op, steps, dt):
        super().__init__(u, L_op)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(u))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.func(self.u)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.u += self.dt*coeff*self.f_list[i].data
        return self.u

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


class BackwardEuler(Timestepper):

    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.u)


class CrankNicolson(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.func.matrix
            self.RHS = self.I + dt/2*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.RHS @ self.u)


class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        super().__init__(u,L_op)
        self.steps = steps
        self.k = [self.u]
        self.past_dt = np.zeros(1,dtype=float)
        pass

    def _step(self, dt):
        self.dt = dt
        if len(self.k) < self.steps:
            steps = len(self.k)
        if len(self.k) >= self.steps:
            steps = self.steps
        self.past_dt = self.past_dt + self.dt
        self.past_dt = np.insert(self.past_dt, 0, 0)
        S = np.zeros((steps+1, steps+1))
        for i in range(steps+1):
            for j in range(steps+1):
                S[i][j] = pow(-1*self.past_dt[j],i)
        b = np.zeros(steps+1,dtype=float)
        b[1] = 1
        a = np.linalg.inv(S) @ b
        LHS = self.func.matrix - a[0]*np.eye(len(self.u),len(self.u))
        RHS = np.zeros(len(self.u),dtype=float)
        for i in range(1,len(a)):
            RHS = RHS + a[i]*self.k[i-1]
        x1 = np.linalg.solve(LHS, RHS)
        self.k.insert(0, x1)
        if steps == self.steps:
            self.k = self.k[:-1]
            self.past_dt = self.past_dt[:-1]
        return self.k[0]



class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])


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
        self.k = [self.X]
        self.past_dt = np.zeros(1,dtype=float)

    def _step(self, dt):
        self.dt = dt
        if len(self.k) < self.steps:
            steps = len(self.k)
        if len(self.k) >= self.steps:
            steps = self.steps
        self.past_dt = self.past_dt + self.dt
        self.past_dt = np.insert(self.past_dt, 0, 0)
        SM = np.zeros((steps+1,steps+1),dtype=float)
        SM[0][0] = 1
        bM = np.zeros(steps+1,dtype=float)
        for i in range(steps+1):
            for j in range(1,steps+1):
                SM[i][j] = pow(-1*self.past_dt[j],i)/factorial(i)       
        bM[1] = 1
        aM = np.linalg.solve(SM,bM) 
        SF = np.zeros((steps,steps),dtype=float)
        bF = np.zeros(steps,dtype=float)
        for i in range(steps):
            for j in range(steps):
                SF[i][j] = pow(-1*self.past_dt[j+1],i)/factorial(i)
        bF[0] = 1
        aF = np.linalg.solve(SF,bF)
        LHS = self.M*aM[0] + self.L
        x1 = 0
        x2 = 0
        for i in range(len(aF)):
            x1 = x1 + aF[i]*self.F(self.k[i])
        for i in range(1,len(aM)):
            x2 = x2 + aM[i]*self.k[i-1].data
        RHS = x1 - self.M @ x2
        X = np.linalg.solve(LHS.todense(),RHS)
        self.k.insert(0, StateVector([X]))
        if steps == self.steps:
            self.k = self.k[:-1]
            self.past_dt = self.past_dt[:-1]
        return self.k[0].data


        

