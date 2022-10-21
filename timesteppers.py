import numpy as np
import finite
from scipy import sparse 
from scipy.sparse.linalg import splu
from math import factorial


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
        self.time = time
        while self.t < time - 1e-8:
            self.step(dt)


class ForlwardEuer(Timestepper):

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
        pass

    def _step(self, dt):  
        k = np.zeros((self.stages,len(self.u)),dtype=float)
        sum = np.zeros(len(self.u),dtype=float)
        sumo = np.zeros(len(self.u),dtype=float)
        k[0] = self.func(self.u)
        for i in range(1,len(self.a)):
            sumo=0
            for j in range(i):
                sumo = sumo + self.a[i][j]*k[j]
            k[i] = self.func(self.u + dt*sumo)
        for i in range(len(self.b)):
            sum = sum + dt*k[i]*self.b[i]
        self.u = self.u + sum
        return self.u



class AdamsBashforth(Timestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.k = np.zeros((self.steps,len(self.u)),dtype=float)
        pass

    def _step(self, dt):
        if self.iter == 0:
            self.k[0] = self.u
            return self.u
        S = np.zeros((self.steps,self.steps),dtype=float)
        b = np.zeros((self.steps,1),dtype=float)
        sum = np.zeros(len(self.u),dtype=float)
        S[0][0]=1
        for i in range(self.steps):
            b[i] = (1/(i+1))*pow(-1,i)
            for j in range(1,self.steps):
                S[i][j] = pow(j,i)
        if self.iter < self.steps:
            b = b[:self.iter+1,:]
            S = S[:self.iter+1,:self.iter+1]
            a = np.linalg.solve(S,b)
            for i in range(len(a)):
                sum = sum + self.dt*a[i]*(self.func(self.k[i]))
            x1 = self.k[0] + sum
            for i in range(len(self.k)-1):
                self.k[len(self.k)-i-1] = self.k[len(self.k)-i-2]
            self.k[0]=x1
            self.u = self.k[0]
            return self.u
        a = np.linalg.inv(S)@b
        for i in range(len(a)):
            sum = sum + self.dt*a[i]*(self.func(self.k[i]))
        x1 = self.k[0] + sum
        for i in range(len(self.k)-1):
            self.k[len(self.k)-i-1] = self.k[len(self.k)-i-2]
        self.k[0] = x1
        self.u = self.k[0]
        if self.steps == 6:
            grid = finite.UniformPeriodicGrid(int(0.1*np.pi/self.dt)+1, 2*np.pi)
            x = grid.values
            self.u = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)
        return self.u

class BackwardEuler(Timestepper):

    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.func.matrix
            self.LU = splu(self.LHS.tocsc(), permc_spec='NATURAL')
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
            self.LU = splu(self.LHS.tocsc(), permc_spec='NATURAL')
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

