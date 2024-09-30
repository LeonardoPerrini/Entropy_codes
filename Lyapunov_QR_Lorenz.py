# Programma che calcola gli esponenti di Lyapunov di un sistema dinamico (Lorenz) e ne mostra il grafico. Leonardo Perrini.
import numpy as np
from scipy.integrate import solve_ivp

#ODE sistema
def func(t, v, sigma, r, b):
    x, y, z = v 
    return [ sigma * (y - x), r * x - y - x * z, x * y - b * z ]

#Jacobiana
def JM(v, sigma, r, b):
    x, y, z = [k for k in v]
    return np.array([[-sigma, sigma, 0], [r - z, -1, -x], [y, x, -b]])

#parametri iniziali
sigma = 10
r = 28
b = 8/3
iters = 10**5
dt = 0.001
tf = iters * dt

U = np.eye(3) 
v0 = np.ones(3) 
lyap = np.zeros(3)

#integro l'ODE, è importante trovarsi in un attrattore per la riuscita dell'algoritmo
sol = solve_ivp(func, [0, tf], v0, t_eval = np.linspace(0, tf, iters), args=(sigma, r, b))
v_n = sol.y.T 

#itero
for i in range(iters):
    v0 = v_n[i]
    A = np.matmul(np.eye(3) + JM(v0, sigma, r, b) * dt, U)

    #decoposizione QR
    Q, R = np.linalg.qr(A)
    lyap = lyap + np.log(abs(R.diagonal()))

    U = Q 

print("Gli esponenti di Lyapunov per il sistema sono: \n",lyap / (iters*dt))