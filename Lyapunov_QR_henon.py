#Programma che calcola gli esponenti di Lyapunov di un sistema dinamico (mappa di Henon) e ne mostra il grafico. Leonardo Perrini.
import numpy as np

#parametri iniziali
a = 1.4
b = 0.3
iters = 1000000

def Henon(u):
    x,y = u
    f = [1 - a*x**2 + y, b*x]
    Df = [[-2*a*x, 1], [b, 0]]
    return np.array(f), np.array(Df)

U = np.eye(2)
v0 = np.array([0.63, 0.19]) 
lyap = np.zeros(2)

for i in range(iters):
    f,Df = Henon(v0)
    A = np.matmul(Df, U)
    Q, R = np.linalg.qr(A)
    lyap = lyap + np.log(abs(R.diagonal()))
    U = Q 
    v0 = f

lyapunov = lyap / (iters) 

print("Gli esponenti di Lyapunov sono: \n", lyapunov)