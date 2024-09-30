# programma che calcola l'esponente di Lyapunov per la mappa logistica. Leonardo Perrini.
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'grid'])

r_min = float(input("inserisci il valore minimo di lambda (in genere 1): "))
r_max = float(input("inserisci il valore massimo di lambda (in genere 4): "))
m = int(input("inserisci il numero di intervalli per il range di suddivisione di lambda (10000 di solito funziona bene): "))

def logistic(r, x):
    #defnisco la funzione logistica
    return r * x * (1 - x)

r = np.linspace(r_min, r_max, m)
iterations = 10000
x = 1e-5 * np.ones(m)
lyapunov = np.zeros(m)

for i in range(iterations):
    x = logistic(r, x)
    # calcoliamo la somma parziale dell'
    # Esponente di Lyapunov.
    lyapunov += np.log(abs(r - 2 * r * x))


fig, ax1 = plt.subplots(1, 1, figsize=(11, 7))
# esponente di Lyapunov negativo.
ax1.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / iterations,
         '.k', ms=.6)
# esponente di Lyapunov positivo.
ax1.plot(r[lyapunov >= 0], lyapunov[lyapunov >= 0] / iterations, '.r', ms=.6)
ax1.set_xlim(r_min, r_max)
ax1.set_ylim(-2, 1)
ax1.set_xlabel('$r$', fontsize=18)
ax1.set_ylabel('$\lambda$', fontsize=18)
ax1.set_title("Esponente di Lyapunov mappa logistica", fontsize=18)
plt.tight_layout()
plt.show()