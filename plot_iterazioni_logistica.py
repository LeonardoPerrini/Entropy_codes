#Questo codice realizza un diagramma a ragnatela per l'iterazione della mappa logistica. Leonardo Perrini.
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'grid'])

r = float(input("inserisci il valore di r: "))
iterazioni = int(input("inserisci il numero di iterazioni: "))
valore_iniziale = float(input("inserisci il valore di x0: "))

def logistic(r, x):
    #defnisco la funzione logistica
    return r * x * (1 - x)

def plot_system(r, x0, n, ax=None):
    #traccia il grafico della funzione e della bisettrice y=x
    t=np.linspace(0,1)
    ax.plot(t, logistic(r,t), 'k', lw=1)
    ax.plot([0,r/4],[0,r/4], 'k', lw=1)
    
    #applico ricorsivamente y=f(x) e grafico le due rette:
    #(x,x)->(x,y)
    #(x,y)->(y,y)
    x=x0
    y_max=float(r/4)
    for i in range(n):
        y=logistic(r,x)
        
        #grafico le due rette
        ax.plot([x,x],[x,y], 'k', lw=0.5)
        ax.plot([x,y],[y,y], 'k', lw=0.5)
        x=y
    
    ax.set_xlim(0,r/4+1/4)
    ax.set_ylim(0,y_max+1/4)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")

fig, ax1=plt.subplots(1, 1, figsize=(8, 6))
plot_system(r, valore_iniziale, iterazioni, ax=ax1)
plt.show()