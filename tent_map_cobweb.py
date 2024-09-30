#Questo codice realizza un diagramma a ragnatela per l'iterazione della mappa a tenda. Leonardo Perrini.
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

r = float(input("inserisci il valore di r: "))
iterazioni=int(input("inserisci il numero di iterazioni: "))
valore_iniziale=float(input("inserisci il valore di x0: "))

def tent(r, x):
    #defnisco la funzione tent
    if x < 0.5:
        return r*x
    else:
        return r*(1-x)

def plot_system(r, x0, n, ax=None):
    #traccia il grafico della funzione e della bisettrice y=x
    t = np.linspace(0,1, 1000)
    ax.plot([0,1/2],[0,r/2], 'k', lw=1)
    ax.plot([1/2,1],[r/2,0], 'k', lw=1)
    ax.plot([0,1],[0,1], 'k', lw=1)
    ax.plot(x0,x0, 'ob', ms=2)
    
    #applico ricorsivamente y=f(x) e grafico le due rette:
    #(x,x)->(x,y)
    #(x,y)->(y,y)
    x=x0
    for i in range(n):
        y=tent(r, x)
        #grafico le due rette
        ax.plot([x,x],[x,y], 'r', lw=0.4)
        ax.plot([x,y],[y,y], 'r', lw=0.4)
        x=y
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(f"$r={r:.3f}\,\,\,\,\, x_0={x0:.3f}$")
    ax.set_box_aspect(1)

fig, ax1=plt.subplots(1, 1, figsize=(8, 6))
plot_system(r, valore_iniziale, iterazioni, ax=ax1)
plt.show()