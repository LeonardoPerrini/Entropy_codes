import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time
from numba import jit
import scienceplots
from scipy.optimize import curve_fit

plt.style.use(['science', 'grid'])

start_time = time.time()    

@jit
def logistic_map(r, x0, N):
    """
    Calcola l'evoluzione della mappa logistica per N iterazioni.
    
    Args:
        r (float): Parametro di controllo della mappa logistica.
        x0 (float): Valore iniziale della mappa logistica.
        N (int): Numero di iterazioni.
    
    Returns:
        np.ndarray: Sequenza di N valori della mappa logistica.
    """
    # Inizializziamo l'array per i risultati
    x = np.empty(N)
    x[0] = x0
    
    # Generiamo la sequenza iterativa
    for i in range(1, N):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    
    return x

@jit
def create_symbolic_sequence(sequence, epsilon):
    """
    Converte una sequenza di numeri continui in una sequenza simbolica basata su una partizione.
    
    Args:
        sequence (list): Sequenza di valori continui (es. generata dalla mappa logistica).
        epsilon (float): Larghezza degli intervalli per la partizione (determina la granularità della suddivisione).
    
    Returns:
        list: Sequenza simbolica (una lista di interi o stringhe).
    """
    # Dividiamo l'intervallo [0, 1] in regioni di ampiezza epsilon
    partition_points = np.arange(0, 1 + epsilon, epsilon)
    symbolic_sequence = []
    
    for value in sequence:
        # Troviamo l'intervallo in cui il valore si trova
        symbol = np.digitize(value, partition_points) - 1  # -1 per avere simboli da 0 in poi
        symbolic_sequence.append(symbol)
    
    return symbolic_sequence

def shannon_entropy(symbolic_sequence, partition_size):
    """
    Calcola l'entropia di Shannon per una sequenza simbolica.
    
    Args:
        symbolic_sequence (list): Sequenza simbolica.
        partition_size (int): Lunghezza delle sequenze simboliche da considerare.
    
    Returns:
        float: Entropia di Shannon per la sequenza fornita.
    """
    # Creiamo tutte le sequenze di lunghezza partition_size
    subsequences = [tuple(symbolic_sequence[i:i + partition_size]) for i in range(len(symbolic_sequence) - partition_size + 1)]
    
    # Contiamo le occorrenze di ogni sottosequenza
    subsequence_counts = Counter(subsequences)
    
    # Calcoliamo la probabilità di ciascuna sottosequenza
    total_subsequences = len(subsequences)
    probabilities = np.array(list(subsequence_counts.values())) / total_subsequences
    
    # Calcoliamo l'entropia di Shannon
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    return entropy

# Parametri della mappa logistica
r = 3.99  # Parametro di controllo della mappa logistica
x0 = 0.1  # Valore iniziale
n_steps = 100000000  # Numero di passi nella mappa logistica

# Parametri della partizione e entropia
epsilon = 0.5  # Larghezza degli intervalli della partizione
#partition_size = 40  # Lunghezza delle sequenze simboliche da considerare

# Generiamo la sequenza dalla mappa logistica
logistic_sequence = logistic_map(r, x0, n_steps)

# Convertiamo la sequenza continua in una sequenza simbolica
symbolic_sequence = create_symbolic_sequence(logistic_sequence, epsilon)

x = []
y = []
Nmin = 1
Nmax = 30

# Calcoliamo l'entropia di Shannon fratto n per ogni partition size
for partition_size in range(Nmin, Nmax+1):
    entropy = shannon_entropy(symbolic_sequence, partition_size)
    print(f"Entropia di Shannon (partition size = {partition_size}): {entropy/partition_size}")
    x.append(entropy)
    y.append(entropy/partition_size)

end_time = time.time()

print(f"Tempo di esecuzione: {end_time - start_time} secondi")
print(f"Entropia di Shannon finale: {entropy}")

#fit lineare
def f(x, a):
    return a*x

popt, pcov = curve_fit(f, np.arange(11, 16), x[10:15], maxfev=10000)

textstr = 'Valore stimato per $h$: $%.3f$' % (popt[0])


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


#plot
fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle(f'Mappa logistica con $r$ = {r}, $x0$ = {x0}, $N$ = {n_steps}, $\epsilon$ = {epsilon}')
ax = axes[0]
ax.set_title('Entropia di Shannon')
ax.set_xlabel('$n$')
ax.set_ylabel('$H$')
ax.plot(np.arange(Nmin, Nmax+1), x, "ob", label='Entropia di Shannon', markersize=2)
ax.plot(np.arange(Nmin, Nmax+1), f(np.arange(Nmin, Nmax+1), *popt), '-r', label='Fit lineare', linewidth=0.5)
ax.text(0.5, 0.1, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
ax.legend()
ax = axes[1]
ax.set_title('Entropia di Kolmogorov-Sinai')
ax.set_xlabel('$n$')
ax.set_ylabel(r'$h_{KS}$')
ax.plot(np.arange(Nmin, Nmax+1), y, "ob", label='Entropia di Kolmogorov-Sinai', markersize=2)
ax.plot(np.arange(Nmin, Nmax+1), np.ones_like(y)*popt[0], '-r', label='Fit lineare', linewidth=0.5) 
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
ax.legend()
fig.tight_layout()
#plt.savefig('KS.pdf')
plt.show()