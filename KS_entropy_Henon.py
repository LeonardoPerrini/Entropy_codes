#programma per calcolare l'entropia di Kolmogorov-Sinai per la mappa di Henon. Leonardo Perrini.
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from numba import jit
import time
from scipy.optimize import curve_fit
import scienceplots

plt.style.use(['science', 'grid'])

start_time = time.time()
# Defining the Henon map function
@jit
def henon_map(a, b, x0, y0, N):
    """
    Computes the evolution of the Henon map for N iterations.
    
    Args:
        a (float): Control parameter for the Henon map.
        b (float): Control parameter for the Henon map.
        x0 (float): Initial x value for the Henon map.
        y0 (float): Initial y value for the Henon map.
        N (int): Number of iterations.
    
    Returns:
        np.ndarray: Two sequences of N values for the x and y coordinates of the Henon map.
    """
    x = np.empty(N)
    y = np.empty(N)
    x[0], y[0] = x0, y0

    for i in range(1, N):
        x[i] = 1 - a * x[i - 1]**2 + y[i - 1]
        y[i] = b * x[i - 1]
    
    return x, y

# Now let's add a function to generate a symbolic sequence from the Henon map
@jit
def create_symbolic_sequence_2D(x_sequence, y_sequence, epsilon):
    """
    Converts the Henon map sequences into a symbolic sequence based on partitions.
    
    Args:
        x_sequence (np.ndarray): Sequence of x values from the Henon map.
        y_sequence (np.ndarray): Sequence of y values from the Henon map.
        epsilon (float): Partition size for symbolic dynamics.
    
    Returns:
        list: Symbolic sequence generated from the Henon map.
    """
    symbolic_sequence = []
    for x, y in zip(x_sequence, y_sequence):
        if x > epsilon and y > epsilon:
            symbolic_sequence.append('0')
        elif x > epsilon and y <= epsilon:
            symbolic_sequence.append('1')
        elif x <= epsilon and y > epsilon:
            symbolic_sequence.append('2')
        else:
            symbolic_sequence.append('3')
    
    return symbolic_sequence

# Function to compute the entropy of the symbolic sequence
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

a, b = 1.4, 0.3  # typical parameters for the Henon map
x0, y0 = 0, 0    # initial conditions
N = 1000000        # number of iterations

# Parametri della partizione e entropia
epsilon = 0.5  # Larghezza degli intervalli della partizione

# Generiamo la sequenza dalla mappa logistica
henon_sequence_x, henon_sequence_y = henon_map(a, b, x0, y0, N)

# Convertiamo la sequenza continua in una sequenza simbolica
symbolic_sequence = create_symbolic_sequence_2D(henon_sequence_x, henon_sequence_y, epsilon)

x = []
y = []
Nmin = 0
Nmax = 25

# Calcoliamo l'entropia di Shannon fratto n per ogni partition size
for partition_size in range(Nmin, Nmax+1):
    entropy = shannon_entropy(symbolic_sequence, partition_size)
    print(f"Entropia di Shannon (partition size = {partition_size}): {entropy/partition_size}")
    x.append(entropy)
    y.append(entropy/partition_size)

end_time = time.time()

print(x)

print(f"Tempo di esecuzione: {end_time - start_time} secondi")
print(f"Entropia di Shannon finale: {entropy}")
#fit lineare
def f(x, a):
    return a*x

popt, pcov = curve_fit(f, np.arange(8, 13), x[8:13], maxfev=10000)

textstr = 'Valore stimato per $h$: $%.3f$' % (popt[0])


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


#plot
fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle(f'Mappa di Henon con $a={a}$ e $b={b}$', fontsize=16)
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
plt.show()