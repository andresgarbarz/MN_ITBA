import numpy as np
from scipy.optimize import fsolve

""" Author: ChatGPT con los prompts de Mariu """

# Definimos la función f(x)
def f(x):
    return x*np.exp(0.5*x) + 1.2*x - 5

# en un ej con solo una raiz figuraba
#Función original: f(x) = x*exp(0.5*x) + 1.2*x - 5
#Derivada: f'(x) = 0.5*x*exp(0.5*x) + exp(0.5*x) + 1.2

# en otro ej figuraba   return np.exp(x) - 3*x**2 - x
# con derivada   return np.exp(x) - 6*x - 1

# Derivada de la función f(x), útil para Newton (opcional)
def df(x):
    return 0.5*x*np.exp(0.5*x) + np.exp(0.5*x) + 1.2

# Generamos valores en el intervalo [0, 4]
x_vals = np.linspace(1, 2, 400)
y_vals = f(x_vals)


# Buscamos soluciones aproximadas con fsolve para saber dónde están las raíces
root1 = fsolve(f, 0.5)[0]
root2 = fsolve(f, 3)[0]
print(f"Raíces aproximadas: {root1:.6f}, {root2:.6f}")

# Proponemos una forma de g(x): x = e^x / (3x + 1)
def g(x):
    return np.exp(x) / (3*x + 1)

# Derivada numérica de g(x)
def g_prime(x):
    h = 1e-3
    return (g(x + h) - g(x - h)) / (2 * h)

# Evaluamos |g'(x)| cerca de cada raíz

# si hay una sola raiz va a poner la unica dos veces
g_prime_r1 = abs(g_prime(root1))
g_prime_r2 = abs(g_prime(root2))
print(f"|g'(raíz menor)| = {g_prime_r1:.6f}")
print(f"|g'(raíz mayor)| = {g_prime_r2:.6f}")

# va a converger si el modulo ese es menor a 1 