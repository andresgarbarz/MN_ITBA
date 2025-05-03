import sympy as sp
import numpy as np
# Definir la variable simbólica
x = sp.Symbol('x')

# Definir la función simbólica
f = x*sp.exp(0.5*x) + 1.2*x - 5
# f = x**3 - 7*x + 2

# Calcular la derivada
derivada = sp.diff(f, x)

# Mostrar resultados
print("Función original: f(x) =", f)
print("Derivada: f'(x) =", derivada)

# correr esto para obtener la derivada simbolica, puedo copiarla en el metodo que quiera correr