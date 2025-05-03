import sympy as sp
import numpy as np
# Definir la variable simbólica
x = sp.Symbol('x')

# Definir la función simbólica
f = x*sp.exp(0.5*x) + 1.2*x - 5
# f = x**3 - 7*x + 2

# Calcular la derivada
for i in range(2):
    f = sp.diff(f, x)
    print(f"f{i+1}(x) = ", f)

# correr esto para obtener la derivada simbolica, puedo copiarla en el metodo que quiera correr