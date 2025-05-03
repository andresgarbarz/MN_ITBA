import sympy as sp

from utils import preprocesar_expresion

# Definir la variable simbólica
x = sp.Symbol('x')

def derivative(f, n):
    for _ in range(n):
        f = sp.diff(f, x)
    return f

def main():
    print("Ingrese la función f(x) (use x como variable):")
    print("Puede usar ^ para exponentes, pi, e, y omitir * entre números y variables")
    f_str = input("f(x) = ")
    
    # crear la funcion
    f = preprocesar_expresion(f_str)
    print(f)
    
    # calcular la derivada
    derivada = derivative(f, 1)
    derivada2 = derivative(f, 2)
    # mostrar resultados
    print("Función original: f(x) =", f)
    print("1era derivada: f'(x) =", derivada)
    print("2da derivada: f''(x) =", derivada2)

if __name__ == "__main__":
    main()
