import numpy as np
from utils import create_function_from_string


def newton_raphson(f, f_prime, f_double_prime, a, b, e, max_iter=None):
    """
    Método de Newton-Raphson para encontrar el cero de una función.
    
    Parámetros:
    f: función f(x)
    f_prime: derivada primera f'(x)
    f_double_prime: derivada segunda f''(x)
    a: extremo izquierdo del intervalo
    b: extremo derecho del intervalo
    e: tolerancia o None si se usa iteraciones
    max_iter: None si se usa tolerancia, número de iteraciones si se usa iteraciones
    
    Retorna:
    x: aproximación del cero
    n: número de iteraciones
    """
    # Verificar que f(a) y f(b) tienen signos opuestos
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    
    # Verificar que f' tiene signo constante en [a,b]
    x_test = np.linspace(a, b, 100)
    f_prime_values = [f_prime(x) for x in x_test]
    if not all(x > 0 for x in f_prime_values) and not all(x < 0 for x in f_prime_values):
        raise ValueError("f' no tiene signo constante en [a,b]")
    
    # Verificar que f'' tiene signo constante en [a,b]
    f_double_prime_values = [f_double_prime(x) for x in x_test]
    if not all(x > 0 for x in f_double_prime_values) and not all(x < 0 for x in f_double_prime_values):
        raise ValueError("f'' no tiene signo constante en [a,b]")
    
    if f_double_prime(a) * f(a) >= 0:
        x0 = a
    else:
        x0 = b
    
    # Iteraciones de Newton-Raphson
    xn = x0
    n = 0
    print(f"Iteración {n}: xn = {xn}, f(xn) = {f(xn)}")
    
    if max_iter is None:
        # Criterio de parada por tolerancia
        while abs(f(xn)) >= e:
            xn = xn - f(xn) / f_prime(xn)
            n += 1
            print(f"Iteración {n}: xn = {xn}, f(xn) = {f(xn)}")
    else:
        # Criterio de parada por iteraciones
        while n < max_iter:
            xn = xn - f(xn) / f_prime(xn)
            n += 1
            print(f"Iteración {n}: xn = {xn}, f(xn) = {f(xn)}")
    
    return xn, n

def main():
    print("Método de Newton-Raphson")
    print("Ingrese la función f(x) (use x como variable):")
    print("Puede usar ^ para exponentes, pi, e, y omitir * entre números y variables")
    f_str = input("f(x) = ")
    print("Ingrese la derivada f'(x):")
    f_prime_str = input("f'(x) = ")
    print("Ingrese la segunda derivada f''(x):")
    f_double_prime_str = input("f''(x) = ")
    
    # Crear funciones a partir de las expresiones
    f = create_function_from_string(f_str)
    f_prime = create_function_from_string(f_prime_str)
    f_double_prime = create_function_from_string(f_double_prime_str)
    
    # Ingresar parámetros
    a = float(input("Ingrese a (extremo izquierdo del intervalo): "))
    b = float(input("Ingrese b (extremo derecho del intervalo): "))
    
    # Elegir criterio de parada
    while True:
        criterio = input("Ingrese 't' para tolerancia o 'i' para iteraciones: ").lower()
        if criterio in ['t', 'i']:
            break
        print("Opción inválida. Ingrese 't' o 'i'")
    
    if criterio == 't':
        e = float(input("Ingrese la tolerancia: "))
        max_iter = None
    else:
        e = None  # No se usa tolerancia
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
    
    try:
        raiz, iteraciones = newton_raphson(f, f_prime, f_double_prime, a, b, e, max_iter)
        print(f"\nRaíz encontrada: {raiz}")
        print(f"Número de iteraciones: {iteraciones}")
        print(f"f({raiz}) = {f(raiz)}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
