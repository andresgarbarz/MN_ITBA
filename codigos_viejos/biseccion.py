import math
from utils import create_function_from_string

def biseccion(f, a, b, e=None, max_steps=None):
    """
    Método de bisección para encontrar la raíz de una función f en el intervalo [a,b]
    
    Parámetros:
    f: función a evaluar
    a: límite inferior del intervalo
    b: límite superior del intervalo
    e: error máximo permitido (None si se usa iteraciones)
    max_steps: número máximo de iteraciones (None si se usa error)
    
    Retorna:
    La aproximación de la raíz y el número de pasos realizados
    """
    # Verificar que la función cambie de signo en el intervalo
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe cambiar de signo en el intervalo [a,b]")
    
    # Si f(a) es positivo, intercambiamos a y b para que f(a) sea negativo
    if f(a) > 0:
        a, b = b, a
    
    # Calcular el número de pasos necesarios si se usa error
    if e is not None:
        L = abs(b - a)
        n = math.ceil(math.log2(L/e) - 1)
    else:
        n = max_steps
    
    # Realizar n+1 iteraciones (n+1 pues empiezo en S0)
    for i in range(n+1):
        c = (a + b) / 2
        if f(c) == 0:
            return c, i + 1
        elif f(c) < 0:
            a = c
        else:
            b = c
        print(f"Paso {i}: S{i} = {c}, f(S{i}) = {f(c)}")
    
    return c, n

if __name__ == "__main__":
    print("Método de Bisección")
    print("-------------------")
    print("Ingrese la función f(x) (use x como variable):")
    print("Puede usar ^ para exponentes, pi, e, y omitir * entre números y variables")
    
    # Obtener la función del usuario
    expr = input("f(x) = ")
    f = create_function_from_string(expr)
    
    # Obtener los límites del intervalo
    a = float(input("Ingrese el límite inferior (a): "))
    b = float(input("Ingrese el límite superior (b): "))
    
    # Elegir criterio de parada
    while True:
        criterio = input("Ingrese 't' para tolerancia o 'p' para pasos: ").lower()
        if criterio in ['t', 'p']:
            break
        print("Opción inválida. Ingrese 't' o 'p'")
    
    if criterio == 't':
        e = float(input("Ingrese el error máximo permitido (e): "))
        max_steps = None
    else:
        e = None
        max_steps = int(input("Ingrese el número máximo de pasos: "))
    
    try:
        raiz, pasos = biseccion(f, a, b, e, max_steps)
        print(f"\nLa raíz aproximada es: {raiz}")
        print(f"Número de pasos realizados: {pasos}")
    except ValueError as error:
        print(f"\nError: {error}")
