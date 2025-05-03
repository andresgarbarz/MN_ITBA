from sympy import symbols, lambdify, sympify
from utils import preprocesar_expresion

def metodo_incrementos(funcion_str, a, b):
    """
    Implementa el método de incrementos para encontrar raíces de una función.
    
    Parámetros:
    funcion_str (str): Función en formato string
    a (float): Punto inicial del intervalo
    b (float): Punto final del intervalo
    
    Retorna:
    tuple: (último valor de x antes del cambio de signo, número de iteraciones)
    """
    # Convertir la función string a una función evaluable
    x = symbols('x')
    funcion = sympify(funcion_str)
    f = lambdify(x, funcion, 'numpy')
    
    # Incremento hardcodeado
    delta = 0.000001
    
    # Inicialización
    x_actual = a
    iteracion = 0
    
    # Bucle principal
    while x_actual <= b:
        # Evaluar la función en el punto actual y en el siguiente
        f_actual = f(x_actual)
        f_siguiente = f(x_actual + delta)
        
        # Verificar si hay cambio de signo
        if f_actual * f_siguiente <= 0:
            # Retornamos el último valor de x antes del cambio de signo
            return x_actual, iteracion
        
        # Actualizar el punto actual
        x_actual += delta
        iteracion += 1
    
    # Si no se encontró raíz
    return None, iteracion

def main():
    print("Método de Incrementos")
    print("Ingrese la función f(x) (use x como variable):")
    print("Puede usar ^ para exponentes, pi, e, y omitir * entre números y variables")
    funcion_str = input("f(x) = ")
    
    # Obtener intervalo
    print("\nIngrese el intervalo [a, b] para buscar raíces:")
    a = float(input("a = "))
    b = float(input("b = "))
    
    # Preprocesar la expresión
    funcion_str = preprocesar_expresion(funcion_str)
    
    raiz, iteraciones = metodo_incrementos(funcion_str, a, b)
    
    if raiz is not None:
        print(f"\nÚltimo valor antes del cambio de signo: {raiz}")
        print(f"Número de iteraciones: {iteraciones}")
    else:
        print("\nNo se encontró raíz en el intervalo especificado")

if __name__ == "__main__":
    main()
