from utils import create_function_from_string

""" NO TIENE CHEQUEOS DE SI LA F HALLADA SIRVE """
""" HAY QUE DESPEJAR X, IGUALAR ESO A g(X), IR A DESMOS Y VER QUE |g'(x)| < 1 PINTE LA RAÍZ, ENTONCES SIGNIFICA QUE SE PUEDE HALLAR CON ESE DESPEJE"""
""" f(x) ACA REPRESENTA g(x) """

def metodo_puntos_fijos(f_str, a, b, error_max=None, max_steps=None):
    """
    Implementa el método de puntos fijos para encontrar una raíz de la función f.
    
    Parámetros:
    f_str (str): Expresión de la función en formato string
    a (float): Extremo inferior del intervalo
    b (float): Extremo superior del intervalo
    error_max (float): Error máximo permitido (None si se usa max_steps)
    max_steps (int): Número máximo de pasos (None si se usa error_max)
    
    Retorna:
    tuple: (aproximación de la raíz, número de pasos, error final)
    """
    # Crear la función a partir del string
    f = create_function_from_string(f_str)
    
    # Inicializar variables
    x_anterior = b  # Tomamos x0 como el extremo b

    print(f"Paso 0: X0 = {x_anterior}")

    x_actual = f(x_anterior)
    pasos = 1
    error = abs(x_anterior - x_actual)
    
    print(f"Paso {pasos}: X{pasos} = {x_actual}, Error{pasos} = {error}")
    
    # Iterar según el criterio elegido
    if error_max is not None:
        # Criterio de parada por error
        while error > error_max:
            x_anterior = x_actual
            x_actual = f(x_anterior)
            pasos += 1
            error = abs(x_anterior - x_actual)
            print(f"Paso {pasos}: X{pasos} = {x_actual}, Error{pasos} = {error}")
    else:
        # Criterio de parada por pasos
        while pasos < max_steps:
            x_anterior = x_actual
            x_actual = f(x_anterior)
            pasos += 1
            error = abs(x_anterior - x_actual)
            print(f"Paso {pasos}: X{pasos} = {x_actual}, Error{pasos} = {error}")
    
    return x_actual, pasos, error

if __name__ == "__main__":
    # Ejemplo de uso
    try:
        # Solicitar datos al usuario
        f_str = input("Ingrese la función f(x): ")
        a = float(input("Ingrese el extremo inferior a: "))
        b = float(input("Ingrese el extremo superior b: "))
        
        # Elegir criterio de parada
        while True:
            criterio = input("Ingrese 'e' para error o 'p' para pasos: ").lower()
            if criterio in ['e', 'p']:
                break
            print("Opción inválida. Ingrese 'e' o 'p'")
        
        if criterio == 'e':
            error_max = float(input("Ingrese el error máximo permitido: "))
            max_steps = None
        else:
            error_max = None
            max_steps = int(input("Ingrese el número máximo de pasos: "))
        
        # Aplicar el método
        raiz, pasos, error_final = metodo_puntos_fijos(f_str, a, b, error_max, max_steps)
        
        # Mostrar resultados
        print(f"\nAproximación de la raíz: {raiz}")
        print(f"Número de pasos: {pasos}")
        print(f"Error final: {error_final}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
