import re
from math import pi, e

def preprocesar_expresion(expr):
    """
    Preprocesa una expresión matemática para hacerla más amigable.
    - Reemplaza ^ por ** para exponentes
    - Agrega * implícito entre números y variables
    - Reemplaza pi y e por sus valores numéricos
    """
    # Reemplazar ^ por **
    expr = expr.replace('^', '**')
    
    # Agregar * implícito entre números y variables
    expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', expr)
    
    # Reemplazar pi y e
    expr = expr.replace('pi', str(pi))
    expr = expr.replace('e', str(e))
    
    return expr

def create_function_from_string(expr):
    """
    Crea una función a partir de una expresión matemática en string
    """
    # Preprocesar la expresión
    expr = preprocesar_expresion(expr)
    
    # Verificar que la expresión solo contenga caracteres seguros
    if not re.match(r'^[0-9+\-*/().x\s^]+$', expr):
        raise ValueError("La expresión contiene caracteres no permitidos")
    
    # Crear la función
    def f(x):
        try:
            return eval(expr)
        except:
            raise ValueError("La expresión no es válida")
    
    return f

if __name__ == "__main__":
    print("Evaluate")
    print("-------------------")
    print("Ingrese la función f(x) (use x como variable):")
    print("Puede usar ^ para exponentes, pi, e, y omitir * entre números y variables")
    
    # Obtener la función del usuario
    expr = input("f(x) = ")
    f = create_function_from_string(expr)

    # Ingrese el valor de x a evaluar
    x = float(input("Ingrese el valor de x: "))

    # Evaluar la función en el valor de x
    print(f"f({x}) = {f(x)}")