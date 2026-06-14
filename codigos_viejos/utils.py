import re
from math import pi, e, log, sin, cos, tan, exp, sqrt
from sympy.core.function import diff
from sympy.core.symbol import Symbol

def derivative(f, n):
    # Definir la variable simbólica
    x = Symbol('x')

    for _ in range(n):
        f = diff(f, x)
    return str(f)

def preprocesar_expresion(expr):
    """
    Preprocesa una expresión matemática para hacerla más amigable.
    - Reemplaza ^ por ** para exponentes
    - Agrega * implícito entre números y variables
    - Reemplaza pi y e por sus valores numéricos sin romper exp(x)
    - Permite exp(x) para exponenciales
    - Reemplaza ln(x) por log(x) (logaritmo natural)
    - Reemplaza logN(x) por log(x, N) (logaritmo en base N)
    """
    # Reemplazar ^ por **
    expr = expr.replace('^', '**')
    
    # Agregar * implícito entre números y variables
    expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', expr)
    
    # Reemplazar pi y e solo cuando aparecen como constantes.
    # No usar str.replace("e", ...) porque rompe exp(x).
    expr = re.sub(r'(?<![A-Za-z])pi(?![A-Za-z])', str(pi), expr)
    expr = re.sub(r'(?<![A-Za-z])e(?![A-Za-z])', str(e), expr)
    
    # Reemplazar ln(x) por log(x)
    expr = re.sub(r'ln\s*\(([^)]+)\)', r'log(\1)', expr)
    
    # Reemplazar logN(x) por log(x, N)
    expr = re.sub(r'log(\d+)\s*\(([^)]+)\)', r'log(\2, \1)', expr)
    
    return expr

def create_function_from_string(expr):
    """
    Crea una función a partir de una expresión matemática en string
    """
    # Preprocesar la expresión
    expr = preprocesar_expresion(expr)
    
    # Verificar que la expresión solo contenga caracteres seguros
    if not re.match(r'^[0-9+\-*/().xyt\s^logsincoetanepi]+$', expr):
        raise ValueError("La expresión contiene caracteres no permitidos. Use solo números, operadores matemáticos básicos, y funciones como sin, cos, tan, log, exp")
    
    # Crear la función
    def f(x):
        try:
            return eval(expr)
        except Exception as e:
            raise ValueError(f"Error al evaluar la expresión: {str(e)}")
    
    return f
