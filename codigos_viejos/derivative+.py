import re
from utils import derivative 
import sympy as sp
x = sp.Symbol('x')

def process_expr(expr):
    """
    Procesa una expresión matemática para que pueda ser evaluada por la función f
    """
    # Reemplazar ^ por **
    expr = expr.replace('^', '**')
    
    # Agregar * implícito entre números y variables
    expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', expr)
    
    # Reemplazar pi y e
    expr = expr.replace('pi', 'sp.pi')
    expr = expr.replace('e**', 'sp.exp')
    
    # Reemplazar ln(x) por log(x)
    expr = re.sub(r'ln\s*\(([^)]+)\)', r'log(\1)', expr)
    
    # Reemplazar logN(x) por log(x, N)
    expr = re.sub(r'log(\d+)\s*\(([^)]+)\)', r'log(\2, \1)', expr)
    
    return expr

def main():
    print("Método de Newton-Raphson")
    print("Ingrese la función f(x) (use x como variable):")
    print("Puede usar ^ para exponentes, pi, e, y omitir * entre números y variables")
    # f_str = "e^(2x) - x - 6"
    f_str = "x^4 + 8x^3 +13x^2 -16x -30"
    
    # Crear funciones a partir de las expresiones
    f = process_expr(f_str)
    f_prime_str = derivative(eval(f), 1)
    f_double_prime_str = derivative(eval(f), 2)

    f_str = f_str.replace("**", "^").replace("exp", "e^")
    f_prime_str = f_prime_str.replace("**", "^").replace("exp", "e^")
    f_double_prime_str = f_double_prime_str.replace("**", "^").replace("exp", "e^")

    print("f(x) =", f_str)
    print("f'(x) =", f_prime_str)
    print("f''(x) =", f_double_prime_str)

if __name__ == "__main__":
    main()
