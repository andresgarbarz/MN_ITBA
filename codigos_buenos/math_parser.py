import re
import sympy as sp

def create_function_from_string(f_str):

    expr = convert_math_expression(f_str)
    
    # Crear la función
    def f(x):
        try:
            return eval(expr)
        except Exception as e:
            raise ValueError(f"Error al evaluar la expresión: {str(e)}")
    
    return f

def derivative(f_str, n):
    x = sp.Symbol('x')
    f = create_function_from_string(f_str)(x)

    for _ in range(n):
        f = sp.diff(f, x)
    return lambda x: f.subs(sp.Symbol('x'), x)

def convert_math_expression(expr):
    """
    Convierte una expresión matemática en string a una expresión Python válida.
    
    Args:
        expr (str): Expresión matemática en formato string
        
    Returns:
        str: Expresión convertida a formato Python
    """
    # Reemplazar exponentes (2^x -> 2**x)
    expr = expr.replace('^', '**')
    
    # Reemplazar constantes matemáticas
    expr = expr.replace('e**', 'sp.exp')
    expr = re.sub(r'(?<!\w)e(?!\w)', 'sp.E', expr)
    expr = expr.replace('pi', 'sp.pi')

    # Reemplazar log(x) por sp.log(x)
    expr = re.sub(r'log\s*\(([^)]+)\)', r'sp.log(\1)', expr)

    # Reemplazar ln(x) por sp.log(x)
    expr = re.sub(r'ln\s*\(([^)]+)\)', r'sp.log(\1)', expr)
    
    # Reemplazar logN(x) por sp.log(x, N)
    expr = re.sub(r'log(\d+)\s*\(([^)]+)\)', r'sp.log(\2, \1)', expr)


    # Reemplazar exp sin paréntesis por exp con paréntesis
    expr = re.sub(r'sp\.exp([^\(])', r'sp.exp(\1)', expr)

    # Agregar * implícito entre números y variables
    expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', expr)
    
    # Reemplazar funciones trigonométricas
    expr = expr.replace('sin', 'sp.sin')
    expr = expr.replace('cos', 'sp.cos')
    expr = expr.replace('tan', 'sp.tan')
    
    return expr

# Ejemplos de uso
if __name__ == "__main__":
    # Ejemplos de conversión
    examples = [
        "2x",
        "sin(x)",
        "2^x",
        "e^x",
        "e^(2x + 1)",
        "2pi",
        "x^2",
        "2sin(x)",
        "cos(2x)",
        "log10(x + 5)",
        "ln(x^2 + 1)",
        "1/x + 3x^(0.5x + 2) - sin(2x+1) + e + log10(x + 5) - x^2"
    ]
    
    # for example in examples:
    #     converted = convert_math_expression(example)
    #     print(f"Original: {example}")
    #     print(f"Convertido: {converted}")
    #     print()

    f_str = convert_math_expression("1/x + 3x^(0.5x + 2) - sin(2x+1) + e + log10(x + 5) - x^2")
    # f_str = convert_math_expression("x^5 + 3x^4 + 2x^3 - x^2 + 3x + 1")
    print("f_str: ", f_str)
    f_prime_str = derivative(f_str, 1)
    print("f_prime_str: ", f_prime_str)
    f_double_prime_str = derivative(f_str, 2)
    print("f_double_prime_str: ", f_double_prime_str)
    f_double_prime_processed = convert_math_expression(f_double_prime_str)
    print("f_double_prime_processed: ", f_double_prime_processed)

    # f = create_function_from_string(f_str)
    # for x in range(10):
    #     print(f"f({x}) = {f(x)}")