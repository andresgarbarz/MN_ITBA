import numpy as np
import matplotlib.pyplot as plt
from math_parser import convert_math_expression, create_function_from_string, derivative
import sympy as sp

class NewtonPolynomial:
    """
    Clase para trabajar con polinomios de Newton
    Permite interpolación, aproximación de derivadas e integrales usando diferencias divididas
    """
    
    def __init__(self, x_points, y_points):
        """
        Inicializa el polinomio de Newton con puntos dados
        
        Args:
            x_points: Lista de coordenadas x de los puntos
            y_points: Lista de coordenadas y de los puntos
        """
        self.x_points = np.array(x_points)
        self.y_points = np.array(y_points)
        self.n = len(x_points)
        
        if len(x_points) != len(y_points):
            raise ValueError("El número de puntos x e y debe ser igual")
        
        # Verificar que no hay puntos x repetidos
        if len(set(x_points)) != len(x_points):
            raise ValueError("Los puntos x deben ser distintos")
        
        # Calcular las diferencias divididas
        self.divided_differences = self._compute_divided_differences()
    
    def _compute_divided_differences(self):
        """
        Calcula las diferencias divididas usando el método de tabla
        
        Returns:
            Lista de diferencias divididas [f[x₀], f[x₀,x₁], f[x₀,x₁,x₂], ...]
        """
        # Crear tabla de diferencias divididas
        table = np.zeros((self.n, self.n))
        
        # Primera columna son los valores de y
        table[:, 0] = self.y_points
        
        # Calcular diferencias divididas
        for j in range(1, self.n):
            for i in range(self.n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (self.x_points[i + j] - self.x_points[i])
        
        # Retornar la primera fila (coeficientes del polinomio de Newton)
        return table[0, :]
    
    def newton_basis(self, x, i):
        """
        Calcula el i-ésimo término base del polinomio de Newton: (x-x₀)(x-x₁)...(x-x_{i-1})
        
        Args:
            x: Punto donde evaluar
            i: Índice del término base (0 para constante, 1 para (x-x₀), etc.)
            
        Returns:
            Valor del i-ésimo término base en x
        """
        if i == 0:
            return 1.0
        
        result = 1.0
        for k in range(i):
            result *= (x - self.x_points[k])
        return result
    
    def evaluate(self, x):
        """
        Evalúa el polinomio de Newton en el punto x
        
        Args:
            x: Punto donde evaluar (puede ser un array)
            
        Returns:
            Valor del polinomio en x
        """
        if isinstance(x, (list, np.ndarray)):
            return np.array([self.evaluate_single(xi) for xi in x])
        else:
            return self.evaluate_single(x)
    
    def evaluate_single(self, x):
        """
        Evalúa el polinomio en un solo punto
        """
        result = 0.0
        for k in range(self.n):
            result += self.divided_differences[k] * self.newton_basis(x, k)
        return result
    
    def derivative_basis(self, x, i):
        """
        Calcula la derivada del i-ésimo término base del polinomio de Newton
        
        Args:
            x: Punto donde evaluar
            i: Índice del término base
            
        Returns:
            Valor de la derivada del i-ésimo término base en x
        """
        if i == 0:
            return 0.0
        
        if i == 1:
            return 1.0
        
        # Derivada de (x-x₀)(x-x₁)...(x-x_{i-1})
        result = 0.0
        for k in range(i):
            term = 1.0
            for j in range(i):
                if j != k:
                    term *= (x - self.x_points[j])
            result += term
        return result
    
    def derivative(self, x):
        """
        Aproxima la derivada del polinomio de Newton
        
        Args:
            x: Punto donde evaluar la derivada (puede ser un array)
            
        Returns:
            Valor de la derivada en x
        """
        if isinstance(x, (list, np.ndarray)):
            return np.array([self.derivative_single(xi) for xi in x])
        else:
            return self.derivative_single(x)
    
    def derivative_single(self, x):
        """
        Evalúa la derivada en un solo punto
        """
        result = 0.0
        for i in range(self.n):
            result += self.divided_differences[i] * self.derivative_basis(x, i)
        return result
    
    def get_coefficients(self):
        """
        Obtiene los coeficientes del polinomio en forma expandida
        
        Returns:
            Lista de coeficientes [a_n, a_{n-1}, ..., a_0] donde P(x) = a_n*x^n + ... + a_0
        """
        x = sp.Symbol('x')
        poly = 0
        
        for k in range(self.n):
            basis = 1
            for j in range(k):
                basis *= (x - self.x_points[j])
            poly += self.divided_differences[k] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Obtener coeficientes
        coeffs = []
        for i in range(self.n):
            coeff = poly_expanded.coeff(x, i)
            coeffs.append(float(coeff))
        
        return coeffs[::-1]  # Invertir para tener orden descendente
    
    def get_polynomial_math_string(self):
        """
        Retorna el polinomio de Newton como string
        
        Returns:
            str: Representación del polinomio como string
        """
        x = sp.Symbol('x')
        poly = 0
        
        for k in range(self.n):
            basis = 1
            for j in range(k):
                basis *= (x - self.x_points[j])
            poly += self.divided_differences[k] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Convertir a string
        return str(poly_expanded)
    
    def get_nth_derivative_string(self, n):
        """
        Retorna la n-ésima derivada del polinomio de Newton como string
        
        Args:
            n: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
            
        Returns:
            str: Representación de la n-ésima derivada como string
        """
        if n < 0:
            raise ValueError("El orden de la derivada debe ser no negativo")
        
        if n == 0:
            return self.get_polynomial_math_string()
        
        x = sp.Symbol('x')
        poly = 0
        
        for k in range(self.n):
            basis = 1
            for j in range(k):
                basis *= (x - self.x_points[j])
            poly += self.divided_differences[k] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Derivar n veces
        derivative = poly_expanded
        for _ in range(n):
            derivative = sp.diff(derivative, x)
        
        # Convertir a string
        return str(derivative)
    
    def get_polynomial_latex(self):
        """
        Retorna el polinomio de Newton en formato LaTeX
        
        Returns:
            str: Representación del polinomio en LaTeX
        """
        x = sp.Symbol('x')
        poly = 0
        
        for k in range(self.n):
            basis = 1
            for j in range(k):
                basis *= (x - self.x_points[j])
            poly += self.divided_differences[k] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Convertir a LaTeX
        return sp.latex(poly_expanded)
    
    def get_nth_derivative_latex(self, n):
        """
        Retorna la n-ésima derivada del polinomio de Newton en formato LaTeX
        
        Args:
            n: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
            
        Returns:
            str: Representación de la n-ésima derivada en LaTeX
        """
        if n < 0:
            raise ValueError("El orden de la derivada debe ser no negativo")
        
        if n == 0:
            return self.get_polynomial_latex()
        
        x = sp.Symbol('x')
        poly = 0
        
        for k in range(self.n):
            basis = 1
            for j in range(k):
                basis *= (x - self.x_points[j])
            poly += self.divided_differences[k] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Derivar n veces
        derivative = poly_expanded
        for _ in range(n):
            derivative = sp.diff(derivative, x)
        
        # Convertir a LaTeX
        return sp.latex(derivative)
    
    def evaluate_nth_derivative(self, x, n):
        """
        Evalúa la n-ésima derivada del polinomio de Newton en el punto x
        
        Args:
            x: Punto donde evaluar (puede ser un array)
            n: Orden de la derivada (0 para el polinomio, 1 para primera derivada, etc.)
            
        Returns:
            Valor de la n-ésima derivada en x
        """
        if n < 0:
            raise ValueError("El orden de la derivada debe ser no negativo")
        
        if n == 0:
            return self.evaluate(x)
        
        # Si n es mayor que el grado del polinomio, la derivada es 0
        if n > len(self.x_points) - 1:
            if isinstance(x, (list, np.ndarray)):
                return np.zeros_like(x)
            else:
                return 0.0
        
        if isinstance(x, (list, np.ndarray)):
            return np.array([self.evaluate_nth_derivative_single(xi, n) for xi in x])
        else:
            return self.evaluate_nth_derivative_single(x, n)
    
    def evaluate_nth_derivative_single(self, x, n):
        """
        Evalúa la n-ésima derivada en un solo punto
        """
        if n == 0:
            return self.evaluate_single(x)
        
        # Si n es mayor que el grado del polinomio, la derivada es 0
        if n > len(self.x_points) - 1:
            return 0.0
        
        # Calcular la n-ésima derivada usando los términos base de Newton
        result = 0.0
        
        for i in range(self.n):
            # Calcular la n-ésima derivada del i-ésimo término base
            basis_deriv = self.nth_derivative_basis(x, i, n)
            result += self.divided_differences[i] * basis_deriv
        
        return result
    
    def nth_derivative_basis(self, x, i, n):
        """
        Calcula la n-ésima derivada del i-ésimo término base del polinomio de Newton
        
        Args:
            x: Punto donde evaluar
            i: Índice del término base
            n: Orden de la derivada
            
        Returns:
            Valor de la n-ésima derivada del i-ésimo término base en x
        """
        if n == 0:
            return self.newton_basis(x, i)
        
        if n == 1:
            return self.derivative_basis(x, i)
        
        # Para derivadas de orden superior, usamos cálculo simbólico
        x_sym = sp.Symbol('x')
        basis = 1
        
        for j in range(i):
            basis *= (x_sym - self.x_points[j])
        
        # Derivar n veces
        derivative = basis
        for _ in range(n):
            derivative = sp.diff(derivative, x_sym)
        
        # Sustituir x y evaluar
        return float(derivative.subs(x_sym, x))
    
    def get_polynomial_string(self):
        """
        Retorna el polinomio de Newton como string en forma expandida, mostrando solo términos con coeficientes no nulos
        """
        coeffs = self.get_coefficients()
        
        poly_str = ""
        first_term = True
        
        for i, coeff in enumerate(coeffs):
            # Solo mostrar términos con coeficientes no nulos
            if abs(coeff) > 1e-10:  # Usar tolerancia para evitar errores de punto flotante
                power = len(coeffs) - 1 - i
                
                if first_term:
                    # Primer término
                    if power == 0:
                        poly_str += f"{coeff:.6f}"
                    elif power == 1:
                        poly_str += f"{coeff:.6f}x"
                    else:
                        poly_str += f"{coeff:.6f}x^{power}"
                    first_term = False
                else:
                    # Términos siguientes
                    if coeff >= 0:
                        poly_str += " + "
                    else:
                        poly_str += " - "
                        coeff = abs(coeff)
                    
                    if power == 0:
                        poly_str += f"{coeff:.6f}"
                    elif power == 1:
                        poly_str += f"{coeff:.6f}x"
                    else:
                        poly_str += f"{coeff:.6f}x^{power}"
        
        # Si todos los coeficientes son cero, retornar "0"
        if not poly_str:
            return "0"
        
        return poly_str
    
    def get_newton_form_string(self):
        """
        Retorna el polinomio de Newton en su forma original (con diferencias divididas)
        
        Returns:
            str: Representación del polinomio en forma de Newton
        """
        poly_str = f"{self.divided_differences[0]:.6f}"
        
        for i in range(1, self.n):
            term_str = f" + {self.divided_differences[i]:.6f}"
            for j in range(i):
                if self.x_points[j] == 0:
                    term_str += "(x)"
                elif self.x_points[j] < 0:
                    term_str += f"(x+{abs(self.x_points[j]):.6f})"
                else:
                    term_str += f"(x-{self.x_points[j]:.6f})"
            poly_str += term_str
        
        return poly_str
    
    def get_divided_differences(self):
        """
        Retorna las diferencias divididas calculadas
        
        Returns:
            Lista de diferencias divididas [f[x₀], f[x₀,x₁], f[x₀,x₁,x₂], ...]
        """
        return self.divided_differences.copy()
    
    def display_divided_differences_table(self):
        """
        Muestra la tabla triangular de diferencias divididas en formato clásico
        
        Returns:
            str: Representación de la tabla como string
        """
        # Crear tabla de diferencias divididas
        table = np.zeros((self.n, self.n))
        
        # Primera columna son los valores de y
        table[:, 0] = self.y_points
        
        # Calcular diferencias divididas
        for j in range(1, self.n):
            for i in range(self.n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (self.x_points[i + j] - self.x_points[i])
        
        # Crear representación de la tabla
        table_str = "Tabla de Diferencias Divididas:\n"
        table_str += "=" * 100 + "\n"
        
        # Encabezado
        header = f"{'x':<12} {'f(x)':<12}"
        for i in range(1, self.n):
            header += f" {'f[x₀,...,x_'+str(i)+']':<15}"
        table_str += header + "\n"
        table_str += "-" * 100 + "\n"
        
        # Filas de la tabla
        for i in range(self.n):
            row = f"{self.x_points[i]:<12.6f} {table[i, 0]:<12.6f}"
            for j in range(1, self.n):
                if i < self.n - j:
                    row += f" {table[i, j]:<15.6f}"
                else:
                    row += f" {'':<15}"
            table_str += row + "\n"
        
        table_str += "=" * 100 + "\n"
        table_str += f"Coeficientes del polinomio de Newton: {[f'{coeff:.6f}' for coeff in self.divided_differences]}\n"
        
        return table_str
    
    def get_divided_differences_table(self):
        """
        Retorna la tabla completa de diferencias divididas como array numpy
        
        Returns:
            numpy.ndarray: Tabla triangular de diferencias divididas
        """
        # Crear tabla de diferencias divididas
        table = np.zeros((self.n, self.n))
        
        # Primera columna son los valores de y
        table[:, 0] = self.y_points
        
        # Calcular diferencias divididas
        for j in range(1, self.n):
            for i in range(self.n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (self.x_points[i + j] - self.x_points[i])
        
        return table
    
    def display_divided_differences_table_compact(self):
        """
        Muestra la tabla triangular de diferencias divididas en formato compacto
        
        Returns:
            str: Representación compacta de la tabla como string
        """
        table = self.get_divided_differences_table()
        
        table_str = "Tabla de Diferencias Divididas (Formato Compacto):\n"
        table_str += "=" * 80 + "\n"
        
        # Encabezado
        header = f"{'x':<8} {'f(x)':<8}"
        for i in range(1, self.n):
            header += f" {'f[x₀..x_'+str(i)+']':<12}"
        table_str += header + "\n"
        table_str += "-" * 80 + "\n"
        
        # Filas de la tabla
        for i in range(self.n):
            row = f"{self.x_points[i]:<8.2f} {table[i, 0]:<8.2f}"
            for j in range(1, self.n):
                if i < self.n - j:
                    row += f" {table[i, j]:<12.4f}"
                else:
                    row += f" {'':<12}"
            table_str += row + "\n"
        
        table_str += "=" * 80 + "\n"
        table_str += f"Coeficientes: {[f'{coeff:.4f}' for coeff in self.divided_differences]}\n"
        
        return table_str
    
    def get_divided_differences_formatted(self, precision=6):
        """
        Retorna las diferencias divididas con formato personalizable
        
        Args:
            precision: Número de decimales para mostrar
            
        Returns:
            str: Diferencias divididas formateadas
        """
        formatted_str = "Diferencias Divididas:\n"
        formatted_str += "-" * 40 + "\n"
        
        for i, diff in enumerate(self.divided_differences):
            formatted_str += f"f[x₀, x₁, ..., x_{i}] = {diff:.{precision}f}\n"
        
        return formatted_str


def interpolate_function(func_str, x_points):
    """
    Interpola una función usando polinomio de Newton
    
    Args:
        func_str: String de la función a interpolar
        x_points: Puntos x para la interpolación
        
    Returns:
        NewtonPolynomial: Objeto del polinomio de Newton
    """
    # Crear función
    f = create_function_from_string(func_str)
    
    # Evaluar función en los puntos dados
    y_points = [f(x) for x in x_points]
    
    # Crear polinomio de Newton
    newton_poly = NewtonPolynomial(x_points, y_points)
    
    return newton_poly


def approximate_nth_derivative(func_str, x_points, x_eval, n):
    """
    Aproxima la n-ésima derivada de una función usando polinomio de Newton
    
    Args:
        func_str: String de la función
        x_points: Puntos para construir el polinomio
        x_eval: Puntos donde evaluar la derivada
        n: Orden de la derivada
        
    Returns:
        Valores de la n-ésima derivada aproximada
    """
    newton_poly = interpolate_function(func_str, x_points)
    return newton_poly.evaluate_nth_derivative(x_eval, n)


def test1():
    points = [(-1, 4.35), (-0.5, 2.0390625), (0, 1), (0.2, 0.8782), (1, 1.875)]
    x_points = [x for x, _ in points]
    y_points = [y for _, y in points]
    newton_poly = NewtonPolynomial(x_points, y_points)
    
    print("=== Polinomio de Newton ===")
    print(f"Puntos: {points}")
    print()
    
    # Mostrar tabla triangular de diferencias divididas
    print(newton_poly.display_divided_differences_table())
    print()
    
    """ # Mostrar diferencias divididas en formato compacto
    print(newton_poly.display_divided_differences_table_compact())
    print()
    
    # Mostrar diferencias divididas en formato personalizado
    print(newton_poly.get_divided_differences_formatted())
    print() """
    
    # Obtener polinomio en forma de Newton
    newton_form = newton_poly.get_newton_form_string()
    print("P(x) en forma de Newton:")
    print(f"P(x) = {newton_form}")
    print()
    
    # Obtener polinomio como string expandido
    poly_str = newton_poly.get_polynomial_string()
    print("P(x) expandido:")
    print(f"P(x) = {poly_str}")
    print()

    # Obtener n derivada como string
    n = 2
    deriv_n_str = newton_poly.get_nth_derivative_string(n)
    print(f"P^({n})(x) = {deriv_n_str}")
    print()

    # Evaluar n derivada en un punto
    x_eval = 2
    deriv_n_x = newton_poly.evaluate_nth_derivative(x_eval, n)
    print(f"P^({n})({x_eval}) = {deriv_n_x}")
    print()

def ejercicio3():
    """
    Resuelve el ejercicio de la Pregunta 3:
    Aproximar f(x) = 1/x en [1, 2] con puntos igualmente espaciados
    y encontrar el coeficiente que multiplica a (x-x₀)(x-x₁)(x-x₂)
    """
    print("=== EJERCICIO PREGUNTA 3 ===")
    print("Función: f(x) = 1/x")
    print("Intervalo: [1, 2]")
    print("Puntos igualmente espaciados: x₀ = 1, x₁ = 1.333, x₂ = 1.666, x₃ = 2")
    print()
    
    # Definir los puntos igualmente espaciados en [1, 2]
    x_points = [1.0, 4/3, 5/3, 2.0]  # 4 puntos igualmente espaciados
    
    # Calcular los valores de f(x) = 1/x en estos puntos
    y_points = [1/x for x in x_points]
    
    print(f"Puntos x: {x_points}")
    print(f"Puntos y = f(x): {y_points}")
    print()
    
    # Crear el polinomio de Newton
    newton_poly = NewtonPolynomial(x_points, y_points)
    
    # Mostrar la tabla de diferencias divididas
    print(newton_poly.display_divided_differences_table())
    print()
    
    # Obtener las diferencias divididas
    divided_diffs = newton_poly.get_divided_differences()
    print(f"Diferencias divididas: {[f'{diff:.6f}' for diff in divided_diffs]}")
    print()
    
    # El coeficiente que multiplica a (x-x₀)(x-x₁)(x-x₂) es la cuarta diferencia dividida
    # que corresponde al índice 3 (ya que empezamos desde 0)
    coeficiente = divided_diffs[3]
    
    print("=== RESPUESTA ===")
    print(f"El coeficiente que multiplica a (x-x₀)(x-x₁)(x-x₂) es: {coeficiente:.6f}")
    print(f"Redondeado a 3 decimales: {coeficiente:.3f}")
    print()
    
    # Mostrar el polinomio en forma de Newton
    newton_form = newton_poly.get_newton_form_string()
    print("Polinomio de Newton:")
    print(f"P(x) = {newton_form}")
    print()
    
    # Verificar que el polinomio interpola correctamente
    print("Verificación:")
    for i, x in enumerate(x_points):
        p_x = newton_poly.evaluate(x)
        f_x = 1/x
        print(f"P({x}) = {p_x:.6f}, f({x}) = {f_x:.6f}, Diferencia = {abs(p_x - f_x):.2e}")
    
    return coeficiente

# Ejemplos de uso
if __name__ == "__main__":
    # test1()
    ejercicio3()