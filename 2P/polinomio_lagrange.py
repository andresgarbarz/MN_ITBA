import re
import numpy as np
import matplotlib.pyplot as plt
from math_parser import convert_math_expression, create_function_from_string, derivative
import sympy as sp

class LagrangePolynomial:
    """
    Clase para trabajar con polinomios de Lagrange
    Permite interpolación, aproximación de derivadas e integrales
    """
    
    def __init__(self, x_points, y_points):
        """
        Inicializa el polinomio de Lagrange con puntos dados
        
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
    
    def lagrange_basis(self, x, i):
        """
        Calcula el i-ésimo polinomio base de Lagrange
        
        Args:
            x: Punto donde evaluar
            i: Índice del polinomio base
            
        Returns:
            Valor del i-ésimo polinomio base en x
        """
        result = 1.0
        for k in range(self.n):
            if i != k:
                result *= (x - self.x_points[k]) / (self.x_points[i] - self.x_points[k])
        return result
    
    def evaluate(self, x):
        """
        Evalúa el polinomio de Lagrange en el punto x
        
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
            result += self.y_points[k] * self.lagrange_basis(x, k)
        return result
    
    def derivative_basis(self, x, i):
        """
        Calcula la derivada del i-ésimo polinomio base de Lagrange
        
        Args:
            x: Punto donde evaluar
            i: Índice del polinomio base
            
        Returns:
            Valor de la derivada del i-ésimo polinomio base en x
        """
        result = 0.0
        for k in range(self.n):
            if k != i:
                term = 1.0
                for j in range(self.n):
                    if j != i and j != k:
                        term *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
                term /= (self.x_points[i] - self.x_points[k])
                result += term
        return result
    
    def derivative(self, x):
        """
        Aproxima la derivada del polinomio de Lagrange
        
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
            result += self.y_points[i] * self.derivative_basis(x, i)
        return result
    
    def get_coefficients(self):
        """
        Obtiene los coeficientes del polinomio en forma expandida
        
        Returns:
            Lista de coeficientes [a_n, a_{n-1}, ..., a_0] donde P(x) = a_n*x^n + ... + a_0
        """
        x = sp.Symbol('x')
        poly = 0
        
        for i in range(self.n):
            basis = 1
            for j in range(self.n):
                if i != j:
                    basis *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
            poly += self.y_points[i] * basis
        
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
        Retorna el polinomio de Lagrange como string
        
        Returns:
            str: Representación del polinomio como string
        """
        x = sp.Symbol('x')
        poly = 0
        
        for i in range(self.n):
            basis = 1
            for j in range(self.n):
                if i != j:
                    basis *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
            poly += self.y_points[i] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Convertir a string
        return str(poly_expanded)
    
    def get_nth_derivative_string(self, n):
        """
        Retorna la n-ésima derivada del polinomio de Lagrange como string
        
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
        
        for i in range(self.n):
            basis = 1
            for j in range(self.n):
                if i != j:
                    basis *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
            poly += self.y_points[i] * basis
        
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
        Retorna el polinomio de Lagrange en formato LaTeX
        
        Returns:
            str: Representación del polinomio en LaTeX
        """
        x = sp.Symbol('x')
        poly = 0
        
        for i in range(self.n):
            basis = 1
            for j in range(self.n):
                if i != j:
                    basis *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
            poly += self.y_points[i] * basis
        
        # Expandir el polinomio
        poly_expanded = sp.expand(poly)
        
        # Convertir a LaTeX
        return sp.latex(poly_expanded)
    
    def get_nth_derivative_latex(self, n):
        """
        Retorna la n-ésima derivada del polinomio de Lagrange en formato LaTeX
        
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
        
        for i in range(self.n):
            basis = 1
            for j in range(self.n):
                if i != j:
                    basis *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
            poly += self.y_points[i] * basis
        
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
        Evalúa la n-ésima derivada del polinomio de Lagrange en el punto x
        
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
        
        # Calcular la n-ésima derivada usando los polinomios base de Lagrange
        result = 0.0
        
        for i in range(self.n):
            # Calcular la n-ésima derivada del i-ésimo polinomio base
            basis_deriv = self.nth_derivative_basis(x, i, n)
            result += self.y_points[i] * basis_deriv
        
        return result
    
    def nth_derivative_basis(self, x, i, n):
        """
        Calcula la n-ésima derivada del i-ésimo polinomio base de Lagrange
        
        Args:
            x: Punto donde evaluar
            i: Índice del polinomio base
            n: Orden de la derivada
            
        Returns:
            Valor de la n-ésima derivada del i-ésimo polinomio base en x
        """
        if n == 0:
            return self.lagrange_basis(x, i)
        
        if n == 1:
            return self.derivative_basis(x, i)
        
        # Para derivadas de orden superior, usamos la fórmula recursiva
        # o calculamos simbólicamente
        x_sym = sp.Symbol('x')
        basis = 1
        
        for j in range(self.n):
            if i != j:
                basis *= (x_sym - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
        
        # Derivar n veces
        derivative = basis
        for _ in range(n):
            derivative = sp.diff(derivative, x_sym)
        
        # Sustituir x y evaluar
        return float(derivative.subs(x_sym, x))
    
    def get_polynomial_string(self):
        """
        Retorna el polinomio de Lagrange como string, mostrando solo términos con coeficientes no nulos
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


def interpolate_function(func_str, x_points):
    """
    Interpola una función usando polinomio de Lagrange
    
    Args:
        func_str: String de la función a interpolar
        x_points: Puntos x para la interpolación
        
    Returns:
        LagrangePolynomial: Objeto del polinomio de Lagrange
    """
    # Crear función
    f = create_function_from_string(func_str)
    
    # Evaluar función en los puntos dados
    y_points = [f(x) for x in x_points]
    
    # Crear polinomio de Lagrange
    lagrange_poly = LagrangePolynomial(x_points, y_points)
    
    return lagrange_poly


def approximate_nth_derivative(func_str, x_points, x_eval, n):
    """
    Aproxima la n-ésima derivada de una función usando polinomio de Lagrange
    
    Args:
        func_str: String de la función
        x_points: Puntos para construir el polinomio
        x_eval: Puntos donde evaluar la derivada
        n: Orden de la derivada
        
    Returns:
        Valores de la n-ésima derivada aproximada
    """
    lagrange_poly = interpolate_function(func_str, x_points)
    return lagrange_poly.evaluate_nth_derivative(x_eval, n)

def ejercicio1():
    points = [(-2, -6), (-1, -1/8), (0,3), (1, 33/8)]
    x_points = [x for x, _ in points]
    y_points = [y for _, y in points]
    lagrange_poly = LagrangePolynomial(x_points, y_points)
    print(f"P(x) = {lagrange_poly.get_polynomial_string()}")
    print(f"P'(x) = {lagrange_poly.get_nth_derivative_string(1)}")

    x_eval = 1
    
    print(f"P'({x_eval}) = {lagrange_poly.evaluate_nth_derivative(x_eval, 1)}")

def ejercicio2():
    points = [(-1, 4.35), (-0.5, 2.0390625), (0, 1), (0.2, 0.8782), (1, 1.875)]
    x_points = [x for x, _ in points]
    y_points = [y for _, y in points]
    lagrange_poly = LagrangePolynomial(x_points, y_points)
    
    print("=== Polinomio de Lagrange ===")
    print(f"Puntos: {points}")
    print()
    
    # Obtener polinomio como string
    poly_str = lagrange_poly.get_polynomial_string()
    print("P(x) =", poly_str)
    print()

    # Obtener n derivada como string
    n = 2
    deriv_n_str = lagrange_poly.get_nth_derivative_string(n)
    print(f"P^({n})(x) = {deriv_n_str}")
    print()

    # Evaluar n derivada en un punto
    x_eval = 2
    deriv_n_x = lagrange_poly.evaluate_nth_derivative(x_eval, n)
    print(f"P^({n})({x_eval}) = {deriv_n_x}")
    print()

def ejercicio3():
    points = [(-2, 9), (-1, 16), (0, 17), (1, 18), (3, 44), (4, 81)]
    x_points = [x for x, _ in points]
    y_points = [y for _, y in points]
    lagrange_poly = LagrangePolynomial(x_points, y_points)

    print("=== Polinomio de Lagrange ===")
    print(f"Puntos: {points}")
    print()

    # Obtener polinomio como string
    poly_str = lagrange_poly.get_polynomial_math_string()
    print("P(x) =", poly_str)
    print()

    # Obtener el grado del polinomio
    match = re.search(r'\*\*(\d+)', poly_str)
    deg = 0
    if match:
        deg = int(match.group(1))
    print(f"Grado del polinomio: {deg}")

# Ejemplos de uso
if __name__ == "__main__":
    ejercicio3()