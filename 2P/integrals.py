import numpy as np
from typing import Callable, Union


class NumericalIntegrator:
    """
    A class for numerical integration methods.
    
    This class provides various numerical methods to approximate definite integrals
    of a function f(x) from a to b.
    """
    
    def __init__(self, f: Callable[[float], float], a: float = 0, b: float = 1):
        """
        Initialize the integrator with a function and integration limits.
        
        Args:
            f: The function to integrate (must accept a float and return a float)
            a: Lower limit of integration (default: 0)
            b: Upper limit of integration (default: 1)
        """
        self.f = f
        self.a = a
        self.b = b
        
        if a >= b:
            raise ValueError("Lower limit 'a' must be less than upper limit 'b'")
    
    def _get_gauss_legendre_weights_and_nodes(self, n: int) -> tuple:
        """
        Get Gauss-Legendre weights and nodes for n-point quadrature.
        
        Args:
            n: Number of points (order of quadrature)
            
        Returns:
            Tuple of (weights, nodes) for the interval [-1, 1]
        """
        # Predefined weights and nodes for common orders
        # These are for the interval [-1, 1]
        gauss_data = {
            2: {
                'nodes': [-0.5773502691896257, 0.5773502691896257],
                'weights': [1.0, 1.0]
            },
            3: {
                'nodes': [-0.7745966692414834, 0.0, 0.7745966692414834],
                'weights': [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
            },
            4: {
                'nodes': [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
                'weights': [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
            },
            5: {
                'nodes': [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
                'weights': [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
            },
            6: {
                'nodes': [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521],
                'weights': [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
            },
            8: {
                'nodes': [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
                'weights': [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763]
            },
            10: {
                'nodes': [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717],
                'weights': [0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
            }
        }
        
        if n not in gauss_data:
            raise ValueError(f"Gauss-Legendre quadrature not available for n={n}. Available orders: {list(gauss_data.keys())}")
        
        return gauss_data[n]['nodes'], gauss_data[n]['weights']
    
    def gauss_legendre(self, n: int = 2) -> float:
        """
        Approximate integral using Gauss-Legendre quadrature.
        
        Args:
            n: Number of points (order of quadrature, default: 2)
                Available orders: 2, 3, 4, 5, 6, 8, 10
            
        Returns:
            Approximation of the integral
        """
        nodes, weights = self._get_gauss_legendre_weights_and_nodes(n)
        
        # Transform nodes from [-1, 1] to [a, b]
        # x = (b-a)/2 * t + (a+b)/2, where t is in [-1, 1]
        transformed_nodes = [(self.b - self.a) / 2 * t + (self.a + self.b) / 2 for t in nodes]
        
        # Evaluate function at transformed nodes
        y_values = [self.f(x) for x in transformed_nodes]
        
        # Apply Gauss-Legendre formula: (b-a)/2 * sum(wi * f(xi))
        result = 0.0
        for i in range(n):
            result += weights[i] * y_values[i]
        
        return (self.b - self.a) / 2 * result
    
    def left_rectangle(self, n: int = 1) -> float:
        """
        Approximate integral using left rectangle method.
        
        Args:
            n: Number of subintervals (default: 1)
            
        Returns:
            Approximation of the integral
        """
        h = (self.b - self.a) / n
        x_points = np.linspace(self.a, self.b - h, n)
        y_values = [self.f(x) for x in x_points]
        return h * sum(y_values)
    
    def right_rectangle(self, n: int = 1) -> float:
        """
        Approximate integral using right rectangle method.
        
        Args:
            n: Number of subintervals (default: 1)
            
        Returns:
            Approximation of the integral
        """
        h = (self.b - self.a) / n
        x_points = np.linspace(self.a + h, self.b, n)
        y_values = [self.f(x) for x in x_points]
        return h * sum(y_values)
    
    def midpoint_rectangle(self, n: int = 1) -> float:
        """
        Approximate integral using midpoint rectangle method.
        
        Args:
            n: Number of subintervals (default: 1)
            
        Returns:
            Approximation of the integral
        """
        h = (self.b - self.a) / n
        x_points = np.linspace(self.a + h/2, self.b - h/2, n)
        y_values = [self.f(x) for x in x_points]
        return h * sum(y_values)
    
    def trapezium_rule(self, n: int = 1) -> float:
        """
        Approximate integral using trapezium rule.
        
        Args:
            n: Number of subintervals (default: 1)
            
        Returns:
            Approximation of the integral
        """
        h = (self.b - self.a) / n
        x_points = np.linspace(self.a, self.b, n + 1)
        y_values = [self.f(x) for x in x_points]
        
        # Apply trapezium rule: h/2 * (f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(xn-1) + f(xn))
        return h/2 * (y_values[0] + 2*sum(y_values[1:-1]) + y_values[-1])
    
    def simpson_one_third(self, n: int = 2) -> float:
        """
        Approximate integral using Simpson's 1/3 rule.
        
        Args:
            n: Number of subintervals (must be even, default: 2)
            
        Returns:
            Approximation of the integral
        """
        if n % 2 != 0:
            n += 1  # Ensure n is even
            
        h = (self.b - self.a) / n
        x_points = np.linspace(self.a, self.b, n + 1)
        y_values = [self.f(x) for x in x_points]
        
        # Apply Simpson's 1/3 rule: h/3 * (f(x0) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + 4*f(xn-1) + f(xn))
        result = y_values[0] + y_values[-1]  # First and last terms
        
        # Add terms with coefficient 4 (odd indices)
        for i in range(1, n, 2):
            result += 4 * y_values[i]
            
        # Add terms with coefficient 2 (even indices, excluding first and last)
        for i in range(2, n, 2):
            result += 2 * y_values[i]
            
        return h/3 * result
    
    def simpson_three_eighths(self, n: int = 3) -> float:
        """
        Approximate integral using Simpson's 3/8 rule.
        
        Args:
            n: Number of subintervals (must be divisible by 3, default: 3)
            
        Returns:
            Approximation of the integral
        """
        if n % 3 != 0:
            n = (n // 3) * 3  # Adjust n to be divisible by 3
            
        h = (self.b - self.a) / n
        x_points = np.linspace(self.a, self.b, n + 1)
        y_values = [self.f(x) for x in x_points]
        
        # Apply Simpson's 3/8 rule: 3h/8 * (f(x0) + 3*f(x1) + 3*f(x2) + 2*f(x3) + ... + 3*f(xn-2) + 3*f(xn-1) + f(xn))
        result = y_values[0] + y_values[-1]  # First and last terms
        
        # Add terms with coefficient 3 (indices not divisible by 3, excluding first and last)
        for i in range(1, n):
            if i % 3 != 0:
                result += 3 * y_values[i]
                
        # Add terms with coefficient 2 (indices divisible by 3, excluding first and last)
        for i in range(3, n, 3):
            result += 2 * y_values[i]
            
        return 3*h/8 * result
    
    def compare_methods(self, n: int = 1) -> dict:
        """
        Compare all integration methods for the same number of subintervals.
        
        Args:
            n: Number of subintervals (default: 1)
            
        Returns:
            Dictionary with results from all methods
        """
        results = {
            'left_rectangle': self.left_rectangle(n),
            'right_rectangle': self.right_rectangle(n),
            'midpoint_rectangle': self.midpoint_rectangle(n),
            'trapezium_rule': self.trapezium_rule(n),
            'simpson_one_third': self.simpson_one_third(2*n),
            'simpson_three_eighths': self.simpson_three_eighths(3*n),
            'gauss_legendre': self.gauss_legendre(2*n)
        }
        return results

def simpson_rule_from_data(y_values: list[float], h: float) -> float:
    """
    Approximate integral using Simpson's 1/3 rule from a list of y values.

    Args:
        y_values: List of function values f(x_i)
        h: Step size between x values (must be constant)

    Returns:
        Approximation of the integral
    """
    n = len(y_values) - 1
    if n < 1:
        return 0.0

    if n % 2 != 0:
        raise ValueError(
            "Simpson's 1/3 rule requires an even number of intervals "
            f"(received {n}, which is odd)."
        )

    # Apply Simpson's 1/3 rule: h/3 * (y0 + 4*y1 + 2*y2 + ... + 4*yn-1 + yn)
    result = y_values[0] + y_values[-1]  # First and last terms

    # Add terms with coefficient 4 (odd indices)
    for i in range(1, n, 2):
        result += 4 * y_values[i]

    # Add terms with coefficient 2 (even indices)
    for i in range(2, n, 2):
        result += 2 * y_values[i]

    return (h / 3) * result

def test1():
    # Example function: f(x) = x^2
    def f(x):
        return x**2
    
    # Create integrator for f(x) = x^2 from 0 to 1
    # The exact integral is 1/3
    integrator = NumericalIntegrator(f, 0, 1)
    
    print("Integrating f(x) = x^2 from 0 to 1")
    print("Exact value: 1/3 ≈ 0.333333...")
    print()
    
    # Compare all methods
    results = integrator.compare_methods()
    
    for method, result in results.items():
        print(f"{method.replace('_', ' ').title()}: {result:.8f}")
    
    print("\n" + "="*50)
    
    # Another example: f(x) = sin(x) from 0 to π
    def g(x):
        return np.sin(x)
    
    integrator2 = NumericalIntegrator(g, 0, np.pi)
    print("Integrating f(x) = sin(x) from 0 to π")
    print("Exact value: 2")
    print()
    
    results2 = integrator2.compare_methods()
    
    for method, result in results2.items():
        print(f"{method.replace('_', ' ').title()}: {result:.8f}")

def ejercicio1():
    def f(x):
        return 0.2 + 25*x + 3*x**2 + 2*x**2

    integrator = NumericalIntegrator(f, 0, 1)
    result = integrator.simpson_one_third(4)

    print(result)

def ejercicio2():
    # Data from the exercise image
    # x = [-4, -2, 0, 2, 4, 6, 8]
    fx = [1, 3, 4, 4, 6, 9, 14]
    
    # The integration interval is from a = -4 to b = 8.
    # The number of subintervals is n = len(fx) - 1 = 6.
    # The step size is h = (b - a) / n = (8 - (-4)) / 6 = 2.
    h = 2
    
    integral_value = simpson_rule_from_data(fx, h)
    print(integral_value)

if __name__ == "__main__":
    ejercicio1()
    ejercicio2()