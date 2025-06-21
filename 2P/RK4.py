"""
Runge-Kutta Method of Order 4 (RK4) for Initial Value Problems

This module implements the RK4 method to solve initial value problems
of the form: dy/dt = f(t, y), y(t0) = y0

The RK4 method is a fourth-order numerical method that provides
good accuracy for most practical problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple, List
import warnings


class RK4Solver:
    """
    A class to solve initial value problems using the Runge-Kutta method of order 4.
    
    The RK4 method uses four evaluations of the derivative function per step:
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h, y_n + h*k3)
    
    Then: y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def __init__(self, step_size: float = 0.1):
        """
        Initialize the RK4 solver.
        
        Args:
            step_size (float): The step size for the numerical integration.
                              Smaller values provide better accuracy but require more computation.
        """
        self.step_size = step_size
        self.solution = None
        self.t_values = None
        self.y_values = None
    
    def solve(self, 
              f: Callable[[float, Union[float, np.ndarray]], Union[float, np.ndarray]], 
              t_span: Tuple[float, float], 
              y0: Union[float, np.ndarray],
              step_size: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the initial value problem using RK4 method.
        
        Args:
            f: The derivative function f(t, y) that defines the ODE dy/dt = f(t, y)
            t_span: Tuple (t_start, t_end) defining the time interval
            y0: Initial condition y(t_start)
            step_size: Optional step size override
            
        Returns:
            Tuple of (t_values, y_values) arrays containing the solution
        """
        if step_size is not None:
            self.step_size = step_size
            
        t_start, t_end = t_span
        
        if t_start >= t_end:
            raise ValueError("t_start must be less than t_end")
            
        # Calculate number of steps
        n_steps = int((t_end - t_start) / self.step_size) + 1
        
        # Initialize arrays
        t_values = np.linspace(t_start, t_end, n_steps)
        y_values = np.zeros((n_steps,))
        
        # Set initial condition
        y_values[0] = y0
        
        # RK4 integration
        for i in range(n_steps - 1):
            t = t_values[i]
            y = y_values[i]
            h = self.step_size
            
            # RK4 coefficients
            k1 = f(t, y)
            k2 = f(t + h/2, y + h*k1/2)
            k3 = f(t + h/2, y + h*k2/2)
            k4 = f(t + h, y + h*k3)
            
            # Update solution
            y_values[i + 1] = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        self.solution = (t_values, y_values)
        self.t_values = t_values
        self.y_values = y_values
        
        return t_values, y_values
    
    def solve_system(self, 
                     f: Callable[[float, np.ndarray], np.ndarray], 
                     t_span: Tuple[float, float], 
                     y0: np.ndarray,
                     step_size: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a system of ODEs using RK4 method.
        
        Args:
            f: The derivative function f(t, y) that returns a vector
            t_span: Tuple (t_start, t_end) defining the time interval
            y0: Initial condition vector y(t_start)
            step_size: Optional step size override
            
        Returns:
            Tuple of (t_values, y_values) where y_values is a 2D array
        """
        if step_size is not None:
            self.step_size = step_size
            
        t_start, t_end = t_span
        
        if t_start >= t_end:
            raise ValueError("t_start must be less than t_end")
            
        # Ensure y0 is a numpy array
        y0 = np.array(y0, dtype=float)
        n_vars = len(y0)
        
        # Calculate number of steps
        n_steps = int((t_end - t_start) / self.step_size) + 1
        
        # Initialize arrays
        t_values = np.linspace(t_start, t_end, n_steps)
        y_values = np.zeros((n_steps, n_vars))
        
        # Set initial condition
        y_values[0] = y0
        
        # RK4 integration
        for i in range(n_steps - 1):
            t = t_values[i]
            y = y_values[i]
            h = self.step_size
            
            # RK4 coefficients
            k1 = f(t, y)
            k2 = f(t + h/2, y + h*k1/2)
            k3 = f(t + h/2, y + h*k2/2)
            k4 = f(t + h, y + h*k3)
            
            # Update solution
            y_values[i + 1] = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        self.solution = (t_values, y_values)
        self.t_values = t_values
        self.y_values = y_values
        
        return t_values, y_values


def solve_ivp_rk4(f: Callable[[float, Union[float, np.ndarray]], Union[float, np.ndarray]], 
                  t_span: Tuple[float, float], 
                  y0: Union[float, np.ndarray],
                  step_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to solve an initial value problem using RK4.
    
    Args:
        f: The derivative function f(t, y)
        t_span: Tuple (t_start, t_end)
        y0: Initial condition
        step_size: Step size for integration
        
    Returns:
        Tuple of (t_values, y_values)
    """
    solver = RK4Solver(step_size)
    return solver.solve(f, t_span, y0)

def example_oscillator():
    """Example: Simple harmonic oscillator d²y/dt² + ω²y = 0"""
    print("Example 3: Simple Harmonic Oscillator")
    
    omega = 2.0  # angular frequency
    
    def f(t, y):
        y_pos, y_vel = y
        return np.array([y_vel, -omega**2 * y_pos])
    
    # Initial conditions: y(0) = 1, y'(0) = 0
    solver = RK4Solver(step_size=0.01)
    t_values, y_values = solver.solve_system(f, (0, 10), [1, 0])
    
    # Exact solution: y(t) = cos(ωt)
    exact_position = np.cos(omega * t_values)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_values, y_values[:, 0], 'b-', linewidth=2, label='RK4 Position')
    plt.plot(t_values, exact_position, 'r--', linewidth=2, label='Exact Position')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Harmonic Oscillator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_values[:, 0], y_values[:, 1], 'g-', linewidth=1)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Space')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def ejercicio1():
    """Exercise 1: Solve y' = sqrt(t+y) with y(0.4) = 0.41, h=0.2, find y(0.8)"""
    # print("Exercise 1: y' = sqrt(t+y)")
    # print("Initial condition: y(0.4) = 0.41")
    # print("Step size: h = 0.2")
    # print("Find: y(0.8)")
    # print()
    
    def f(t, y):
        return np.sqrt(t + y)
    
    # Initial conditions
    t0 = 0.4
    y0 = 0.41
    h = 0.2
    t_target = 0.8
    
    # Create solver and solve
    solver = RK4Solver(step_size=h)
    t_values, y_values = solver.solve(f, (t0, t_target), y0)
    
    # Get the final value (y(0.8))
    final_y = y_values[-1]
    
    print(f"Solución: y({t_target}) = {final_y}")
    print()
    
    # # Plot the solution
    # plt.figure(figsize=(10, 6))
    # plt.plot(t_values, y_values, 'b-', linewidth=2, label='RK4 Solution')
    # plt.plot(t0, y0, 'ro', markersize=8, label='Initial Point')
    # plt.plot(t_target, final_y, 'go', markersize=8, label='Target Point')
    # plt.xlabel('Time (t)')
    # plt.ylabel('y(t)')
    # plt.title('Solution of y\' = sqrt(t+y) using RK4')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()
    
    return final_y
    

if __name__ == "__main__":
    print("Runge-Kutta Method of Order 4 (RK4) Implementation")
    print("=" * 50)
        
    # Run examples
    # example_oscillator()
    ejercicio1()
    
    print("\nAll examples completed!")
