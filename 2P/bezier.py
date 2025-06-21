import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

def binomial_coefficient(n: int, k: int) -> int:
    """
    Calculate the binomial coefficient C(n,k)
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use Pascal's triangle method for efficiency
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result

def bernstein_polynomial(n: int, k: int, t: float) -> float:
    """
    Calculate the k-th Bernstein polynomial of degree n at parameter t
    B_k^n(t) = C(n,k) * t^k * (1-t)^(n-k)
    """
    return binomial_coefficient(n, k) * (t ** k) * ((1 - t) ** (n - k))

def bezier_curve_point(control_points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """
    Evaluate a Bézier curve at parameter t (0 <= t <= 1)
    
    Args:
        control_points: List of (x, y) control points
        t: Parameter value between 0 and 1
    
    Returns:
        (x, y) point on the Bézier curve
    """
    if not control_points:
        raise ValueError("Control points list cannot be empty")
    
    n = len(control_points) - 1  # Degree of the curve
    x, y = 0.0, 0.0
    
    for k, (px, py) in enumerate(control_points):
        bernstein = bernstein_polynomial(n, k, t)
        x += px * bernstein
        y += py * bernstein
    
    return (x, y)

def bezier_curve_evaluate(control_points: List[Tuple[float, float]], t_values: List[float]) -> List[Tuple[float, float]]:
    """
    Evaluate a Bézier curve at multiple parameter values
    
    Args:
        control_points: List of (x, y) control points
        t_values: List of parameter values between 0 and 1
    
    Returns:
        List of (x, y) points on the Bézier curve
    """
    return [bezier_curve_point(control_points, t) for t in t_values]

def find_y_for_x(control_points: List[Tuple[float, float]], target_x: float, tolerance: float = 1e-6, max_iterations: int = 100) -> Optional[float]:
    """
    Find the y-value on the Bézier curve for a given x-value using binary search
    
    Args:
        control_points: List of (x, y) control points
        target_x: The x-value to find the corresponding y-value for
        tolerance: Tolerance for convergence
        max_iterations: Maximum number of iterations for binary search
    
    Returns:
        y-value if found, None if no solution exists
    """
    # Get the x-range of the curve
    x_coords = [p[0] for p in control_points]
    min_x, max_x = min(x_coords), max(x_coords)
    
    # Check if target_x is within the curve's x-range
    if target_x < min_x or target_x > max_x:
        return None
    
    # Binary search on the parameter t
    t_left, t_right = 0.0, 1.0
    
    for _ in range(max_iterations):
        t_mid = (t_left + t_right) / 2
        x_mid, _ = bezier_curve_point(control_points, t_mid)
        
        if abs(x_mid - target_x) < tolerance:
            _, y_mid = bezier_curve_point(control_points, t_mid)
            return y_mid
        
        if x_mid < target_x:
            t_left = t_mid
        else:
            t_right = t_mid
    
    # If we reach here, return the best approximation
    t_final = (t_left + t_right) / 2
    _, y_final = bezier_curve_point(control_points, t_final)
    return y_final

def plot_bezier_curve(control_points: List[Tuple[float, float]], num_points: int = 100, show_control_polygon: bool = True):
    """
    Plot the Bézier curve and optionally the control polygon
    
    Args:
        control_points: List of (x, y) control points
        num_points: Number of points to generate for the curve
        show_control_polygon: Whether to show the control polygon
    """
    # Generate curve points
    t_values = np.linspace(0, 1, num_points)
    curve_points = bezier_curve_evaluate(control_points, t_values)
    
    # Extract x and y coordinates
    x_curve = [p[0] for p in curve_points]
    y_curve = [p[1] for p in curve_points]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot the curve
    plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Bézier Curve')
    
    # Plot control points and polygon
    if show_control_polygon:
        x_control = [p[0] for p in control_points]
        y_control = [p[1] for p in control_points]
        plt.plot(x_control, y_control, 'r--', alpha=0.7, label='Control Polygon')
        plt.plot(x_control, y_control, 'ro', markersize=8, label='Control Points')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bézier Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

def main():
    """
    Example usage of the Bézier curve functions
    """
    # Example control points for a quadratic Bézier curve
    control_points = [(0, 0), (2, 4), (4, 0)]
    
    print("Control Points:", control_points)
    
    # Evaluate at specific parameter values
    t_values = [0, 0.25, 0.5, 0.75, 1.0]
    curve_points = bezier_curve_evaluate(control_points, t_values)
    
    print("\nCurve points at different t values:")
    for t, (x, y) in zip(t_values, curve_points):
        print(f"t={t}: ({x:.3f}, {y:.3f})")
    
    # Find y-value for a specific x-value
    target_x = 2.0
    y_value = find_y_for_x(control_points, target_x)
    
    if y_value is not None:
        print(f"\nFor x = {target_x}, y = {y_value:.3f}")
    else:
        print(f"\nNo y-value found for x = {target_x}")
    
    # Plot the curve
    plot_bezier_curve(control_points)

if __name__ == "__main__":
    main()
