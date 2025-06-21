# Bézier Curve Implementation

This module provides a complete implementation of Bézier curves with functions to evaluate curves at given points and find y-values for specific x-values.

## Features

- **Bézier curve evaluation**: Calculate points on a Bézier curve for given parameter values
- **X-to-Y mapping**: Find the y-value on a curve for a given x-value using binary search
- **Visualization**: Plot Bézier curves with control points and control polygon
- **Support for any degree**: Works with linear, quadratic, cubic, and higher-degree Bézier curves

## Functions

### `bezier_curve_point(control_points, t)`

Evaluates a Bézier curve at parameter `t` (0 ≤ t ≤ 1).

**Parameters:**

- `control_points`: List of (x, y) control points
- `t`: Parameter value between 0 and 1

**Returns:**

- `(x, y)`: Point on the Bézier curve

### `bezier_curve_evaluate(control_points, t_values)`

Evaluates a Bézier curve at multiple parameter values.

**Parameters:**

- `control_points`: List of (x, y) control points
- `t_values`: List of parameter values between 0 and 1

**Returns:**

- List of `(x, y)` points on the Bézier curve

### `find_y_for_x(control_points, target_x, tolerance=1e-6, max_iterations=100)`

Finds the y-value on the Bézier curve for a given x-value using binary search.

**Parameters:**

- `control_points`: List of (x, y) control points
- `target_x`: The x-value to find the corresponding y-value for
- `tolerance`: Tolerance for convergence (default: 1e-6)
- `max_iterations`: Maximum number of iterations for binary search (default: 100)

**Returns:**

- `y_value`: y-value if found, `None` if no solution exists

### `plot_bezier_curve(control_points, num_points=100, show_control_polygon=True)`

Plots the Bézier curve and optionally the control polygon.

**Parameters:**

- `control_points`: List of (x, y) control points
- `num_points`: Number of points to generate for the curve (default: 100)
- `show_control_polygon`: Whether to show the control polygon (default: True)

## Usage Examples

### Basic Usage

```python
from bezier import bezier_curve_point, find_y_for_x

# Define control points for a quadratic Bézier curve
control_points = [(0, 0), (2, 4), (4, 0)]

# Evaluate at t = 0.5
x, y = bezier_curve_point(control_points, 0.5)
print(f"Point at t=0.5: ({x}, {y})")

# Find y-value for x = 2
y_value = find_y_for_x(control_points, 2.0)
print(f"y-value for x=2: {y_value}")
```

### Different Curve Types

```python
# Linear Bézier curve (2 control points)
linear_points = [(0, 0), (4, 2)]

# Quadratic Bézier curve (3 control points)
quadratic_points = [(0, 0), (2, 4), (4, 0)]

# Cubic Bézier curve (4 control points)
cubic_points = [(0, 0), (1, 3), (3, 3), (4, 0)]

# Higher-degree curve (5 control points)
complex_points = [(0, 0), (1, 2), (2, -1), (3, 3), (4, 0)]
```

### Visualization

```python
from bezier import plot_bezier_curve

# Plot a curve with control points
control_points = [(0, 0), (2, 4), (4, 0)]
plot_bezier_curve(control_points)
```

## Mathematical Background

A Bézier curve of degree n is defined by n+1 control points P₀, P₁, ..., Pₙ and is given by:

B(t) = Σᵏ₌₀ⁿ C(n,k) × tᵏ × (1-t)ⁿ⁻ᵏ × Pₖ

where:

- C(n,k) is the binomial coefficient
- t is the parameter (0 ≤ t ≤ 1)
- Pₖ are the control points

The curve starts at P₀ (t=0) and ends at Pₙ (t=1), with intermediate control points influencing the shape of the curve.

## Dependencies

- `numpy`: For numerical operations
- `matplotlib`: For plotting functionality

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running Examples

Run the main example:

```bash
python bezier.py
```
