import numpy as np
import matplotlib.pyplot as plt

"""
Author: Santi
Date: 2025
Description: Implementación de métodos numéricos para resolver problemas de valor inicial
             Incluye métodos de Euler, Euler Modificado, Heun, RK2 y RK4
"""


def rk2(func, start, end, a, b, p, q, y0, N):
    """
    Implementación del método RK2 genérico.
    
    Args:
        func: Función que define la EDO dy/dt = f(t,y)
        start: Tiempo inicial
        end: Tiempo final
        a, b: Coeficientes del método
        p, q: Parámetros para calcular k2
        y0: Condición inicial
        N: Número de pasos
        
    Returns:
        Lista de puntos (t,y) de la solución
    """
    yk = np.array([y0, 0]) if not hasattr(y0, '__len__') else np.array(y0)
    tk = start
    h = (end-start)/N
    points = [(tk,yk)]
    for k in range(N):
        k1 = h*func(tk, yk)
        k2 = h*func(tk + p*h, yk + q*k1)
        yk = yk + a*k1 + b*k2
        tk = tk + h
        points.append((tk,yk))
    return points

def euler(func, start, end, y0, N):
    """
    Implementación del método de Euler.
    
    Args:
        func: Función que define la EDO dy/dt = f(t,y)
        start: Tiempo inicial
        end: Tiempo final
        y0: Condición inicial
        N: Número de pasos
        
    Returns:
        Lista de puntos (t,y) de la solución
    """
    yk = np.array([y0, 0]) if not hasattr(y0, '__len__') else np.array(y0)
    tk = start
    h = (end-start)/N
    points = [(tk,yk)]
    for k in range(N):
        yk = yk + h*func(tk,yk)
        tk = tk + h
        points.append((tk,yk))
    return points

def modified_euler(func, start, end, y0, N):
    """
    Implementación del método de Euler Modificado (RK2 con parámetros específicos).
    Este método usa el punto medio para calcular la pendiente.
    
    Args:
        func: Función que define la EDO dy/dt = f(t,y)
        start: Tiempo inicial
        end: Tiempo final
        y0: Condición inicial
        N: Número de pasos
        
    Returns:
        Lista de puntos (t,y) de la solución
    """
    a = 0
    b = 1
    p = 0.5
    q = 0.5
    return rk2(func, start, end, a, b, p, q, y0, N)

def heun(func, start, end, y0, N):
    """
    Implementación del método de Heun (RK2 con parámetros específicos).
    
    Args:
        func: Función que define la EDO dy/dt = f(t,y)
        start: Tiempo inicial
        end: Tiempo final
        y0: Condición inicial
        N: Número de pasos
        
    Returns:
        Lista de puntos (t,y) de la solución
    """
    a = 0.5
    b = 0.5
    p = 1
    q = 1
    return rk2(func, start, end, a, b, p, q, y0, N)

def derivative_t(func_ty,t,y):
    h = 1e-8
    return (func_ty(t+h,y)-func_ty(t,y))/h

def derivative_y(func_ty,t,y):
    h = 1e-8
    return (func_ty(t,y+h)-func_ty(t,y))/h

def taylor(func, start, end, y0, N):
    yk = np.array([y0, 0]) if not hasattr(y0, '__len__') else np.array(y0)
    tk = start
    h = (end-start)/N
    points = [(tk,yk)]
    for k in range(N):
        fp = derivative_t(func,tk,yk) + derivative_y(func,tk,yk)*func(tk,yk)
        yk = yk + h*func(tk,yk) + (h*h)*0.5*fp
        tk = tk + h
        points.append((tk,yk))
    return points

def rk4(func, start, end, y0, N):
    """
    Implementación del método RK4.
    
    Args:
        func: Función que define la EDO dy/dt = f(t,y)
        start: Tiempo inicial
        end: Tiempo final
        y0: Condición inicial
        N: Número de pasos
        
    Returns:
        Lista de puntos (t,y) de la solución
    """
    yk = np.array([y0, 0]) if not hasattr(y0, '__len__') else np.array(y0)
    tk = start
    h = (end-start)/N
    points = [(tk,yk)]
    for k in range(N):
        k1 = func(tk, yk)
        k2 = func(tk + 0.5*h, yk + 0.5*h*k1)
        k3 = func(tk + 0.5*h, yk + 0.5*h*k2)
        k4 = func(tk + h, yk + h*k3)
        yk = yk + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        tk = tk + h
        points.append((tk,yk))
    return points

def rk_osc(method, f, K, t0, y0, v0, N):
    b = t0 + osc_to_x(K)
    return method(f, t0, b, [y0,v0], N)

def osc_to_x(k):
    return k * 2 * np.pi / w

def exact_solution_f(t):
    return 2 * np.cos(w*t)

def exact_solution_g(t):
    return 3*np.exp(2*t)

def exact_derivative_f(t):
    return -2*w*np.sin(w*t)

def exact_derivative_g(t):
    return 6*np.exp(2*t)

def exact_derivative_approx(solution_func, t):
    h = 1e-9
    return (solution_func(t+h) - solution_func(t)) / h

methodNames = {
    heun: "Heun's Method (RK2)",
    rk4: "Runge-Kutta 4th Order Method",
    euler: "Euler's Method",
    modified_euler: "Modified Euler's Method",
    taylor: "Taylor's Method"
}

def plot(range,points, exact_solution, error_points, yRange=None, title="Runge-Kutta Method"):
    a, b = range
    t_values, y_values = [p[0] for p in points], [p[1][0] for p in points]
    t_error, y_error = [p[0] for p in error_points], [p[1][0] for p in error_points]

    if exact_solution is not None:
        t_exact = np.linspace(a, b, 1000)
        y_exact = [exact_solution(t) for t in t_exact]
        plt.plot(t_exact, y_exact, label="Exact", color="red", linestyle='--')

    plt.plot(t_values, y_values, label="Numerical", color="blue")
    plt.plot(t_error, y_error, label="Error check", color="green", linestyle='--')
    plt.xlim(a, b)
    if yRange is not None: plt.ylim(yRange)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

def solve_f(method, startX, k, N, plotRange=None, yRange=None,plotExact=False,y0=None,v0=None):
    print("Solving f")
    endX = startX + k * 2 * np.pi/w
    if y0 is None: y0 = exact_solution_f(a)
    if v0 is None: v0 = exact_derivative_f(a)
    points = rk_osc(method, f, k, startX, y0, v0, N)
    
    #calculate error
    y0bar = points[-1][1][0]
    v0bar = points[-1][1][1]
    t0bar = points[-1][0]
    error_points = rk_osc(method, f, -k, t0bar, y0bar, v0bar, N)
    error = abs(points[0][1][0] - error_points[-1][1][0])
    print(f"Global error E={error}")

    #plot
    if plotRange is None:
        plotRange = (startX, endX)
    plot(plotRange,points,(exact_solution_f if plotExact else None),error_points,yRange,methodNames[method])
    return points

def solve_g(method, startX, endX, N, plotRange=None,yRange=None,plotExact=False,y0=None):
    print("Solving g")
    if y0 is None: y0 = exact_solution_g(startX)
    points = method(g, startX, endX, np.array([y0,0]), N)
    
    #calculate error
    y0bar = points[-1][1][0]
    error_points = method(g, b, a, y0bar, N)
    error = abs(points[0][1][0] - error_points[-1][1][0])
    print(f"Global error E={error}")

    #plot
    if plotRange is None:
        plotRange = (startX, endX)
    plot(plotRange,points,(exact_solution_g if plotExact else None),error_points,yRange,methodNames[method])
    return points

def evaluate(points,t):
    return min(points, key=lambda p: abs(p[0]-t))[1][0]

#oscillator
def f(t,y):
    y1,y2 = y
    return np.array([y2, -(w**2)*y1])

#first order (y' = g(t,y))
def g(t,y):
    # return (1+2*t)*(np.sqrt(y))
    # return t + 1 - y
    # return 2*np.exp(-5*t) -(9/10)*y
    # return 1 + y**2
    return (t**3)*y -1.25*y 

w = 2
# h = (b-a)/N
N = 8
a = 0
b = 2
y0 = 1
point = 2

k = 10 # oscillator

#solve_f(heun, a, k, N)
#solve_f(rk4, a, k, N)

while True:
    method = input("Enter 'e' for Euler, 'm' for Modified Euler, 'h' for Heun, or 't' for Taylor: ").lower()
    if method == 'e':
        points = solve_g(euler, a, b, N, y0=y0)
        break
    elif method == 'm':
        points = solve_g(modified_euler, a, b, N, y0=y0)
        break
    elif method == 'h':
        points = solve_g(heun, a, b, N, y0=y0)
        break
    elif method == 't':
        points = solve_g(taylor, a, b, N, y0=y0)
        break
    else:
        print("Invalid method. Please enter 'e' for Euler, 'm' for Modified Euler, 'h' for Heun, or 't' for Taylor.")

print(evaluate(points,point)) # evaluate g(point)