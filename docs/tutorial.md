# Tutorial rapido para usar los metodos

Este tutorial indica que archivo de `codigos_andy` usar para cada metodo de la guia y que datos ingresar o editar.

Para correr cualquier archivo, ubicarse en la carpeta del proyecto:

```bash
cd /Users/rodri/ITBA/metodos
```

## 1. Ecuaciones No Lineales

Estos metodos buscan raices de una funcion:

```text
f(x) = 0
```

### 1.1 Metodo de Biseccion

Archivo:

```bash
codigos_andy/codigos_viejos/biseccion.py
```

Comando:

```bash
python3 codigos_andy/codigos_viejos/biseccion.py
```

Que ingresar:

```text
f(x) = funcion original
a = extremo izquierdo
b = extremo derecho
t = si queres cortar por tolerancia
p = si queres cortar por cantidad de pasos
```

Condicion importante:

```text
f(a) y f(b) tienen que tener signos opuestos
```

Ejemplo:

```text
f(x) = x^2 - 2
a = 1
b = 2
t
e = 0.000001
```

El error que pide es la tolerancia maxima. Si el enunciado no da tolerancia, `0.000001` es una eleccion razonable para obtener varias cifras decimales.

### 1.2 Metodo de Newton

Archivo recomendado:

```bash
codigos_andy/codigos_buenos/newton.py
```

Comando:

```bash
python3 codigos_andy/codigos_buenos/newton.py
```

Que ingresar:

```text
f(x) = funcion original
a = extremo izquierdo del intervalo
b = extremo derecho del intervalo
t = si queres cortar por tolerancia
i = si queres cortar por iteraciones
```

Ejemplo:

```text
f(x) = x^2 - 2
a = 1
b = 2
t
tolerancia = 0.000001
```

El codigo calcula automaticamente `f'(x)` y `f''(x)`.

Condiciones que chequea el codigo:

```text
f(a) y f(b) deben tener signos opuestos
f'(x) debe tener signo constante en [a,b]
f''(x) debe tener signo constante en [a,b]
```

### 1.3 Metodo de los Puntos Fijos

Archivo:

```bash
codigos_andy/codigos_viejos/puntos_fijos.py
```

Comando:

```bash
python3 codigos_andy/codigos_viejos/puntos_fijos.py
```

Importante: aunque el programa pide `f(x)`, en realidad tenes que ingresar `g(x)`.

El metodo no usa directamente:

```text
f(x) = 0
```

Primero tenes que despejar:

```text
x = g(x)
```

Despues ingresas esa `g(x)`.

Que ingresar:

```text
Ingrese la funcion f(x): g(x)
a = extremo inferior
b = extremo superior
e = si queres cortar por error
p = si queres cortar por pasos
```

Ejemplo:

Si:

```text
f(x) = x^2 - 2
```

Un despeje posible es:

```text
x = 2/x
```

Entonces en el programa se ingresa:

```text
Ingrese la funcion f(x): 2/x
a = 1
b = 2
e
error maximo = 0.000001
```

Condicion importante:

```text
Conviene verificar que |g'(x)| < 1 cerca de la raiz.
```

Si no se cumple, puede no converger.

El error que muestra el codigo es:

```text
Error_n = |x_n - x_(n-1)|
```

## 2. Ecuaciones Diferenciales Ordinarias

Estos metodos resuelven problemas de valor inicial:

```text
y' = f(t,y)
y(t0) = y0
```

Archivo para Euler, Taylor, Heun y Runge-Kutta 4:

```bash
codigos_andy/codigos_viejos/pvi.py
```

Comando:

```bash
python3 codigos_andy/codigos_viejos/pvi.py
```

Este archivo esta hardcodeado. Para cada ejercicio tenes que editar la funcion y los datos.

### Datos que hay que cambiar en PVI

Primero, editar la funcion `g(t,y)`:

```python
def g(t,y):
    return ...
```

Ahi va la ecuacion diferencial:

```text
y' = f(t,y)
```

Ejemplo:

Si el enunciado dice:

$$
y' = (t - y) / 2
$$

poner:

```python
def g(t,y):
    return (t - y) / 2
```

Despues editar:

```python
N = ...
a = ...
b = ...
y0 = ...
point = ...
```

Significado:

```text
a = t inicial
b = t final
y0 = condicion inicial
point = punto donde queres evaluar la solucion
N = cantidad de pasos
```

Si el enunciado da el paso `h`, calcular:

```text
N = (b - a) / h
```

Ejemplo:

```text
t en [0,3], h = 0.5
N = (3 - 0) / 0.5 = 6
```

Entonces:

```python
N = 6
a = 0
b = 3
y0 = 1
point = 3
```

### 2.1 Metodo de Euler

Usar:

```bash
codigos_andy/codigos_viejos/pvi.py
```

Editar `g`, `N`, `a`, `b`, `y0` y `point`.

Correr:

```bash
python3 codigos_andy/codigos_viejos/pvi.py
```

Cuando pregunte:

```text
Enter 'e' for Euler, 'm' for Modified Euler, 'h' for Heun, or 't' for Taylor:
```

Ingresar:

```text
e
```

### 2.2 Metodo de Taylor de orden 2

Usar:

```bash
codigos_andy/codigos_viejos/pvi.py
```

Editar `g`, `N`, `a`, `b`, `y0` y `point`.

Correr:

```bash
python3 codigos_andy/codigos_viejos/pvi.py
```

Cuando pregunte el metodo, ingresar:

```text
t
```

Nota: este codigo aproxima numericamente las derivadas parciales que necesita Taylor:

```text
f_t(t,y)
f_y(t,y)
```

### 2.3 Metodo de Heun de orden 2

Usar:

```bash
codigos_andy/codigos_viejos/pvi.py
```

Editar `g`, `N`, `a`, `b`, `y0` y `point`.

Correr:

```bash
python3 codigos_andy/codigos_viejos/pvi.py
```

Cuando pregunte el metodo, ingresar:

```text
h
```

El mismo archivo tambien tiene Euler modificado. Si queres usarlo, ingresar:

```text
m
```

### 2.4 Metodo de Runge-Kutta de orden 4

Usar:

```bash
codigos_andy/codigos_viejos/pvi.py
```

La funcion `rk4` ya existe en el archivo, pero el menu interactivo actual no tiene opcion para RK4.

Para usar RK4, editar el final del archivo y reemplazar temporalmente el bloque del menu por:

```python
points = solve_g(rk4, a, b, N, y0=y0)
print(evaluate(points, point))
```

Antes de correr, igual tenes que editar:

```python
def g(t,y):
    return ...

N = ...
a = ...
b = ...
y0 = ...
point = ...
```

Despues correr:

```bash
python3 codigos_andy/codigos_viejos/pvi.py
```

## Resumen rapido

| Tema | Metodo | Archivo | Como se usa |
|---|---|---|---|
| Ecuaciones no lineales | Biseccion | `codigos_andy/codigos_viejos/biseccion.py` | Correr e ingresar `f(x)`, intervalo y tolerancia/pasos |
| Ecuaciones no lineales | Newton | `codigos_andy/codigos_buenos/newton.py` | Correr e ingresar `f(x)`, intervalo y tolerancia/iteraciones |
| Ecuaciones no lineales | Punto fijo | `codigos_andy/codigos_viejos/puntos_fijos.py` | Correr e ingresar `g(x)`, no `f(x)` |
| EDO/PVI | Euler | `codigos_andy/codigos_viejos/pvi.py` | Editar datos y elegir `e` |
| EDO/PVI | Taylor orden 2 | `codigos_andy/codigos_viejos/pvi.py` | Editar datos y elegir `t` |
| EDO/PVI | Heun orden 2 | `codigos_andy/codigos_viejos/pvi.py` | Editar datos y elegir `h` |
| EDO/PVI | RK4 | `codigos_andy/codigos_viejos/pvi.py` | Editar datos y llamar `solve_g(rk4, a, b, N, y0=y0)` |

## Notas de escritura de funciones

En los codigos viejos se pueden usar expresiones como:

```text
x^2
sin(x)
cos(x)
tan(x)
exp(x)
log(x)
sqrt(x)
pi
e
```

Si una funcion no existe en el parser, conviene escribirla con funciones basicas.

Ejemplo:

```text
cosh(u) = (exp(u) + exp(-u)) / 2
```

Entonces:

```text
cosh(10/x)
```

se ingresa como:

```text
(exp(10/x)+exp(-10/x))/2
```
