# Métodos Numéricos

Este repositorio contiene implementaciones de diferentes métodos numéricos para la resolución de ecuaciones y problemas de valor inicial.

## Implementaciones

### Métodos de Búsqueda de Raíces

- `biseccion.py`: Implementación del método de bisección. **Incluye verificación de las condiciones necesarias** para su correcta aplicación (teorema de Bolzano).
- `newton.py`: Implementación del método de Newton-Raphson. **Incluye verificación de las condiciones necesarias** para su convergencia.
- `puntos_fijos.py`: Implementación del método de punto fijo. **No incluye verificación** de si la función cumple con las condiciones necesarias para la convergencia.

### Problemas de Valor Inicial

- `pvi.py`: Implementación de varios métodos para resolver problemas de valor inicial (Euler, Euler mejorado, Runge-Kutta). Este es el único archivo que:
  - No recibe los datos por input
  - Desarrollado por [@santiagolifischtz](https://github.com/santiagolifischtz)

## Créditos

El archivo `pvi.py` fue desarrollado por Santiago. Agradecemos su contribución al proyecto.

## Uso

Cada archivo contiene instrucciones específicas sobre cómo ejecutarlo y qué datos necesita como entrada, excepto `pvi.py` que tiene los datos predefinidos en el código.
