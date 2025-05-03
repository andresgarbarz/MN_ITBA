# Métodos Numéricos

Este repositorio contiene implementaciones de diferentes métodos numéricos para la resolución de ecuaciones y problemas de valor inicial. El proyecto está organizado en dos directorios principales: `codigos_buenos` y `codigos_viejos`.

## Estructura del Proyecto

### Directorio `codigos_buenos`

Contiene las versiones más recientes y mejoradas de los métodos numéricos, que son permisivas en el input del usuario y estan verificadas por @andresgarbarz(https://github.com/andresgarbarz):

- `newton.py`: Implementación del método de Newton-Raphson
- `math_parser.py`: Herramienta para el análisis y evaluación de expresiones matemáticas

### Directorio `codigos_viejos`

Contiene las implementaciones originales y versiones anteriores de los métodos:

- `biseccion.py`: Implementación del método de bisección
- `newton.py`: Implementación original del método de Newton-Raphson
- `puntos_fijos.py` y `puntos_fijos2.py`: Implementaciones del método de punto fijo
- `pvi.py`: Implementación de métodos para resolver problemas de valor inicial (Euler, Euler mejorado, Runge-Kutta)
- `derivative.py` y `derivative+.py`: Implementaciones para cálculo de derivadas
- `utils.py` y `utils+.py`: Utilidades y funciones auxiliares
- `evaluate.py`: Herramienta para evaluación de expresiones matemáticas
- `incrementos.py`: Implementación de métodos de incrementos

## Características

- Implementaciones verificadas de métodos numéricos fundamentales
- Soporte para diferentes tipos de problemas matemáticos
- Herramientas auxiliares para el análisis y evaluación de expresiones
- Versiones mejoradas y optimizadas de los métodos originales

## Uso

Cada archivo contiene instrucciones específicas sobre cómo ejecutarlo y qué datos necesita como entrada. Las versiones más recientes en `codigos_buenos` incluyen mejoras en la interfaz de usuario y manejo de errores.

## Créditos

El archivo `pvi.py` fue desarrollado originalmente por [@santiagolifischtz](https://github.com/santiagolifischtz) y levemente modificado por [@andresgarbarz](https://github.com/andresgarbarz).
