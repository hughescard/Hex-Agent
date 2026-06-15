# Proyecto de IA – Agente Autónomo para HEX

## 1. Descripción breve

Este proyecto implementa un agente autónomo para el juego **HEX**.  
La entrega final está concentrada en un único archivo principal, `solution.py`, donde la clase pública evaluada en torneo es `SmartPlayer`.

El agente final utiliza un enfoque de **Monte Carlo Tree Search (MCTS)** guiado por heurísticas estratégicas para:

- detectar tácticas inmediatas
- priorizar jugadas con progreso de conexión
- mantener una exploración controlada del árbol
- usar rollouts ligeros y eficientes dentro del presupuesto de tiempo

Además, el repositorio conserva una **interfaz gráfica local** para simular partidas visualmente y observar el comportamiento del agente sin necesidad de scripts externos.

## 2. Agente final de entrega

La clase final que debe considerarse como entrega es:

- `SmartPlayer` en `Guillermo_Hughes_Cardona/solution.py`

Esa clase ya está configurada con los hiperparámetros finales elegidos para el torneo. No hace falta pasar argumentos por línea de comandos ni editar configuración externa para usar la versión final.

También existe:

- `SmartPlayerMCTS`: implementación base del motor MCTS sobre la que se apoya `SmartPlayer`

En la versión actual del proyecto, `SmartPlayer` es simplemente el agente final listo para evaluación, construido sobre esa implementación MCTS.

## 3. Estructura del proyecto

Los archivos realmente importantes del proyecto son:

- `Guillermo_Hughes_Cardona/solution.py`  
  Contiene el agente final de entrega. Aquí vive la clase `SmartPlayer` y la implementación del motor MCTS.

- `Guillermo_Hughes_Cardona/board.py`  
  Implementa el tablero de HEX, clonación del estado, vecindad hexagonal y verificación de conexión ganadora.

- `Guillermo_Hughes_Cardona/player.py`  
  Define la clase base `Player`, que establece la interfaz esperada por el proyecto.

- `Guillermo_Hughes_Cardona/hex_match_gui.py`  
  Interfaz gráfica local para cargar agentes y ejecutar partidas visuales.

- `Guillermo_Hughes_Cardona/HexAgent.pdf`  
  Informe final del proyecto con la estrategia y el enfoque general del agente.

- `Guillermo_Hughes_Cardona/experiments/`  
  Carpeta secundaria con scripts y resultados de experimentación. No forma parte del flujo principal de entrega, pero se conserva como apoyo para análisis y tuning.

## 4. Estrategia usada

El agente final combina una capa táctica con un motor MCTS.

### Tácticas inmediatas

Antes de entrar en búsqueda, el agente revisa situaciones forzadas:

- victoria inmediata
- bloqueo inmediato de derrota

Esto evita gastar iteraciones del árbol en jugadas obligatorias.

### Heurísticas estructurales

El agente usa conocimiento geométrico del tablero para identificar patrones útiles, especialmente:

- progreso hacia el eje ganador
- estructura local de conexión
- reconocimiento de bridges válidos
- evaluación rápida de jugadas candidatas

Estas heurísticas no sustituyen al MCTS, sino que lo guían para que el árbol priorice movimientos prometedores.

### Monte Carlo Tree Search

La decisión principal del agente final se toma mediante MCTS:

- selección con UCT
- expansión con branching factor limitado
- simulaciones rápidas guiadas
- retropropagación root-centric

El diseño está optimizado para funcionar dentro de un presupuesto temporal fijo por jugada.

### Rollouts guiados

Los rollouts no son completamente aleatorios.  
Primero comprueban tácticas inmediatas y, si no existen, usan una política barata basada en señales locales como:

- progreso axial
- vecindad con piedras propias o rivales
- actividad cercana
- detección local barata de bridge

### Tuning

Durante el desarrollo se probaron múltiples configuraciones de hiperparámetros.  
El proyecto principal ya quedó fijado con la configuración elegida para entrega, mientras que las utilidades de tuning se movieron a `experiments/`.

## 5. Requisitos de ejecución

Recomendado:

- **Python 3.10 o superior**

Dependencias:

- no se requieren librerías externas adicionales para ejecutar el agente
- la interfaz gráfica usa `tkinter`, que normalmente viene con Python en instalaciones de escritorio

Si `tkinter` no está disponible en tu sistema, el agente igualmente puede importarse y utilizarse, pero la interfaz visual no arrancará hasta instalar el soporte gráfico correspondiente de Python.

## 6. Cómo ejecutar una simulación local

La forma recomendada de probar el agente visualmente es usar la interfaz gráfica:

```bash
python3 Guillermo_Hughes_Cardona/hex_match_gui.py
```

Ejecuta ese comando **desde la raíz del proyecto**.

### Qué debería ver el usuario

Se abrirá una ventana llamada **HEX Match Viewer** con:

- selección de archivo para Jugador 1 y Jugador 2
- controles para crear una nueva partida
- botones para jugar un turno o jugar automáticamente
- visualización del tablero
- panel de estado y registro de jugadas

### Cómo usar la UI

1. Abre la interfaz con el comando anterior.
2. Por defecto, ambos jugadores cargan `solution.py`.
3. Si quieres, puedes cambiar cualquiera de los dos jugadores seleccionando otro archivo `.py`.
4. El archivo cargado debe definir una clase pública llamada `SmartPlayer`.
5. Ajusta el tamaño del tablero si quieres.
6. Usa:
   - `Nueva partida` para reiniciar
   - `Jugar un turno` para avanzar manualmente
   - `Auto` para dejar jugar a los agentes automáticamente
   - `Detener` para pausar la partida

### Flujo real soportado por la UI

La interfaz está diseñada para enfrentar **dos agentes Python** entre sí.  
No hay modo humano vs agente implementado en la versión actual.

## 7. Cómo ejecutar pruebas o benchmarks

El flujo principal de entrega no depende de scripts de benchmark, pero se conserva material de apoyo en:

- `Guillermo_Hughes_Cardona/experiments/test_player.py`
- `Guillermo_Hughes_Cardona/experiments/test_player_mcts.py`
- `Guillermo_Hughes_Cardona/experiments/tune_mcts.py`

Estos archivos sirven para:

- comparar agentes
- medir rendimiento
- barrer hiperparámetros

Al estar movidos a `experiments/`, quedan explícitamente fuera del flujo principal de entrega.

## 8. Qué se entrega y qué queda como apoyo

### Parte de entrega

Lo esencial para la entrega es:

- `Guillermo_Hughes_Cardona/solution.py`
- `Guillermo_Hughes_Cardona/board.py`
- `Guillermo_Hughes_Cardona/player.py`
- `Guillermo_Hughes_Cardona/HexAgent.pdf`

La clase de evaluación es:

- `SmartPlayer`

### Parte de apoyo local

Para pruebas visuales locales se conserva:

- `Guillermo_Hughes_Cardona/hex_match_gui.py`

### Parte experimental

Para análisis y desarrollo adicional se conserva:

- `Guillermo_Hughes_Cardona/experiments/`

## 9. Notas finales

- El archivo principal del proyecto es `solution.py`.
- La clase final de entrega es `SmartPlayer`.
- El proyecto ya está preparado para ejecución local y para uso en torneo sin configuración adicional.
- La interfaz gráfica se conserva como herramienta práctica para validar partidas y observar el comportamiento del agente.
