# Proyecto de IA – Agente Autónomo para HEX

## 1. Descripción breve

Este repositorio implementa un agente para el juego **HEX**.  
El archivo principal del proyecto es `Guillermo_Hughes_Cardona/solution.py`, y la clase pública que representa la entrega final es `SmartPlayer`.

El agente final utiliza un enfoque de **Monte Carlo Tree Search (MCTS)** guiado por heurísticas estratégicas y tácticas inmediatas. Además, el proyecto incluye una **interfaz gráfica local** para visualizar partidas entre agentes y scripts auxiliares para pruebas y comparación.

## 2. Agente final de entrega

La clase que debe considerarse como agente final de entrega es:

- `SmartPlayer` en `Guillermo_Hughes_Cardona/solution.py`

También existe:

- `SmartPlayerMCTS`, que contiene la implementación base del motor MCTS sobre la que se construye `SmartPlayer`

En la versión actual del repositorio, `SmartPlayer` es la clase que debe usarse para torneo y evaluación.

## 3. Estructura del proyecto

Los archivos principales que actualmente forman parte del repositorio son:

- `Guillermo_Hughes_Cardona/solution.py`  
  Implementación del agente final de HEX.

- `Guillermo_Hughes_Cardona/board.py`  
  Lógica del tablero, clonación de estado, vecindad hexagonal y verificación de conexión ganadora.

- `Guillermo_Hughes_Cardona/player.py`  
  Clase base `Player` esperada por el proyecto.

- `Guillermo_Hughes_Cardona/hex_match_gui.py`  
  Interfaz gráfica local para cargar agentes y ver partidas de forma visual.

- `Guillermo_Hughes_Cardona/test_player.py`  
  Script simple para ejecutar partidas automáticas entre instancias de `SmartPlayer`.

- `Guillermo_Hughes_Cardona/test_player_mcts.py`  
  Script de benchmark/comparación entre el agente MCTS y otros agentes compatibles.

- `README.md`  
  Documento de descripción general y uso del proyecto.

## 4. Estrategia usada

El agente final combina una capa táctica con un motor MCTS.

### Tácticas inmediatas

Antes de entrar en búsqueda, el agente revisa:

- victoria inmediata
- bloqueo inmediato de derrota

Esto evita desperdiciar iteraciones del árbol en jugadas forzadas.

### Heurísticas estructurales

El agente usa señales geométricas del tablero para orientar la búsqueda, entre ellas:

- progreso hacia el eje ganador
- conectividad local
- reconocimiento de bridges válidos
- priorización de jugadas candidatas

Estas heurísticas guían la exploración, pero no reemplazan la búsqueda MCTS.

### Monte Carlo Tree Search

La decisión principal se toma con MCTS:

- selección con UCT
- expansión con branching factor controlado
- simulaciones rápidas
- retropropagación root-centric

El agente está ajustado para operar bajo un presupuesto temporal por turno.

### Rollouts guiados

Los rollouts no son completamente aleatorios.  
Primero revisan tácticas inmediatas y luego usan una política ligera basada en señales locales como:

- progreso axial
- vecindad con piedras propias
- vecindad con piedras rivales
- actividad local del tablero

## 5. Requisitos de ejecución

Recomendado:

- **Python 3.10 o superior**

Dependencias:

- no se requieren librerías externas para ejecutar el agente
- la interfaz gráfica usa `tkinter`, que normalmente viene incluida con Python en instalaciones de escritorio

Si tu instalación de Python no incluye `tkinter`, el agente seguirá siendo importable, pero la interfaz visual no arrancará hasta instalar el soporte gráfico correspondiente.

## 6. Cómo ejecutar una simulación local

La forma principal de ejecutar una simulación visual local es:

```bash
python3 Guillermo_Hughes_Cardona/hex_match_gui.py
```

Ese comando debe ejecutarse desde la raíz del proyecto.

### Qué hace la interfaz

La ventana **HEX Match Viewer** permite:

- cargar un archivo para el Jugador 1
- cargar un archivo para el Jugador 2
- crear una partida nueva
- jugar turno a turno
- ejecutar la partida automáticamente
- visualizar el tablero y el registro de eventos

### Flujo de uso

1. Ejecuta la interfaz con el comando anterior.
2. Por defecto, la UI carga `solution.py` para ambos jugadores.
3. Puedes seleccionar otros archivos `.py` siempre que definan una clase pública llamada `SmartPlayer`.
4. Elige el tamaño del tablero.
5. Usa los botones:
   - `Nueva partida`
   - `Jugar un turno`
   - `Auto`
   - `Detener`

### Alcance real de la UI

La UI actual está diseñada para **agente vs agente**.  
No implementa modo humano vs agente.

## 7. Cómo ejecutar pruebas o benchmarks

Además de la UI, el repositorio contiene dos scripts auxiliares:

- `python3 Guillermo_Hughes_Cardona/test_player.py`  
  Ejecuta partidas automáticas entre instancias de `SmartPlayer`.

- `python3 Guillermo_Hughes_Cardona/test_player_mcts.py`  
  Permite comparar el agente MCTS con el baseline u otros agentes compatibles.

Estos scripts son útiles para validación y benchmarking, pero no forman parte de la API principal del agente.

## 8. Qué corresponde a la entrega

La parte esencial de la entrega es:

- `Guillermo_Hughes_Cardona/solution.py`
- `Guillermo_Hughes_Cardona/board.py`
- `Guillermo_Hughes_Cardona/player.py`

La clase de evaluación es:

- `SmartPlayer`

La interfaz gráfica y los scripts de benchmark se mantienen como herramientas locales de apoyo.

## 9. Notas finales

- El archivo principal del proyecto es `solution.py`.
- La clase final de entrega es `SmartPlayer`.
- El repositorio actual también conserva utilidades para pruebas y visualización local.
- Si en el futuro se limpia el repositorio para una entrega mínima, los archivos auxiliares podrían separarse de la versión principal.
