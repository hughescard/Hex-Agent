from __future__ import annotations

import argparse
import importlib.util
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional

from board import HexBoard
from player import Player
from solution import SmartPlayer, SmartPlayerMCTS


AgentFactory = Callable[[int], Player]
MetricDict = dict[str, Any]
APP_DIR = Path(__file__).resolve().parent


DEFAULT_BOARD_SIZE = 5
DEFAULT_NUM_GAMES = 20
DEFAULT_SEEDS = [11, 29, 47, 83]
DEFAULT_MCTS_MAX_TIME = 4.8
DEFAULT_EXPLORATION_C = 1.2
DEFAULT_ROLLOUT_TOP_K = 6
DEFAULT_MAX_ROLLOUT_DEPTH = 80


def validate_legal_move(board: HexBoard, move: tuple[int, int] | None) -> tuple[int, int]:
    """Validate the move returned by an agent."""
    if move is None:
        raise AssertionError("El jugador devolvio None")
    if not isinstance(move, tuple) or len(move) != 2:
        raise AssertionError(f"La jugada debe ser una tupla (row, col): {move}")

    row, col = move
    if not isinstance(row, int) or not isinstance(col, int):
        raise AssertionError(f"La jugada debe contener enteros: {move}")
    if not (0 <= row < board.size and 0 <= col < board.size):
        raise AssertionError(f"Jugada fuera del tablero: {move}")
    if board.board[row][col] != 0:
        raise AssertionError(f"Jugada ilegal sobre casilla ocupada: {move}")

    return row, col


def build_baseline_factory() -> AgentFactory:
    """Create the baseline heuristic player factory."""
    return lambda player_id: SmartPlayer(player_id)


def build_mcts_factory(
    *,
    max_time: float,
    exploration_c: float,
    rollout_top_k: int,
    max_rollout_depth: int,
    debug: bool,
) -> AgentFactory:
    """Create the MCTS player factory with configured parameters."""
    return lambda player_id: SmartPlayerMCTS(
        player_id,
        max_time=max_time,
        exploration_c=exploration_c,
        rollout_top_k=rollout_top_k,
        max_rollout_depth=max_rollout_depth,
        debug=debug,
    )


def load_external_agent_class(file_path: Path) -> type[Player]:
    """Load a player class from an external Python file in the project directory."""
    module_name = f"benchmark_agent_{file_path.stem}_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el modulo: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for class_name in ("SmartPlayer", "SmartPlayerMCTS", "SmartPlayer1"):
        candidate = getattr(module, class_name, None)
        if isinstance(candidate, type) and issubclass(candidate, Player):
            return candidate

    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, Player) and value is not Player:
            return value

    raise RuntimeError(f"No se encontro una clase de agente compatible en {file_path.name}")


def discover_peer_agent_factories() -> list[tuple[str, AgentFactory]]:
    """Discover companion agent scripts named solution_*.py if they exist locally."""
    factories: list[tuple[str, AgentFactory]] = []

    for path in sorted(APP_DIR.glob("solution_*.py")):
        if path.name == "solution.py":
            continue
        try:
            agent_cls = load_external_agent_class(path)
        except Exception:
            continue
        factories.append((path.stem, lambda player_id, cls=agent_cls: cls(player_id)))

    return factories


def make_empty_metrics(
    agent_a_name: str,
    agent_b_name: str,
    board_size: int,
    num_games: int,
    seeds: list[int],
) -> MetricDict:
    """Initialize the metrics container for a match."""
    return {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "board_size": board_size,
        "num_games": num_games,
        "seeds": seeds[:],
        "wins_a": 0,
        "wins_b": 0,
        "draws": 0,
        "wins_a_as_p1": 0,
        "wins_a_as_p2": 0,
        "wins_b_as_p1": 0,
        "wins_b_as_p2": 0,
        "agent_a_move_times": [],
        "agent_b_move_times": [],
        "errors": [],
        "game_results": [],
        "total_match_time": 0.0,
    }


def finalize_metrics(metrics: MetricDict) -> MetricDict:
    """Compute derived statistics from raw metrics."""
    total_games = metrics["num_games"]
    wins_a = metrics["wins_a"]
    wins_b = metrics["wins_b"]
    draws = metrics["draws"]

    metrics["win_rate_a"] = (wins_a / total_games) * 100 if total_games else 0.0
    metrics["win_rate_b"] = (wins_b / total_games) * 100 if total_games else 0.0
    metrics["draw_rate"] = (draws / total_games) * 100 if total_games else 0.0

    games_a_as_p1 = total_games // 2 + (total_games % 2)
    games_a_as_p2 = total_games // 2
    games_b_as_p1 = games_a_as_p2
    games_b_as_p2 = games_a_as_p1

    metrics["win_rate_a_as_p1"] = (metrics["wins_a_as_p1"] / games_a_as_p1) * 100 if games_a_as_p1 else 0.0
    metrics["win_rate_a_as_p2"] = (metrics["wins_a_as_p2"] / games_a_as_p2) * 100 if games_a_as_p2 else 0.0
    metrics["win_rate_b_as_p1"] = (metrics["wins_b_as_p1"] / games_b_as_p1) * 100 if games_b_as_p1 else 0.0
    metrics["win_rate_b_as_p2"] = (metrics["wins_b_as_p2"] / games_b_as_p2) * 100 if games_b_as_p2 else 0.0

    for key in ("agent_a_move_times", "agent_b_move_times"):
        times = metrics[key]
        avg_key = key.replace("_move_times", "_avg_move_time")
        max_key = key.replace("_move_times", "_max_move_time")
        metrics[avg_key] = sum(times) / len(times) if times else 0.0
        metrics[max_key] = max(times) if times else 0.0

    return metrics


def record_win(metrics: MetricDict, winner_label: str, winner_player_id: int) -> None:
    """Register a win by agent label and color."""
    if winner_label == "A":
        metrics["wins_a"] += 1
        if winner_player_id == 1:
            metrics["wins_a_as_p1"] += 1
        else:
            metrics["wins_a_as_p2"] += 1
    else:
        metrics["wins_b"] += 1
        if winner_player_id == 1:
            metrics["wins_b_as_p1"] += 1
        else:
            metrics["wins_b_as_p2"] += 1


def play_single_game(
    agent_a_factory: AgentFactory,
    agent_b_factory: AgentFactory,
    *,
    board_size: int,
    game_index: int,
    game_seed: int,
    verbose: bool,
    metrics: MetricDict,
) -> None:
    """Play one game, update metrics, and handle failures cleanly."""
    board = HexBoard(board_size)
    random.seed(game_seed)

    a_is_player1 = game_index < (metrics["num_games"] // 2 + metrics["num_games"] % 2)
    agent_a_player_id = 1 if a_is_player1 else 2
    agent_b_player_id = 2 if a_is_player1 else 1

    players = {
        agent_a_player_id: agent_a_factory(agent_a_player_id),
        agent_b_player_id: agent_b_factory(agent_b_player_id),
    }
    labels = {
        agent_a_player_id: "A",
        agent_b_player_id: "B",
    }

    current_player = 1
    max_turns = board.size * board.size
    result_text = "draw"

    for turn_index in range(max_turns):
        current_label = labels[current_player]
        start = time.perf_counter()
        try:
            move = players[current_player].play(board)
            row, col = validate_legal_move(board, move)
        except Exception as exc:
            loser_label = current_label
            winner_player = 2 if current_player == 1 else 1
            winner_label = labels[winner_player]
            record_win(metrics, winner_label, winner_player)
            error_text = (
                f"Game {game_index + 1}: error de agente {loser_label} "
                f"(player {current_player}) en turno {turn_index + 1}: {exc}"
            )
            metrics["errors"].append(error_text)
            result_text = f"{winner_label} gana por error/ilegal"
            break
        finally:
            elapsed = time.perf_counter() - start
            if current_label == "A":
                metrics["agent_a_move_times"].append(elapsed)
            else:
                metrics["agent_b_move_times"].append(elapsed)

        if not board.place_piece(row, col, current_player):
            winner_player = 2 if current_player == 1 else 1
            winner_label = labels[winner_player]
            record_win(metrics, winner_label, winner_player)
            error_text = (
                f"Game {game_index + 1}: no se pudo aplicar jugada de {current_label} "
                f"(player {current_player}) en {(row, col)}"
            )
            metrics["errors"].append(error_text)
            result_text = f"{winner_label} gana por aplicacion invalida"
            break

        if board.check_connection(current_player):
            winner_label = labels[current_player]
            record_win(metrics, winner_label, current_player)
            result_text = f"{winner_label} gana como player {current_player}"
            break

        current_player = 2 if current_player == 1 else 1
    else:
        metrics["draws"] += 1
        result_text = "draw por tablero lleno"

    metrics["game_results"].append(
        {
            "game": game_index + 1,
            "seed": game_seed,
            "agent_a_as_player": agent_a_player_id,
            "agent_b_as_player": agent_b_player_id,
            "result": result_text,
        }
    )

    if verbose:
        print(
            f"Game {game_index + 1}/{metrics['num_games']} | seed={game_seed} | "
            f"A=P{agent_a_player_id} B=P{agent_b_player_id} | {result_text}"
        )


def run_match(
    agent1_cls: AgentFactory,
    agent2_cls: AgentFactory,
    num_games: int,
    board_size: int,
    *,
    seeds: list[int],
    verbose: bool = False,
    agent1_name: str = "AgentA",
    agent2_name: str = "AgentB",
) -> MetricDict:
    """Run a color-balanced match and return aggregate metrics."""
    metrics = make_empty_metrics(agent1_name, agent2_name, board_size, num_games, seeds)
    match_start = time.perf_counter()

    for game_index in range(num_games):
        seed = seeds[game_index % len(seeds)] + 1009 * game_index
        play_single_game(
            agent1_cls,
            agent2_cls,
            board_size=board_size,
            game_index=game_index,
            game_seed=seed,
            verbose=verbose,
            metrics=metrics,
        )

    metrics["total_match_time"] = time.perf_counter() - match_start
    return finalize_metrics(metrics)


def print_match_summary(results: MetricDict) -> None:
    """Print a compact, readable summary of a match result."""
    print(f"Match: {results['agent_a']} vs {results['agent_b']}")
    print(f"Tablero: {results['board_size']}x{results['board_size']}")
    print(f"Partidas: {results['num_games']}")
    print(f"Seeds base: {results['seeds']}")
    print(
        f"Victorias {results['agent_a']}: {results['wins_a']} "
        f"({results['win_rate_a']:.1f}%)"
    )
    print(
        f"Victorias {results['agent_b']}: {results['wins_b']} "
        f"({results['win_rate_b']:.1f}%)"
    )
    print(f"Draws: {results['draws']} ({results['draw_rate']:.1f}%)")
    print(
        f"{results['agent_a']} como P1: {results['wins_a_as_p1']} "
        f"({results['win_rate_a_as_p1']:.1f}%) | "
        f"como P2: {results['wins_a_as_p2']} ({results['win_rate_a_as_p2']:.1f}%)"
    )
    print(
        f"{results['agent_b']} como P1: {results['wins_b_as_p1']} "
        f"({results['win_rate_b_as_p1']:.1f}%) | "
        f"como P2: {results['wins_b_as_p2']} ({results['win_rate_b_as_p2']:.1f}%)"
    )
    print(
        f"Tiempo medio/jugada {results['agent_a']}: {results['agent_a_avg_move_time']:.4f}s | "
        f"max: {results['agent_a_max_move_time']:.4f}s"
    )
    print(
        f"Tiempo medio/jugada {results['agent_b']}: {results['agent_b_avg_move_time']:.4f}s | "
        f"max: {results['agent_b_max_move_time']:.4f}s"
    )
    print(f"Tiempo total del match: {results['total_match_time']:.3f}s")
    if results["errors"]:
        print("Errores detectados:")
        for error in results["errors"][:5]:
            print(f"- {error}")
        if len(results["errors"]) > 5:
            print(f"- ... y {len(results['errors']) - 5} errores mas")
    print()


def parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list."""
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        return DEFAULT_SEEDS[:]
    return [int(chunk) for chunk in parts]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark experimental para SmartPlayer vs SmartPlayerMCTS.")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--num-games", type=int, default=DEFAULT_NUM_GAMES)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--mcts-max-time", type=float, default=DEFAULT_MCTS_MAX_TIME)
    parser.add_argument("--mcts-exploration-c", type=float, default=DEFAULT_EXPLORATION_C)
    parser.add_argument("--rollout-top-k", type=int, default=DEFAULT_ROLLOUT_TOP_K)
    parser.add_argument("--max-rollout-depth", type=int, default=DEFAULT_MAX_ROLLOUT_DEPTH)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-mcts", action="store_true")
    parser.add_argument("--skip-peers", action="store_true")
    return parser


def main() -> None:
    """Run baseline, MCTS, and optionally peer-script comparisons."""
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)

    baseline_factory = build_baseline_factory()
    mcts_factory = build_mcts_factory(
        max_time=args.mcts_max_time,
        exploration_c=args.mcts_exploration_c,
        rollout_top_k=args.rollout_top_k,
        max_rollout_depth=args.max_rollout_depth,
        debug=args.debug_mcts,
    )

    scheduled_matches: list[tuple[str, AgentFactory, str, AgentFactory]] = [
        ("SmartPlayer", baseline_factory, "SmartPlayerMCTS", mcts_factory),
        ("SmartPlayerMCTS", mcts_factory, "SmartPlayer", baseline_factory),
    ]

    if not args.skip_peers:
        for peer_name, peer_factory in discover_peer_agent_factories():
            scheduled_matches.append(("SmartPlayer", baseline_factory, peer_name, peer_factory))
            scheduled_matches.append(("SmartPlayerMCTS", mcts_factory, peer_name, peer_factory))

    if not scheduled_matches:
        print("No hay emparejamientos disponibles.")
        return

    for agent_a_name, agent_a_factory, agent_b_name, agent_b_factory in scheduled_matches:
        results = run_match(
            agent_a_factory,
            agent_b_factory,
            args.num_games,
            args.board_size,
            seeds=seeds,
            verbose=args.verbose,
            agent1_name=agent_a_name,
            agent2_name=agent_b_name,
        )
        print_match_summary(results)


if __name__ == "__main__":
    main()
