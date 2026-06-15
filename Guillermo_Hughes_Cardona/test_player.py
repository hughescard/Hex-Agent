from board import HexBoard
from solution import SmartPlayer

BOARD_SIZE = 5
TOTAL_GAMES = 20


def validate_legal_move(board: HexBoard, move: tuple[int, int] | None) -> tuple[int, int]:
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


def play_single_game(starting_player: int) -> int:
    board = HexBoard(BOARD_SIZE)
    players = {
        1: SmartPlayer(1),
        2: SmartPlayer(2),
    }

    current_player = starting_player
    max_turns = board.size * board.size

    for _turn in range(max_turns):
        move = players[current_player].play(board)
        row, col = validate_legal_move(board, move)

        if not board.place_piece(row, col, current_player):
            raise AssertionError(f"No se pudo aplicar la jugada: {(row, col)}")

        if board.check_connection(current_player):
            return current_player

        current_player = 2 if current_player == 1 else 1

    raise AssertionError("La partida termino sin ganador tras llenar el tablero")


def run_benchmark() -> None:
    starts = [1] * (TOTAL_GAMES // 2) + [2] * (TOTAL_GAMES // 2)
    wins = {1: 0, 2: 0}

    for starting_player in starts:
        winner = play_single_game(starting_player)
        wins[winner] += 1

    total_games = wins[1] + wins[2]
    player_1_pct = (wins[1] / total_games) * 100 if total_games else 0.0
    player_2_pct = (wins[2] / total_games) * 100 if total_games else 0.0

    print("Resumen SmartPlayer vs SmartPlayer")
    print(f"Tablero: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Total de partidas: {total_games}")
    print(f"Victorias jugador 1: {wins[1]} ({player_1_pct:.1f}%)")
    print(f"Victorias jugador 2: {wins[2]} ({player_2_pct:.1f}%)")


if __name__ == "__main__":
    run_benchmark()
