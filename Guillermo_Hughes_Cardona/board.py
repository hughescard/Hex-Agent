from collections import deque

class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clone(self):
        new_board = HexBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        return new_board

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = player_id
        return True

    def get_neighbors(self, row, col):
        # even-r layout
        if row % 2 == 0:
            directions = [(-1, 0), (-1, -1), (0, -1), (0, 1), (1, 0), (1, -1)]
        else:
            directions = [(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 1), (1, 0)]

        neighbors = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append((r, c))
        return neighbors

    def check_connection(self, player_id: int) -> bool:
        visited = set()
        queue = deque()

        # jugador 1 → izquierda a derecha
        if player_id == 1:
            for r in range(self.size):
                if self.board[r][0] == player_id:
                    queue.append((r, 0))
                    visited.add((r, 0))

            target_col = self.size - 1

            while queue:
                r, c = queue.popleft()
                if c == target_col:
                    return True

                for nr, nc in self.get_neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == player_id:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # jugador 2 → arriba a abajo
        else:
            for c in range(self.size):
                if self.board[0][c] == player_id:
                    queue.append((0, c))
                    visited.add((0, c))

            target_row = self.size - 1

            while queue:
                r, c = queue.popleft()
                if r == target_row:
                    return True

                for nr, nc in self.get_neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == player_id:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return False