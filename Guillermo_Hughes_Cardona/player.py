class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # 1 o 2

    def play(self, board):
        raise NotImplementedError("¡Implementa este método!")