import importlib.util
import math
import queue
import sys
import threading
import time
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, Button, Canvas, DoubleVar, Entry
from tkinter import Frame, Label, Scrollbar, Spinbox, StringVar, Text, Tk, filedialog, messagebox
from tkinter import ttk

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from board import HexBoard


DEFAULT_AGENT_1 = APP_DIR / "solution.py"
DEFAULT_AGENT_2 = APP_DIR / "solution_Camilo.py"
MIN_BOARD_SIZE = 3
MAX_BOARD_SIZE = 11


class AgentLoadError(Exception):
    pass


def load_smart_player_class(file_path: Path):
    if not file_path.exists():
        raise AgentLoadError(f"No existe el archivo: {file_path}")
    if file_path.suffix != ".py":
        raise AgentLoadError(f"El archivo no es Python: {file_path}")

    module_name = f"hex_agent_{abs(hash((str(file_path), time.time_ns())))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AgentLoadError(f"No se pudo cargar el modulo: {file_path}")

    module = importlib.util.module_from_spec(spec)
    added_paths = []

    for path in (str(file_path.parent), str(APP_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
            added_paths.append(path)

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AgentLoadError(f"Error importando {file_path.name}: {exc}") from exc
    finally:
        for path in reversed(added_paths):
            try:
                sys.path.remove(path)
            except ValueError:
                pass

    player_class = getattr(module, "SmartPlayer", None)
    if player_class is None:
        raise AgentLoadError(f"{file_path.name} no define la clase SmartPlayer")

    return player_class


class SectionFrame(Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent, bg="#fdfaf4", bd=1, relief="solid")
        Label(
            self,
            text=title,
            bg="#d9c7a2",
            fg="#1f2a37",
            anchor="w",
            padx=10,
            pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(fill="x")
        self.body = Frame(self, bg="#fdfaf4", padx=10, pady=10)
        self.body.pack(fill=BOTH, expand=True)


class HexMatchGUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("HEX Match Viewer")
        self.root.geometry("1500x920")
        self.root.minsize(1280, 820)

        self.agent1_path = StringVar(value=str(DEFAULT_AGENT_1))
        self.agent2_path = StringVar(value=str(DEFAULT_AGENT_2))
        self.status_var = StringVar(value="Selecciona dos agentes y crea una partida nueva.")
        self.turn_var = StringVar(value="Turno: -")
        self.result_var = StringVar(value="Resultado: -")
        self.agent1_label_var = StringVar(value=f"Jugador 1: {DEFAULT_AGENT_1.name}")
        self.agent2_label_var = StringVar(value=f"Jugador 2: {DEFAULT_AGENT_2.name}")
        self.delay_var = DoubleVar(value=0.35)

        self.board = HexBoard(5)
        self.players = {}
        self.agent_names = {}
        self.current_player_id = 1
        self.turn_number = 1
        self.last_move = None
        self.running = False
        self.worker_active = False
        self.game_over = False
        self.worker_queue: queue.Queue = queue.Queue()
        self.state_text = None

        self._build_layout()
        self._append_log("Interfaz inicializada.")
        self._draw_board()
        self._refresh_state_box()
        self.root.after(50, self._poll_worker_queue)

    def _build_layout(self) -> None:
        root_frame = Frame(self.root, bg="#f4f1ea")
        root_frame.pack(fill=BOTH, expand=True)

        left_panel = Frame(root_frame, bg="#f4f1ea", width=540)
        left_panel.pack(side=LEFT, fill="y", padx=18, pady=18)
        left_panel.pack_propagate(False)

        right_panel = Frame(root_frame, bg="#e9e4d8")
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=(0, 18), pady=18)
        board_area = Frame(right_panel, bg="#e9e4d8")
        board_area.pack(side=LEFT, fill=BOTH, expand=True)
        board_header = Frame(board_area, bg="#e9e4d8")
        board_header.pack(fill="x", padx=18, pady=(10, 0))
        Label(
            board_header,
            text="Jugador 1 conecta izquierda-derecha | Jugador 2 conecta arriba-abajo",
            bg="#e9e4d8",
            fg="#5a4636",
            font=("Helvetica", 12, "bold"),
            wraplength=900,
            justify=LEFT,
        ).pack(anchor="w")
        status_panel = Frame(right_panel, bg="#f4f1ea", width=360)
        status_panel.pack(side=RIGHT, fill="y", padx=(14, 0))
        status_panel.pack_propagate(False)

        Label(
            left_panel,
            text="HEX Match Viewer",
            font=("Helvetica", 22, "bold"),
            bg="#f4f1ea",
            fg="#1f2a37",
        ).pack(anchor="w", pady=(0, 14))

        agents_frame = SectionFrame(left_panel, "Agentes")
        agents_frame.pack(fill="x", pady=(0, 14))
        self._build_file_selector(
            agents_frame.body,
            label_text="Jugador 1",
            variable=self.agent1_path,
            browse_command=lambda: self._choose_file(self.agent1_path),
        )
        self._build_file_selector(
            agents_frame.body,
            label_text="Jugador 2",
            variable=self.agent2_path,
            browse_command=lambda: self._choose_file(self.agent2_path),
        )

        settings_frame = SectionFrame(left_panel, "Partida")
        settings_frame.pack(fill="x", pady=(0, 14))

        size_row = Frame(settings_frame.body, bg="#fdfaf4")
        size_row.pack(fill="x", pady=(0, 10))
        Label(size_row, text="Tamano del tablero", bg="#fdfaf4", fg="#1f2a37").pack(side=LEFT)
        self.size_spinbox = Spinbox(size_row, from_=MIN_BOARD_SIZE, to=MAX_BOARD_SIZE, width=5, justify="center")
        self.size_spinbox.delete(0, END)
        self.size_spinbox.insert(0, "5")
        self.size_spinbox.pack(side=RIGHT)

        delay_row = Frame(settings_frame.body, bg="#fdfaf4")
        delay_row.pack(fill="x")
        Label(delay_row, text="Pausa entre jugadas", bg="#fdfaf4", fg="#1f2a37").pack(side=LEFT)
        ttk.Scale(
            delay_row,
            from_=0.0,
            to=1.5,
            variable=self.delay_var,
            orient="horizontal",
            length=170,
        ).pack(side=RIGHT)

        controls_frame = SectionFrame(left_panel, "Controles")
        controls_frame.pack(fill="x", pady=(0, 14))

        Button(
            controls_frame.body,
            text="Nueva partida",
            command=self.reset_match,
            bg="#1f6f78",
            fg="white",
            relief="flat",
            font=("Helvetica", 10),
        ).pack(fill="x", pady=(0, 8))
        Button(
            controls_frame.body,
            text="Jugar un turno",
            command=self.play_one_turn,
            bg="#f4a261",
            fg="#122027",
            relief="flat",
        ).pack(fill="x", pady=(0, 8))
        Button(
            controls_frame.body,
            text="Auto",
            command=self.start_auto_play,
            bg="#2a9d8f",
            fg="white",
            relief="flat",
        ).pack(fill="x", pady=(0, 8))
        Button(
            controls_frame.body,
            text="Detener",
            command=self.stop_auto_play,
            bg="#bc4749",
            fg="white",
            relief="flat",
        ).pack(fill="x")

        state_frame = SectionFrame(status_panel, "Estado")
        state_frame.pack(fill=BOTH, expand=True)

        Label(
            state_frame.body,
            textvariable=self.agent1_label_var,
            bg="#fdfaf4",
            fg="#1f2a37",
            wraplength=280,
            justify=LEFT,
        ).pack(anchor="w")
        Label(
            state_frame.body,
            textvariable=self.agent2_label_var,
            bg="#fdfaf4",
            fg="#1f2a37",
            wraplength=280,
            justify=LEFT,
        ).pack(anchor="w", pady=(6, 0))
        Label(state_frame.body, textvariable=self.turn_var, bg="#fdfaf4", fg="#1f2a37").pack(anchor="w", pady=(10, 0))
        Label(state_frame.body, textvariable=self.result_var, bg="#fdfaf4", fg="#1f2a37").pack(anchor="w", pady=(4, 0))
        Label(
            state_frame.body,
            textvariable=self.status_var,
            bg="#fdfaf4",
            fg="#3b4c5a",
            wraplength=280,
            justify=LEFT,
        ).pack(anchor="w", pady=(10, 0))

        state_row = Frame(state_frame.body, bg="#fdfaf4")
        state_row.pack(fill="both", expand=True, pady=(10, 0))

        state_scroll = Scrollbar(state_row, orient=VERTICAL)
        state_scroll.pack(side=RIGHT, fill="y")

        self.state_text = Text(
            state_row,
            height=18,
            wrap="word",
            yscrollcommand=state_scroll.set,
            bg="#fffdf7",
            fg="#17212b",
            relief="solid",
            bd=1,
            padx=8,
            pady=8,
            font=("Helvetica", 11),
        )
        self.state_text.pack(side=LEFT, fill="both", expand=True)
        state_scroll.config(command=self.state_text.yview)

        log_frame = SectionFrame(left_panel, "Registro")
        log_frame.pack(fill=BOTH, expand=True)
        log_frame.pack_propagate(False)

        log_scroll = Scrollbar(log_frame.body, orient=VERTICAL)
        log_scroll.pack(side=RIGHT, fill="y")

        self.log_text = Text(
            log_frame.body,
            wrap="word",
            yscrollcommand=log_scroll.set,
            bg="#fffdf7",
            fg="#17212b",
            relief="flat",
        )
        self.log_text.pack(fill=BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

        board_frame = Frame(board_area, bg="#e9e4d8")
        board_frame.pack(fill=BOTH, expand=True, padx=18, pady=18)

        self.canvas = Canvas(board_frame, bg="#f7f2e7", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_board())

    def _build_file_selector(self, parent: Frame, label_text: str, variable: StringVar, browse_command) -> None:
        Label(parent, text=label_text, bg="#fdfaf4", fg="#1f2a37").pack(anchor="w")
        row = Frame(parent, bg="#fdfaf4")
        row.pack(fill="x", pady=(4, 10))
        Entry(row, textvariable=variable).pack(side=LEFT, fill="x", expand=True)
        Button(row, text="Buscar", command=browse_command, relief="flat", bg="#d9c7a2").pack(side=RIGHT, padx=(8, 0))

    def _choose_file(self, variable: StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Selecciona un agente",
            filetypes=[("Python", "*.py"), ("Todos", "*.*")],
            initialdir=str(APP_DIR),
        )
        if selected:
            variable.set(selected)

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(END, f"[{timestamp}] {message}\n")
        self.log_text.see(END)

    def _refresh_state_box(self) -> None:
        if self.state_text is None:
            return

        lines = [
            f"Estado: {self.status_var.get()}",
            f"Turno: {self.turn_var.get()}",
            f"Resultado: {self.result_var.get()}",
            f"Jugador activo: {self.agent_names.get(self.current_player_id, 'n/a')}",
        ]
        if self.last_move is not None:
            lines.append(f"Ultima jugada: {self.last_move}")

        self.state_text.config(state="normal")
        self.state_text.delete("1.0", END)
        self.state_text.insert("1.0", "\n".join(lines))
        self.state_text.config(state="disabled")

    def _load_agents(self) -> bool:
        path1 = Path(self.agent1_path.get()).expanduser()
        path2 = Path(self.agent2_path.get()).expanduser()

        try:
            player1_class = load_smart_player_class(path1)
            player2_class = load_smart_player_class(path2)
            self.players = {
                1: player1_class(1),
                2: player2_class(2),
            }
        except Exception as exc:
            messagebox.showerror("Error cargando agentes", str(exc))
            self.status_var.set("No se pudieron cargar los agentes.")
            return False

        self.agent_names = {1: path1.stem, 2: path2.stem}
        self.agent1_label_var.set(f"Jugador 1: {path1.name}")
        self.agent2_label_var.set(f"Jugador 2: {path2.name}")
        self._append_log(f"Agente 1 cargado: {path1}")
        self._append_log(f"Agente 2 cargado: {path2}")
        self._refresh_state_box()
        return True

    def reset_match(self) -> None:
        if self.worker_active:
            self.status_var.set("Espera a que termine la jugada actual para reiniciar.")
            return

        try:
            board_size = int(self.size_spinbox.get())
        except ValueError:
            messagebox.showerror("Tamano invalido", "El tamano del tablero debe ser un entero.")
            return

        if not (MIN_BOARD_SIZE <= board_size <= MAX_BOARD_SIZE):
            messagebox.showerror("Tamano invalido", f"El tamano debe estar entre {MIN_BOARD_SIZE} y {MAX_BOARD_SIZE}.")
            return

        if not self._load_agents():
            return

        self.board = HexBoard(board_size)
        self.current_player_id = 1
        self.turn_number = 1
        self.last_move = None
        self.running = False
        self.worker_active = False
        self.game_over = False
        self.turn_var.set("Turno: 1")
        self.result_var.set("Resultado: En curso")
        self.status_var.set("Partida reiniciada.")
        self._append_log(f"Nueva partida {board_size}x{board_size}.")
        self._refresh_state_box()
        self._draw_board()

    def start_auto_play(self) -> None:
        if self.game_over:
            self.status_var.set("La partida termino. Crea una nueva para continuar.")
            self._refresh_state_box()
            return
        if not self.players and not self._load_agents():
            return
        self.running = True
        self.status_var.set("Auto play activo.")
        self._request_next_move()

    def stop_auto_play(self) -> None:
        self.running = False
        self.status_var.set("Auto play detenido.")
        self._refresh_state_box()

    def play_one_turn(self) -> None:
        if self.game_over:
            self.status_var.set("La partida termino. Crea una nueva para continuar.")
            return
        if not self.players and not self._load_agents():
            return
        self.running = False
        self._request_next_move()

    def _request_next_move(self) -> None:
        if self.worker_active or self.game_over:
            return

        if not any(0 in row for row in self.board.board):
            self._finish_game("Tablero lleno sin ganador.")
            return

        player_id = self.current_player_id
        player = self.players[player_id]
        board_snapshot = self.board.clone()
        self.worker_active = True
        self.status_var.set(f"Calculando jugada de {self.agent_names[player_id]}...")

        worker = threading.Thread(
            target=self._compute_move_worker,
            args=(player_id, player, board_snapshot),
            daemon=True,
        )
        worker.start()

    def _compute_move_worker(self, player_id: int, player, board_snapshot: HexBoard) -> None:
        start_time = time.perf_counter()
        try:
            move = player.play(board_snapshot)
            elapsed = time.perf_counter() - start_time
            self.worker_queue.put(("move", player_id, move, elapsed))
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.worker_queue.put(("error", player_id, str(exc), elapsed))

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                item = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_worker_message(item)
        self.root.after(50, self._poll_worker_queue)

    def _handle_worker_message(self, item) -> None:
        self.worker_active = False
        kind = item[0]

        if kind == "error":
            _kind, player_id, error_message, elapsed = item
            self.running = False
            self.game_over = True
            self.result_var.set(f"Resultado: error de jugador {player_id}")
            self.status_var.set(f"Error en {self.agent_names.get(player_id, f'jugador {player_id}')}.")
            self._append_log(
                f"Error de {self.agent_names.get(player_id, player_id)} tras {elapsed:.3f}s: {error_message}"
            )
            self._refresh_state_box()
            return

        _kind, player_id, move, elapsed = item
        if player_id != self.current_player_id or self.game_over:
            return

        if not self._is_legal_move(move):
            self.running = False
            self.game_over = True
            winner = 2 if player_id == 1 else 1
            self.result_var.set(f"Resultado: gana jugador {winner} por jugada ilegal")
            self.status_var.set("Se detecto una jugada ilegal.")
            self._append_log(f"Jugada ilegal de {self.agent_names[player_id]} tras {elapsed:.3f}s: {move}")
            self._refresh_state_box()
            self._draw_board()
            return

        row, col = move
        self.board.place_piece(row, col, player_id)
        self.last_move = (row, col)
        self._append_log(
            f"Turno {self.turn_number}: {self.agent_names[player_id]} ({player_id}) -> {move} en {elapsed:.3f}s"
        )
        self._refresh_state_box()

        if self.board.check_connection(player_id):
            self._finish_game(f"Gana {self.agent_names[player_id]} como jugador {player_id}.")
            return

        if not any(0 in row_values for row_values in self.board.board):
            self._finish_game("Tablero lleno sin ganador.")
            return

        self.current_player_id = 2 if self.current_player_id == 1 else 1
        self.turn_number += 1
        self.turn_var.set(f"Turno: {self.turn_number}")
        self.status_var.set(f"Listo. Siguiente jugador: {self.agent_names[self.current_player_id]}.")
        self._draw_board()

        if self.running:
            delay_ms = int(self.delay_var.get() * 1000)
            self.root.after(delay_ms, self._request_next_move)

    def _finish_game(self, message: str) -> None:
        self.running = False
        self.game_over = True
        self.result_var.set(f"Resultado: {message}")
        self.status_var.set(message)
        self._append_log(message)
        self._refresh_state_box()
        self._draw_board()

    def _is_legal_move(self, move) -> bool:
        if not isinstance(move, tuple) or len(move) != 2:
            return False
        row, col = move
        if not isinstance(row, int) or not isinstance(col, int):
            return False
        if not (0 <= row < self.board.size and 0 <= col < self.board.size):
            return False
        return self.board.board[row][col] == 0

    def _draw_board(self) -> None:
        self.canvas.delete("all")

        width = max(self.canvas.winfo_width(), 800)
        height = max(self.canvas.winfo_height(), 700)
        size = self.board.size
        radius = min(34, max(18, int(min(width, height) / (size * 2.2))))
        horiz = math.sqrt(3) * radius
        vert = 1.5 * radius
        board_width = horiz * (size - 1) + horiz / 2 + 2 * radius
        board_height = vert * (size - 1) + 2 * radius
        offset_x = max(70, (width - board_width) / 2)
        offset_y = max(80, (height - board_height) / 2)

        self.canvas.create_line(
            offset_x - 30,
            offset_y + radius,
            offset_x - 30,
            offset_y + board_height - radius,
            fill="#d97706",
            width=6,
        )
        self.canvas.create_line(
            offset_x + board_width + 10,
            offset_y + radius,
            offset_x + board_width + 10,
            offset_y + board_height - radius,
            fill="#d97706",
            width=6,
        )
        self.canvas.create_line(
            offset_x + horiz / 2,
            offset_y - 28,
            offset_x + board_width - horiz / 2,
            offset_y - 28,
            fill="#0f766e",
            width=6,
        )
        self.canvas.create_line(
            offset_x + horiz / 2,
            offset_y + board_height + 10,
            offset_x + board_width - horiz / 2,
            offset_y + board_height + 10,
            fill="#0f766e",
            width=6,
        )

        for row in range(size):
            for col in range(size):
                center_x = offset_x + col * horiz + (row % 2) * (horiz / 2) + radius
                center_y = offset_y + row * vert + radius
                points = self._hex_points(center_x, center_y, radius)
                fill = "#efe2c1"
                outline = "#9a7b4f"

                cell = self.board.board[row][col]
                if cell == 1:
                    fill = "#e07a5f"
                    outline = "#9d3f2e"
                elif cell == 2:
                    fill = "#3d405b"
                    outline = "#202437"

                line_width = 4 if self.last_move == (row, col) else 2
                self.canvas.create_polygon(points, fill=fill, outline=outline, width=line_width)

                if self.last_move == (row, col):
                    self.canvas.create_text(
                        center_x,
                        center_y,
                        text="●",
                        fill="#f8fafc" if cell == 2 else "#111827",
                        font=("Helvetica", max(12, radius // 2), "bold"),
                    )

    def _hex_points(self, center_x: float, center_y: float, radius: float) -> list[float]:
        points = []
        for angle in range(30, 390, 60):
            radians = math.radians(angle)
            points.extend(
                [
                    center_x + radius * math.cos(radians),
                    center_y + radius * math.sin(radians),
                ]
            )
        return points


def main() -> None:
    root = Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    HexMatchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
