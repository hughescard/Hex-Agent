from __future__ import annotations

import heapq
import math
import random
import time
from typing import Any, Iterable, Optional

from player import Player
from board import HexBoard


Cell = tuple[int, int]
BridgeRecord = tuple[Cell, Cell, Cell, Cell]
BridgeCandidate = dict[str, Any]


class HexHeuristicMixin:
    """Reusable Hex heuristics shared by the baseline agent and the MCTS agent."""

    player_id: int
    previous_board: Optional[HexBoard]
    debug: bool

    # =========================================================
    # Basic utilities
    # =========================================================

    def opponent_id(self, player: int) -> int:
        """Return the opponent id for an arbitrary player."""
        return 2 if player == 1 else 1

    def opponent(self) -> int:
        """Return the opponent player id for this agent."""
        return self.opponent_id(self.player_id)

    def all_empty_cells(self, board: HexBoard) -> list[Cell]:
        """Return every empty cell on the board."""
        empties: list[Cell] = []
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] == 0:
                    empties.append((row, col))
        return empties

    def valid_moves_near_action(self, board: HexBoard) -> list[Cell]:
        """Prefer empty cells adjacent to existing stones; fallback to all empties."""
        all_empty: list[Cell] = []
        near_action: list[Cell] = []

        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] != 0:
                    continue

                move = (row, col)
                all_empty.append(move)

                for nr, nc in board.get_neighbors(row, col):
                    if board.board[nr][nc] != 0:
                        near_action.append(move)
                        break

        return near_action if near_action else all_empty

    def move_diff(self, prev_board: Optional[HexBoard], curr_board: HexBoard) -> Optional[Cell]:
        """Return the single newly occupied cell between two boards, if detectable."""
        if prev_board is None or prev_board.size != curr_board.size:
            return None

        detected: Optional[Cell] = None
        for row in range(curr_board.size):
            for col in range(curr_board.size):
                prev_value = prev_board.board[row][col]
                curr_value = curr_board.board[row][col]
                if prev_value == curr_value:
                    continue
                if prev_value == 0 and curr_value != 0 and detected is None:
                    detected = (row, col)
                    continue
                return None
        return detected

    def axis_progress(self, a: Cell, b: Cell, player: int) -> int:
        """Measure progress along the winning axis for a player."""
        if player == 1:
            return abs(a[1] - b[1])
        return abs(a[0] - b[0])

    def lateral_drift(self, a: Cell, b: Cell, player: int) -> int:
        """Measure drift away from the winning axis for a player."""
        if player == 1:
            return abs(a[0] - b[0])
        return abs(a[1] - b[1])

    def cell_goal_progress(self, board: HexBoard, cell: Cell, player: int) -> float:
        """Return a positional value aligned with the player's target direction."""
        row, col = cell
        center = (board.size - 1) / 2.0
        if player == 1:
            return 18.0 * col - 3.0 * abs(row - center)
        return 18.0 * row - 3.0 * abs(col - center)

    def oriented_bridge(self, a: Cell, b: Cell, player: int) -> bool:
        """Check whether a bridge-like relation advances in the correct orientation."""
        main = self.axis_progress(a, b, player)
        side = self.lateral_drift(a, b, player)
        return main >= 1 and main >= side

    def orientation_score(self, a: Cell, b: Cell, player: int) -> float:
        """Return a geometric orientation score for a bridge from a player's perspective."""
        main = self.axis_progress(a, b, player)
        side = self.lateral_drift(a, b, player)

        score = 100.0 * main - 55.0 * side
        if main > side:
            score += 80.0
        if main >= side + 1:
            score += 40.0
        return score

    def clone_and_play(self, board: HexBoard, move: Cell, player: int) -> HexBoard:
        """Clone a board and apply a move."""
        clone = board.clone()
        clone.place_piece(move[0], move[1], player)
        return clone

    def board_full(self, board: HexBoard) -> bool:
        """Return True when no empty cells remain."""
        return not any(0 in row for row in board.board)

    def winner_on_board(self, board: HexBoard) -> int:
        """Return winner id if someone already connected, else 0."""
        if board.check_connection(1):
            return 1
        if board.check_connection(2):
            return 2
        return 0

    # =========================================================
    # Debug
    # =========================================================

    def debug_log(self, message: str) -> None:
        """Print a compact debug line only when debug mode is enabled."""
        if self.debug:
            print(f"[{self.__class__.__name__} {self.player_id}] {message}")

    def summarize_bridge_candidate(self, candidate: BridgeCandidate) -> str:
        """Build a short one-line summary for a bridge candidate."""
        return (
            f"A={candidate['anchor_a']} B={candidate['anchor_b']} "
            f"score={candidate['orientation_score']:.0f} "
            f"live={candidate['is_live']} "
            f"threat={candidate['is_immediate_threat']} "
            f"build={candidate['is_buildable']}"
        )

    def debug_top_bridges(self, label: str, candidates: list[BridgeCandidate], limit: int = 2) -> None:
        """Print only the top bridge candidates for a category."""
        if not self.debug:
            return
        if not candidates:
            self.debug_log(f"{label}: none")
            return
        summary = " | ".join(self.summarize_bridge_candidate(candidate) for candidate in candidates[:limit])
        self.debug_log(f"{label}: {summary}")

    # =========================================================
    # Layer 1: immediate tactics
    # =========================================================

    def immediate_winning_move_for_player(self, board: HexBoard, player: int) -> Optional[Cell]:
        """Return an immediate winning move for a specific player if one exists."""
        for move in self.all_empty_cells(board):
            clone = self.clone_and_play(board, move, player)
            if clone.check_connection(player):
                return move
        return None

    def immediate_threat_cells_for_player(self, board: HexBoard, player: int) -> list[Cell]:
        """Return cells that must be occupied to stop the opponent's immediate win."""
        enemy = self.opponent_id(player)
        threats: list[Cell] = []

        for move in self.all_empty_cells(board):
            clone = self.clone_and_play(board, move, enemy)
            if clone.check_connection(enemy):
                threats.append(move)

        return threats

    def immediate_winning_move(self, board: HexBoard) -> Optional[Cell]:
        """Return an immediate winning move for this player if one exists."""
        return self.immediate_winning_move_for_player(board, self.player_id)

    def immediate_block_move_for_player(self, board: HexBoard, player: int) -> Optional[Cell]:
        """Return a move that blocks an opponent immediate win if required."""
        threats = self.immediate_threat_cells_for_player(board, player)
        if not threats:
            return None
        if len(threats) == 1:
            return threats[0]

        best_move: Optional[Cell] = None
        best_score = float("-inf")
        for move in threats:
            score = self.score_move_for_player(board, move, player)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def immediate_block_move(self, board: HexBoard) -> Optional[Cell]:
        """Return a move that blocks an opponent immediate win if required."""
        return self.immediate_block_move_for_player(board, self.player_id)

    def defend_broken_bridge(self, board: HexBoard) -> Optional[Cell]:
        """Answer a direct hit on one support of a previously live own bridge."""
        enemy_last_move = self.move_diff(self.previous_board, board)
        if enemy_last_move is None or self.previous_board is None:
            return None

        for a, b, support1, support2 in self.live_bridges(self.previous_board, self.player_id):
            if enemy_last_move == support1 and board.board[support2[0]][support2[1]] == 0:
                self.debug_log(f"defend_broken_bridge: enemy hit {support1}, answering with {support2}")
                return support2
            if enemy_last_move == support2 and board.board[support1[0]][support1[1]] == 0:
                self.debug_log(f"defend_broken_bridge: enemy hit {support2}, answering with {support1}")
                return support1
        return None

    # =========================================================
    # Layer 2: pattern detection
    # =========================================================

    def common_neighbors(self, board: HexBoard, a: Cell, b: Cell) -> list[Cell]:
        """Return common neighbors between two cells."""
        return list(set(board.get_neighbors(*a)) & set(board.get_neighbors(*b)))

    def is_adjacent(self, board: HexBoard, a: Cell, b: Cell) -> bool:
        """Check whether two cells are direct neighbors."""
        return b in board.get_neighbors(*a)

    def bridge_supports(self, board: HexBoard, a: Cell, b: Cell) -> Optional[tuple[Cell, Cell]]:
        """Return the two support cells if a and b form a valid geometric two-bridge."""
        if a == b or self.is_adjacent(board, a, b):
            return None
        if a[0] == b[0] or a[1] == b[1]:
            return None

        common = self.common_neighbors(board, a, b)
        if len(common) != 2:
            return None
        support1, support2 = sorted(common)
        return support1, support2

    def supports_blocked_by_opponent(
        self,
        board: HexBoard,
        supports: tuple[Cell, Cell],
        player: int,
    ) -> bool:
        """Check whether any bridge support is already occupied by the opponent."""
        enemy = self.opponent_id(player)
        for row, col in supports:
            if board.board[row][col] == enemy:
                return True
        return False

    def bridge_candidates_for(self, board: HexBoard, player: int) -> list[BridgeCandidate]:
        """Build structured bridge candidates from a given player's perspective."""
        candidates: list[BridgeCandidate] = []
        seen: set[tuple[Cell, Cell]] = set()

        for row_a in range(board.size):
            for col_a in range(board.size):
                a = (row_a, col_a)
                for row_b in range(board.size):
                    for col_b in range(board.size):
                        b = (row_b, col_b)
                        key = tuple(sorted((a, b)))
                        if a == b or key in seen:
                            continue
                        seen.add(key)

                        supports = self.bridge_supports(board, a, b)
                        if supports is None:
                            continue

                        support1, support2 = supports
                        a_value = board.board[a[0]][a[1]]
                        b_value = board.board[b[0]][b[1]]
                        support1_value = board.board[support1[0]][support1[1]]
                        support2_value = board.board[support2[0]][support2[1]]

                        axis = self.axis_progress(a, b, player)
                        drift = self.lateral_drift(a, b, player)
                        score = self.orientation_score(a, b, player)
                        supports_blocked = self.supports_blocked_by_opponent(board, supports, player)

                        is_live = (
                            a_value == player
                            and b_value == player
                            and support1_value == 0
                            and support2_value == 0
                        )
                        is_immediate_threat = (
                            not supports_blocked
                            and ((a_value == player and b_value == 0) or (b_value == player and a_value == 0))
                        )
                        is_buildable = (
                            not supports_blocked
                            and (
                                (a_value == 0 and b_value == 0)
                                or (a_value == player and b_value == 0)
                                or (b_value == player and a_value == 0)
                            )
                        )

                        candidates.append(
                            {
                                "anchor_a": a,
                                "anchor_b": b,
                                "support_1": support1,
                                "support_2": support2,
                                "owner_perspective": player,
                                "axis_progress": axis,
                                "lateral_drift": drift,
                                "orientation_score": score,
                                "is_live": is_live,
                                "is_immediate_threat": is_immediate_threat,
                                "is_buildable": is_buildable,
                            }
                        )

        candidates.sort(key=lambda candidate: candidate["orientation_score"], reverse=True)
        return candidates

    def live_bridges(self, board: HexBoard, player: int) -> list[BridgeRecord]:
        """Return existing bridges whose two supports are both empty."""
        bridges: list[BridgeRecord] = []
        for candidate in self.bridge_candidates_for(board, player):
            if not candidate["is_live"]:
                continue
            bridges.append(
                (
                    candidate["anchor_a"],
                    candidate["anchor_b"],
                    candidate["support_1"],
                    candidate["support_2"],
                )
            )
        return bridges

    def bridge_corridor_pressure(self, board: HexBoard, cell: Cell, player: int) -> float:
        """Measure how naturally a cell sits on the player's connection corridor."""
        return self.cell_goal_progress(board, cell, player)

    def enemy_bridge_priority_for_player(
        self,
        board: HexBoard,
        candidate: BridgeCandidate,
        player: int,
    ) -> float:
        """Rank enemy bridge threats using only defensive bridge geometry criteria."""
        enemy = candidate["owner_perspective"]
        anchor_a = candidate["anchor_a"]
        anchor_b = candidate["anchor_b"]
        support_1 = candidate["support_1"]
        support_2 = candidate["support_2"]

        occupied_anchors = 0
        if board.board[anchor_a[0]][anchor_a[1]] == enemy:
            occupied_anchors += 1
        if board.board[anchor_b[0]][anchor_b[1]] == enemy:
            occupied_anchors += 1

        corridor_score = max(
            self.bridge_corridor_pressure(board, anchor_a, enemy),
            self.bridge_corridor_pressure(board, anchor_b, enemy),
        )

        usable_supports = 0
        for support in (support_1, support_2):
            if board.board[support[0]][support[1]] != player:
                usable_supports += 1

        score = candidate["orientation_score"]
        score += 90.0 * occupied_anchors
        score += 4.0 * corridor_score
        score += 35.0 * usable_supports
        if candidate["is_immediate_threat"]:
            score += 1000.0
        elif candidate["is_buildable"]:
            score += 400.0
        return score

    def enemy_bridge_priority(self, board: HexBoard, candidate: BridgeCandidate) -> float:
        """Rank enemy bridge threats against this player."""
        return self.enemy_bridge_priority_for_player(board, candidate, self.player_id)

    def enemy_bridge_candidates_for_player(self, board: HexBoard, player: int) -> list[BridgeCandidate]:
        """Return enemy bridge threats ordered by defensive urgency for a given player."""
        enemy = self.opponent_id(player)
        candidates: list[BridgeCandidate] = []
        for candidate in self.bridge_candidates_for(board, enemy):
            if not candidate["is_immediate_threat"] and not candidate["is_buildable"]:
                continue
            candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: (
                1 if candidate["is_immediate_threat"] else 0,
                self.enemy_bridge_priority_for_player(board, candidate, player),
            ),
            reverse=True,
        )
        return candidates

    def enemy_bridge_candidates(self, board: HexBoard) -> list[BridgeCandidate]:
        """Return enemy bridge threats ordered by defensive urgency."""
        return self.enemy_bridge_candidates_for_player(board, self.player_id)

    def block_enemy_bridge_for_player(self, board: HexBoard, player: int) -> Optional[Cell]:
        """Block the highest-priority enemy bridge by taking its key empty anchor."""
        candidates = self.enemy_bridge_candidates_for_player(board, player)
        if not candidates:
            return None

        for candidate in candidates:
            empty_anchors: list[Cell] = []
            for anchor in (candidate["anchor_a"], candidate["anchor_b"]):
                if board.board[anchor[0]][anchor[1]] == 0:
                    empty_anchors.append(anchor)

            if not empty_anchors:
                continue

            empty_anchors.sort(
                key=lambda anchor: self.bridge_corridor_pressure(
                    board,
                    anchor,
                    candidate["owner_perspective"],
                ),
                reverse=True,
            )
            return empty_anchors[0]

        return None

    def block_enemy_bridge(self, board: HexBoard) -> Optional[Cell]:
        """Block the highest-priority enemy bridge by taking its key empty anchor."""
        return self.block_enemy_bridge_for_player(board, self.player_id)

    # =========================================================
    # Layer 4: attack
    # =========================================================

    def borders_touched_by_player(self, board: HexBoard, player: int) -> set[str]:
        """Return which winning-side borders are already touched by the player's stones."""
        touched: set[str] = set()
        size = board.size

        for row in range(size):
            for col in range(size):
                if board.board[row][col] != player:
                    continue
                if player == 1:
                    if col == 0:
                        touched.add("start")
                    if col == size - 1:
                        touched.add("goal")
                else:
                    if row == 0:
                        touched.add("start")
                    if row == size - 1:
                        touched.add("goal")

        return touched

    def touches_border_side(self, board: HexBoard, cell: Cell, player: int, side: str) -> bool:
        """Check whether a cell lies on the requested winning border side."""
        row, col = cell
        if player == 1:
            return col == 0 if side == "start" else col == board.size - 1
        return row == 0 if side == "start" else row == board.size - 1

    def count_bridge_links_from_move(self, board: HexBoard, move: Cell, player: int) -> int:
        """Estimate how strongly a move links into the player's bridge network."""
        if board.board[move[0]][move[1]] != 0:
            return -999

        clone = self.clone_and_play(board, move, player)
        links = 0

        for candidate in self.bridge_candidates_for(clone, player):
            if not candidate["is_live"]:
                continue
            if move not in (candidate["anchor_a"], candidate["anchor_b"]):
                continue

            other_anchor = candidate["anchor_b"] if candidate["anchor_a"] == move else candidate["anchor_a"]
            if clone.board[other_anchor[0]][other_anchor[1]] == player:
                links += 3
            else:
                links += 1

            for anchor in (candidate["anchor_a"], candidate["anchor_b"]):
                for nr, nc in clone.get_neighbors(*anchor):
                    if clone.board[nr][nc] == player and (nr, nc) != move:
                        links += 1

        return links

    def my_bridge_priority_for_player(
        self,
        board: HexBoard,
        candidate: BridgeCandidate,
        player: int,
    ) -> float:
        """Rank own bridge candidates using only bridge-building offensive criteria."""
        anchor_a = candidate["anchor_a"]
        anchor_b = candidate["anchor_b"]
        support_1 = candidate["support_1"]
        support_2 = candidate["support_2"]

        occupied_anchors = 0
        empty_anchors: list[Cell] = []
        for anchor in (anchor_a, anchor_b):
            value = board.board[anchor[0]][anchor[1]]
            if value == player:
                occupied_anchors += 1
            elif value == 0:
                empty_anchors.append(anchor)

        supports_free = 0
        for support in (support_1, support_2):
            if board.board[support[0]][support[1]] == 0:
                supports_free += 1

        touched_borders = self.borders_touched_by_player(board, player)
        border_projection = 0.0
        for anchor in (anchor_a, anchor_b):
            if self.touches_border_side(board, anchor, player, "start") and "start" not in touched_borders:
                border_projection += 90.0
            if self.touches_border_side(board, anchor, player, "goal") and "goal" not in touched_borders:
                border_projection += 90.0

        best_link_gain = 0
        for anchor in empty_anchors:
            best_link_gain = max(best_link_gain, self.count_bridge_links_from_move(board, anchor, player))

        score = candidate["orientation_score"]
        score += 160.0 * occupied_anchors
        score += 70.0 * supports_free
        score += 65.0 * best_link_gain
        score += border_projection

        if candidate["is_immediate_threat"]:
            score += 220.0
        elif candidate["is_buildable"]:
            score += 120.0

        if candidate["lateral_drift"] > candidate["axis_progress"]:
            score -= 240.0
        if best_link_gain <= 0:
            score -= 140.0

        return score

    def my_bridge_priority(self, board: HexBoard, candidate: BridgeCandidate) -> float:
        """Rank own bridge candidates for this player."""
        return self.my_bridge_priority_for_player(board, candidate, self.player_id)

    def my_bridge_candidates(self, board: HexBoard) -> list[BridgeRecord]:
        """Return own live bridge patterns ordered by offensive value."""
        candidates = self.bridge_candidates_for(board, self.player_id)
        candidates = [
            candidate
            for candidate in candidates
            if candidate["is_live"] or candidate["is_immediate_threat"] or candidate["is_buildable"]
        ]
        candidates.sort(key=lambda candidate: self.my_bridge_priority(board, candidate), reverse=True)

        bridges: list[BridgeRecord] = []
        for candidate in candidates:
            bridges.append(
                (
                    candidate["anchor_a"],
                    candidate["anchor_b"],
                    candidate["support_1"],
                    candidate["support_2"],
                )
            )
        return bridges

    def build_my_bridge_for_player(self, board: HexBoard, player: int) -> Optional[Cell]:
        """Build the most valuable own bridge for a given player."""
        candidates = self.bridge_candidates_for(board, player)
        best_move: Optional[Cell] = None
        best_score = float("-inf")

        for candidate in candidates:
            if not candidate["is_buildable"]:
                continue
            move_options: list[Cell] = []
            for move in (candidate["anchor_a"], candidate["anchor_b"]):
                if board.board[move[0]][move[1]] != 0:
                    continue
                move_options.append(move)

            for move in move_options:
                link_gain = self.count_bridge_links_from_move(board, move, player)
                if link_gain <= 0 and not candidate["is_immediate_threat"]:
                    continue

                move_score = self.my_bridge_priority_for_player(board, candidate, player)
                move_score += 40.0 * link_gain

                adjacent_own = 0
                for nr, nc in board.get_neighbors(*move):
                    if board.board[nr][nc] == player:
                        adjacent_own += 1
                if link_gain == 0 and adjacent_own > 0:
                    move_score -= 120.0

                if move_score > best_score:
                    best_score = move_score
                    best_move = move

        return best_move

    def build_my_bridge(self, board: HexBoard) -> Optional[Cell]:
        """Build the most valuable own bridge, avoiding decorative or merely adjacent moves."""
        return self.build_my_bridge_for_player(board, self.player_id)

    # =========================================================
    # Layer 5: fallback
    # =========================================================

    def connection_cost(self, cell_value: int, player: int) -> float:
        """Return path cost for a cell from a player's perspective."""
        if cell_value == player:
            return 0.0
        if cell_value == 0:
            return 1.0
        return 999999.0

    def shortest_connection_path(self, board: HexBoard, player: int) -> tuple[float, list[Cell], set[Cell]]:
        """Return minimal path cost, ordered best path and empty cells on that path."""
        size = board.size
        dist: dict[Cell, float] = {}
        parent: dict[Cell, Optional[Cell]] = {}
        heap: list[tuple[float, int, int]] = []

        if player == 1:
            starts: Iterable[Cell] = [(row, 0) for row in range(size)]
            is_goal = lambda cell: cell[1] == size - 1
        else:
            starts = [(0, col) for col in range(size)]
            is_goal = lambda cell: cell[0] == size - 1

        for cell in starts:
            row, col = cell
            cost = self.connection_cost(board.board[row][col], player)
            if cost >= 999999.0:
                continue
            dist[cell] = cost
            parent[cell] = None
            heapq.heappush(heap, (cost, row, col))

        best_goal: Optional[Cell] = None
        best_goal_cost = float("inf")

        while heap:
            cost, row, col = heapq.heappop(heap)
            cell = (row, col)
            if cost != dist.get(cell):
                continue

            if is_goal(cell):
                best_goal = cell
                best_goal_cost = cost
                break

            for nr, nc in board.get_neighbors(row, col):
                next_cost = self.connection_cost(board.board[nr][nc], player)
                if next_cost >= 999999.0:
                    continue
                candidate = cost + next_cost
                next_cell = (nr, nc)
                if candidate < dist.get(next_cell, float("inf")):
                    dist[next_cell] = candidate
                    parent[next_cell] = cell
                    heapq.heappush(heap, (candidate, nr, nc))

        ordered_path: list[Cell] = []
        path_empties: set[Cell] = set()
        if best_goal is not None:
            reversed_path: list[Cell] = []
            cursor: Optional[Cell] = best_goal
            while cursor is not None:
                reversed_path.append(cursor)
                row, col = cursor
                if board.board[row][col] == 0:
                    path_empties.add(cursor)
                cursor = parent.get(cursor)
            ordered_path = list(reversed(reversed_path))

        return best_goal_cost, ordered_path, path_empties

    def dedupe_cells(self, cells: Iterable[Cell]) -> list[Cell]:
        """Preserve order while removing duplicate cells."""
        unique: list[Cell] = []
        seen: set[Cell] = set()

        for cell in cells:
            if cell in seen:
                continue
            seen.add(cell)
            unique.append(cell)

        return unique

    def candidate_pool(self, board: HexBoard) -> list[Cell]:
        """Return fallback candidates from bridges, paths and local action."""
        enemy = self.opponent()
        my_candidates = self.bridge_candidates_for(board, self.player_id)
        enemy_candidates = self.enemy_bridge_candidates(board)
        _my_cost, my_ordered_path, _my_path_empties = self.shortest_connection_path(board, self.player_id)
        _enemy_cost, enemy_ordered_path, _enemy_path_empties = self.shortest_connection_path(board, enemy)

        ordered: list[Cell] = []

        for candidate in enemy_candidates:
            for anchor in (candidate["anchor_a"], candidate["anchor_b"]):
                if board.board[anchor[0]][anchor[1]] == 0:
                    ordered.append(anchor)

        for candidate in my_candidates:
            if not candidate["is_buildable"] and not candidate["is_immediate_threat"]:
                continue
            for anchor in (candidate["anchor_a"], candidate["anchor_b"]):
                if board.board[anchor[0]][anchor[1]] == 0:
                    ordered.append(anchor)

        for cell in my_ordered_path:
            if board.board[cell[0]][cell[1]] == 0:
                ordered.append(cell)
        for cell in enemy_ordered_path:
            if board.board[cell[0]][cell[1]] == 0:
                ordered.append(cell)
        for move in self.valid_moves_near_action(board):
            ordered.append(move)

        deduped = [cell for cell in self.dedupe_cells(ordered) if board.board[cell[0]][cell[1]] == 0]
        if not deduped:
            return self.all_empty_cells(board)
        return deduped[:20]

    def move_creates_bridge(self, board: HexBoard, move: Cell, player: int) -> bool:
        """Return True if playing move forms at least one valid bridge with an existing stone."""
        if board.board[move[0]][move[1]] != 0:
            return False

        clone = self.clone_and_play(board, move, player)
        for row in range(clone.size):
            for col in range(clone.size):
                if (row, col) == move:
                    continue
                if clone.board[row][col] != player:
                    continue
                if self.bridge_supports(clone, move, (row, col)) is not None:
                    return True
        return False

    def score_move_for_player(self, board: HexBoard, move: Cell, player: int) -> float:
        """Evaluate a fallback move using path pressure and resulting bridge structure."""
        current_my_cost, current_my_ordered_path, current_my_path = self.shortest_connection_path(board, player)
        current_enemy_cost, current_enemy_ordered_path, current_enemy_path = self.shortest_connection_path(
            board, self.opponent_id(player)
        )
        clone = self.clone_and_play(board, move, player)
        my_cost, my_ordered_path, my_path = self.shortest_connection_path(clone, player)
        enemy_cost, enemy_ordered_path, enemy_path = self.shortest_connection_path(clone, self.opponent_id(player))
        live_bridges_after = len(self.live_bridges(clone, player))
        live_bridges_before = len(self.live_bridges(board, player))
        bridge_gain = live_bridges_after - live_bridges_before
        creates_bridge = self.move_creates_bridge(board, move, player)

        score = 0.0
        score += 110.0 * self.cell_goal_progress(board, move, player)
        score += 130.0 * (current_my_cost - my_cost)
        score += 95.0 * (enemy_cost - current_enemy_cost)
        score += 150.0 * bridge_gain

        if move in my_path or move in my_ordered_path:
            score += 140.0
        if move in current_my_path or move in current_my_ordered_path:
            score += 80.0
        if (
            move in enemy_path
            or move in enemy_ordered_path
            or move in current_enemy_path
            or move in current_enemy_ordered_path
        ):
            score += 110.0

        for nr, nc in board.get_neighbors(*move):
            if board.board[nr][nc] == player:
                score += 10.0 if creates_bridge else -22.0
            elif board.board[nr][nc] == self.opponent_id(player):
                score += 12.0

        if not creates_bridge:
            adjacent_own = 0
            for nr, nc in board.get_neighbors(*move):
                if board.board[nr][nc] == player:
                    adjacent_own += 1
            score -= 24.0 * adjacent_own

        return score

    def score_move(self, board: HexBoard, move: Cell) -> float:
        """Evaluate a fallback move for this player."""
        return self.score_move_for_player(board, move, self.player_id)

    def strategic_fallback_move_for_player(self, board: HexBoard, player: int) -> Cell:
        """Choose the best remaining move for an arbitrary player."""
        candidates = self.candidate_pool(board) if player == self.player_id else self.all_empty_cells(board)
        best_move = candidates[0]
        best_score = float("-inf")

        for move in candidates:
            score = self.score_move_for_player(board, move, player)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def strategic_fallback_move(self, board: HexBoard) -> Cell:
        """Choose the best remaining move from the fallback candidate pool."""
        return self.strategic_fallback_move_for_player(board, self.player_id)

    # =========================================================
    # Shared heuristic bias for MCTS
    # =========================================================

    def heuristic_move_score_for_player(self, board: HexBoard, move: Cell, player: int) -> float:
        """Rank a move using lightweight tactical and structural heuristics."""
        if board.board[move[0]][move[1]] != 0:
            return float("-inf")

        enemy = self.opponent_id(player)
        threat_cells = set(self.immediate_threat_cells_for_player(board, player))
        score = 0.0
        score += 35.0 * self.cell_goal_progress(board, move, player)

        clone = self.clone_and_play(board, move, player)
        if clone.check_connection(player):
            score += 100000.0

        if move in threat_cells:
            score += 80000.0

        bridge_block = self.block_enemy_bridge_for_player(board, player)
        if bridge_block == move:
            score += 12000.0

        build_move = self.build_my_bridge_for_player(board, player)
        if build_move == move:
            score += 9000.0

        if self.move_creates_bridge(board, move, player):
            score += 2200.0

        score += 140.0 * max(0, self.count_bridge_links_from_move(board, move, player))

        clone = self.clone_and_play(board, move, player)
        score += 250.0 * len(self.live_bridges(clone, player))

        for nr, nc in board.get_neighbors(*move):
            if board.board[nr][nc] == player:
                score += 35.0
            elif board.board[nr][nc] == enemy:
                score += 20.0

        return score

    def ordered_moves_for_player(self, board: HexBoard, player: int) -> list[Cell]:
        """Return all legal moves ordered by heuristic value."""
        moves = self.all_empty_cells(board)
        moves.sort(
            key=lambda move: (
                self.heuristic_move_score_for_player(board, move, player),
                random.random(),
            ),
            reverse=True,
        )
        return moves

    def rollout_policy_move(self, board: HexBoard, player: int, top_k: int) -> Cell:
        """Guided playout policy with light tactical bias."""
        move = self.immediate_winning_move_for_player(board, player)
        if move is not None:
            return move

        move = self.immediate_block_move_for_player(board, player)
        if move is not None:
            return move

        move = self.block_enemy_bridge_for_player(board, player)
        if move is not None:
            return move

        move = self.build_my_bridge_for_player(board, player)
        if move is not None:
            return move

        ordered = self.ordered_moves_for_player(board, player)
        shortlist = ordered[: max(1, min(top_k, len(ordered)))]
        return random.choice(shortlist)


class SmartPlayer(HexHeuristicMixin, Player):
    """Baseline heuristic player kept intact as a deterministic reference agent."""

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.previous_board = None
        self.debug = False

    def play(self, board: HexBoard) -> Cell:
        """Choose a move using strict tactical-to-strategic layer ordering."""
        enemy_last_move = self.move_diff(self.previous_board, board)
        if self.debug:
            self.debug_log(f"enemy_last_move={enemy_last_move}")
            self.debug_top_bridges("enemy_bridges", self.enemy_bridge_candidates(board))
            own_candidates = self.bridge_candidates_for(board, self.player_id)
            own_candidates.sort(key=lambda candidate: self.my_bridge_priority(board, candidate), reverse=True)
            self.debug_top_bridges("my_bridges", own_candidates)

        reason = "strategic_fallback_move"
        move = self.immediate_winning_move(board)
        if move is not None:
            reason = "immediate_winning_move"
        if move is None:
            move = self.immediate_block_move(board)
            if move is not None:
                reason = "immediate_block_move"
        if move is None:
            move = self.defend_broken_bridge(board)
            if move is not None:
                reason = "defend_broken_bridge"
        if move is None:
            move = self.block_enemy_bridge(board)
            if move is not None:
                reason = "block_enemy_bridge"
        if move is None:
            move = self.build_my_bridge(board)
            if move is not None:
                reason = "build_my_bridge"
        if move is None:
            move = self.strategic_fallback_move(board)

        self.debug_log(f"chosen_move={move} reason={reason}")
        self.previous_board = self.clone_and_play(board, move, self.player_id)
        return move


class _MCTSNode:
    """Single-threaded MCTS node with root-centric win accounting."""

    def __init__(
        self,
        board: HexBoard,
        player_to_move: int,
        parent: Optional[_MCTSNode] = None,
        move_from_parent: Optional[Cell] = None,
        untried_moves: Optional[list[Cell]] = None,
    ):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children: list[_MCTSNode] = []
        self.untried_moves = untried_moves or []
        self.visits = 0
        self.wins = 0.0

    def is_terminal(self) -> bool:
        """Return True if the position is finished."""
        return (
            self.board.check_connection(1)
            or self.board.check_connection(2)
            or not any(0 in row for row in self.board.board)
        )

    def fully_expanded(self) -> bool:
        """Return True if no untried moves remain."""
        return not self.untried_moves


class SmartPlayerMCTS(HexHeuristicMixin, Player):
    """MCTS agent guided by the same bridge and tactical heuristics as the baseline."""

    def __init__(
        self,
        player_id: int,
        max_time: float = 4.8,
        exploration_c: float = 1.2,
        rollout_top_k: int = 6,
        max_rollout_depth: int = 80,
        debug: bool = False,
    ):
        super().__init__(player_id)
        self.previous_board = None
        self.max_time = max_time
        self.exploration_c = exploration_c
        self.rollout_top_k = rollout_top_k
        self.max_rollout_depth = max_rollout_depth
        self.debug = debug

    def create_node(
        self,
        board: HexBoard,
        player_to_move: int,
        parent: Optional[_MCTSNode] = None,
        move_from_parent: Optional[Cell] = None,
    ) -> _MCTSNode:
        """Create a node with heuristically ordered legal moves."""
        return _MCTSNode(
            board=board,
            player_to_move=player_to_move,
            parent=parent,
            move_from_parent=move_from_parent,
            untried_moves=self.ordered_moves_for_player(board, player_to_move),
        )

    def uct_score(self, parent_visits: int, child: _MCTSNode) -> float:
        """Compute standard UCT score from the root player's perspective."""
        if child.visits == 0:
            return float("inf")
        exploitation = child.wins / child.visits
        exploration = self.exploration_c * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration

    def select_child(self, node: _MCTSNode) -> _MCTSNode:
        """Select the next node using UCT."""
        return max(node.children, key=lambda child: self.uct_score(max(1, node.visits), child))

    def expand(self, node: _MCTSNode) -> _MCTSNode:
        """Expand one untried move from the node."""
        move = node.untried_moves.pop(0)
        child_board = self.clone_and_play(node.board, move, node.player_to_move)
        child = self.create_node(
            board=child_board,
            player_to_move=self.opponent_id(node.player_to_move),
            parent=node,
            move_from_parent=move,
        )
        node.children.append(child)
        return child

    def simulate(self, board: HexBoard, player_to_move: int) -> int:
        """Run a guided rollout and return the winner id, or 0 if unresolved."""
        current_player = player_to_move
        depth = 0

        while depth < self.max_rollout_depth:
            winner = self.winner_on_board(board)
            if winner != 0:
                return winner
            if self.board_full(board):
                return 0

            move = self.rollout_policy_move(board, current_player, self.rollout_top_k)
            if board.place_piece(move[0], move[1], current_player) is False:
                legal = self.all_empty_cells(board)
                if not legal:
                    return 0
                move = random.choice(legal)
                board.place_piece(move[0], move[1], current_player)

            current_player = self.opponent_id(current_player)
            depth += 1

        return self.winner_on_board(board)

    def backpropagate(self, node: _MCTSNode, winner: int, root_player: int) -> None:
        """Propagate root-centric wins and visit counts up the tree."""
        current: Optional[_MCTSNode] = node
        while current is not None:
            current.visits += 1
            if winner == root_player:
                current.wins += 1.0
            elif winner == 0:
                current.wins += 0.5
            current = current.parent

    def run_mcts(self, board: HexBoard) -> tuple[Cell, int, _MCTSNode]:
        """Run MCTS from the current position until the time budget expires."""
        root = self.create_node(board.clone(), self.player_id)
        deadline = time.perf_counter() + self.max_time
        iterations = 0

        while time.perf_counter() < deadline:
            node = root

            while node.fully_expanded() and node.children and not node.is_terminal():
                node = self.select_child(node)

            if not node.is_terminal() and node.untried_moves:
                node = self.expand(node)

            rollout_board = node.board.clone()
            winner = self.simulate(rollout_board, node.player_to_move)
            self.backpropagate(node, winner, self.player_id)
            iterations += 1

        if not root.children:
            fallback = self.strategic_fallback_move(board)
            return fallback, iterations, root

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move_from_parent or self.strategic_fallback_move(board), iterations, root

    def debug_root_summary(self, root: _MCTSNode, iterations: int, best_move: Cell) -> None:
        """Print a compact root summary for the MCTS search."""
        if not self.debug:
            return

        top_children = sorted(root.children, key=lambda child: child.visits, reverse=True)[:3]
        if not top_children:
            self.debug_log(f"mcts: iterations={iterations} best={best_move} children=none")
            return

        best_child = top_children[0]
        summary = " | ".join(
            f"{child.move_from_parent}:v={child.visits},wr={(child.wins / child.visits):.2f}"
            for child in top_children
            if child.visits > 0
        )
        self.debug_log(
            f"mcts: iterations={iterations} best={best_move} best_visits={best_child.visits} top={summary}"
        )

    def play(self, board: HexBoard) -> Cell:
        """Use hard tactics first; otherwise use MCTS as the main decision engine."""
        move = self.immediate_winning_move(board)
        if move is not None:
            if self.debug:
                self.debug_log(f"hard_tactic=immediate_winning_move move={move}")
            self.previous_board = self.clone_and_play(board, move, self.player_id)
            return move

        move = self.immediate_block_move(board)
        if move is not None:
            if self.debug:
                self.debug_log(f"hard_tactic=immediate_block_move move={move}")
            self.previous_board = self.clone_and_play(board, move, self.player_id)
            return move

        move = self.defend_broken_bridge(board)
        if move is not None:
            if self.debug:
                self.debug_log(f"hard_tactic=defend_broken_bridge move={move}")
            self.previous_board = self.clone_and_play(board, move, self.player_id)
            return move

        move, iterations, root = self.run_mcts(board)
        if self.debug:
            self.debug_root_summary(root, iterations, move)

        self.previous_board = self.clone_and_play(board, move, self.player_id)
        return move
