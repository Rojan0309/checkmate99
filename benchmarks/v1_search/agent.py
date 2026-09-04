"""A compact classical chess engine for the AI Chessathon.

The implementation deliberately uses python-chess for reliable move generation. Strength comes
from iterative-deepening principal-variation search, tactical quiescence, a persistent
transposition table, and a tapered handcrafted evaluation. No files, network, subprocesses, or
third-party engines are used at runtime.
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Final

import chess

INF: Final = 40_000
MATE: Final = 32_000
MATE_BOUND: Final = 31_000
MAX_PLY: Final = 96
TT_LIMIT: Final = 250_000

EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

PIECE_VALUE: Final = (0, 100, 320, 330, 500, 900, 0)
PHASE_VALUE: Final = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE: Final = 24

# PeSTO-style middlegame/endgame piece-square values. The tables are indexed from White's
# point of view; Black's squares are mirrored vertically.
MG_VALUE: Final = (0, 82, 337, 365, 477, 1025, 0)
EG_VALUE: Final = (0, 94, 281, 297, 512, 936, 0)

MG_TABLES: Final = (
    (),
    (
        0, 0, 0, 0, 0, 0, 0, 0,
        98, 134, 61, 95, 68, 126, 34, -11,
        -6, 7, 26, 31, 65, 56, 25, -20,
        -14, 13, 6, 21, 23, 12, 17, -23,
        -27, -2, -5, 12, 17, 6, 10, -25,
        -26, -4, -4, -10, 3, 3, 33, -12,
        -35, -1, -20, -23, -15, 24, 38, -22,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    (
        -167, -89, -34, -49, 61, -97, -15, -107,
        -73, -41, 72, 36, 23, 62, 7, -17,
        -47, 60, 37, 65, 84, 129, 73, 44,
        -9, 17, 19, 53, 37, 69, 18, 22,
        -13, 4, 16, 13, 28, 19, 21, -8,
        -23, -9, 12, 10, 19, 17, 25, -16,
        -29, -53, -12, -3, -1, 18, -14, -19,
        -105, -21, -58, -33, -17, -28, -19, -23,
    ),
    (
        -29, 4, -82, -37, -25, -42, 7, -8,
        -26, 16, -18, -13, 30, 59, 18, -47,
        -16, 37, 43, 40, 35, 50, 37, -2,
        -4, 5, 19, 50, 37, 37, 7, -2,
        -6, 13, 13, 26, 34, 12, 10, 4,
        0, 15, 15, 15, 14, 27, 18, 10,
        4, 15, 16, 0, 7, 21, 33, 1,
        -33, -3, -14, -21, -13, -12, -39, -21,
    ),
    (
        32, 42, 32, 51, 63, 9, 31, 43,
        27, 32, 58, 62, 80, 67, 26, 44,
        -5, 19, 26, 36, 17, 45, 61, 16,
        -24, -11, 7, 26, 24, 35, -8, -20,
        -36, -26, -12, -1, 9, -7, 6, -23,
        -45, -25, -16, -17, 3, 0, -5, -33,
        -44, -16, -20, -9, -1, 11, -6, -71,
        -19, -13, 1, 17, 16, 7, -37, -26,
    ),
    (
        -28, 0, 29, 12, 59, 44, 43, 45,
        -24, -39, -5, 1, -16, 57, 28, 54,
        -13, -17, 7, 8, 29, 56, 47, 57,
        -27, -27, -16, -16, -1, 17, -2, 1,
        -9, -26, -9, -10, -2, -4, 3, -3,
        -14, 2, -11, -2, -5, 2, 14, 5,
        -35, -8, 11, 2, 8, 15, -3, 1,
        -1, -18, -9, 10, -15, -25, -31, -50,
    ),
    (
        -65, 23, 16, -15, -56, -34, 2, 13,
        29, -1, -20, -7, -8, -4, -38, -29,
        -9, 24, 2, -16, -20, 6, 22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49, -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1, 7, -8, -64, -43, -16, 9, 8,
        -15, 36, 12, -54, 8, -28, 24, 14,
    ),
)

EG_TABLES: Final = (
    (),
    (
        0, 0, 0, 0, 0, 0, 0, 0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100, 85, 67, 56, 53, 82, 84,
        32, 24, 13, 5, -2, 4, 17, 17,
        13, 9, -3, -7, -7, -8, 3, -1,
        4, 7, -6, 1, 0, -5, -1, -8,
        13, 8, 8, 10, 13, 0, 2, -7,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    (
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25, -8, -25, -2, -9, -25, -24, -52,
        -24, -20, 10, 9, -1, -9, -19, -41,
        -17, 3, 22, 22, 22, 11, 8, -18,
        -18, -6, 16, 25, 16, 17, 4, -18,
        -23, -3, -1, 15, 10, -3, -20, -22,
        -42, -20, -10, -5, -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ),
    (
        -14, -21, -11, -8, -7, -9, -17, -24,
        -8, -4, 7, -12, -3, -13, -4, -14,
        2, -8, 0, -1, -2, 6, 0, 4,
        -3, 9, 12, 9, 14, 10, 3, 2,
        -6, 3, 13, 19, 7, 10, -3, -9,
        -12, -3, 8, 10, 13, 3, -7, -15,
        -14, -18, -7, -1, 4, -9, -15, -27,
        -23, -9, -23, -5, -9, -16, -5, -17,
    ),
    (
        13, 10, 18, 15, 12, 12, 8, 5,
        11, 13, 13, 11, -3, 3, 8, 3,
        7, 7, 7, 5, 4, -3, -5, -3,
        4, 3, 13, 1, 2, 1, -1, 2,
        3, 5, 8, 4, -5, -6, -8, -11,
        -4, 0, -5, -1, -7, -12, -8, -16,
        -6, -6, 0, 2, -9, -9, -11, -3,
        -9, 2, 3, -1, -5, -13, 4, -20,
    ),
    (
        -9, 22, 22, 27, 27, 19, 10, 20,
        -17, 20, 32, 41, 58, 25, 30, 0,
        -20, 6, 9, 49, 47, 35, 19, 9,
        3, 22, 24, 45, 57, 40, 57, 36,
        -18, 28, 19, 47, 31, 34, 39, 23,
        -16, -27, 15, 6, 9, 17, 10, 5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43, -5, -32, -20, -41,
    ),
    (
        -74, -35, -18, -18, -11, 15, 4, -17,
        -12, 17, 14, 17, 17, 38, 23, 11,
        10, 17, 23, 15, 20, 45, 44, 13,
        -8, 22, 24, 27, 26, 33, 26, 3,
        -18, -4, 21, 24, 27, 23, 9, -11,
        -19, -3, 11, 21, 23, 16, 7, -9,
        -27, -11, 4, 13, 14, 4, -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ),
)

FILE_MASKS: Final = tuple(chess.BB_FILES[file_index] for file_index in range(8))
ADJACENT_FILES: Final = tuple(
    (chess.BB_FILES[file_index - 1] if file_index else 0)
    | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
    for file_index in range(8)
)


class SearchTimeout(Exception):
    """Raised internally to unwind an incomplete iteration."""


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    bound: int
    move: chess.Move | None
    age: int


def _key(board: chess.Board) -> Hashable:
    return board._transposition_key()


def _relative_rank(color: chess.Color, square: chess.Square) -> int:
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _passed_mask(color: chess.Color, square: chess.Square) -> int:
    file_index = chess.square_file(square)
    files = FILE_MASKS[file_index] | ADJACENT_FILES[file_index]
    rank = chess.square_rank(square)
    ahead = ~((1 << ((rank + 1) * 8)) - 1) if color == chess.WHITE else (1 << (rank * 8)) - 1
    return files & ahead & chess.BB_ALL


PASSED_MASKS: Final = tuple(
    tuple(_passed_mask(color, square) for square in chess.SQUARES)
    for color in (chess.BLACK, chess.WHITE)
)


def evaluate(board: chess.Board) -> int:
    """Return a tapered positional score from the side-to-move's perspective."""
    mg = 0
    eg = 0
    phase = 0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for piece_type in chess.PIECE_TYPES:
            pieces = board.pieces_mask(piece_type, color)
            count = pieces.bit_count()
            phase += PHASE_VALUE[piece_type] * count
            for square in chess.scan_reversed(pieces):
                table_square = chess.square_mirror(square) if color == chess.WHITE else square
                mg += sign * (MG_VALUE[piece_type] + MG_TABLES[piece_type][table_square])
                eg += sign * (EG_VALUE[piece_type] + EG_TABLES[piece_type][table_square])

        bishops = board.pieces_mask(chess.BISHOP, color)
        if bishops.bit_count() >= 2:
            mg += sign * 28
            eg += sign * 38

        pawns = board.pieces_mask(chess.PAWN, color)
        enemy_pawns = board.pieces_mask(chess.PAWN, not color)
        for file_index in range(8):
            file_count = (pawns & FILE_MASKS[file_index]).bit_count()
            if file_count > 1:
                mg -= sign * 11 * (file_count - 1)
                eg -= sign * 16 * (file_count - 1)

        for square in chess.scan_reversed(pawns):
            file_index = chess.square_file(square)
            if not pawns & ADJACENT_FILES[file_index]:
                mg -= sign * 9
                eg -= sign * 14
            if not enemy_pawns & PASSED_MASKS[int(color)][square]:
                rank = _relative_rank(color, square)
                mg += sign * (rank * rank * 3)
                eg += sign * (rank * rank * 7)

        own_pawn_files = 0
        for file_index in range(8):
            if pawns & FILE_MASKS[file_index]:
                own_pawn_files |= 1 << file_index
        all_pawns = pawns | enemy_pawns
        for square in chess.scan_reversed(board.pieces_mask(chess.ROOK, color)):
            file_index = chess.square_file(square)
            if not all_pawns & FILE_MASKS[file_index]:
                mg += sign * 20
                eg += sign * 12
            elif not own_pawn_files & (1 << file_index):
                mg += sign * 10

    phase = min(phase, MAX_PHASE)
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    score += 10 if board.turn == chess.WHITE else -10
    return score if board.turn == chess.WHITE else -score


class Engine:
    def __init__(self) -> None:
        self.tt: dict[Hashable, TTEntry] = {}
        self.age = 0
        self.deadline = 0.0
        self.nodes = 0
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history = [0] * (2 * 64 * 64)
        self.repetitions: dict[Hashable, int] = {}

    def choose_move(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("get_move called in a terminal position")
        if len(legal_moves) == 1:
            return legal_moves[0]

        root_key = _key(board)
        self.repetitions[root_key] = self.repetitions.get(root_key, 0) + 1
        self.age += 1
        self.nodes = 0
        for pair in self.killers:
            pair[0] = pair[1] = None

        # A hard deadline is checked within the tree; completed iterations stop at the softer
        # target. The reserve scales up with the clock to cover IPC and scheduler jitter.
        clock_ms = max(1, time_left_ms)
        reserve_ms = max(15, min(500, clock_ms // 20))
        usable_ms = max(1, clock_ms - reserve_ms)
        soft_ms = min(4_500, max(5, int(clock_ms * 0.035)))
        hard_ms = min(usable_ms, max(soft_ms + 5, int(soft_ms * 1.65)))
        started = time.perf_counter()
        self.deadline = started + hard_ms / 1000.0

        entry = self.tt.get(root_key)
        ordered = self._ordered_moves(board, legal_moves, entry.move if entry else None, 0)
        best_move = ordered[0]
        previous_score = 0

        for depth in range(1, 64):
            if depth > 1 and (time.perf_counter() - started) * 1000.0 >= soft_ms:
                break
            try:
                if depth >= 4:
                    alpha = max(-INF, previous_score - 35)
                    beta = min(INF, previous_score + 35)
                    score, move = self._root(board, depth, alpha, beta)
                    if score <= alpha or score >= beta:
                        score, move = self._root(board, depth, -INF, INF)
                else:
                    score, move = self._root(board, depth, -INF, INF)
            except SearchTimeout:
                break
            best_move = move
            previous_score = score
            if abs(score) >= MATE_BOUND:
                break

        self._record_played_position(board, best_move)
        self._trim_tt()
        return best_move

    def _root(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> tuple[int, chess.Move]:
        original_alpha = alpha
        key = _key(board)
        entry = self.tt.get(key)
        moves = self._ordered_moves(
            board, list(board.legal_moves), entry.move if entry else None, 0
        )
        best_move = moves[0]
        best_score = -INF

        for move_index, move in enumerate(moves):
            board.push(move)
            child_key = _key(board)
            self.repetitions[child_key] = self.repetitions.get(child_key, 0) + 1
            try:
                if move_index == 0:
                    score = -self._search(board, depth - 1, -beta, -alpha, 1, True, True)
                else:
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, 1, False, True)
                    if alpha < score < beta:
                        score = -self._search(board, depth - 1, -beta, -alpha, 1, True, True)
            finally:
                self._pop_repetition(child_key)
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
        self.tt[key] = TTEntry(depth, self._score_to_tt(best_score, 0), bound, best_move, self.age)
        return best_score, best_move

    def _search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        pv_node: bool,
        allow_null: bool,
    ) -> int:
        self._tick()
        if ply >= MAX_PLY - 1:
            return evaluate(board)

        key = _key(board)
        if self.repetitions.get(key, 0) >= 3 or board.halfmove_clock >= 100:
            return 0

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        original_alpha = alpha
        entry = self.tt.get(key)
        tt_move = entry.move if entry else None
        if entry is not None and entry.depth >= depth:
            tt_score = self._score_from_tt(entry.score, ply)
            if entry.bound == EXACT:
                return tt_score
            if entry.bound == LOWER and tt_score >= beta:
                return tt_score
            if entry.bound == UPPER and tt_score <= alpha:
                return tt_score

        static_eval = evaluate(board) if not in_check else -INF

        # Null-move pruning: if even passing the turn beats beta, ordinary moves are unlikely to
        # matter. Restrict it to positions with non-pawn material to avoid zugzwang endgames.
        if (
            allow_null
            and not pv_node
            and not in_check
            and depth >= 3
            and static_eval >= beta
            and self._has_non_pawn_material(board, board.turn)
        ):
            reduction = 2 + depth // 5
            board.push(chess.Move.null())
            null_key = _key(board)
            self.repetitions[null_key] = self.repetitions.get(null_key, 0) + 1
            try:
                score = -self._search(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False, False
                )
            finally:
                self._pop_repetition(null_key)
                board.pop()
            if score >= beta:
                return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        moves = self._ordered_moves(board, moves, tt_move, ply)

        best_score = -INF
        best_move: chess.Move | None = None
        for move_index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and move.promotion is None
            gives_check = board.gives_check(move)

            # Shallow futility pruning avoids quiet moves that cannot plausibly raise alpha.
            if (
                depth == 1
                and not pv_node
                and not in_check
                and is_quiet
                and not gives_check
                and static_eval + 120 <= alpha
                and move_index > 0
            ):
                continue

            board.push(move)
            child_key = _key(board)
            self.repetitions[child_key] = self.repetitions.get(child_key, 0) + 1
            try:
                reduction = 0
                if (
                    depth >= 3
                    and move_index >= 3
                    and is_quiet
                    and not gives_check
                    and not in_check
                    and move not in self.killers[min(ply, MAX_PLY - 1)]
                ):
                    reduction = 1 + int(depth >= 6 and move_index >= 8)

                if move_index == 0:
                    score = -self._search(
                        board, depth - 1, -beta, -alpha, ply + 1, pv_node, True
                    )
                else:
                    score = -self._search(
                        board,
                        depth - 1 - reduction,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        False,
                        True,
                    )
                    if reduction and score > alpha:
                        score = -self._search(
                            board, depth - 1, -alpha - 1, -alpha, ply + 1, False, True
                        )
                    if alpha < score < beta:
                        score = -self._search(
                            board, depth - 1, -beta, -alpha, ply + 1, pv_node, True
                        )
            finally:
                self._pop_repetition(child_key)
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if is_quiet:
                    self._store_killer(move, ply)
                    history_index = self._history_index(board.turn, move)
                    self.history[history_index] += depth * depth
                    if self.history[history_index] > 1_000_000:
                        self.history = [value // 2 for value in self.history]
                break

        if best_move is None:
            # Every move can only be skipped by futility pruning; the static score is then a safe
            # fail-low result.
            return static_eval
        bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
        self.tt[key] = TTEntry(
            depth, self._score_to_tt(best_score, ply), bound, best_move, self.age
        )
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        key = _key(board)
        if self.repetitions.get(key, 0) >= 3 or board.halfmove_clock >= 100:
            return 0
        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            if ply >= MAX_PLY - 1:
                return alpha
            moves = [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion is not None
            ]

        moves = self._ordered_moves(board, moves, None, min(ply, MAX_PLY - 1))
        stand_pat = evaluate(board) if not in_check else -INF
        for move in moves:
            if not in_check and move.promotion is None:
                victim = self._victim_value(board, move)
                if stand_pat + victim + 180 < alpha:
                    continue
            board.push(move)
            child_key = _key(board)
            self.repetitions[child_key] = self.repetitions.get(child_key, 0) + 1
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                self._pop_repetition(child_key)
                board.pop()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    def _ordered_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        color = board.turn
        killer_pair = self.killers[min(ply, MAX_PLY - 1)]

        def move_score(move: chess.Move) -> int:
            if move == tt_move:
                return 20_000_000
            if move.promotion is not None:
                return 10_000_000 + PIECE_VALUE[move.promotion]
            if board.is_capture(move):
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                return 8_000_000 + 16 * self._victim_value(board, move) - PIECE_VALUE[attacker]
            if move == killer_pair[0]:
                return 7_000_000
            if move == killer_pair[1]:
                return 6_900_000
            return self.history[self._history_index(color, move)]

        moves.sort(key=move_score, reverse=True)
        return moves

    @staticmethod
    def _victim_value(board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return PIECE_VALUE[chess.PAWN]
        return PIECE_VALUE[board.piece_type_at(move.to_square) or 0]

    @staticmethod
    def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
        return bool(
            board.pieces_mask(chess.KNIGHT, color)
            | board.pieces_mask(chess.BISHOP, color)
            | board.pieces_mask(chess.ROOK, color)
            | board.pieces_mask(chess.QUEEN, color)
        )

    @staticmethod
    def _history_index(color: chess.Color, move: chess.Move) -> int:
        return int(color) * 4096 + move.from_square * 64 + move.to_square

    def _store_killer(self, move: chess.Move, ply: int) -> None:
        pair = self.killers[min(ply, MAX_PLY - 1)]
        if move != pair[0]:
            pair[1] = pair[0]
            pair[0] = move

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes & 511 == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def _pop_repetition(self, key: Hashable) -> None:
        count = self.repetitions[key] - 1
        if count:
            self.repetitions[key] = count
        else:
            del self.repetitions[key]

    def _record_played_position(self, board: chess.Board, move: chess.Move) -> None:
        board.push(move)
        child_key = _key(board)
        self.repetitions[child_key] = self.repetitions.get(child_key, 0) + 1
        board.pop()

    def _trim_tt(self) -> None:
        if len(self.tt) <= TT_LIMIT:
            return
        cutoff = self.age - 2
        self.tt = {key: entry for key, entry in self.tt.items() if entry.age >= cutoff}
        if len(self.tt) > TT_LIMIT:
            self.tt.clear()

    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score + ply
        if score <= -MATE_BOUND:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score - ply
        if score <= -MATE_BOUND:
            return score + ply
        return score


_ENGINE = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move, with a deterministic legal fallback on search failure."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"  # The referee never requests a move from a terminal position.
    fallback = legal_moves[0]
    try:
        move = _ENGINE.choose_move(board, time_left_ms)
        return move.uci() if move in board.legal_moves else fallback.uci()
    except Exception:
        # Reliability is worth more than diagnostics in a rated game. The already-generated
        # fallback remains legal even if an unexpected search edge case occurs.
        return fallback.uci()


