"""A classical chess agent with iterative deepening and positional evaluation."""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import chess

INF: Final = 40_000
MATE: Final = 32_000
MATE_BOUND: Final = 31_000
MAX_PLY: Final = 96
TT_LIMIT: Final = 250_000
EVAL_CACHE_LIMIT: Final = 150_000

EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

PIECE_VALUE: Final = (0, 100, 320, 330, 500, 900, 0)
PHASE_VALUE: Final = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE: Final = 24

FILE_MASKS: Final = tuple(chess.BB_FILES[file_index] for file_index in range(8))
ADJACENT_FILES: Final = tuple(
    (chess.BB_FILES[file_index - 1] if file_index else 0)
    | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
    for file_index in range(8)
)


class SearchTimeout(Exception):
    pass


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    bound: int
    move: chess.Move | None
    age: int


def _key(board: chess.Board) -> Hashable:
    return board._transposition_key()


def _tt_key(board: chess.Board) -> tuple[Hashable, int]:
    return _key(board), board.halfmove_clock


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
    mg = 0
    eg = 0
    phase = 0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        own_pieces = board.occupied_co[color]
        for piece_type in chess.PIECE_TYPES:
            pieces = board.pieces_mask(piece_type, color)
            count = pieces.bit_count()
            phase += PHASE_VALUE[piece_type] * count
            material = PIECE_VALUE[piece_type] * count
            mg += sign * material
            eg += sign * material
            for square in chess.scan_reversed(pieces):
                file_index = chess.square_file(square)
                rank = chess.square_rank(square)
                centrality = 14 - abs(2 * file_index - 7) - abs(2 * rank - 7)
                mobility = (board.attacks_mask(square) & ~own_pieces).bit_count()
                if piece_type == chess.PAWN:
                    advance = _relative_rank(color, square)
                    mg += sign * (advance * 2 + centrality // 4)
                    eg += sign * (advance * 5 + centrality // 4)
                elif piece_type == chess.KNIGHT:
                    mg += sign * (centrality * 3 + mobility * 2)
                    eg += sign * (centrality * 2 + mobility * 2)
                elif piece_type == chess.BISHOP:
                    mg += sign * (centrality + mobility * 2)
                    eg += sign * (centrality + mobility * 2)
                elif piece_type in (chess.ROOK, chess.QUEEN):
                    mg += sign * mobility
                    eg += sign * mobility
                else:
                    mg -= sign * centrality * 2
                    eg += sign * centrality * 3

        bishops = board.pieces_mask(chess.BISHOP, color)
        if bishops.bit_count() >= 2:
            mg += sign * 28
            eg += sign * 38

        pawns = board.pieces_mask(chess.PAWN, color)
        enemy_pawns = board.pieces_mask(chess.PAWN, not color)
        all_pawns = pawns | enemy_pawns
        for square in chess.scan_reversed(board.pieces_mask(chess.ROOK, color)):
            file_index = chess.square_file(square)
            if not all_pawns & FILE_MASKS[file_index]:
                mg += sign * 20
                eg += sign * 12
            elif not pawns & FILE_MASKS[file_index]:
                mg += sign * 10

        king_square = board.king(color)
        if king_square is not None:
            shield_rank = chess.square_rank(king_square) + (1 if color else -1)
            if 0 <= shield_rank < 8:
                king_file = chess.square_file(king_square)
                for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
                    shield_square = chess.square(file_index, shield_rank)
                    if pawns & chess.BB_SQUARES[shield_square]:
                        mg += sign * 8

    pawn_mg, pawn_eg = _pawn_structure(
        board.pieces_mask(chess.PAWN, chess.WHITE),
        board.pieces_mask(chess.PAWN, chess.BLACK),
    )
    mg += pawn_mg
    eg += pawn_eg

    phase = min(phase, MAX_PHASE)
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    score += 10 if board.turn == chess.WHITE else -10
    return score if board.turn == chess.WHITE else -score


@lru_cache(maxsize=32_768)
def _pawn_structure(white_pawns: int, black_pawns: int) -> tuple[int, int]:
    mg = 0
    eg = 0
    for color, pawns, enemy_pawns in (
        (chess.WHITE, white_pawns, black_pawns),
        (chess.BLACK, black_pawns, white_pawns),
    ):
        sign = 1 if color == chess.WHITE else -1
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
    return mg, eg


class Engine:
    def __init__(self) -> None:
        self.tt: dict[tuple[Hashable, int], TTEntry] = {}
        self.eval_cache: dict[Hashable, int] = {}
        self.age = 0
        self.deadline = 0.0
        self.time_check_mask = 511
        self.nodes = 0
        self.last_depth = 0
        self.last_score = 0
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history = [0] * (2 * 64 * 64)
        self.repetitions: dict[Hashable, int] = {}
        self.repeated_positions = 0
        self.repetition_tainted = False
        self.null_search = 0

    def choose_move(
        self,
        board: chess.Board,
        time_left_ms: int,
        legal_moves: list[chess.Move] | None = None,
    ) -> chess.Move:
        legal_moves = list(board.legal_moves) if legal_moves is None else legal_moves
        if not legal_moves:
            raise ValueError("get_move called in a terminal position")

        root_key = _key(board)
        self._push_repetition(root_key)
        self.age += 1
        self.nodes = 0
        self.last_depth = 0
        self.repetition_tainted = False
        self.null_search = 0
        if len(legal_moves) == 1:
            self._record_played_position(board, legal_moves[0])
            self._trim_tt()
            return legal_moves[0]
        for pair in self.killers:
            pair[0] = pair[1] = None

        clock_ms = max(1, time_left_ms)
        if clock_ms <= 30:
            self._record_played_position(board, legal_moves[0])
            return legal_moves[0]
        reserve_ms = max(15, min(500, clock_ms // 20))
        usable_ms = max(1, clock_ms - reserve_ms)
        soft_ms = min(4_500, max(5, int(clock_ms * 0.035)))
        hard_ms = min(usable_ms, max(soft_ms + 5, int(soft_ms * 1.65)))
        started = time.perf_counter()
        self.deadline = started + hard_ms / 1000.0
        if clock_ms < 250:
            self.time_check_mask = 7
        elif clock_ms < 1_000:
            self.time_check_mask = 31
        elif clock_ms < 5_000:
            self.time_check_mask = 127
        else:
            self.time_check_mask = 511

        entry = self.tt.get(_tt_key(board))
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
            self.last_depth = depth
            self.last_score = score
            if abs(score) >= MATE_BOUND:
                break

        self._record_played_position(board, best_move)
        self._trim_tt()
        return best_move

    def _root(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> tuple[int, chess.Move]:
        original_alpha = alpha
        key = _tt_key(board)
        entry = self.tt.get(key)
        moves = self._ordered_moves(
            board, list(board.legal_moves), entry.move if entry else None, 0
        )
        best_move = moves[0]
        best_score = -INF

        for move_index, move in enumerate(moves):
            board.push(move)
            child_key = _key(board)
            self._push_repetition(child_key)
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
        if not self.repetition_tainted and self.repeated_positions == 0:
            self.tt[key] = TTEntry(
                depth, self._score_to_tt(best_score, 0), bound, best_move, self.age
            )
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
        position_key = _key(board)
        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        moves: list[chess.Move] | None = None
        if self._is_insufficient_material(board):
            return 0
        if self._needs_draw_check(board, position_key):
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply if in_check else 0
            if self._is_rule_draw(board, position_key, moves):
                return 0
        if ply >= MAX_PLY - 1:
            if moves is None:
                moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply if in_check else 0
            return self._evaluate(board, position_key)

        original_alpha = alpha
        key = _tt_key(board)
        entry = self.tt.get(key)
        tt_move = entry.move if entry else None
        tt_allowed = not self.repetition_tainted and self.repeated_positions == 0
        if entry is not None and entry.age == self.age and entry.depth >= depth and tt_allowed:
            tt_score = self._score_from_tt(entry.score, ply)
            if entry.bound == EXACT:
                return tt_score
            if entry.bound == LOWER and tt_score >= beta:
                return tt_score
            if entry.bound == UPPER and tt_score <= alpha:
                return tt_score

        static_eval = self._evaluate(board, position_key) if not in_check else -INF

        if moves is None:
            moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        # Non-pawn material reduces null-move errors in zugzwang endgames.
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
            self.null_search += 1
            try:
                score = -self._search(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False, False
                )
            finally:
                self.null_search -= 1
                board.pop()
            if score >= beta:
                return score

        moves = self._ordered_moves(board, moves, tt_move, ply)

        best_score = -INF
        best_move: chess.Move | None = None
        for move_index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and move.promotion is None

            can_prune = (
                depth == 1
                and not pv_node
                and not in_check
                and is_quiet
                and static_eval + 120 <= alpha
                and move_index > 0
            )
            gives_check = can_prune and board.gives_check(move)
            if can_prune and not gives_check:
                continue

            reduction = 0
            can_reduce = (
                depth >= 3
                and move_index >= 3
                and is_quiet
                and not in_check
                and move not in self.killers[min(ply, MAX_PLY - 1)]
            )
            if can_reduce and not gives_check:
                if board.gives_check(move):
                    gives_check = True
                else:
                    reduction = 1 + int(depth >= 6 and move_index >= 8)

            board.push(move)
            child_key = _key(board)
            self._push_repetition(child_key)
            try:
                if move_index == 0:
                    score = -self._search(board, depth - 1, -beta, -alpha, ply + 1, pv_node, True)
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
            return static_eval
        bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
        if not self.repetition_tainted and self.repeated_positions == 0:
            self.tt[key] = TTEntry(
                depth, self._score_to_tt(best_score, ply), bound, best_move, self.age
            )
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        position_key = _key(board)
        in_check = board.is_check()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return -MATE + ply if in_check else 0
        if self._is_rule_draw(board, position_key, legal_moves):
            return 0
        if ply >= MAX_PLY - 1:
            return self._evaluate(board, position_key)

        if in_check:
            moves = legal_moves
        else:
            stand_pat = self._evaluate(board, position_key)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            moves = [
                move for move in legal_moves if board.is_capture(move) or move.promotion is not None
            ]

        moves = self._ordered_moves(board, moves, None, min(ply, MAX_PLY - 1))
        for move in moves:
            if not in_check and move.promotion is None:
                victim = self._victim_value(board, move)
                if stand_pat + victim + 180 < alpha:
                    continue
            board.push(move)
            child_key = _key(board)
            self._push_repetition(child_key)
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
        if self.nodes & self.time_check_mask == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def _is_insufficient_material(board: chess.Board) -> bool:
        if board.pawns or board.rooks or board.queens:
            return False
        return board.is_insufficient_material()

    def _needs_draw_check(self, board: chess.Board, position_key: Hashable) -> bool:
        return self.null_search == 0 and (
            board.halfmove_clock >= 99
            or self.repetitions.get(position_key, 0) >= 3
            or self.repeated_positions > 0
        )

    def _is_rule_draw(
        self,
        board: chess.Board,
        position_key: Hashable,
        legal_moves: list[chess.Move],
    ) -> bool:
        if self._is_insufficient_material(board):
            return True
        if self.null_search:
            return False

        if self.repetitions.get(position_key, 0) >= 3:
            self.repetition_tainted = True
            return True
        if board.halfmove_clock >= 100:
            return True
        if board.halfmove_clock >= 99:
            for move in legal_moves:
                if board.is_zeroing(move):
                    continue
                board.push(move)
                try:
                    if board.is_fifty_moves():
                        return True
                finally:
                    board.pop()

        if self.repeated_positions:
            for move in legal_moves:
                board.push(move)
                try:
                    if self.repetitions.get(_key(board), 0) >= 2:
                        self.repetition_tainted = True
                        return True
                finally:
                    board.pop()
        return False

    def _push_repetition(self, key: Hashable) -> None:
        count = self.repetitions.get(key, 0)
        self.repetitions[key] = count + 1
        if count == 1:
            self.repeated_positions += 1

    def _pop_repetition(self, key: Hashable) -> None:
        count = self.repetitions[key]
        if count == 2:
            self.repeated_positions -= 1
        if count > 1:
            self.repetitions[key] = count - 1
        else:
            del self.repetitions[key]

    def _record_played_position(self, board: chess.Board, move: chess.Move) -> None:
        board.push(move)
        child_key = _key(board)
        self._push_repetition(child_key)
        board.pop()

    def _trim_tt(self) -> None:
        if len(self.eval_cache) > EVAL_CACHE_LIMIT:
            self.eval_cache.clear()
        if len(self.tt) <= TT_LIMIT:
            return
        cutoff = self.age - 2
        self.tt = {key: entry for key, entry in self.tt.items() if entry.age >= cutoff}
        if len(self.tt) > TT_LIMIT:
            self.tt.clear()

    def _evaluate(self, board: chess.Board, key: Hashable | None = None) -> int:
        position_key = _key(board) if key is None else key
        try:
            return self.eval_cache[position_key]
        except KeyError:
            score = evaluate(board)
            self.eval_cache[position_key] = score
            return score

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
    """Return a legal move in UCI notation."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"  # The referee never requests a move from a terminal position.
    fallback = legal_moves[0]
    try:
        move = _ENGINE.choose_move(board, time_left_ms, legal_moves)
        return move.uci() if move in board.legal_moves else fallback.uci()
    except Exception:
        return fallback.uci()
