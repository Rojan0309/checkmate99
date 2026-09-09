"""A classical chess agent: PVS/negamax search with SEE-driven ordering and pruning,
adaptive time management, and a tapered positional evaluation."""

from __future__ import annotations

import math
import sys
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
ASSUMED_INCREMENT_MS: Final = 100  # conservative nudge; published rule is 0.5s/move

EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

# Used for material scoring in evaluate() -- king intentionally 0 so it never enters a sum.
PIECE_VALUE: Final = (0, 100, 320, 330, 500, 900, 0)
# Used inside SEE, where a "captured king" must never look free.
SEE_VALUE: Final = (0, 100, 320, 330, 500, 900, 20_000)
PHASE_VALUE: Final = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE: Final = 24

FILE_MASKS: Final = tuple(chess.BB_FILES[file_index] for file_index in range(8))
ADJACENT_FILES: Final = tuple(
    (chess.BB_FILES[file_index - 1] if file_index else 0)
    | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
    for file_index in range(8)
)

LMR_MAX_DEPTH: Final = 64
LMR_MAX_INDEX: Final = 64
LMR_TABLE: Final = tuple(
    tuple(
        int(0.5 + math.log(depth) * math.log(move_index) * 0.5)
        if depth > 0 and move_index > 0
        else 0
        for move_index in range(LMR_MAX_INDEX)
    )
    for depth in range(LMR_MAX_DEPTH)
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


def _tt_key(board: chess.Board, position_key: Hashable | None = None) -> tuple[Hashable, int]:
    key = _key(board) if position_key is None else position_key
    return key, board.halfmove_clock


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


# ---------------------------------------------------------------------------
# Static Exchange Evaluation
#
# python-chess exposes attackers_mask(color, square, occupied) on recent versions,
# which is exactly what SEE needs (it lets us ask "who attacks this square" against
# a *hypothetical* occupancy as pieces are removed during the simulated exchange,
# correctly revealing x-ray attackers). We probe for it once at import time and fall
# back to a same-board approximation (no x-ray awareness, but still far better than
# no SEE at all) if it isn't available. Every public entry point is wrapped so a
# bug here degrades to a cheap heuristic instead of ever crashing the search.
# ---------------------------------------------------------------------------


def _detect_attackers_mask_supports_occupied() -> bool:
    try:
        probe = chess.Board()
        probe.attackers_mask(chess.WHITE, chess.E4, probe.occupied)
        return True
    except (TypeError, AttributeError):
        return False


_ATTACKERS_SUPPORTS_OCCUPIED: Final = _detect_attackers_mask_supports_occupied()


def _attackers_for_see(board: chess.Board, color: chess.Color, square: chess.Square, occupied: int) -> int:
    if _ATTACKERS_SUPPORTS_OCCUPIED:
        return board.attackers_mask(color, square, occupied)
    return int(board.attackers(color, square)) & occupied


def _least_valuable_attacker(
    board: chess.Board, attackers: int, color: chess.Color
) -> tuple[int | None, int]:
    for piece_type in range(1, 7):
        subset = attackers & board.pieces_mask(piece_type, color)
        if subset:
            square = (subset & -subset).bit_length() - 1
            return square, piece_type
    return None, 0


def _see_impl(board: chess.Board, move: chess.Move) -> int:
    to_square = move.to_square
    from_square = move.from_square
    mover_color = board.turn

    if board.is_en_passant(move):
        captured_value = PIECE_VALUE[chess.PAWN]
        capture_removal_square = to_square + (-8 if mover_color == chess.WHITE else 8)
    else:
        captured_piece_type = board.piece_type_at(to_square)
        captured_value = PIECE_VALUE[captured_piece_type] if captured_piece_type else 0
        capture_removal_square = to_square

    attacker_piece_type = board.piece_type_at(from_square) or chess.PAWN
    if move.promotion:
        promotion_gain = PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
        current_value = SEE_VALUE[move.promotion]
    else:
        promotion_gain = 0
        current_value = SEE_VALUE[attacker_piece_type]

    occupied = board.occupied
    occupied &= ~chess.BB_SQUARES[from_square]
    if capture_removal_square != to_square:
        occupied &= ~chess.BB_SQUARES[capture_removal_square]
    occupied |= chess.BB_SQUARES[to_square]

    gains = [captured_value + promotion_gain]
    side = not mover_color

    for _ in range(32):
        attackers = _attackers_for_see(board, side, to_square, occupied)
        square, piece_type = _least_valuable_attacker(board, attackers, side)
        if square is None:
            break
        gains.append(current_value - gains[-1])
        occupied &= ~chess.BB_SQUARES[square]
        current_value = SEE_VALUE[piece_type]
        side = not side

    for index in range(len(gains) - 1, 0, -1):
        gains[index - 1] = -max(-gains[index - 1], gains[index])
    return gains[0]


def see(board: chess.Board, move: chess.Move) -> int:
    """Net material result (centipawns) of the full capture sequence a capture or
    promotion touches off, assuming best play by both sides. Falls back to a cheap
    heuristic on any unexpected error, so a SEE bug can never crash the search."""
    try:
        return _see_impl(board, move)
    except Exception:
        captured_piece_type = board.piece_type_at(move.to_square)
        captured_value = PIECE_VALUE[captured_piece_type] if captured_piece_type else 0
        attacker_piece_type = board.piece_type_at(move.from_square) or chess.PAWN
        return captured_value - PIECE_VALUE[attacker_piece_type] // 10


def _king_zone(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    zone = 0
    for rank_offset in (-1, 0, 1):
        target_rank = king_rank + rank_offset
        if not 0 <= target_rank < 8:
            continue
        for file_offset in (-1, 0, 1):
            target_file = king_file + file_offset
            if not 0 <= target_file < 8:
                continue
            zone |= chess.BB_SQUARES[chess.square(target_file, target_rank)]
    return zone


def evaluate(board: chess.Board) -> int:
    mg = 0
    eg = 0
    phase = 0

    white_king_zone = _king_zone(board, chess.WHITE)
    black_king_zone = _king_zone(board, chess.BLACK)

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        own_pieces = board.occupied_co[color]
        enemy_king_zone = black_king_zone if color == chess.WHITE else white_king_zone
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
                attacks = board.attacks_mask(square)
                mobility = (attacks & ~own_pieces).bit_count()
                if piece_type not in (chess.PAWN, chess.KING):
                    pressure = (attacks & enemy_king_zone).bit_count()
                    # Weighted well above the old value: this is one of the few signals
                    # the engine has that a king hunt is developing, and it needs to be
                    # strong enough to outweigh a tempting pawn grab elsewhere on the board.
                    mg += sign * pressure * 10
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
            if _relative_rank(color, square) == 6:
                mg += sign * 12
                eg += sign * 22

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
        self.countermoves: dict[tuple[int, int], chess.Move] = {}
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

        soft_ms, hard_ms = self._time_budget(board, clock_ms)
        started = time.perf_counter()
        self.deadline = started + hard_ms / 1000.0
        if clock_ms < 250:
            self.time_check_mask = 1
        elif clock_ms < 1_000:
            self.time_check_mask = 7
        elif clock_ms < 5_000:
            self.time_check_mask = 31
        else:
            self.time_check_mask = 127

        entry = self.tt.get(_tt_key(board))
        last_move = board.peek() if board.move_stack else None
        countermove = (
            self.countermoves.get((last_move.from_square, last_move.to_square))
            if last_move is not None
            else None
        )
        ordered = self._ordered_moves(
            board, legal_moves, entry.move if entry else None, 0, countermove
        )
        best_move = ordered[0][0]
        previous_score = 0
        last_best_move: chess.Move | None = None
        stable_count = 0
        target_ms = float(soft_ms)

        for depth in range(1, 64):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if depth > 1 and elapsed_ms >= target_ms:
                break
            try:
                if depth >= 4:
                    window = 25
                    alpha = max(-INF, previous_score - window)
                    beta = min(INF, previous_score + window)
                    attempts = 0
                    while True:
                        score, move = self._root(board, depth, alpha, beta)
                        attempts += 1
                        if score <= alpha and alpha > -INF and attempts < 4:
                            window *= 3
                            alpha = max(-INF, previous_score - window)
                        elif score >= beta and beta < INF and attempts < 4:
                            window *= 3
                            beta = min(INF, previous_score + window)
                        elif score <= alpha or score >= beta:
                            score, move = self._root(board, depth, -INF, INF)
                            break
                        else:
                            break
                else:
                    score, move = self._root(board, depth, -INF, INF)
            except SearchTimeout:
                break

            changed = last_best_move is not None and move != last_best_move
            dropped = last_best_move is not None and score < previous_score - 40
            stable_count = 0 if (changed or dropped) else stable_count + 1

            best_move = move
            last_best_move = move
            previous_score = score
            self.last_depth = depth
            self.last_score = score

            if abs(score) >= MATE_BOUND:
                break

            if (changed or dropped) and depth >= 4:
                target_ms = min(float(hard_ms), target_ms * 1.7)
            elif stable_count >= 3 and depth >= 7:
                target_ms = max(soft_ms * 0.4, target_ms * 0.75)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > 0:
            nps = int(self.nodes / (elapsed_ms / 1000.0))
        else:
            nps = 0
        print(
            f"[depth={self.last_depth} nodes={self.nodes} nps={nps} "
            f"time={elapsed_ms:.0f}ms clock={clock_ms}ms score={self.last_score}]",
            file=sys.stderr,
        )

        self._record_played_position(board, best_move)
        self._trim_tt()
        return best_move

    def _time_budget(self, board: chess.Board, clock_ms: int) -> tuple[int, int]:
        non_king_material = sum(
            PIECE_VALUE[piece_type]
            * (
                board.pieces_mask(piece_type, chess.WHITE) | board.pieces_mask(piece_type, chess.BLACK)
            ).bit_count()
            for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        )
        if non_king_material > 5_000:
            moves_to_go = 32
        elif non_king_material > 2_500:
            moves_to_go = 24
        else:
            moves_to_go = 16

        # Reserve is deliberately generous: SEE-driven ordering makes each node more
        # expensive than before, so the periodic clock check (see time_check_mask)
        # can overshoot by more real time than it used to at critically low clocks.
        reserve_ms = max(150, min(2000, clock_ms // 6))
        usable_ms = max(1, clock_ms - reserve_ms)
        per_move_budget = usable_ms / moves_to_go + ASSUMED_INCREMENT_MS * 0.4
        soft_ms = max(15.0, min(usable_ms * 0.4, per_move_budget))
        hard_ms = max(soft_ms + 15.0, min(usable_ms * 0.75, soft_ms * 2.0))
        return int(soft_ms), int(hard_ms)

    def _root(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> tuple[int, chess.Move]:
        original_alpha = alpha
        key = _tt_key(board)
        entry = self.tt.get(key)
        last_move = board.peek() if board.move_stack else None
        countermove = (
            self.countermoves.get((last_move.from_square, last_move.to_square))
            if last_move is not None
            else None
        )
        ordered = self._ordered_moves(
            board, list(board.legal_moves), entry.move if entry else None, 0, countermove
        )
        best_move = ordered[0][0]
        best_score = -INF

        for move_index, (move, _is_quiet, _is_capture, _see_value) in enumerate(ordered):
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

        mate_alpha = max(alpha, -MATE + ply)
        mate_beta = min(beta, MATE - ply - 1)
        if mate_alpha >= mate_beta:
            return mate_alpha
        alpha, beta = mate_alpha, mate_beta

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
        key = _tt_key(board, position_key)
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

        if (
            not pv_node
            and not in_check
            and depth <= 7
            and abs(beta) < MATE_BOUND
            and static_eval - 85 * depth >= beta
        ):
            return static_eval

        if not pv_node and not in_check and depth == 1 and static_eval + 300 <= alpha:
            razor_score = self._quiescence(board, alpha, beta, ply)
            if razor_score <= alpha:
                return razor_score

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

        last_move = board.peek() if board.move_stack else None
        countermove = (
            self.countermoves.get((last_move.from_square, last_move.to_square))
            if last_move is not None
            else None
        )
        ordered = self._ordered_moves(board, moves, tt_move, ply, countermove)

        best_score = -INF
        best_move: chess.Move | None = None
        for move_index, (move, is_quiet, is_capture, see_value) in enumerate(ordered):
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

            if (
                not pv_node
                and not in_check
                and depth <= 3
                and move_index > 0
                and is_capture
                and see_value < -60 * depth
            ):
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
                    reduction = LMR_TABLE[min(depth, LMR_MAX_DEPTH - 1)][min(move_index, LMR_MAX_INDEX - 1)]
                    if pv_node:
                        reduction = max(0, reduction - 1)
                    reduction = max(0, min(reduction, depth - 1))

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
                    if last_move is not None:
                        self.countermoves[(last_move.from_square, last_move.to_square)] = move
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
            stand_pat = -INF
        else:
            stand_pat = self._evaluate(board, position_key)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            moves = [
                move for move in legal_moves if board.is_capture(move) or move.promotion is not None
            ]

        ordered = self._ordered_moves(board, moves, None, min(ply, MAX_PLY - 1), None)
        for move, _is_quiet, _is_capture, see_value in ordered:
            if not in_check and move.promotion is None:
                if see_value < 0:
                    continue
                if stand_pat + see_value + 120 < alpha:
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
        countermove: chess.Move | None,
    ) -> list[tuple[chess.Move, bool, bool, int]]:
        color = board.turn
        killer_pair = self.killers[min(ply, MAX_PLY - 1)]
        scored: list[tuple[int, chess.Move, bool, bool, int]] = []

        for move in moves:
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None
            is_quiet = not is_capture and not is_promotion
            see_value = 0

            if move == tt_move:
                score = 1_000_000_000
            elif is_capture or is_promotion:
                see_value = see(board, move)
                score = (100_000_000 + see_value) if see_value >= 0 else (-100_000_000 + see_value)
            elif move == killer_pair[0]:
                score = 90_000_000
            elif move == killer_pair[1]:
                score = 89_000_000
            elif countermove is not None and move == countermove:
                score = 88_000_000
            else:
                score = self.history[self._history_index(color, move)]

            scored.append((score, move, is_quiet, is_capture, see_value))

        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

        scored.sort(key=lambda item: item[0], reverse=True)
        return [(move, is_quiet, is_capture, see_value) for _, move, is_quiet, is_capture, see_value in scored]

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
        if self.nodes % 50_000 == 0:
            self._trim_tt()
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


def _emergency_move(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
    """A cheap, non-search fallback used only if the main search raises unexpectedly:
    take an immediate mate if one exists, otherwise the best SEE-scored capture,
    otherwise the first quiet move. Favours safety over speed."""
    best_move = legal_moves[0]
    best_score = -INF
    for move in legal_moves:
        if board.is_capture(move) or move.promotion is not None:
            score = see(board, move)
        else:
            score = 0
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


ENGINE = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"  # The referee never requests a move from a terminal position.

    move: chess.Move | None = None
    try:
        candidate = ENGINE.choose_move(board, time_left_ms, legal_moves)
        if candidate in board.legal_moves:
            move = candidate
    except Exception:
        move = None

    if move is not None:
        return move.uci()

    fallback = _emergency_move(board, legal_moves)
    try:
        ENGINE._record_played_position(board, fallback)
        ENGINE._trim_tt()
    except Exception:
        pass
    return fallback.uci()