"""A classical chess agent: PVS/negamax search with SEE-driven ordering and pruning,
adaptive time management, and a tapered positional evaluation."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import chess
import chess.syzygy

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

# Passed pawns become nonlinear assets in the endgame.  Indexes are relative
# ranks (a white pawn on the seventh rank, for example, has index 6).
PASSED_MG_BONUS: Final = (0, 0, 8, 18, 38, 75, 140, 0)
PASSED_EG_BONUS: Final = (0, 4, 12, 30, 70, 150, 200, 0)

FILE_MASKS: Final = tuple(chess.BB_FILES[file_index] for file_index in range(8))
ADJACENT_FILES: Final = tuple(
    (chess.BB_FILES[file_index - 1] if file_index else 0)
    | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
    for file_index in range(8)
)
KING_DISTANCE: Final = tuple(
    tuple(chess.square_distance(first, second) for second in chess.SQUARES)
    for first in chess.SQUARES
)


def _open_tablebase() -> chess.syzygy.Tablebase | None:
    """Open the optional shipped subset; any filesystem/data issue disables it safely."""
    try:
        path = Path(__file__).resolve().parent / "syzygy"
        return chess.syzygy.open_tablebase(str(path)) if path.is_dir() else None
    except Exception:
        return None


TABLEBASE: Final = _open_tablebase()

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


def _attackers_for_see(
    board: chess.Board, color: chess.Color, square: chess.Square, occupied: int
) -> int:
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
    queen_count = board.queens.bit_count()
    rook_count = board.rooks.bit_count()

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
                    # Central kings are desirable only as major-piece danger fades.
                    # A queen on the board makes generic centralisation actively risky;
                    # specialised mating guidance below handles the attacking king.
                    king_activity = 0 if queen_count else 2 if rook_count else 4
                    eg += sign * centrality * king_activity

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
    if phase <= 8:
        eg += _endgame_adjustment(board)
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
                passer_mg = PASSED_MG_BONUS[rank]
                passer_eg = PASSED_EG_BONUS[rank]

                pawn_attackers = chess.BB_PAWN_ATTACKS[not color][square] & pawns
                if pawn_attackers:
                    passer_mg += 4 + rank * 3
                    passer_eg += 8 + rank * 7

                neighbours = pawns & ADJACENT_FILES[file_index]
                if any(
                    abs(_relative_rank(color, neighbour) - rank) <= 1
                    and not enemy_pawns & PASSED_MASKS[int(color)][neighbour]
                    for neighbour in chess.scan_reversed(neighbours)
                ):
                    passer_mg += 5 + rank * 3
                    passer_eg += 10 + rank * 8

                promotion_square = chess.square(file_index, 7 if color else 0)
                path = chess.between(square, promotion_square) | chess.BB_SQUARES[promotion_square]
                if not (white_pawns | black_pawns) & path:
                    passer_mg += rank * 2
                    passer_eg += rank * 5

                mg += sign * passer_mg
                eg += sign * passer_eg
    return mg, eg


def _material_vector(board: chess.Board, color: chess.Color) -> tuple[int, int, int, int, int]:
    return (
        board.pieces_mask(chess.PAWN, color).bit_count(),
        board.pieces_mask(chess.KNIGHT, color).bit_count(),
        board.pieces_mask(chess.BISHOP, color).bit_count(),
        board.pieces_mask(chess.ROOK, color).bit_count(),
        board.pieces_mask(chess.QUEEN, color).bit_count(),
    )


def _endgame_class(board: chess.Board) -> str | None:
    """A small, position-independent material classifier used by evaluation and tests."""
    white = _material_vector(board, chess.WHITE)
    black = _material_vector(board, chess.BLACK)

    for strong, weak in ((white, black), (black, white)):
        if sum(weak) == 0:
            if strong == (0, 0, 0, 0, 1):
                return "KQK"
            if strong == (0, 0, 0, 1, 0):
                return "KRK"
            if strong == (0, 0, 2, 0, 0):
                bishops = board.bishops
                if bishops & chess.BB_LIGHT_SQUARES and bishops & chess.BB_DARK_SQUARES:
                    return "KBBK"
            if strong == (0, 1, 1, 0, 0):
                return "KBNK"
            if strong[0] == 1 and sum(strong[1:]) == 0:
                return "KPK"
            if strong[0] > 1 and sum(strong[1:]) == 0:
                return "KPPK"
            if strong[4] >= 1:
                return "KQXK"
            if strong[3] >= 1:
                return "KRXK"

        if (
            strong == (0, 0, 0, 0, 1)
            and weak[0] == weak[3] == weak[4] == 0
            and weak[1] + weak[2] == 1
        ):
            return "KQ_MINOR"
        if (
            strong == (0, 0, 0, 1, 0)
            and weak[0] == weak[3] == weak[4] == 0
            and weak[1] + weak[2] == 1
        ):
            return "KR_MINOR_DRAWISH"

    if (
        white[1] == black[1] == white[3] == black[3] == white[4] == black[4] == 0
        and white[2] == black[2] == 1
    ):
        white_bishop = next(iter(board.pieces(chess.BISHOP, chess.WHITE)))
        black_bishop = next(iter(board.pieces(chess.BISHOP, chess.BLACK)))
        same_color = bool(chess.BB_SQUARES[white_bishop] & chess.BB_LIGHT_SQUARES) == bool(
            chess.BB_SQUARES[black_bishop] & chess.BB_LIGHT_SQUARES
        )
        return "SAME_BISHOPS" if same_color else "OPPOSITE_BISHOPS"

    if (
        white[1] == black[1] == white[2] == black[2] == white[4] == black[4] == 0
        and white[3] == black[3] == 1
        and white[0] + black[0] > 0
    ):
        return "ROOK_PAWNS"
    if (
        white[1] == black[1] == white[2] == black[2] == white[3] == black[3] == 0
        and white[4] == black[4] == 1
        and white[0] + black[0] > 0
    ):
        return "QUEEN_PAWNS"
    return None


def _only_king(board: chess.Board, color: chess.Color) -> bool:
    return board.occupied_co[color] == board.kings & board.occupied_co[color]


def _mating_guidance(board: chess.Board, endgame_class: str | None) -> int:
    if endgame_class not in {"KQK", "KRK", "KBBK", "KQXK", "KRXK", "KQ_MINOR"}:
        return 0

    if _only_king(board, chess.BLACK):
        strong = chess.WHITE
    elif _only_king(board, chess.WHITE):
        strong = chess.BLACK
    elif endgame_class == "KQ_MINOR":
        strong = chess.WHITE if board.queens & board.occupied_co[chess.WHITE] else chess.BLACK
    else:
        return 0

    weak = not strong
    strong_king = board.king(strong)
    weak_king = board.king(weak)
    if strong_king is None or weak_king is None:
        return 0

    weak_file = chess.square_file(weak_king)
    weak_rank = chess.square_rank(weak_king)
    edge_distance = 3 - min(weak_file, 7 - weak_file, weak_rank, 7 - weak_rank)

    attacked = 0
    for square in chess.scan_reversed(board.occupied_co[strong]):
        attacked |= board.attacks_mask(square)
    safe_ring = chess.BB_KING_ATTACKS[weak_king] & ~attacked
    restriction = 8 - safe_ring.bit_count()
    proximity = 7 - KING_DISTANCE[strong_king][weak_king]

    scale = 1 if endgame_class == "KQ_MINOR" else 2
    bonus = (edge_distance * 28 + restriction * 9 + proximity * 8) * scale

    majors = (board.queens | board.rooks) & board.occupied_co[strong]
    for square in chess.scan_reversed(majors):
        if KING_DISTANCE[weak_king][square] <= 1 and KING_DISTANCE[strong_king][square] > 1:
            bonus -= 700

    return bonus if strong == chess.WHITE else -bonus


def _is_passed(board: chess.Board, color: chess.Color, square: chess.Square) -> bool:
    return not (board.pieces_mask(chess.PAWN, not color) & PASSED_MASKS[int(color)][square])


def _promotion_path(
    board: chess.Board, color: chess.Color, square: chess.Square
) -> tuple[chess.Square, ...]:
    rank = _relative_rank(color, square)
    if not 1 <= rank <= 6:
        return ()
    step = 8 if color == chess.WHITE else -8
    first = square + step
    if board.occupied & chess.BB_SQUARES[first]:
        return ()

    path: list[chess.Square] = []
    current = square
    if rank == 1:
        double = square + 2 * step
        if not board.occupied & chess.BB_SQUARES[double]:
            current = double
            path.append(current)
    while _relative_rank(color, current) < 7:
        current += step
        if board.occupied & chess.BB_SQUARES[current]:
            return ()
        path.append(current)
    return tuple(path)


def _king_can_catch_pawn(
    board: chess.Board,
    color: chess.Color,
    square: chess.Square,
    path: tuple[chess.Square, ...],
) -> bool:
    enemy_king = board.king(not color)
    own_king = board.king(color)
    if enemy_king is None:
        return False
    own_pawns = board.pieces_mask(chess.PAWN, color) & ~chess.BB_SQUARES[square]
    for move_number, target in enumerate(path, start=1):
        king_moves = move_number if board.turn != color else move_number - 1
        if KING_DISTANCE[enemy_king][target] > king_moves:
            continue
        protected = own_king is not None and KING_DISTANCE[own_king][target] <= 1
        protected |= bool(chess.BB_PAWN_ATTACKS[not color][target] & own_pawns)
        if not protected:
            return True
    return False


def _queen_line_attack(from_square: chess.Square, to_square: chess.Square, occupied: int) -> bool:
    file_delta = abs(chess.square_file(from_square) - chess.square_file(to_square))
    rank_delta = abs(chess.square_rank(from_square) - chess.square_rank(to_square))
    if file_delta != 0 and rank_delta != 0 and file_delta != rank_delta:
        return False
    return not bool(chess.between(from_square, to_square) & occupied)


def _promotion_info(
    board: chess.Board, color: chess.Color, square: chess.Square
) -> tuple[int, chess.Square, bool] | None:
    if not _is_passed(board, color, square):
        return None
    path = _promotion_path(board, color, square)
    if not path or _king_can_catch_pawn(board, color, square, path):
        return None
    plies = 2 * len(path) - int(board.turn == color)
    promotion_square = path[-1]
    enemy_king = board.king(not color)
    occupied = (board.occupied & ~chess.BB_SQUARES[square]) | chess.BB_SQUARES[promotion_square]
    promotes_with_check = enemy_king is not None and _queen_line_attack(
        promotion_square, enemy_king, occupied
    )
    return plies, promotion_square, promotes_with_check


def _pawn_race_adjustment(board: chess.Board) -> int:
    if board.knights or board.bishops or board.rooks or board.queens:
        return 0

    racers: dict[chess.Color, list[tuple[int, chess.Square, bool, chess.Square]]] = {
        chess.WHITE: [],
        chess.BLACK: [],
    }
    for color in (chess.WHITE, chess.BLACK):
        for square in chess.scan_reversed(board.pieces_mask(chess.PAWN, color)):
            info = _promotion_info(board, color, square)
            if info is not None:
                racers[color].append((*info, square))

    white = min(racers[chess.WHITE], default=None)
    black = min(racers[chess.BLACK], default=None)
    if white is None and black is None:
        return 0
    if black is None:
        return 100 + max(0, 13 - white[0]) * 8 if white is not None else 0
    if white is None:
        return -(100 + max(0, 13 - black[0]) * 8)

    white_tempo = white[0] - int(white[2])
    black_tempo = black[0] - int(black[2])
    if white_tempo == black_tempo:
        return 0

    white_first = white_tempo < black_tempo
    faster = white if white_first else black
    slower = black if white_first else white
    advantage = 130 + abs(white_tempo - black_tempo) * 40
    occupied_after = (board.occupied & ~chess.BB_SQUARES[faster[3]]) | chess.BB_SQUARES[faster[1]]
    if _queen_line_attack(faster[1], slower[3], occupied_after) or _queen_line_attack(
        faster[1], slower[1], occupied_after
    ):
        advantage += 70
    return advantage if white_first else -advantage


def _passed_pawn_adjustment(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        own_king = board.king(color)
        enemy_king = board.king(not color)
        enemy_stoppers = (board.rooks | board.queens) & board.occupied_co[not color]
        own_rooks = board.rooks & board.occupied_co[color]
        for square in chess.scan_reversed(board.pieces_mask(chess.PAWN, color)):
            if not _is_passed(board, color, square):
                continue
            rank = _relative_rank(color, square)
            step = 8 if color == chess.WHITE else -8
            front = square + step
            promotion_square = chess.square(chess.square_file(square), 7 if color else 0)
            bonus = 0

            path = chess.between(square, promotion_square) | chess.BB_SQUARES[promotion_square]
            path_clear = not bool(board.occupied & path)
            if path_clear:
                bonus += rank * 7

            if own_king is not None and enemy_king is not None:
                target = front if 0 <= front < 64 else square
                distance_scale = 4 + rank * 2
                bonus += (
                    KING_DISTANCE[enemy_king][target] - KING_DISTANCE[own_king][target]
                ) * distance_scale

            if board.attackers_mask(color, square) & ~chess.BB_SQUARES[square]:
                bonus += 5 + rank * 5

            rooks_on_file = own_rooks & FILE_MASKS[chess.square_file(square)]
            for rook_square in chess.scan_reversed(rooks_on_file):
                if _relative_rank(color, rook_square) < rank and not (
                    chess.between(rook_square, square) & board.occupied
                ):
                    bonus += 18 + rank * 4
                    break

            if enemy_stoppers and (
                board.is_attacked_by(not color, front)
                or board.is_attacked_by(not color, promotion_square)
            ):
                bonus -= 18 + rank * 7

            if rank == 6 and not board.occupied & chess.BB_SQUARES[front]:
                bonus += 110 if board.turn == color else 70
            score += sign * bonus
    return score


def _opposition_adjustment(board: chess.Board) -> int:
    if board.knights or board.bishops or board.rooks or board.queens:
        return 0
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None or KING_DISTANCE[white_king][black_king] != 2:
        return 0
    same_file = chess.square_file(white_king) == chess.square_file(black_king)
    same_rank = chess.square_rank(white_king) == chess.square_rank(black_king)
    if not (same_file or same_rank):
        return 0
    return 18 if board.turn == chess.BLACK else -18


def _endgame_adjustment(board: chess.Board) -> int:
    endgame_class = _endgame_class(board)
    score = _mating_guidance(board, endgame_class)
    score += _passed_pawn_adjustment(board)
    score += _pawn_race_adjustment(board)
    score += _opposition_adjustment(board)

    if endgame_class == "OPPOSITE_BISHOPS":
        pawn_balance = (
            board.pieces_mask(chess.PAWN, chess.WHITE).bit_count()
            - board.pieces_mask(chess.PAWN, chess.BLACK).bit_count()
        )
        score -= max(-90, min(90, pawn_balance * 24))
    elif endgame_class == "KR_MINOR_DRAWISH":
        white_has_rook = bool(board.rooks & board.occupied_co[chess.WHITE])
        score += -130 if white_has_rook else 130
    return score


def _tablebase_move(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move | None:
    """Choose an exact root move only when every child is covered by the shipped subset."""
    if TABLEBASE is None or board.occupied.bit_count() > 4:
        return None
    try:
        TABLEBASE.probe_wdl(board)
        ranked: list[tuple[tuple[int, int, int], chess.Move]] = []
        root_color = board.turn
        for move in legal_moves:
            board.push(move)
            try:
                outcome = board.outcome(claim_draw=True)
                if outcome is not None:
                    wdl = 0 if outcome.winner is None else 2 if outcome.winner == root_color else -2
                    distance = 0
                else:
                    wdl = -TABLEBASE.probe_wdl(board)
                    distance = abs(TABLEBASE.probe_dtz(board))
            finally:
                board.pop()

            if wdl > 0:
                distance_score = -distance
            elif wdl < 0:
                distance_score = distance
            else:
                distance_score = 0
            promotion_score = PIECE_VALUE[move.promotion] if move.promotion else 0
            ranked.append(((wdl, distance_score, promotion_score), move))
        return max(ranked, key=lambda item: item[0])[1] if ranked else None
    except Exception:
        return None


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

        clock_ms = max(1, time_left_ms)
        if clock_ms <= 100:
            self._record_played_position(board, legal_moves[0])
            return legal_moves[0]

        tablebase_move = _tablebase_move(board, legal_moves)
        if tablebase_move is not None:
            self._record_played_position(board, tablebase_move)
            self._trim_tt()
            print(f"[tablebase move={tablebase_move.uci()}]", file=sys.stderr)
            return tablebase_move
        for pair in self.killers:
            pair[0] = pair[1] = None

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
        nps = int(self.nodes / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
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
                board.pieces_mask(piece_type, chess.WHITE)
                | board.pieces_mask(piece_type, chess.BLACK)
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
        endgame_sensitive = self._zugzwang_sensitive(board)

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
            and not endgame_sensitive
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
            moving_piece = board.piece_type_at(move.from_square)
            critical_pawn_push = moving_piece == chess.PAWN and (
                _relative_rank(board.turn, move.to_square) >= 5
                or _is_passed(board, board.turn, move.from_square)
            )
            can_prune = (
                depth == 1
                and not pv_node
                and not in_check
                and is_quiet
                and not critical_pawn_push
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
                and not critical_pawn_push
                and move not in self.killers[min(ply, MAX_PLY - 1)]
            )
            if can_reduce and not gives_check:
                if board.gives_check(move):
                    gives_check = True
                else:
                    reduction = LMR_TABLE[min(depth, LMR_MAX_DEPTH - 1)][
                        min(move_index, LMR_MAX_INDEX - 1)
                    ]
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

            moving_piece = board.piece_type_at(move.from_square)
            if (
                is_quiet
                and moving_piece == chess.PAWN
                and _is_passed(board, color, move.from_square)
            ):
                rank = _relative_rank(color, move.to_square)
                score += 1_000_000 + rank * rank * 40_000

            scored.append((score, move, is_quiet, is_capture, see_value))

        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            (move, is_quiet, is_capture, see_value)
            for _, move, is_quiet, is_capture, see_value in scored
        ]

    @staticmethod
    def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
        return bool(
            board.pieces_mask(chess.KNIGHT, color)
            | board.pieces_mask(chess.BISHOP, color)
            | board.pieces_mask(chess.ROOK, color)
            | board.pieces_mask(chess.QUEEN, color)
        )

    @staticmethod
    def _zugzwang_sensitive(board: chess.Board) -> bool:
        pawns = board.pawns.bit_count()
        non_pawns = (board.knights | board.bishops | board.rooks | board.queens).bit_count()
        if pawns == 0 and non_pawns <= 3:
            return True
        if board.queens:
            return False
        return pawns <= 8 and non_pawns <= 2

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
        score = see(board, move) if board.is_capture(move) or move.promotion is not None else 0
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
