"""The submission entrypoint. The platform imports this file and calls get_move."""

import chess
import random

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 10**6


def material(board: chess.Board, side: chess.Color) -> int:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )


def evaluate(board: chess.Board) -> int:
    """Static evaluation: material + mate detection."""
    mover = board.turn
    if board.is_checkmate():
        return -MATE  # current player is mated
    return material(board, mover)


def minimax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """Minimax search with alpha-beta pruning."""
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    mover = board.turn
    best_score = -MATE

    for move in board.legal_moves:
        board.push(move)
        score = -minimax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best_score:
            best_score = score

        alpha = max(alpha, score)
        if alpha >= beta:
            break  # prune

    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)

    # Choose depth based on time left
    depth = 3 if time_left_ms > 2000 else 2

    best_score = -MATE
    best_moves: list[chess.Move] = []

    for move in board.legal_moves:
        board.push(move)
        score = -minimax(board, depth - 1, -MATE, MATE)
        board.pop()

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return random.choice(best_moves).uci()
