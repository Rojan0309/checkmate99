"""Frozen copy of the repository's original random-move starter."""

import random

import chess


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()
