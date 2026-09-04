"""Reliability smoke test over random reachable positions and emergency clocks."""

from __future__ import annotations

import random
import time

import chess

import agent


def random_position(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for _ in range(rng.randrange(0, 100)):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(moves))
    return board


def main() -> None:
    rng = random.Random(20260904)
    clocks = (1, 5, 20, 31, 100, 500)
    maximum = {clock: 0.0 for clock in clocks}
    tested = 0
    for index in range(120):
        board = random_position(rng)
        if board.is_game_over(claim_draw=True):
            continue
        clock = clocks[index % len(clocks)]
        started = time.perf_counter()
        uci = agent.get_move(board.fen(), clock)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        maximum[clock] = max(maximum[clock], elapsed_ms)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise AssertionError(f"illegal move {uci} for {board.fen()}")
        tested += 1
    print(f"{tested} legal random-position replies")
    print("maximum milliseconds: " + ", ".join(f"{k}ms={v:.1f}" for k, v in maximum.items()))


if __name__ == "__main__":
    main()
