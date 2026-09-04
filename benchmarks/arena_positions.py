"""Head-to-head arena over several fixed, balanced opening positions.

This lives outside ``harness`` so the competition mirror remains untouched. Each opening is
played twice with colors reversed, making deterministic engine comparisons less dependent on a
single starting-position game.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local

OPENINGS = (
    ("start", ()),
    ("ruy-lopez", ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6")),
    ("sicilian", ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6")),
    ("french", ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6")),
    ("caro-kann", ("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4")),
    ("qgd", ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6")),
    ("slav", ("d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6")),
    (
        "kings-indian",
        ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"),
    ),
    ("english", ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "f8b4")),
)


def opening_fen(moves: tuple[str, ...]) -> str:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board.fen()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=3_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=200)
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    for name, moves in OPENINGS:
        fen = opening_fen(moves)
        for agent_is_white in (True, False):
            white, black = (agent, opponent) if agent_is_white else (opponent, agent)
            outcome = play_match(
                local(white),
                local(black),
                arguments.base_ms,
                arguments.increment_ms,
                ply_cap=arguments.ply_cap,
                start_fen=fen,
            )
            terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
            if outcome.result in {"draw", "void"}:
                draws += 1
                marker = "="
            elif (outcome.result == "white") == agent_is_white:
                wins += 1
                marker = "+"
            else:
                losses += 1
                marker = "-"
            color = "white" if agent_is_white else "black"
            print(f"{name:13} as {color:5}: {marker} {outcome.termination}", flush=True)

    games = wins + draws + losses
    score = (wins + draws / 2) / games
    print(f"\n+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{key} {value}" for key, value in terminations.items()))
    broken = {key: value for key, value in terminations.items() if key in FAILED_TERMINATIONS}
    if broken:
        raise SystemExit(
            "agent failure: " + ", ".join(f"{key} {value}" for key, value in broken.items())
        )


if __name__ == "__main__":
    main()
