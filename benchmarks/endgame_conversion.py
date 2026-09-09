"""Run repeatable conversion games from representative winning endgames."""

from __future__ import annotations

import argparse
import io
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from harness.referee import play_match
from harness.sandbox import local


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    fen: str
    winning_color: chess.Color


CASES = (
    Case("kqk_corner_a", "KQK", "8/8/8/7k/8/2K5/8/Q7 w - - 0 1", chess.WHITE),
    Case("kqk_corner_e", "KQK", "8/2K5/8/8/8/7k/8/4Q3 w - - 0 1", chess.WHITE),
    Case("kqk_black_e", "KQK", "4q3/8/8/8/2k5/8/8/6K1 b - - 0 1", chess.BLACK),
    Case("kqk_black_b", "KQK", "1q6/8/8/8/5k2/8/8/7K b - - 0 1", chess.BLACK),
    Case("kqk_wide", "KQK", "8/8/6k1/8/7Q/8/1K6/8 w - - 0 1", chess.WHITE),
    Case("kqk_black_wide", "KQK", "8/8/K7/5k2/8/8/8/2q5 b - - 0 1", chess.BLACK),
    Case("krk_corner_a", "KRK", "8/8/8/7k/8/2K5/8/R7 w - - 0 1", chess.WHITE),
    Case("krk_corner_e", "KRK", "8/2K5/8/8/8/7k/8/4R3 w - - 0 1", chess.WHITE),
    Case("krk_black_e", "KRK", "4r3/8/8/8/2k5/8/8/6K1 b - - 0 1", chess.BLACK),
    Case("krk_black_b", "KRK", "1r6/8/8/8/5k2/8/8/7K b - - 0 1", chess.BLACK),
    Case("krk_wide", "KRK", "8/8/6k1/8/1R6/8/1K6/8 w - - 0 1", chess.WHITE),
    Case("krk_black_wide", "KRK", "8/8/K7/5k2/8/8/8/2r5 b - - 0 1", chess.BLACK),
    Case("outside_passer", "PAWN", "8/7k/8/P7/8/8/8/K7 w - - 0 1", chess.WHITE),
    Case("promotion_check_race", "PAWN", "7k/8/P7/8/8/7p/8/K7 w - - 0 1", chess.WHITE),
    Case("connected_passers", "PAWN", "7k/8/8/2PP4/8/8/8/K7 w - - 0 1", chess.WHITE),
    Case("queen_extra_pawns", "MATERIAL", "7k/8/8/8/1PP5/8/Q7/K7 w - - 0 1", chess.WHITE),
    Case("rook_extra_pawns", "MATERIAL", "7k/8/8/8/1PP5/8/R7/K7 w - - 0 1", chess.WHITE),
    Case("two_bishops", "MATERIAL", "8/8/8/7k/8/2K5/8/1BB5 w - - 0 1", chess.WHITE),
    Case("bishop_knight", "MATERIAL", "8/8/8/7k/8/2K5/8/1BN5 w - - 0 1", chess.WHITE),
)


def _plies(pgn: str) -> int:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return len(list(game.mainline_moves())) if game is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/friend"))
    parser.add_argument("--group", choices=sorted({case.group for case in CASES}))
    parser.add_argument("--case")
    parser.add_argument("--base-ms", type=int, default=1_500)
    parser.add_argument("--increment-ms", type=int, default=50)
    parser.add_argument("--ply-cap", type=int, default=150)
    parser.add_argument("--show-pgn-on-fail", action="store_true")
    args = parser.parse_args()

    cases = [
        case
        for case in CASES
        if (args.group is None or case.group == args.group)
        and (args.case is None or case.name == args.case)
    ]
    results: dict[str, list[bool]] = defaultdict(list)
    plies: dict[str, list[int]] = defaultdict(list)
    terminations: Counter[str] = Counter()
    started = time.perf_counter()

    for case in cases:
        agent = args.agent.resolve()
        opponent = args.opponent.resolve()
        white = local(agent if case.winning_color == chess.WHITE else opponent)
        black = local(opponent if case.winning_color == chess.WHITE else agent)
        outcome = play_match(
            white,
            black,
            args.base_ms,
            args.increment_ms,
            ply_cap=args.ply_cap,
            start_fen=case.fen,
        )
        won = outcome.result == ("white" if case.winning_color == chess.WHITE else "black")
        game_plies = _plies(outcome.pgn)
        results[case.group].append(won)
        plies[case.group].append(game_plies)
        terminations[outcome.termination] += 1
        print(
            f"{case.name}: {'PASS' if won else 'FAIL'} "
            f"({outcome.result}, {outcome.termination}, {game_plies} plies)"
        )
        if not won and args.show_pgn_on_fail:
            print(outcome.pgn)

    for group in sorted(results):
        successes = sum(results[group])
        average = sum(plies[group]) / len(plies[group])
        print(f"{group}: {successes}/{len(results[group])} ({average:.1f} mean plies)")
    details = ", ".join(f"{name}={count}" for name, count in sorted(terminations.items()))
    print(f"terminations: {details}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
