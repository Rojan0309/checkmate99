"""Micro-profile static evaluation without involving move-search timing."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import timeit
from pathlib import Path
from types import ModuleType

import chess

POSITIONS = (
    ("start", chess.STARTING_FEN),
    ("rook_end", "8/5pk1/6p1/3P4/3R4/6P1/5PK1/7r w - - 0 1"),
    ("pawn_end", "8/7k/8/P7/8/8/8/K7 w - - 0 1"),
)


def _load(path: Path) -> ModuleType:
    name = f"profiled_agent_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("agent.py"))
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    module = _load(args.agent)

    for name, fen in POSITIONS:
        board = chess.Board(fen)
        elapsed = timeit.timeit(
            lambda position=board: module.evaluate(position), number=args.iterations
        )
        microseconds = elapsed * 1_000_000 / args.iterations
        print(f"{name}: {microseconds:.1f} us/eval")


if __name__ == "__main__":
    main()
