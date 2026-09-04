"""Interleaved fixed-depth timing for two snapshot-compatible agent modules."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chess


def load_agent(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def search(module: Any, depth: int) -> tuple[float, int, str]:
    board = chess.Board()
    engine = module.Engine()
    engine.deadline = time.perf_counter() + 60.0
    engine.repetitions[module._key(board)] = 1
    started = time.perf_counter()
    score, move = engine._root(board, depth, -module.INF, module.INF)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    return elapsed_ms, int(engine.nodes), f"{score}:{move}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    arguments = parser.parse_args()
    modules = (
        load_agent("speed_first", arguments.first),
        load_agent("speed_second", arguments.second),
    )
    samples: list[list[float]] = [[], []]
    signatures: list[set[tuple[int, str]]] = [set(), set()]
    for _ in range(arguments.rounds):
        for index, module in enumerate(modules):
            elapsed, nodes, signature = search(module, arguments.depth)
            samples[index].append(elapsed)
            signatures[index].add((nodes, signature))
            print(index + 1, f"{elapsed:.1f} ms", nodes, signature, flush=True)
    for index in range(2):
        print(
            index + 1,
            f"median {statistics.median(samples[index]):.1f} ms",
            sorted(signatures[index]),
        )


if __name__ == "__main__":
    main()
