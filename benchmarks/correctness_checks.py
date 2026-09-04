"""Focused competition-correctness regressions for the root agent."""

from __future__ import annotations

import time

import chess

import agent


def search_engine(board: chess.Board) -> agent.Engine:
    engine = agent.Engine()
    engine.deadline = time.perf_counter() + 60.0
    engine.age = 1
    engine._push_repetition(agent._key(board))
    return engine


def assert_legal_reply(fen: str, clock_ms: int = 1) -> None:
    board = chess.Board(fen)
    uci = agent.get_move(fen, clock_ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, (fen, uci)


def main() -> None:
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.is_stalemate()
    assert search_engine(stalemate)._quiescence(stalemate, -agent.INF, agent.INF, 3) == 0
    assert agent.get_move(stalemate.fen(), 1) == "0000"

    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 80")
    assert mate.is_checkmate()
    assert search_engine(mate)._quiescence(mate, -agent.INF, agent.INF, 3) == -agent.MATE + 3
    repeated_mate_engine = search_engine(mate)
    repeated_mate_engine.repetitions[agent._key(mate)] = 3
    repeated_mate_engine.repeated_positions = 1
    assert repeated_mate_engine._quiescence(mate, -agent.INF, agent.INF, 3) == (
        -agent.MATE + 3
    )
    assert search_engine(mate)._search(mate, 1, -agent.INF, agent.INF, 3, True, True) == (
        -agent.MATE + 3
    )
    assert agent.get_move(mate.fen(), 1) == "0000"

    fifty = chess.Board("8/8/8/8/8/2k5/4R3/2K5 w - - 100 50")
    assert fifty.can_claim_fifty_moves()
    assert search_engine(fifty)._quiescence(fifty, -agent.INF, agent.INF, 1) == 0
    fifty_before = chess.Board("8/8/8/8/8/2k5/4R3/2K5 w - - 99 50")
    assert fifty_before.can_claim_fifty_moves()
    assert search_engine(fifty_before)._quiescence(fifty_before, -agent.INF, agent.INF, 1) == 0

    insufficient = chess.Board("8/8/8/8/8/2k5/4B3/2K5 w - - 0 1")
    assert insufficient.is_insufficient_material()
    assert search_engine(insufficient)._quiescence(
        insufficient, -agent.INF, agent.INF, 1
    ) == 0

    starting = chess.Board()
    repeated_child = starting.copy(stack=False)
    repeated_child.push_uci("g1f3")
    repetition_engine = search_engine(starting)
    repetition_engine.repetitions[agent._key(repeated_child)] = 2
    repetition_engine.repeated_positions = 1
    assert repetition_engine._is_rule_draw(
        starting, agent._key(starting), list(starting.legal_moves)
    )
    assert repetition_engine.repetition_tainted

    repeated_game = chess.Board()
    positions = [agent._key(repeated_game)]
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        repeated_game.push_uci(uci)
        positions.append(agent._key(repeated_game))
    assert repeated_game.can_claim_threefold_repetition()
    game_history_engine = agent.Engine()
    for position_key in positions:
        game_history_engine._push_repetition(position_key)
    game_history_engine.deadline = time.perf_counter() + 60.0
    assert game_history_engine._quiescence(
        repeated_game, -agent.INF, agent.INF, 1
    ) == 0

    no_clock = chess.Board(chess.STARTING_FEN)
    clock_99 = chess.Board(chess.STARTING_FEN.replace(" 0 1", " 99 50"))
    assert agent._key(no_clock) == agent._key(clock_99)
    assert agent._tt_key(no_clock) != agent._tt_key(clock_99)
    tt_engine = search_engine(no_clock)
    tt_engine.tt[agent._tt_key(no_clock)] = agent.TTEntry(99, 12_345, agent.EXACT, None, 0)
    assert tt_engine._search(no_clock, 1, -agent.INF, agent.INF, 0, True, True) != 12_345

    promotion = chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")
    promotion_moves = {move.uci() for move in promotion.legal_moves if move.from_square == chess.A7}
    assert promotion_moves == {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}
    search_engine(promotion)._ordered_moves(
        promotion, list(promotion.legal_moves), None, 0
    )

    en_passant = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    ep_move = chess.Move.from_uci("e5d6")
    assert ep_move in en_passant.legal_moves and en_passant.is_en_passant(ep_move)
    assert agent.Engine._victim_value(en_passant, ep_move) == agent.PIECE_VALUE[chess.PAWN]

    castling = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert chess.Move.from_uci("e1g1") in castling.legal_moves
    assert chess.Move.from_uci("e1c1") in castling.legal_moves

    forced = chess.Board("r7/1k3p2/7p/ppp3bn/Pr2P1RP/5PN1/1q2R3/1K6 w - - 0 62")
    forced_moves = list(forced.legal_moves)
    assert [move.uci() for move in forced_moves] == ["e2b2"]
    forced_engine = agent.Engine()
    forced_move = forced_engine.choose_move(forced, 1, forced_moves)
    assert forced_engine.repetitions[agent._key(forced)] == 1
    forced.push(forced_move)
    assert forced_engine.repetitions[agent._key(forced)] == 1

    edge_fens = (
        promotion.fen(),
        en_passant.fen(),
        castling.fen(),
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
        chess.STARTING_FEN.replace(" w ", " b "),
    )
    for fen in edge_fens:
        assert_legal_reply(fen)

    print("competition correctness checks passed")


if __name__ == "__main__":
    main()
