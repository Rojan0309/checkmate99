from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import chess

import agent


def white_score(board: chess.Board) -> int:
    score = agent.evaluate(board)
    return score if board.turn == chess.WHITE else -score


class MaterialClassTests(unittest.TestCase):
    def test_major_and_mating_classes(self) -> None:
        cases = {
            "8/8/8/7k/8/2K5/8/Q7 w - - 0 1": "KQK",
            "8/8/8/7k/8/2K5/8/R7 w - - 0 1": "KRK",
            "8/8/8/7k/8/2K5/8/1BB5 w - - 0 1": "KBBK",
            "8/8/8/7k/8/2K5/8/1BN5 w - - 0 1": "KBNK",
            "8/8/5n2/7k/8/2K5/8/Q7 w - - 0 1": "KQ_MINOR",
            "8/8/5n2/7k/8/2K5/8/R7 w - - 0 1": "KR_MINOR_DRAWISH",
            "8/8/8/7k/8/2K5/P7/Q7 w - - 0 1": "KQXK",
        }
        for fen, expected in cases.items():
            with self.subTest(expected):
                self.assertEqual(agent._endgame_class(chess.Board(fen)), expected)

    def test_pawn_and_bishop_classes(self) -> None:
        cases = {
            "7k/8/8/P7/8/8/8/K7 w - - 0 1": "KPK",
            "7k/8/8/2PP4/8/8/8/K7 w - - 0 1": "KPPK",
            "4kb2/7p/8/8/8/8/P7/2B1K3 w - - 0 1": "SAME_BISHOPS",
            "4b3/7p/7k/8/8/8/P7/2B1K3 w - - 0 1": "OPPOSITE_BISHOPS",
            "4k2r/7p/8/8/8/8/P7/R3K3 w - - 0 1": "ROOK_PAWNS",
            "4k2q/7p/8/8/8/8/P7/Q3K3 w - - 0 1": "QUEEN_PAWNS",
        }
        for fen, expected in cases.items():
            with self.subTest(expected):
                self.assertEqual(agent._endgame_class(chess.Board(fen)), expected)


class PassedPawnTests(unittest.TestCase):
    def test_advanced_passer_bonus_is_nonlinear(self) -> None:
        fourth = chess.Board("7k/8/8/8/2P5/8/8/K7 w - - 0 1")
        seventh = chess.Board("7k/2P5/8/8/8/8/8/K7 w - - 0 1")
        self.assertGreater(white_score(seventh) - white_score(fourth), 250)

    def test_connected_passers_outscore_separated_passers(self) -> None:
        connected = chess.Board("7k/8/8/2PP4/8/8/8/K7 w - - 0 1")
        separated = chess.Board("7k/8/8/2P2P2/8/8/8/K7 w - - 0 1")
        self.assertGreater(white_score(connected), white_score(separated))

    def test_friendly_king_proximity_helps_passer(self) -> None:
        near = chess.Board("7k/8/8/4P3/4K3/8/8/8 w - - 0 1")
        far = chess.Board("7k/8/8/4P3/8/8/8/K7 w - - 0 1")
        self.assertGreater(white_score(near), white_score(far))

    def test_rook_belongs_behind_passed_pawn(self) -> None:
        behind = chess.Board("7k/8/8/P7/8/8/R7/2K5 w - - 0 1")
        aside = chess.Board("7k/8/8/P7/8/8/6R1/2K5 w - - 0 1")
        self.assertGreater(
            agent._passed_pawn_adjustment(behind), agent._passed_pawn_adjustment(aside)
        )

    def test_immediate_enemy_promotion_is_a_large_threat(self) -> None:
        advanced = chess.Board("7k/8/8/8/8/8/2p5/K7 b - - 0 1")
        distant = chess.Board("7k/8/8/2p5/8/8/8/K7 b - - 0 1")
        self.assertLess(white_score(advanced), white_score(distant) - 200)

    def test_realising_promotion_improves_score(self) -> None:
        board = chess.Board("8/2PP4/8/4k3/8/8/8/3K4 w - - 0 1")
        before = white_score(board)
        board.push_uci("c7c8q")
        self.assertGreater(white_score(board), before)


class PawnRaceTests(unittest.TestCase):
    def test_square_rule_inside_and_outside(self) -> None:
        outside = chess.Board("8/7k/8/P7/8/8/8/K7 w - - 0 1")
        inside = chess.Board("8/2k5/8/P7/8/8/8/K7 w - - 0 1")
        self.assertIsNotNone(agent._promotion_info(outside, chess.WHITE, chess.A5))
        self.assertIsNone(agent._promotion_info(inside, chess.WHITE, chess.A5))

    def test_promotion_with_check_wins_race_tempo(self) -> None:
        board = chess.Board("7k/8/P7/8/8/7p/8/K7 w - - 0 1")
        self.assertGreater(agent._pawn_race_adjustment(board), 0)

    def test_connected_passers_have_a_queening_route(self) -> None:
        board = chess.Board("7k/8/8/2PP4/8/8/8/K7 w - - 0 1")
        infos = [
            agent._promotion_info(board, chess.WHITE, square)
            for square in board.pieces(chess.PAWN, chess.WHITE)
        ]
        self.assertTrue(any(info is not None for info in infos))


class PromotionTests(unittest.TestCase):
    UNDERPROMOTION_FEN = "6Q1/2P1k3/8/8/3Q4/2K5/8/8 w - - 0 61"

    def test_all_four_promotions_are_kept(self) -> None:
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        promotions = {move.promotion for move in board.legal_moves if move.from_square == chess.A7}
        self.assertEqual(promotions, {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT})

    def test_knight_underpromotion_avoids_stalemate_and_mates(self) -> None:
        board = chess.Board(self.UNDERPROMOTION_FEN)
        outcomes: dict[int, tuple[bool, bool]] = {}
        for move in board.legal_moves:
            if move.from_square != chess.C7 or move.promotion is None:
                continue
            board.push(move)
            hip = (board.is_checkmate(), board.is_stalemate())
            board.pop()
            outcomes[move.promotion] = hip
        self.assertEqual(outcomes[chess.KNIGHT], (True, False))
        self.assertEqual(outcomes[chess.QUEEN], (False, True))

        engine = agent.Engine()
        engine.deadline = time.perf_counter() + 2.0
        score, move = engine._root(board, 1, -agent.INF, agent.INF)
        self.assertEqual(move, chess.Move.from_uci("c7c8n"))
        self.assertGreaterEqual(score, agent.MATE_BOUND)


class DrawPolicyTests(unittest.TestCase):
    @staticmethod
    def _seed_repeating_child(engine: agent.Engine, board: chess.Board) -> chess.Move:
        move = list(board.legal_moves)[-1]
        board.push(move)
        key = agent._key(board)
        board.pop()
        engine.repetitions[key] = 2
        engine.repeated_positions = 1
        return move

    def test_winning_side_avoids_available_repetition(self) -> None:
        board = chess.Board("8/8/8/7k/8/2K5/8/Q7 w - - 0 1")
        engine = agent.Engine()
        repetition = self._seed_repeating_child(engine, board)
        engine.deadline = time.perf_counter() + 2.0
        score, move = engine._root(board, 1, -agent.INF, agent.INF)
        self.assertNotEqual(move, repetition)
        self.assertGreater(score, 0)

    def test_losing_side_takes_available_repetition(self) -> None:
        board = chess.Board("4q3/8/8/8/2k5/8/8/6K1 w - - 0 1")
        engine = agent.Engine()
        repetition = self._seed_repeating_child(engine, board)
        engine.deadline = time.perf_counter() + 2.0
        score, move = engine._root(board, 1, -agent.INF, agent.INF)
        self.assertEqual(move, repetition)
        self.assertEqual(score, 0)

    def test_rule_draws_and_insufficient_material(self) -> None:
        engine = agent.Engine()
        self.assertTrue(
            engine._is_insufficient_material(chess.Board("8/8/8/8/8/2k5/8/K7 w - - 0 1"))
        )
        self.assertTrue(
            engine._is_insufficient_material(chess.Board("8/8/8/7k/8/2K5/8/2B5 w - - 0 1"))
        )
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(stalemate.is_stalemate())
        fifty = chess.Board("7k/8/8/8/8/2K5/8/R7 w - - 100 80")
        self.assertTrue(fifty.is_fifty_moves())

    def test_sparse_pawn_and_rook_endings_disable_risky_pruning(self) -> None:
        engine = agent.Engine()
        pawn_ending = chess.Board("8/7k/8/P7/8/8/8/K7 w - - 0 1")
        rook_ending = chess.Board("7k/7p/8/8/P7/8/R7/2K4r w - - 0 1")
        middlegame = chess.Board()
        self.assertTrue(engine._zugzwang_sensitive(pawn_ending))
        self.assertTrue(engine._zugzwang_sensitive(rook_ending))
        self.assertFalse(engine._zugzwang_sensitive(middlegame))


class TablebaseTests(unittest.TestCase):
    def test_subset_is_available_and_covers_selected_classes(self) -> None:
        self.assertIsNotNone(agent.TABLEBASE)
        assert agent.TABLEBASE is not None
        positions = (
            "8/8/8/7k/8/2K5/8/Q7 w - - 0 1",
            "8/8/8/7k/8/2K5/8/R7 w - - 0 1",
            "8/8/8/7k/8/2K5/8/1BN5 w - - 0 1",
            "8/7k/8/P7/8/8/8/K7 w - - 0 1",
        )
        for fen in positions:
            with self.subTest(fen):
                board = chess.Board(fen)
                agent.TABLEBASE.probe_wdl(board)
                self.assertIsNotNone(agent._tablebase_move(board, list(board.legal_moves)))

    def test_known_kpk_win_and_rook_pawn_draw(self) -> None:
        assert agent.TABLEBASE is not None
        winning = chess.Board("8/7k/8/P7/8/8/8/K7 w - - 0 1")
        drawn = chess.Board("k7/P7/2K5/8/8/8/8/8 w - - 0 1")
        self.assertGreater(agent.TABLEBASE.probe_wdl(winning), 0)
        self.assertEqual(agent.TABLEBASE.probe_wdl(drawn), 0)

    def test_queen_wins_but_bare_rook_vs_minor_is_drawish(self) -> None:
        assert agent.TABLEBASE is not None
        queen_knight = chess.Board("8/8/5n2/7k/8/2K5/8/Q7 w - - 0 1")
        rook_knight = chess.Board("8/8/5n2/7k/8/2K5/8/R7 w - - 0 1")
        rook_bishop = chess.Board("8/8/5b2/7k/8/2K5/8/R7 w - - 0 1")
        self.assertGreater(agent.TABLEBASE.probe_wdl(queen_knight), 0)
        self.assertEqual(agent.TABLEBASE.probe_wdl(rook_knight), 0)
        self.assertEqual(agent.TABLEBASE.probe_wdl(rook_bishop), 0)

    def test_missing_tablebase_falls_back_cleanly(self) -> None:
        board = chess.Board("8/8/8/7k/8/2K5/8/Q7 w - - 0 1")
        with patch.object(agent, "TABLEBASE", None):
            self.assertIsNone(agent._tablebase_move(board, list(board.legal_moves)))

    def test_critical_clock_skips_tablebase_probe(self) -> None:
        board = chess.Board("8/8/8/7k/8/2K5/8/Q7 w - - 0 1")
        legal_moves = list(board.legal_moves)
        with patch.object(agent, "_tablebase_move", side_effect=AssertionError("probe called")):
            move = agent.Engine().choose_move(board, 100, legal_moves)
        self.assertIn(move, legal_moves)


if __name__ == "__main__":
    unittest.main()
