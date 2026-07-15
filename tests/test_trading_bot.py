from __future__ import annotations

import unittest

from app.bot.broker import BracketOrderRequest, MockBroker
from app.bot.config import load_bot_config
from app.bot.service import TradingBotService
from app.models import TradeCandidate


class MockBrokerTests(unittest.TestCase):
    def test_mock_broker_connects_and_reports_paper_account(self):
        broker = MockBroker()
        broker.connect()
        summary = broker.account_summary()
        self.assertTrue(summary.is_paper)
        self.assertTrue(broker.health_check()["connected"])

    def test_bracket_order_builds_parent_and_protection(self):
        broker = MockBroker()
        broker.seed_quote("AAPL", last=200.0)
        broker.connect()
        orders = broker.place_bracket_order(
            BracketOrderRequest(
                ticker="AAPL",
                side="BUY",
                quantity=10,
                entry_price=199.5,
                stop_price=194.0,
                target_price_1=208.0,
                target_price_2=212.0,
            )
        )
        self.assertGreaterEqual(len(orders), 3)
        self.assertTrue(any(order.child_role == "stop" for order in orders))


class BotServicePreviewTests(unittest.TestCase):
    def test_disabled_mode_preview_blocks_submission(self):
        service = TradingBotService()
        service._broker = MockBroker()
        service._broker.connect()
        service._broker.seed_quote("MSFT", last=450.0)
        candidate = TradeCandidate(
            candidate_id="cand-1",
            ticker="MSFT",
            basket="manual",
            status="monitoring",
            preferred_entry=449.0,
            stop_loss=440.0,
            take_profit_1=468.0,
            take_profit_2=475.0,
        )
        preview = service._build_execution_preview(candidate, side="BUY", order_type="LIMIT")
        self.assertIn("trading_disabled", preview["rejection_codes"])
        self.assertFalse(preview["eligible"])

    def test_quantity_below_one_is_rejected(self):
        service = TradingBotService()
        service._broker = MockBroker(cash_balance=50.0)
        service._broker.connect()
        service._broker.seed_quote("NVDA", last=150.0)
        candidate = TradeCandidate(
            candidate_id="cand-2",
            ticker="NVDA",
            basket="manual",
            status="monitoring",
            preferred_entry=150.0,
            stop_loss=149.5,
            take_profit_1=153.0,
            take_profit_2=None,
        )
        preview = service._build_execution_preview(candidate, side="BUY", order_type="LIMIT")
        self.assertIn("quantity_below_one", preview["rejection_codes"])


if __name__ == "__main__":
    unittest.main()
