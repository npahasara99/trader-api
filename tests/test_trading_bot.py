from __future__ import annotations

from dataclasses import replace
import unittest

from app.bot.broker import BracketOrderRequest, BrokerAccountSummary, BrokerError, IBKRBroker, MockBroker
from app.bot.enums import TradingMode
from app.bot.service import TradingBotService
from app.models import TradeCandidate


class MockBrokerTests(unittest.TestCase):
    def test_ibkr_sentinel_prices_are_not_usable_quotes(self):
        self.assertIsNone(IBKRBroker._price(-1))
        self.assertIsNone(IBKRBroker._price(float("nan")))
        self.assertEqual(IBKRBroker._price(501.25), 501.25)

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
    @staticmethod
    def _isolate_risk_queries(service: TradingBotService) -> None:
        service.exposure_status = lambda: {"capital_utilization_pct": 0.0, "capital_in_use": 0.0}
        service.risk_status = lambda: {"open_portfolio_risk": 0.0}

    def test_disabled_mode_preview_blocks_submission(self):
        service = TradingBotService()
        service._config = replace(service._config, trading_mode=TradingMode.DISABLED, ibkr_read_only=False)
        service._broker = MockBroker()
        service._broker.connect()
        service._broker.seed_quote("MSFT", last=450.0)
        self._isolate_risk_queries(service)
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
        self._isolate_risk_queries(service)
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

    def test_required_paper_account_rejects_live_account(self):
        service = TradingBotService()
        live_account = BrokerAccountSummary(
            account_id="U1234567",
            account_type="individual",
            is_paper=False,
            cash_balance=10_000.0,
            buying_power=20_000.0,
            net_liquidation_value=10_000.0,
        )
        with self.assertRaises(BrokerError):
            service._validate_broker_account(live_account)


if __name__ == "__main__":
    unittest.main()
