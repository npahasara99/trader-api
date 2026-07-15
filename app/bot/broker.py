from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import time
import uuid


@dataclass
class BrokerQuote:
    ticker: str
    last: float | None
    bid: float | None = None
    ask: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "mock"

    @property
    def spread_pct(self) -> float | None:
        if self.bid in (None, 0) or self.ask is None:
            return None
        return ((self.ask - self.bid) / self.bid) * 100.0


@dataclass
class BrokerAccountSummary:
    account_id: str
    account_type: str
    is_paper: bool
    cash_balance: float
    buying_power: float
    net_liquidation_value: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPositionSnapshot:
    ticker: str
    quantity: int
    average_cost: float
    market_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None


@dataclass
class BrokerOrderSnapshot:
    broker_order_id: str
    ticker: str
    side: str
    order_type: str
    quantity: int
    status: str
    limit_price: float | None = None
    stop_price: float | None = None
    parent_order_id: str | None = None
    child_role: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerExecutionSnapshot:
    execution_id: str
    broker_order_id: str
    ticker: str
    side: str
    quantity: int
    price: float
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commission: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BracketOrderRequest:
    ticker: str
    side: str
    quantity: int
    entry_price: float
    stop_price: float
    target_price_1: float
    target_price_2: float | None = None
    order_type: str = "LIMIT"
    allow_extended_hours: bool = False
    client_order_key: str | None = None


class BrokerError(RuntimeError):
    pass


class BrokerInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def reconnect(self) -> None: ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]: ...

    @abstractmethod
    def account_summary(self) -> BrokerAccountSummary: ...

    @abstractmethod
    def positions(self) -> list[BrokerPositionSnapshot]: ...

    @abstractmethod
    def open_orders(self) -> list[BrokerOrderSnapshot]: ...

    @abstractmethod
    def completed_orders(self) -> list[BrokerOrderSnapshot]: ...

    @abstractmethod
    def executions(self) -> list[BrokerExecutionSnapshot]: ...

    @abstractmethod
    def current_quote(self, ticker: str) -> BrokerQuote: ...

    @abstractmethod
    def place_order(self, *, ticker: str, side: str, quantity: int, order_type: str, limit_price: float | None = None, stop_price: float | None = None, parent_order_id: str | None = None, child_role: str | None = None) -> BrokerOrderSnapshot: ...

    @abstractmethod
    def place_bracket_order(self, request: BracketOrderRequest) -> list[BrokerOrderSnapshot]: ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot: ...

    @abstractmethod
    def modify_order(self, broker_order_id: str, *, limit_price: float | None = None, stop_price: float | None = None, quantity: int | None = None) -> BrokerOrderSnapshot: ...

    @abstractmethod
    def close_position(self, ticker: str) -> list[BrokerOrderSnapshot]: ...

    @abstractmethod
    def flatten_all_positions(self) -> list[BrokerOrderSnapshot]: ...


class MockBroker(BrokerInterface):
    def __init__(self, *, account_id: str = "DU1234567", is_paper: bool = True, cash_balance: float = 100_000.0) -> None:
        self._connected = False
        self._account = BrokerAccountSummary(
            account_id=account_id,
            account_type="paper" if is_paper else "live",
            is_paper=is_paper,
            cash_balance=cash_balance,
            buying_power=cash_balance * 2.0,
            net_liquidation_value=cash_balance,
        )
        self._positions: dict[str, BrokerPositionSnapshot] = {}
        self._open_orders: dict[str, BrokerOrderSnapshot] = {}
        self._completed_orders: dict[str, BrokerOrderSnapshot] = {}
        self._executions: list[BrokerExecutionSnapshot] = []
        self._quotes: dict[str, BrokerQuote] = {}

    def seed_quote(self, ticker: str, *, last: float, bid: float | None = None, ask: float | None = None) -> None:
        self._quotes[ticker.upper()] = BrokerQuote(
            ticker=ticker.upper(),
            last=last,
            bid=bid if bid is not None else last * 0.998,
            ask=ask if ask is not None else last * 1.002,
            source="mock",
        )

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def reconnect(self) -> None:
        self.disconnect()
        time.sleep(0.01)
        self.connect()

    def health_check(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "account_id": self._account.account_id[-4:],
            "is_paper": self._account.is_paper,
        }

    def account_summary(self) -> BrokerAccountSummary:
        self._require_connection()
        self._account.net_liquidation_value = self._account.cash_balance + sum((p.market_value or 0.0) for p in self._positions.values())
        return self._account

    def positions(self) -> list[BrokerPositionSnapshot]:
        self._require_connection()
        return list(self._positions.values())

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        return list(self._open_orders.values())

    def completed_orders(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        return list(self._completed_orders.values())

    def executions(self) -> list[BrokerExecutionSnapshot]:
        self._require_connection()
        return list(self._executions)

    def current_quote(self, ticker: str) -> BrokerQuote:
        self._require_connection()
        ticker = ticker.upper()
        quote = self._quotes.get(ticker)
        if quote is None:
            raise BrokerError(f"Mock quote unavailable for {ticker}")
        return quote

    def place_order(self, *, ticker: str, side: str, quantity: int, order_type: str, limit_price: float | None = None, stop_price: float | None = None, parent_order_id: str | None = None, child_role: str | None = None) -> BrokerOrderSnapshot:
        self._require_connection()
        broker_order_id = str(uuid.uuid4())
        order = BrokerOrderSnapshot(
            broker_order_id=broker_order_id,
            ticker=ticker.upper(),
            side=side.upper(),
            order_type=order_type.upper(),
            quantity=int(quantity),
            status="submitted",
            limit_price=limit_price,
            stop_price=stop_price,
            parent_order_id=parent_order_id,
            child_role=child_role,
        )
        self._open_orders[broker_order_id] = order
        if child_role in {"stop", "take_profit_1", "take_profit_2"}:
            return order
        fill_price = limit_price or self.current_quote(ticker).last or 0.0
        self._fill_order(order, fill_price=fill_price)
        return order

    def place_bracket_order(self, request: BracketOrderRequest) -> list[BrokerOrderSnapshot]:
        parent = self.place_order(
            ticker=request.ticker,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.entry_price,
        )
        stop = BrokerOrderSnapshot(
            broker_order_id=str(uuid.uuid4()),
            ticker=request.ticker.upper(),
            side="SELL" if request.side.upper() == "BUY" else "BUY",
            order_type="STOP",
            quantity=request.quantity,
            status="submitted",
            stop_price=request.stop_price,
            parent_order_id=parent.broker_order_id,
            child_role="stop",
        )
        target_1 = BrokerOrderSnapshot(
            broker_order_id=str(uuid.uuid4()),
            ticker=request.ticker.upper(),
            side="SELL" if request.side.upper() == "BUY" else "BUY",
            order_type="LIMIT",
            quantity=request.quantity if request.target_price_2 is None else max(1, request.quantity // 2),
            status="submitted",
            limit_price=request.target_price_1,
            parent_order_id=parent.broker_order_id,
            child_role="take_profit_1",
        )
        self._open_orders[stop.broker_order_id] = stop
        self._open_orders[target_1.broker_order_id] = target_1
        orders = [parent, stop, target_1]
        if request.target_price_2 is not None:
            target_2 = BrokerOrderSnapshot(
                broker_order_id=str(uuid.uuid4()),
                ticker=request.ticker.upper(),
                side="SELL" if request.side.upper() == "BUY" else "BUY",
                order_type="LIMIT",
                quantity=max(1, request.quantity - target_1.quantity),
                status="submitted",
                limit_price=request.target_price_2,
                parent_order_id=parent.broker_order_id,
                child_role="take_profit_2",
            )
            self._open_orders[target_2.broker_order_id] = target_2
            orders.append(target_2)
        return orders

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot:
        self._require_connection()
        order = self._open_orders.pop(broker_order_id)
        order.status = "cancelled"
        self._completed_orders[broker_order_id] = order
        return order

    def modify_order(self, broker_order_id: str, *, limit_price: float | None = None, stop_price: float | None = None, quantity: int | None = None) -> BrokerOrderSnapshot:
        self._require_connection()
        order = self._open_orders[broker_order_id]
        if limit_price is not None:
            order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        if quantity is not None:
            order.quantity = int(quantity)
        return order

    def close_position(self, ticker: str) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        position = self._positions.get(ticker.upper())
        if position is None or position.quantity == 0:
            return []
        side = "SELL" if position.quantity > 0 else "BUY"
        order = self.place_order(
            ticker=ticker.upper(),
            side=side,
            quantity=abs(position.quantity),
            order_type="LIMIT",
            limit_price=self.current_quote(ticker).last,
        )
        return [order]

    def flatten_all_positions(self) -> list[BrokerOrderSnapshot]:
        out: list[BrokerOrderSnapshot] = []
        for ticker in list(self._positions.keys()):
            out.extend(self.close_position(ticker))
        return out

    def _fill_order(self, order: BrokerOrderSnapshot, *, fill_price: float) -> None:
        execution = BrokerExecutionSnapshot(
            execution_id=str(uuid.uuid4()),
            broker_order_id=order.broker_order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=max(1.0, round(order.quantity * 0.005, 2)),
        )
        self._executions.append(execution)
        order.status = "filled"
        self._completed_orders[order.broker_order_id] = order
        self._open_orders.pop(order.broker_order_id, None)
        signed_qty = order.quantity if order.side == "BUY" else -order.quantity
        existing = self._positions.get(order.ticker)
        if existing is None:
            self._positions[order.ticker] = BrokerPositionSnapshot(
                ticker=order.ticker,
                quantity=signed_qty,
                average_cost=fill_price,
                market_price=fill_price,
                market_value=signed_qty * fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
        else:
            new_qty = existing.quantity + signed_qty
            if new_qty == 0:
                self._positions.pop(order.ticker, None)
            else:
                existing.quantity = new_qty
                existing.average_cost = fill_price if signed_qty > 0 else existing.average_cost
                existing.market_price = fill_price
                existing.market_value = new_qty * fill_price
        cash_change = fill_price * order.quantity
        if order.side == "BUY":
            self._account.cash_balance -= cash_change
        else:
            self._account.cash_balance += cash_change
        self._account.buying_power = self._account.cash_balance * 2.0

    def _require_connection(self) -> None:
        if not self._connected:
            raise BrokerError("Broker is disconnected")


class IBKRBroker(BrokerInterface):
    def __init__(self, *, host: str, port: int, client_id: int, account_id: str | None, read_only: bool = False) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_id = account_id
        self.read_only = read_only
        self._ib = None

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except Exception as exc:
            raise BrokerError("ib_insync is required for IBKR broker support") from exc
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id, readonly=self.read_only)

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()

    def reconnect(self) -> None:
        self.disconnect()
        time.sleep(0.5)
        self.connect()

    def health_check(self) -> dict[str, Any]:
        return {"connected": bool(self._ib and self._ib.isConnected()), "host": self.host, "port": self.port}

    def account_summary(self) -> BrokerAccountSummary:
        self._require_connection()
        values = {item.tag: item.value for item in self._ib.accountSummary()}
        account_id = self.account_id or str(values.get("AccountCode") or "")
        account_type = str(values.get("AccountType") or "unknown").lower()
        return BrokerAccountSummary(
            account_id=account_id,
            account_type=account_type,
            is_paper=account_id.startswith("DU"),
            cash_balance=float(values.get("TotalCashValue") or 0.0),
            buying_power=float(values.get("BuyingPower") or 0.0),
            net_liquidation_value=float(values.get("NetLiquidation") or 0.0),
            raw=values,
        )

    def positions(self) -> list[BrokerPositionSnapshot]:
        self._require_connection()
        out: list[BrokerPositionSnapshot] = []
        for pos in self._ib.positions():
            out.append(BrokerPositionSnapshot(ticker=pos.contract.symbol, quantity=int(pos.position), average_cost=float(pos.avgCost or 0.0)))
        return out

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        out: list[BrokerOrderSnapshot] = []
        for trade in self._ib.openTrades():
            out.append(BrokerOrderSnapshot(
                broker_order_id=str(trade.order.orderId),
                ticker=trade.contract.symbol,
                side=trade.order.action,
                order_type=trade.order.orderType,
                quantity=int(trade.order.totalQuantity),
                status=str(trade.orderStatus.status),
                limit_price=float(trade.order.lmtPrice) if trade.order.lmtPrice else None,
                stop_price=float(trade.order.auxPrice) if trade.order.auxPrice else None,
                raw={"status": trade.orderStatus.status},
            ))
        return out

    def completed_orders(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        return []

    def executions(self) -> list[BrokerExecutionSnapshot]:
        self._require_connection()
        out: list[BrokerExecutionSnapshot] = []
        for fill in self._ib.fills():
            out.append(BrokerExecutionSnapshot(
                execution_id=str(fill.execution.execId),
                broker_order_id=str(fill.execution.orderId),
                ticker=fill.contract.symbol,
                side=fill.execution.side,
                quantity=int(fill.execution.shares),
                price=float(fill.execution.price),
                executed_at=fill.execution.time if isinstance(fill.execution.time, datetime) else datetime.now(timezone.utc),
                raw={"permId": fill.execution.permId},
            ))
        return out

    def current_quote(self, ticker: str) -> BrokerQuote:
        self._require_connection()
        from ib_insync import Stock
        contract = Stock(ticker.upper(), "SMART", "USD")
        self._ib.qualifyContracts(contract)
        ticker_data = self._ib.reqTickers(contract)[0]
        return BrokerQuote(
            ticker=ticker.upper(),
            last=float(ticker_data.marketPrice()) if ticker_data.marketPrice() is not None else None,
            bid=float(ticker_data.bid) if ticker_data.bid is not None else None,
            ask=float(ticker_data.ask) if ticker_data.ask is not None else None,
            source="ibkr",
        )

    def place_order(self, *, ticker: str, side: str, quantity: int, order_type: str, limit_price: float | None = None, stop_price: float | None = None, parent_order_id: str | None = None, child_role: str | None = None) -> BrokerOrderSnapshot:
        raise BrokerError("Direct single-order placement is not exposed; use place_bracket_order")

    def place_bracket_order(self, request: BracketOrderRequest) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        from ib_insync import LimitOrder, StopOrder, Stock
        contract = Stock(request.ticker.upper(), "SMART", "USD")
        self._ib.qualifyContracts(contract)
        parent = LimitOrder(request.side.upper(), request.quantity, request.entry_price, transmit=False)
        target = LimitOrder("SELL" if request.side.upper() == "BUY" else "BUY", request.quantity, request.target_price_1, parentId=parent.orderId, transmit=False)
        stop = StopOrder("SELL" if request.side.upper() == "BUY" else "BUY", request.quantity, request.stop_price, parentId=parent.orderId, transmit=True)
        trades = [
            self._ib.placeOrder(contract, parent),
            self._ib.placeOrder(contract, target),
            self._ib.placeOrder(contract, stop),
        ]
        return [
            BrokerOrderSnapshot(
                broker_order_id=str(trade.order.orderId),
                ticker=request.ticker.upper(),
                side=trade.order.action,
                order_type=trade.order.orderType,
                quantity=int(trade.order.totalQuantity),
                status=str(trade.orderStatus.status),
                limit_price=float(trade.order.lmtPrice) if trade.order.lmtPrice else None,
                stop_price=float(trade.order.auxPrice) if trade.order.auxPrice else None,
                parent_order_id=str(trade.order.parentId) if trade.order.parentId else None,
            )
            for trade in trades
        ]

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot:
        raise BrokerError("IBKR cancel-by-id requires reconciliation lookup and is not yet supported directly")

    def modify_order(self, broker_order_id: str, *, limit_price: float | None = None, stop_price: float | None = None, quantity: int | None = None) -> BrokerOrderSnapshot:
        raise BrokerError("IBKR order modification requires open-order reconciliation and is not yet supported directly")

    def close_position(self, ticker: str) -> list[BrokerOrderSnapshot]:
        raise BrokerError("IBKR close position should flow through orchestrator-managed exit logic")

    def flatten_all_positions(self) -> list[BrokerOrderSnapshot]:
        raise BrokerError("IBKR flatten-all should flow through orchestrator-managed kill-switch logic")

    def _require_connection(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise BrokerError("IBKR broker is disconnected")
