from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from queue import Empty, Queue
from threading import Event, Thread, current_thread
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
        self._commands: Queue = Queue()
        self._worker_thread: Thread | None = None
        self._worker_ready = Event()
        self._worker_error: Exception | None = None
        self._shutdown = Event()

    def connect(self) -> None:
        self._call(self._connect_internal)

    def disconnect(self) -> None:
        if self._worker_thread is None:
            return
        self._call(self._disconnect_internal)

    def reconnect(self) -> None:
        self._call(self._reconnect_internal)

    def health_check(self) -> dict[str, Any]:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return {
                "connected": False,
                "host": self.host,
                "port": self.port,
                "client_id": self.client_id,
                "read_only": self.read_only,
            }
        return self._call(self._health_check_internal)

    def account_summary(self) -> BrokerAccountSummary:
        return self._call(self._account_summary_internal)

    def positions(self) -> list[BrokerPositionSnapshot]:
        return self._call(self._positions_internal)

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        return self._call(self._open_orders_internal)

    def completed_orders(self) -> list[BrokerOrderSnapshot]:
        return self._call(self._completed_orders_internal)

    def executions(self) -> list[BrokerExecutionSnapshot]:
        return self._call(self._executions_internal)

    def current_quote(self, ticker: str) -> BrokerQuote:
        return self._call(self._current_quote_internal, ticker)

    def place_order(self, *, ticker: str, side: str, quantity: int, order_type: str, limit_price: float | None = None, stop_price: float | None = None, parent_order_id: str | None = None, child_role: str | None = None) -> BrokerOrderSnapshot:
        raise BrokerError("Direct single-order placement is not exposed; use place_bracket_order")

    def place_bracket_order(self, request: BracketOrderRequest) -> list[BrokerOrderSnapshot]:
        return self._call(self._place_bracket_order_internal, request)

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot:
        return self._call(self._cancel_order_internal, broker_order_id)

    def modify_order(self, broker_order_id: str, *, limit_price: float | None = None, stop_price: float | None = None, quantity: int | None = None) -> BrokerOrderSnapshot:
        return self._call(
            self._modify_order_internal,
            broker_order_id,
            limit_price=limit_price,
            stop_price=stop_price,
            quantity=quantity,
        )

    def close_position(self, ticker: str) -> list[BrokerOrderSnapshot]:
        return self._call(self._close_position_internal, ticker)

    def flatten_all_positions(self) -> list[BrokerOrderSnapshot]:
        return self._call(self._flatten_all_positions_internal)

    def _connect_internal(self) -> None:
        if self._ib is None:
            raise BrokerError("IBKR worker is not initialized")
        if self._ib.isConnected():
            return
        connected = self._ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            readonly=self.read_only,
            timeout=10,
        )
        if not connected or not self._ib.isConnected():
            raise BrokerError(f"Unable to connect to TWS at {self.host}:{self.port}")

    def _disconnect_internal(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def _reconnect_internal(self) -> None:
        self._disconnect_internal()
        time.sleep(0.5)
        self._connect_internal()

    def _health_check_internal(self) -> dict[str, Any]:
        return {
            "connected": bool(self._ib and self._ib.isConnected()),
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "read_only": self.read_only,
        }

    def _account_summary_internal(self) -> BrokerAccountSummary:
        self._require_connection()
        account_id = self._resolve_account_id()
        rows = self._ib.accountSummary(account_id)
        values = {item.tag: item.value for item in rows if not account_id or item.account == account_id}
        default_account_type = "paper" if account_id.upper().startswith("DU") else "unknown"
        account_type = str(values.get("AccountType") or default_account_type).lower()
        return BrokerAccountSummary(
            account_id=account_id,
            account_type=account_type,
            is_paper=account_id.upper().startswith("DU"),
            cash_balance=self._number(values.get("TotalCashValue")) or 0.0,
            buying_power=self._number(values.get("BuyingPower")) or 0.0,
            net_liquidation_value=self._number(values.get("NetLiquidation")) or 0.0,
            raw=values,
        )

    def _positions_internal(self) -> list[BrokerPositionSnapshot]:
        self._require_connection()
        account_id = self._resolve_account_id()
        out: list[BrokerPositionSnapshot] = []
        for pos in self._ib.positions(account_id):
            out.append(
                BrokerPositionSnapshot(
                    ticker=pos.contract.symbol,
                    quantity=int(pos.position),
                    average_cost=float(pos.avgCost or 0.0),
                )
            )
        return out

    def _open_orders_internal(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        return [self._trade_snapshot(trade) for trade in self._ib.openTrades()]

    def _completed_orders_internal(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        try:
            trades = self._ib.reqCompletedOrders(apiOnly=False)
        except Exception:
            return []
        return [self._trade_snapshot(trade) for trade in trades]

    def _executions_internal(self) -> list[BrokerExecutionSnapshot]:
        self._require_connection()
        out: list[BrokerExecutionSnapshot] = []
        for fill in self._ib.fills():
            out.append(
                BrokerExecutionSnapshot(
                    execution_id=str(fill.execution.execId),
                    broker_order_id=str(fill.execution.orderId),
                    ticker=fill.contract.symbol,
                    side=fill.execution.side,
                    quantity=int(fill.execution.shares),
                    price=float(fill.execution.price),
                    executed_at=fill.execution.time if isinstance(fill.execution.time, datetime) else datetime.now(timezone.utc),
                    commission=self._number(getattr(fill.commissionReport, "commission", None)),
                    raw={"permId": fill.execution.permId},
                )
            )
        return out

    def _current_quote_internal(self, ticker: str) -> BrokerQuote:
        self._require_connection()
        from ib_insync import Stock

        symbol = ticker.upper()
        contract = Stock(symbol, "SMART", "USD")
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            raise BrokerError(f"IBKR could not qualify {symbol}")

        # Live, frozen, delayed, then delayed-frozen. This keeps closed-market
        # checks useful without falsely labelling delayed data as live.
        market_data_types = ((1, "ibkr_live"), (2, "ibkr_frozen"), (3, "ibkr_delayed"), (4, "ibkr_delayed_frozen"))
        for market_data_type, source in market_data_types:
            self._ib.reqMarketDataType(market_data_type)
            ticker_data = self._ib.reqTickers(contract)[0]
            last = self._price(ticker_data.marketPrice())
            used_close = False
            if last is None:
                last = self._price(getattr(ticker_data, "close", None))
                used_close = last is not None
            bid = self._price(getattr(ticker_data, "bid", None))
            ask = self._price(getattr(ticker_data, "ask", None))
            if last is not None:
                return BrokerQuote(
                    ticker=symbol,
                    last=last,
                    bid=bid,
                    ask=ask,
                    timestamp=datetime.now(timezone.utc),
                    source=f"{source}_close" if used_close else source,
                )
        raise BrokerError(f"IBKR quote unavailable for {symbol}")

    def _place_bracket_order_internal(self, request: BracketOrderRequest) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        if self.read_only:
            raise BrokerError("IBKR connection is read-only")
        if request.quantity < 1 or request.entry_price <= 0 or request.stop_price <= 0 or request.target_price_1 <= 0:
            raise BrokerError("Bracket order contains invalid quantity or price levels")
        self._require_connection()
        from ib_insync import LimitOrder, StopOrder, Stock

        contract = Stock(request.ticker.upper(), "SMART", "USD")
        if not self._ib.qualifyContracts(contract):
            raise BrokerError(f"IBKR could not qualify {request.ticker.upper()}")
        account_id = self._resolve_account_id()
        parent_id = self._ib.client.getReqId()
        target_id = self._ib.client.getReqId()
        stop_id = self._ib.client.getReqId()
        exit_side = "SELL" if request.side.upper() == "BUY" else "BUY"
        common = {"account": account_id, "outsideRth": request.allow_extended_hours}
        parent = LimitOrder(
            request.side.upper(),
            request.quantity,
            request.entry_price,
            orderId=parent_id,
            transmit=False,
            **common,
        )
        target = LimitOrder(
            exit_side,
            request.quantity,
            request.target_price_1,
            orderId=target_id,
            parentId=parent_id,
            transmit=False,
            **common,
        )
        stop = StopOrder(
            exit_side,
            request.quantity,
            request.stop_price,
            orderId=stop_id,
            parentId=parent_id,
            transmit=True,
            **common,
        )
        trades = [self._ib.placeOrder(contract, order) for order in (parent, target, stop)]
        self._ib.sleep(0.25)
        roles = ("entry", "take_profit_1", "stop")
        return [self._trade_snapshot(trade, child_role=role) for trade, role in zip(trades, roles)]

    def _cancel_order_internal(self, broker_order_id: str) -> BrokerOrderSnapshot:
        self._require_connection()
        trade = self._find_open_trade(broker_order_id)
        self._ib.cancelOrder(trade.order)
        self._ib.sleep(0.2)
        return self._trade_snapshot(trade)

    def _modify_order_internal(self, broker_order_id: str, *, limit_price: float | None = None, stop_price: float | None = None, quantity: int | None = None) -> BrokerOrderSnapshot:
        self._require_connection()
        if self.read_only:
            raise BrokerError("IBKR connection is read-only")
        trade = self._find_open_trade(broker_order_id)
        if limit_price is not None:
            trade.order.lmtPrice = float(limit_price)
        if stop_price is not None:
            trade.order.auxPrice = float(stop_price)
        if quantity is not None:
            if quantity < 1:
                raise BrokerError("Order quantity must be at least one share")
            trade.order.totalQuantity = int(quantity)
        updated = self._ib.placeOrder(trade.contract, trade.order)
        self._ib.sleep(0.2)
        return self._trade_snapshot(updated)

    def _close_position_internal(self, ticker: str) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        if self.read_only:
            raise BrokerError("IBKR connection is read-only")
        from ib_insync import MarketOrder

        symbol = ticker.upper()
        account_id = self._resolve_account_id()
        position = next((item for item in self._ib.positions(account_id) if item.contract.symbol.upper() == symbol), None)
        if position is None or int(position.position) == 0:
            return []
        for trade in self._ib.openTrades():
            if trade.contract.symbol.upper() == symbol:
                self._ib.cancelOrder(trade.order)
        side = "SELL" if position.position > 0 else "BUY"
        order = MarketOrder(side, abs(int(position.position)), account=account_id, outsideRth=False)
        trade = self._ib.placeOrder(position.contract, order)
        self._ib.sleep(0.25)
        return [self._trade_snapshot(trade, child_role="manual_exit")]

    def _flatten_all_positions_internal(self) -> list[BrokerOrderSnapshot]:
        self._require_connection()
        snapshots: list[BrokerOrderSnapshot] = []
        account_id = self._resolve_account_id()
        symbols = [item.contract.symbol for item in self._ib.positions(account_id) if int(item.position) != 0]
        for symbol in symbols:
            snapshots.extend(self._close_position_internal(symbol))
        return snapshots

    def _find_open_trade(self, broker_order_id: str):
        order_id = str(broker_order_id)
        trade = next((item for item in self._ib.openTrades() if str(item.order.orderId) == order_id), None)
        if trade is None:
            raise BrokerError(f"Open IBKR order {broker_order_id} was not found")
        return trade

    def _resolve_account_id(self) -> str:
        self._require_connection()
        managed = [str(item) for item in self._ib.managedAccounts()]
        if self.account_id:
            if managed and self.account_id not in managed:
                raise BrokerError("Configured IBKR account is not available in the connected TWS session")
            return self.account_id
        if not managed:
            raise BrokerError("No managed IBKR accounts were returned by TWS")
        return managed[0]

    def _trade_snapshot(self, trade, *, child_role: str | None = None) -> BrokerOrderSnapshot:
        parent_id = int(getattr(trade.order, "parentId", 0) or 0)
        return BrokerOrderSnapshot(
            broker_order_id=str(trade.order.orderId),
            ticker=trade.contract.symbol,
            side=str(trade.order.action),
            order_type=str(trade.order.orderType),
            quantity=int(trade.order.totalQuantity),
            status=str(trade.orderStatus.status),
            limit_price=self._number(getattr(trade.order, "lmtPrice", None)),
            stop_price=self._number(getattr(trade.order, "auxPrice", None)),
            parent_order_id=str(parent_id) if parent_id else None,
            child_role=child_role,
            raw={
                "status": trade.orderStatus.status,
                "perm_id": getattr(trade.order, "permId", None),
                "client_id": getattr(trade.order, "clientId", None),
            },
        )

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and abs(number) < 1e100 else None

    @classmethod
    def _price(cls, value: Any) -> float | None:
        number = cls._number(value)
        return number if number is not None and number > 0 else None

    def _ensure_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_ready.clear()
        self._worker_error = None
        self._shutdown.clear()
        self._worker_thread = Thread(target=self._worker_loop, name=f"ibkr-{self.client_id}", daemon=True)
        self._worker_thread.start()
        if not self._worker_ready.wait(timeout=10):
            raise BrokerError("Timed out while initializing the IBKR event-loop worker")
        if self._worker_error is not None:
            raise BrokerError(str(self._worker_error)) from self._worker_error

    def _worker_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from ib_insync import IB

            self._ib = IB()
        except Exception as exc:
            self._worker_error = BrokerError("ib_insync is required for IBKR broker support")
            self._worker_error.__cause__ = exc
            self._worker_ready.set()
            loop.close()
            return
        self._worker_ready.set()
        try:
            while not self._shutdown.is_set():
                try:
                    function, args, kwargs, future = self._commands.get(timeout=0.05)
                except Empty:
                    if self._ib.isConnected():
                        self._ib.sleep(0.01)
                    continue
                try:
                    future.set_result(function(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
        finally:
            if self._ib is not None and self._ib.isConnected():
                self._ib.disconnect()
            loop.close()

    def _call(self, function, *args, **kwargs):
        self._ensure_worker()
        if current_thread() is self._worker_thread:
            return function(*args, **kwargs)
        future: Future = Future()
        self._commands.put((function, args, kwargs, future))
        try:
            return future.result(timeout=30)
        except FutureTimeoutError as exc:
            raise BrokerError("Timed out waiting for the IBKR event-loop worker") from exc

    def _require_connection(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise BrokerError("IBKR broker is disconnected")
