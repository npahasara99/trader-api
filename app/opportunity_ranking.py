"""Transparent daily-opportunity scoring built on completed planner rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import DEFAULT_PLANNING_CONFIG, PlanningConfig


GRADE_ORDER = {"F": 0, "D": 1, "C": 2, "B": 3, "B+": 4, "A-": 5, "A": 6, "A+": 7}


@dataclass(frozen=True)
class PortfolioSnapshot:
    max_positions: int
    open_positions: int
    available_position_slots: int
    trading_budget: float
    capital_in_use: float
    available_capital: float
    sector_exposures: dict[str, int]
    correlation_exposures: dict[str, int]
    positions: tuple[dict, ...]


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float) -> float:
    return round(max(0.0, min(10.0, float(value))), 4)


def _component(row: Any, name: str, default: float = 5.0) -> float:
    components = _value(row, "component_scores", None) or {}
    direct = _value(row, name, None)
    value = direct if direct is not None else components.get(name, default)
    return _clip(_safe_float(value, default))


def grade_for_score(score: float) -> str:
    if score >= 9.2:
        return "A+"
    if score >= 8.6:
        return "A"
    if score >= 8.0:
        return "A-"
    if score >= 7.3:
        return "B+"
    if score >= 6.5:
        return "B"
    if score >= 5.2:
        return "C"
    if score >= 3.5:
        return "D"
    return "F"


def grade_at_least(grade: str, minimum: str) -> bool:
    return GRADE_ORDER.get(grade, -1) >= GRADE_ORDER.get(minimum, 99)


def correlation_group_for_metadata(metadata: dict | None) -> str:
    metadata = metadata or {}
    sector = str(metadata.get("sector") or "Unknown").strip()
    industry = str(metadata.get("industry") or "Unknown").strip()
    text = f"{sector} {industry}".lower()
    mappings = (
        (("semiconductor equipment",), "Semiconductor_Equipment"),
        (("semiconductor",), "AI_Semiconductors"),
        (("bank",), "Large_US_Banks"),
        (("oil", "gas", "energy"), "Energy_Producers"),
        (("aerospace", "defense"), "Aerospace_Defense"),
        (("reit", "real estate"), "US_REITs"),
        (("utility",), "US_Utilities"),
        (("pharma", "biotech"), "Biopharma"),
        (("software",), "Large_Cap_Software"),
        (("interactive media", "internet content"), "Internet_Platforms"),
        (("automobile", "auto"), "Automakers"),
    )
    for needles, group in mappings:
        if any(needle in text for needle in needles):
            return group
    return "_".join(part for part in (sector, industry) if part and part != "Unknown").replace(" ", "_")[:80] or "Unknown"


def build_portfolio_snapshot(
    positions: Iterable[dict],
    *,
    metadata_by_ticker: dict[str, dict],
    max_positions: int,
    trading_budget: float,
) -> PortfolioSnapshot:
    normalized: list[dict] = []
    sector_counts: dict[str, int] = {}
    correlation_counts: dict[str, int] = {}
    capital_in_use = 0.0
    for source in positions:
        ticker = str(source.get("ticker") or "").upper()
        metadata = metadata_by_ticker.get(ticker, {})
        sector = str(metadata.get("sector") or source.get("sector") or "Unknown")
        correlation_group = str(
            source.get("correlation_group") or correlation_group_for_metadata(metadata)
        )
        quantity = abs(_safe_float(source.get("quantity")))
        average_entry = _safe_float(source.get("average_entry_price"))
        capital_in_use += quantity * average_entry
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        correlation_counts[correlation_group] = correlation_counts.get(correlation_group, 0) + 1
        normalized.append(
            {
                **source,
                "ticker": ticker,
                "sector": sector,
                "correlation_group": correlation_group,
            }
        )
    open_count = len(normalized)
    return PortfolioSnapshot(
        max_positions=max_positions,
        open_positions=open_count,
        available_position_slots=max(max_positions - open_count, 0),
        trading_budget=round(trading_budget, 2),
        capital_in_use=round(capital_in_use, 2),
        available_capital=round(max(trading_budget - capital_in_use, 0.0), 2),
        sector_exposures=sector_counts,
        correlation_exposures=correlation_counts,
        positions=tuple(normalized),
    )


def build_raw_setup_profile(row: Any, *, config: PlanningConfig = DEFAULT_PLANNING_CONFIG) -> dict:
    """Score objective setup quality without portfolio exposure penalties."""

    mtf = _safe_float(_value(row, "multi_timeframe_alignment_score", None), 5.0)
    catalyst_macro = (
        _component(row, "catalyst_score") + _component(row, "macro_score")
    ) / 2.0
    components = {
        "trend": _component(row, "trend_score"),
        "price_location": _component(row, "price_location_score"),
        "support_confluence": _component(row, "support_confluence_score"),
        "multi_timeframe": _clip(mtf),
        "relative_strength": _component(row, "relative_strength_score"),
        "volatility": _component(row, "volatility_suitability_score"),
        "liquidity": _component(row, "liquidity_score"),
        "target_realism": _component(row, "target_realism_score"),
        "reward_risk": _component(row, "reward_risk_score"),
        "catalyst_macro": _clip(catalyst_macro),
        "confirmation": _component(row, "confirmation_score"),
    }
    weights = config.raw_setup_weights
    weighted = sum(components[key] * weights[key] for key in weights) / max(sum(weights.values()), 1e-9)
    exclusion_reasons: list[str] = []
    if not bool(_value(row, "universe_eligible", True)):
        exclusion_reasons.extend(list(_value(row, "universe_rejection_reasons", []) or []))
        weighted -= 2.0
    if _value(row, "executable_stop_technically_valid", True) is False:
        exclusion_reasons.append("no_technically_valid_executable_stop")
        weighted -= 2.5
    if bool(_value(row, "tp_too_optimistic_flag", False)):
        exclusion_reasons.append("target_unrealistic")
        weighted -= 0.8
    if str(_value(row, "enhanced_trend_state", "")) == "structural_breakdown":
        exclusion_reasons.append("structural_breakdown")
        weighted -= 1.2
    score = _clip(weighted)
    return {
        "raw_setup_score": score,
        "grade": grade_for_score(score),
        "raw_setup_components": components,
        "raw_setup_weights": dict(weights),
        "exclusion_reasons": list(dict.fromkeys(exclusion_reasons)),
    }


def _current_rr(row: Any, current_price: float) -> float | None:
    stop = _safe_float(_value(row, "stop_loss", None), 0.0)
    tp1 = _safe_float(_value(row, "take_profit_1", None), 0.0)
    risk = current_price - stop
    reward = tp1 - current_price
    if current_price <= 0 or risk <= 0:
        return None
    return reward / risk


def _entry_proximity_score(row: Any, current_price: float, atr: float, config: PlanningConfig) -> tuple[float, float | None]:
    low = _safe_float(_value(row, "preferred_entry_low", None), 0.0)
    high = _safe_float(_value(row, "preferred_entry_high", None), 0.0)
    preferred = _safe_float(_value(row, "preferred_entry", None), 0.0)
    if low <= 0 or high <= 0:
        low = high = preferred
    if preferred <= 0 or current_price <= 0:
        return 0.0, None
    if low <= current_price <= high:
        return 9.0, 0.0
    distance = min(abs(current_price - low), abs(current_price - high))
    distance_pct = distance / current_price
    distance_atr = distance / max(atr, current_price * 0.005)
    if distance_atr <= config.actionability_entry_near_atr:
        score = 8.0 - distance_atr * 1.5
    elif distance_atr <= 1.5:
        score = 6.5 - (distance_atr - config.actionability_entry_near_atr) * 2.0
    else:
        score = 4.5 - (distance_atr - 1.5) * 1.4
    return _clip(score), round(distance_pct, 6)


def _alignment_score(value: str) -> float:
    return {
        "risk_on": 8.5,
        "neutral": 6.0,
        "trendless": 5.0,
        "high_volatility": 4.0,
        "risk_off": 2.8,
    }.get(value.lower(), 5.0)


def build_daily_actionability_profile(
    row: Any,
    *,
    market_regime: str,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
) -> dict:
    """Measure whether the saved numeric plan is executable at its current scan price."""

    current_price = _safe_float(_value(row, "current_price", None) or _value(row, "last", None), 0.0)
    atr = _safe_float(_value(row, "atr", None), current_price * 0.02)
    trigger = _safe_float(_value(row, "confirmation_trigger_price", None), 0.0)
    stop = _safe_float(_value(row, "stop_loss", None), 0.0)
    invalidation = _safe_float(_value(row, "invalidation_level", None), stop)
    tp1 = _safe_float(_value(row, "take_profit_1", None), 0.0)
    base_entry_state = str(_value(row, "entry_status", None) or "awaiting_confirmation")
    volume_confirmed = bool(_value(row, "volume_confirmed", False))
    confirmation_state = str(_value(row, "confirmation_state", None) or "")
    price_confirmed = bool(_value(row, "price_confirmed", False)) or bool(trigger and current_price >= trigger)

    entry_proximity, distance_to_entry_pct = _entry_proximity_score(row, current_price, atr, config)
    current_rr = _current_rr(row, current_price)
    if current_rr is None or current_rr <= 0:
        rr_score = 0.0
    elif current_rr >= 2.5:
        rr_score = 10.0
    elif current_rr >= 1.5:
        rr_score = 7.0 + (current_rr - 1.5) * 3.0
    elif current_rr >= 1.0:
        rr_score = 4.5 + (current_rr - 1.0) * 5.0
    else:
        rr_score = current_rr * 4.5

    sector_relative = _safe_float(_value(row, "sector_relative_strength", None), 0.0)
    sector_score = _clip(5.0 + sector_relative * 70.0)
    components = {
        "confirmation": _component(row, "confirmation_score"),
        "entry_proximity": entry_proximity,
        "current_reward_risk": _clip(rr_score),
        "target_reachability": _component(row, "hold_window_reachability_score"),
        "market_alignment": _alignment_score(market_regime),
        "sector_alignment": sector_score,
        "volume_confirmation": _component(row, "volume_confirmation_score"),
        "liquidity": _component(row, "liquidity_score"),
    }
    weights = config.daily_actionability_weights
    weighted = sum(components[key] * weights[key] for key in weights) / max(sum(weights.values()), 1e-9)

    extended_threshold = max(atr * config.actionability_extended_atr, current_price * config.actionability_extended_pct)
    exclusion_reasons: list[str] = []
    if current_price <= max(invalidation, stop):
        state = "invalidated"
        exclusion_reasons.append("invalidated")
        weighted = 0.0
    elif tp1 > 0 and current_price >= tp1:
        state = "missed"
        exclusion_reasons.extend(["target_already_reached", "poor_current_rr"])
        weighted = min(weighted, 2.0)
    elif current_rr is not None and current_rr < config.actionability_missed_current_rr:
        state = "missed"
        exclusion_reasons.append("poor_current_rr")
        weighted = min(weighted, 3.0)
    elif trigger > 0 and current_price > trigger + extended_threshold:
        state = "extended"
        exclusion_reasons.append("too_extended")
        weighted = min(weighted - 1.5, 5.0)
    elif base_entry_state in {"extended", "missed", "invalidated"}:
        state = base_entry_state
        exclusion_reasons.append(base_entry_state)
        weighted = min(weighted, 4.0 if state == "extended" else 2.0)
    elif price_confirmed and (volume_confirmed or confirmation_state == "confirmed"):
        state = "confirmed"
        weighted += 0.5
    elif base_entry_state == "too_early":
        state = "too_early"
    elif base_entry_state in {"in_price_zone", "awaiting_confirmation"} or not price_confirmed:
        state = "awaiting_confirmation"
    else:
        state = "in_price_zone"

    if _value(row, "executable_stop_technically_valid", True) is False:
        exclusion_reasons.append("no_technically_valid_executable_stop")
        weighted -= 3.0
    if current_rr is not None and current_rr < config.actionability_min_current_rr:
        if "poor_current_rr" not in exclusion_reasons:
            exclusion_reasons.append("poor_current_rr")
        weighted -= 1.1

    score = _clip(weighted)
    if state == "confirmed" and score >= config.min_actionability_score:
        state = "actionable"

    waiting_for: list[dict] = []
    if state in {"too_early", "in_price_zone", "awaiting_confirmation"}:
        if trigger > 0 and current_price < trigger:
            waiting_for.append({"type": "price_above", "value": round(trigger, 6)})
        if not volume_confirmed:
            waiting_for.append({"type": "volume_confirmation", "value": None})
        if market_regime in {"risk_off", "high_volatility"}:
            waiting_for.append({"type": "market_stabilization", "value": None})

    return {
        "actionability_score": score,
        "actionability_state": state,
        "confirmation_status": "confirmed" if price_confirmed and volume_confirmed else "pending",
        "current_reward_risk": None if current_rr is None else round(current_rr, 4),
        "distance_to_preferred_entry_pct": distance_to_entry_pct,
        "actionability_components": components,
        "actionability_weights": dict(weights),
        "waiting_for": waiting_for,
        "exclusion_reasons": exclusion_reasons,
    }


def build_portfolio_fit_profile(
    *,
    sector: str,
    correlation_group: str,
    portfolio: PortfolioSnapshot,
    selected: list[dict] | None = None,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
) -> dict:
    selected = selected or []
    selected_sector = sum(1 for item in selected if item.get("sector") == sector)
    selected_correlation = sum(1 for item in selected if item.get("correlation_group") == correlation_group)
    existing_sector = portfolio.sector_exposures.get(sector, 0)
    existing_correlation = portfolio.correlation_exposures.get(correlation_group, 0)
    sector_count = existing_sector + selected_sector
    correlation_count = existing_correlation + selected_correlation
    # New same-day trades receive a stronger soft penalty than existing exposure
    # so similarly rated candidates diversify before a hard limit is reached.
    sector_penalty = existing_sector * 0.65 + selected_sector * 1.2
    correlation_penalty = existing_correlation * 0.95 + selected_correlation * 2.0
    reasons: list[str] = []
    if sector_count >= config.max_open_positions_per_sector:
        sector_penalty += 2.2
        reasons.append("sector_overexposure")
    if correlation_count >= config.max_open_positions_per_correlation_group:
        correlation_penalty += 2.4
        reasons.append("correlation_overexposure")
    position_limit_penalty = 0.0
    if portfolio.available_position_slots <= len(selected):
        position_limit_penalty = 10.0
        reasons.append("position_limit_reached")
    elif portfolio.available_capital <= 0:
        position_limit_penalty = 10.0
        reasons.append("available_capital_exhausted")
    score = _clip(10.0 - sector_penalty - correlation_penalty - position_limit_penalty)
    return {
        "portfolio_fit_score": score,
        "sector_concentration_penalty": round(sector_penalty, 4),
        "correlation_penalty": round(correlation_penalty, 4),
        "position_limit_penalty": round(position_limit_penalty, 4),
        "portfolio_exclusion_reasons": reasons,
    }


def _candidate_from_row(
    row: Any,
    *,
    metadata: dict,
    market_regime: str,
    portfolio: PortfolioSnapshot,
    config: PlanningConfig,
) -> dict:
    raw = build_raw_setup_profile(row, config=config)
    actionability = build_daily_actionability_profile(row, market_regime=market_regime, config=config)
    sector = str(metadata.get("sector") or "Unknown")
    industry = str(metadata.get("industry") or "Unknown")
    correlation_group = correlation_group_for_metadata(metadata)
    portfolio_fit = build_portfolio_fit_profile(
        sector=sector,
        correlation_group=correlation_group,
        portfolio=portfolio,
        config=config,
    )
    market_sector = (
        actionability["actionability_components"]["market_alignment"]
        + actionability["actionability_components"]["sector_alignment"]
    ) / 2.0
    weights = config.trade_today_weights
    trade_today_score = _clip(
        raw["raw_setup_score"] * weights["raw_setup"]
        + actionability["actionability_score"] * weights["actionability"]
        + portfolio_fit["portfolio_fit_score"] * weights["portfolio_fit"]
        + market_sector * weights["market_sector_alignment"]
    )
    candidate = {
        "ticker": str(_value(row, "ticker") or ""),
        "company_name": metadata.get("company_name"),
        "sector": sector,
        "industry": industry,
        "correlation_group": correlation_group,
        "setup_type": _value(row, "enhanced_trend_state", None) or _value(row, "setup_type", None),
        "current_price": _value(row, "current_price", None),
        "preferred_entry": _value(row, "preferred_entry", None),
        "confirmation_trigger": _value(row, "confirmation_trigger_price", None),
        "stop_loss": _value(row, "stop_loss", None),
        "take_profit_1": _value(row, "take_profit_1", None),
        "take_profit_2": _value(row, "take_profit_2", None),
        "risk_reward": _value(row, "reward_risk", None),
        "planner_action": _value(row, "final_action", None),
        "action": str(_value(row, "final_action", None) or "WAIT").upper(),
        **raw,
        **actionability,
        **portfolio_fit,
        "trade_today_score": trade_today_score,
        "trade_today_weights": dict(weights),
        "row": row,
    }
    candidate["exclusion_reasons"] = list(
        dict.fromkeys(
            raw["exclusion_reasons"]
            + actionability["exclusion_reasons"]
            + portfolio_fit["portfolio_exclusion_reasons"]
        )
    )
    return candidate


def rank_daily_opportunities(
    rows: list[Any],
    *,
    metadata_by_ticker: dict[str, dict],
    market_regime: str,
    portfolio: PortfolioSnapshot,
    best_setups_count: int,
    best_trades_max: int,
    next_to_trigger_count: int,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
) -> dict:
    """Build the three leaderboards while preserving raw setup independence."""

    candidates: list[dict] = []
    failures: list[dict] = []
    for row in rows:
        ticker = str(_value(row, "ticker", ""))
        if _value(row, "preferred_entry", None) is None or _value(row, "stop_loss", None) is None:
            failures.append(
                {
                    "ticker": ticker,
                    "reason": _value(row, "scan_rejection_reason", None) or "missing_required_data",
                    "details": _value(row, "strategy_reason", None),
                }
            )
            continue
        candidates.append(
            _candidate_from_row(
                row,
                metadata=metadata_by_ticker.get(ticker, {}),
                market_regime=market_regime,
                portfolio=portfolio,
                config=config,
            )
        )

    candidates.sort(key=lambda item: (item["raw_setup_score"], item["actionability_score"]), reverse=True)
    best_setups = [{**item, "rank": rank} for rank, item in enumerate(candidates[:best_setups_count], start=1)]

    next_candidates = [
        item
        for item in candidates
        if item["raw_setup_score"] >= max(config.min_raw_setup_score - 1.0, 0.0)
        and item["actionability_state"] in {"too_early", "in_price_zone", "awaiting_confirmation", "confirmed"}
        and item["waiting_for"]
        and "invalidated" not in item["exclusion_reasons"]
    ]
    next_candidates.sort(key=lambda item: (item["actionability_score"], item["raw_setup_score"]), reverse=True)
    next_to_trigger = [
        {**item, "rank": rank}
        for rank, item in enumerate(next_candidates[:next_to_trigger_count], start=1)
    ]

    selected: list[dict] = []
    eligible_pool = list(candidates)
    limit = min(best_trades_max, config.max_new_trades_per_day, portfolio.available_position_slots)
    while eligible_pool and len(selected) < limit:
        qualified: list[dict] = []
        for candidate in eligible_pool:
            dynamic_fit = build_portfolio_fit_profile(
                sector=candidate["sector"],
                correlation_group=candidate["correlation_group"],
                portfolio=portfolio,
                selected=selected,
                config=config,
            )
            market_sector = (
                candidate["actionability_components"]["market_alignment"]
                + candidate["actionability_components"]["sector_alignment"]
            ) / 2.0
            weights = config.trade_today_weights
            dynamic_score = _clip(
                candidate["raw_setup_score"] * weights["raw_setup"]
                + candidate["actionability_score"] * weights["actionability"]
                + dynamic_fit["portfolio_fit_score"] * weights["portfolio_fit"]
                + market_sector * weights["market_sector_alignment"]
            )
            qualifies = (
                grade_at_least(candidate["grade"], config.min_actionable_grade)
                and candidate["raw_setup_score"] >= config.min_raw_setup_score
                and candidate["actionability_score"] >= config.min_actionability_score
                and dynamic_fit["portfolio_fit_score"] >= config.min_portfolio_fit_score
                and candidate["actionability_state"] == "actionable"
                and str(candidate["planner_action"] or "").upper() == "BUY"
                and not candidate["exclusion_reasons"]
            )
            if qualifies:
                qualified.append(
                    {**candidate, **dynamic_fit, "trade_today_score": dynamic_score, "action": "BUY"}
                )
        if not qualified:
            break
        chosen = max(qualified, key=lambda item: item["trade_today_score"])
        selected.append(chosen)
        eligible_pool = [item for item in eligible_pool if item["ticker"] != chosen["ticker"]]

    selected = [{**item, "rank": rank} for rank, item in enumerate(selected, start=1)]

    return {
        "all_candidates": candidates,
        "best_setups": best_setups,
        "best_trades_today": selected,
        "next_to_trigger": next_to_trigger,
        "failures": failures,
    }
