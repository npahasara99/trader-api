from __future__ import annotations

import json
from dataclasses import dataclass

from .config import PlanningConfig


ALLOWED_ACTIONS = {"BUY", "WAIT", "AVOID"}


@dataclass
class LLMReviewResult:
    llm_action: str
    setup_type: str
    confidence: float
    consensus_view: str
    entry_assessment: str
    stop_assessment: str
    take_profit_assessment: str
    rationale: list[str]
    confirmation_needed: str
    key_risk: str
    risk_tuning_reason: str
    llm_quality_score: float


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _collect_constructive_traits(payload: dict, config: PlanningConfig) -> list[str]:
    traits: list[str] = []
    trend_state = str(payload.get("trend_state") or "range")
    composite = _to_float(payload.get("composite_score"))
    entry_quality = _to_float(payload.get("entry_quality_score"))
    relative_strength_score = _to_float(payload.get("relative_strength_score"), 5.0)
    support_quality_score = _to_float(payload.get("support_quality_score"), 5.0)
    reward_risk = payload.get("reward_risk") or {}
    rr1 = _to_float(reward_risk.get("tp1"))
    expected_return = payload.get("expected_return")
    volume_context = payload.get("volume_context") or {}
    reversal_state = str(volume_context.get("reversal_volume_state") or "unknown")
    earnings = payload.get("earnings") or {}
    days_to_earnings = earnings.get("days_to_earnings")

    if trend_state == "pullback_in_uptrend":
        traits.append("pullback_in_uptrend")
    elif trend_state == "uptrend":
        traits.append("uptrend")
    elif trend_state == "weak_breakdown_risk" and relative_strength_score >= max(5.4, config.wait_min_relative_strength_score):
        traits.append("weak_breakdown_but_relative_strength_holding")

    if relative_strength_score >= max(5.6, config.wait_min_relative_strength_score):
        traits.append("relative_strength_supportive")
    if support_quality_score >= 5.4:
        traits.append("support_confluence_present")
    if entry_quality >= config.wait_min_entry_quality:
        traits.append("entry_not_terrible")
    if rr1 >= config.min_reward_risk_for_wait:
        traits.append("reward_risk_monitorable")
    if composite >= config.wait_min_composite_score:
        traits.append("composite_score_monitorable")
    if expected_return is not None and _to_float(expected_return) > 0:
        traits.append("positive_expectancy")
    if reversal_state == "weak_bounce":
        traits.append("early_reversal_attempt")
    if bool(payload.get("entry_requires_confirmation")):
        traits.append("confirmation_is_main_blocker")
    if days_to_earnings is None or int(days_to_earnings) > config.earnings_penalty_near_days:
        traits.append("no_near_earnings_blocker")

    return traits


def classify_final_action(*, payload: dict, config: PlanningConfig) -> dict:
    """Three-bucket action classifier.

    Intent:
    - BUY stays strict.
    - WAIT is for constructive-but-unconfirmed setups.
    - AVOID is reserved for materially weak or unattractive setups.
    """
    trend_state = str(payload.get("trend_state") or "range")
    market_regime = str(payload.get("market_regime") or "neutral")
    composite = _to_float(payload.get("composite_score"))
    buy_threshold = int(payload.get("buy_threshold") or 6)
    entry_quality = _to_float(payload.get("entry_quality_score"))
    relative_strength_score = _to_float(payload.get("relative_strength_score"), 5.0)
    support_quality_score = _to_float(payload.get("support_quality_score"), 5.0)
    volume_confirmation_score = _to_float(payload.get("volume_confirmation_score"), 5.0)
    reward_risk = payload.get("reward_risk") or {}
    rr1 = _to_float(reward_risk.get("tp1"))
    rr2 = _to_float(reward_risk.get("tp2"))
    expected_return = payload.get("expected_return")
    expected_return_val = _to_float(expected_return, 0.0) if expected_return is not None else None
    prob_tp = payload.get("prob_tp")
    prob_sl = payload.get("prob_sl")
    prob_tp_val = _to_float(prob_tp, 0.0) if prob_tp is not None else None
    prob_sl_val = _to_float(prob_sl, 0.0) if prob_sl is not None else None
    entry_requires_confirmation = bool(payload.get("entry_requires_confirmation"))
    earnings = payload.get("earnings") or {}
    days_to_earnings = earnings.get("days_to_earnings")
    volume_context = payload.get("volume_context") or {}
    selloff_state = str(volume_context.get("selloff_volume_state") or "unknown")
    reversal_state = str(volume_context.get("reversal_volume_state") or "unknown")

    constructive_traits = _collect_constructive_traits(payload, config)
    buy_blockers: list[str] = []
    avoid_reasons: list[str] = []
    wait_reasons: list[str] = []
    severity = 0.0

    if days_to_earnings is not None and int(days_to_earnings) <= config.earnings_hard_block_days:
        buy_blockers.append("earnings_too_close")
        avoid_reasons.append("earnings_event_risk_too_close")
        severity += 3.0

    if trend_state in {"downtrend"}:
        buy_blockers.append("trend_downtrend")
        avoid_reasons.append("trend_is_downtrend")
        severity += config.avoid_downtrend_penalty
    elif trend_state == "weak_breakdown_risk":
        buy_blockers.append("trend_needs_repair")
        avoid_reasons.append("trend_in_weak_breakdown_risk")
        severity += config.avoid_weak_breakdown_penalty
        if relative_strength_score >= config.wait_min_relative_strength_score:
            severity -= 0.85
            wait_reasons.append("weak_structure_has_positive_offsets")

    if relative_strength_score < 4.8:
        buy_blockers.append("relative_strength_weak")
        avoid_reasons.append("relative_strength_is_weak")
        severity += config.avoid_negative_rs_penalty
    elif relative_strength_score >= config.wait_min_relative_strength_score:
        wait_reasons.append("relative_strength_is_supportive")

    if entry_quality < config.wait_min_entry_quality:
        buy_blockers.append("entry_quality_poor")
        avoid_reasons.append("entry_quality_is_poor")
        severity += config.avoid_poor_entry_penalty
    elif entry_quality < config.buy_min_entry_quality:
        buy_blockers.append("entry_quality_not_ready")
        wait_reasons.append("entry_quality_is_monitorable_but_not_buy_ready")

    if support_quality_score < 4.8:
        avoid_reasons.append("support_quality_is_weak")
        severity += config.avoid_weak_support_penalty
    elif support_quality_score >= config.buy_min_support_quality_score:
        wait_reasons.append("support_zone_is_valid")

    if reversal_state == "no_confirmation":
        buy_blockers.append("no_confirmation")
        avoid_reasons.append("reversal_confirmation_absent")
        severity += config.avoid_no_confirmation_penalty
    elif reversal_state == "weak_bounce":
        buy_blockers.append("weak_reversal_confirmation")
        wait_reasons.append("bounce_is_starting_but_not_confirmed")
        severity += config.avoid_weak_bounce_penalty

    if selloff_state == "heavy_distribution":
        buy_blockers.append("heavy_distribution")
        wait_reasons.append("distribution_needs_to_cool_off")
        severity += 1.0

    if entry_requires_confirmation:
        buy_blockers.append("entry_requires_confirmation")
        wait_reasons.append("confirmation_is_still_required")

    if expected_return_val is not None:
        if expected_return_val <= 0:
            buy_blockers.append("expected_return_non_positive")
            avoid_reasons.append("expected_return_is_not_positive")
            severity += config.avoid_negative_expectancy_penalty
        else:
            wait_reasons.append("expected_return_is_positive")

    if prob_tp_val is not None and prob_sl_val is not None and prob_sl_val >= prob_tp_val:
        buy_blockers.append("prob_sl_not_better_than_prob_tp")
        avoid_reasons.append("probability_profile_favors_downside")
        severity += config.avoid_prob_penalty

    if rr1 < config.min_reward_risk_for_wait:
        buy_blockers.append("reward_risk_poor")
        avoid_reasons.append("reward_risk_to_tp1_is_poor")
        severity += config.avoid_poor_rr_penalty
    elif rr1 < config.min_reward_risk_for_buy or rr2 < config.min_reward_risk_tp2_for_buy:
        buy_blockers.append("reward_risk_not_buy_ready")
        wait_reasons.append("reward_risk_is_monitorable_but_not_strong")

    if composite < (buy_threshold - config.avoid_bad_composite_gap):
        avoid_reasons.append("composite_score_materially_below_buy_threshold")
        severity += 1.4
    elif composite < buy_threshold:
        wait_reasons.append("composite_score_is_below_buy_threshold_but_close_enough_to_monitor")
        severity += 0.35

    if market_regime == "risk_off" and trend_state in {"downtrend", "weak_breakdown_risk"}:
        avoid_reasons.append("risk_off_regime_amplifies_weak_structure")
        severity += config.avoid_risk_off_weak_trend_penalty

    buy_ok = (
        composite >= buy_threshold
        and trend_state in {"uptrend", "pullback_in_uptrend"}
        and not entry_requires_confirmation
        and entry_quality >= config.buy_min_entry_quality
        and relative_strength_score >= config.buy_min_relative_strength_score
        and support_quality_score >= config.buy_min_support_quality_score
        and volume_confirmation_score >= config.buy_min_volume_confirmation_score
        and rr1 >= config.min_reward_risk_for_buy
        and rr2 >= config.min_reward_risk_tp2_for_buy
        and (expected_return_val is None or expected_return_val > 0)
        and (prob_tp_val is None or prob_sl_val is None or prob_tp_val > prob_sl_val)
        and not (days_to_earnings is not None and int(days_to_earnings) <= config.earnings_hard_block_days)
    )

    monitorable_setup = bool(len(constructive_traits) >= 2 and composite >= config.wait_min_composite_score and entry_quality >= config.wait_min_entry_quality)
    severity_threshold = config.avoid_severity_threshold_risk_off if market_regime == "risk_off" else config.avoid_severity_threshold
    avoid_ok = bool(severity >= severity_threshold and not monitorable_setup)

    if buy_ok:
        final_action = "BUY"
        action_reason_bucket = "confirmed_setup"
    elif avoid_ok:
        final_action = "AVOID"
        action_reason_bucket = "structurally_weak"
    else:
        final_action = "WAIT"
        action_reason_bucket = "constructive_but_unconfirmed"

    if final_action == "WAIT" and not wait_reasons:
        wait_reasons.append("setup_is_not_buy_ready_but_lacks_enough_damage_to_discard")
    if final_action == "AVOID" and not avoid_reasons:
        avoid_reasons.append("multiple_weak_factors_reduce_priority")

    return {
        "final_action": final_action,
        "action_reason_bucket": action_reason_bucket,
        "monitorable_setup": bool(final_action == "WAIT" or monitorable_setup),
        "avoid_severity_score": round(max(0.0, severity), 3),
        "wait_reason": "; ".join(wait_reasons[:4]) if wait_reasons else None,
        "avoid_reason": "; ".join(avoid_reasons[:4]) if avoid_reasons else None,
        "buy_blockers": buy_blockers[:8],
        "constructive_traits": constructive_traits[:8],
    }


def build_llm_prompt(payload: dict) -> str:
    """Create a provider-agnostic JSON-only prompt using structured planner output."""
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are reviewing a US swing-trade setup. Use only the provided structured data. "
        "Do not invent price levels or unsupported indicators. "
        "Classify the setup as BUY, WAIT, or AVOID. "
        "Use WAIT when the setup has constructive traits but confirmation, timing, or payoff quality is not yet sufficient. "
        "Use AVOID only when the setup is materially weak, relative strength is poor, confirmation is absent or low quality, "
        "and the setup is not worth prioritizing. "
        "Do not classify a setup as AVOID merely because it is not ready yet. "
        "Differentiate between 'not ready' and 'not attractive'. "
        "Treat pullback_in_uptrend with strong relative strength but weak reversal confirmation as WAIT unless other severe negatives exist. "
        "Treat weak_breakdown_risk with positive offsets as potentially WAIT, not automatically AVOID. "
        "Treat no_confirmation + negative expectancy + weak relative strength as AVOID. "
        "Return JSON only with fields: "
        "llm_action, setup_type, confidence, consensus_view, entry_assessment, stop_assessment, "
        "take_profit_assessment, rationale, confirmation_needed, key_risk, risk_tuning_reason. "
        f"Input={compact}"
    )


def parse_llm_response(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    action = str(data.get("llm_action") or "").upper().strip()
    if action not in ALLOWED_ACTIONS:
        return None
    return data


def deterministic_review(*, payload: dict, config: PlanningConfig) -> LLMReviewResult:
    trend_state = str(payload.get("trend_state") or "range")
    reward_risk = payload.get("reward_risk") or {}
    earnings = payload.get("earnings") or {}
    entry_quality = _to_float(payload.get("entry_quality_score"))
    composite = _to_float(payload.get("composite_score"))
    stop_too_tight = bool(payload.get("stop_too_tight_flag"))
    tp_too_optimistic = bool(payload.get("tp_too_optimistic_flag"))
    rr1 = _to_float(reward_risk.get("tp1"))
    days_to_earnings = earnings.get("days_to_earnings")
    volume_context = payload.get("volume_context") or {}
    selloff_state = str(volume_context.get("selloff_volume_state") or "unknown")
    reversal_state = str(volume_context.get("reversal_volume_state") or "unknown")

    classification = classify_final_action(payload=payload, config=config)
    action = str(classification["final_action"])
    constructive_traits = list(classification.get("constructive_traits") or [])
    buy_blockers = list(classification.get("buy_blockers") or [])
    avoid_reason = classification.get("avoid_reason")
    wait_reason = classification.get("wait_reason")

    reasons: list[str] = []
    if action == "WAIT":
        reasons.append("The setup has enough constructive traits to stay on the watchlist, but it is not ready to execute.")
    elif action == "AVOID":
        reasons.append("The setup has stacked weaknesses and does not justify priority in the current market context.")
    else:
        reasons.append("The setup is confirmed enough to act on without forcing timing.")

    if trend_state in {"uptrend", "pullback_in_uptrend"}:
        reasons.append("Trend structure is still constructive rather than fully broken.")
    elif trend_state == "weak_breakdown_risk":
        reasons.append("Structure is damaged enough that confirmation matters before taking risk.")
    else:
        reasons.append("Trend structure is weak and reduces upside reliability.")

    if entry_quality >= 7.0:
        entry_assessment = "Entry sits near meaningful confluence support and is actionable if confirmation is present."
    elif entry_quality >= config.wait_min_entry_quality:
        entry_assessment = "Entry is monitorable, but it still needs better timing or confirmation."
    else:
        entry_assessment = "Entry quality is poor enough that the setup is hard to prioritize."

    if stop_too_tight:
        stop_assessment = "Stop is structurally derived but may still be vulnerable to normal noise."
    else:
        stop_assessment = "Stop sits beyond invalidation with a volatility buffer."

    if tp_too_optimistic:
        take_profit_assessment = "Targets stretch the likely hold-window move and need stronger trend support."
    elif rr1 >= config.min_reward_risk_for_buy:
        take_profit_assessment = "Targets look realistic for a swing setup and are supported by structure."
    else:
        take_profit_assessment = "Targets are usable, but the payoff profile is not strong enough for an immediate buy."

    if reversal_state == "confirmed_bounce":
        reasons.append("Reversal volume is supportive rather than fading.")
    elif reversal_state == "weak_bounce":
        reasons.append("Reversal volume is early and not yet convincing.")
    elif reversal_state == "no_confirmation":
        reasons.append("There is still no credible reversal confirmation.")

    if selloff_state == "heavy_distribution":
        reasons.append("Recent selling volume still looks distributive.")

    if days_to_earnings is not None:
        reasons.append(f"Earnings are {int(days_to_earnings)} days away, which still matters for timing risk.")

    confidence = 0.52 if action == "WAIT" else 0.79 if action == "AVOID" else 0.76
    key_risk = (
        str(avoid_reason)
        if action == "AVOID" and avoid_reason
        else str(wait_reason)
        if action == "WAIT" and wait_reason
        else "Confirmation and payoff profile are aligned."
    )
    consensus_view = (
        "Confirmed setup with enough alignment to act now."
        if action == "BUY"
        else "Constructive watchlist setup that still needs confirmation or cleaner timing."
        if action == "WAIT"
        else "Structurally weak setup that is not worth prioritizing right now."
    )
    rationale = reasons[:3] + ([f"Constructive traits: {', '.join(constructive_traits[:3])}."] if constructive_traits else [])
    if action == "AVOID" and buy_blockers:
        rationale.append(f"Main blockers: {', '.join(buy_blockers[:3])}.")

    return LLMReviewResult(
        llm_action=action,
        setup_type=trend_state,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        consensus_view=consensus_view,
        entry_assessment=entry_assessment,
        stop_assessment=stop_assessment,
        take_profit_assessment=take_profit_assessment,
        rationale=rationale[:4],
        confirmation_needed=str(payload.get("confirmation_trigger") or "Need a stable close above short-term resistance."),
        key_risk=key_risk,
        risk_tuning_reason=(
            f"trend={trend_state}; rr1={rr1:.2f}; entry_q={entry_quality:.2f}; "
            f"selloff={selloff_state}; reversal={reversal_state}; composite={composite:.2f}; "
            f"severity={classification['avoid_severity_score']:.2f}"
        ),
        llm_quality_score=round(max(0.0, min(10.0, composite + (1.1 if action == "BUY" else 0.25 if action == "WAIT" else -0.8))), 3),
    )


def review_setup(*, payload: dict, config: PlanningConfig, provider: str | None = None, model: str | None = None, style: str | None = None) -> dict:
    """Provider-isolated reasoning hook.

    The current code uses a deterministic fallback so API routes stay reliable without
    a live LLM provider. A future provider implementation can call
    `build_llm_prompt()` and `parse_llm_response()` here.
    """
    review = deterministic_review(payload=payload, config=config)
    return {
        "llm_action": review.llm_action,
        "setup_type": review.setup_type,
        "confidence": review.confidence,
        "consensus_view": review.consensus_view,
        "entry_assessment": review.entry_assessment,
        "stop_assessment": review.stop_assessment,
        "take_profit_assessment": review.take_profit_assessment,
        "rationale": review.rationale,
        "confirmation_needed": review.confirmation_needed,
        "key_risk": review.key_risk,
        "risk_tuning_reason": review.risk_tuning_reason,
        "llm_quality_score": review.llm_quality_score,
        "provider": provider,
        "model": model,
        "style": style,
        "prompt_preview": build_llm_prompt(payload),
    }
