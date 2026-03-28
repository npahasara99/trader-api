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


def build_llm_prompt(payload: dict) -> str:
    """Create a provider-agnostic JSON-only prompt using structured planner output."""
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are reviewing a US swing-trade setup. Use only the provided structured data. "
        "Do not invent price levels or unsupported indicators. Return JSON only with fields: "
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
    entry_quality = float(payload.get("entry_quality_score") or 0.0)
    composite = float(payload.get("composite_score") or 0.0)
    entry_requires_confirmation = bool(payload.get("entry_requires_confirmation"))
    stop_too_tight = bool(payload.get("stop_too_tight_flag"))
    tp_too_optimistic = bool(payload.get("tp_too_optimistic_flag"))
    rr1 = float(reward_risk.get("tp1") or 0.0)
    days_to_earnings = earnings.get("days_to_earnings")
    volume_context = payload.get("volume_context") or {}
    selloff_state = str(volume_context.get("selloff_volume_state") or "unknown")
    reversal_state = str(volume_context.get("reversal_volume_state") or "unknown")

    reasons: list[str] = []
    if trend_state in {"uptrend", "pullback_in_uptrend"}:
        reasons.append("Trend structure remains constructive rather than broken.")
    elif trend_state == "range":
        reasons.append("The chart is range-bound, so follow-through is less reliable.")
    else:
        reasons.append("Structure is weakening and needs confirmation before risking capital.")

    if entry_quality >= 7.0:
        entry_assessment = "Entry sits near meaningful confluence support and looks reasonable."
    elif entry_quality >= 5.0:
        entry_assessment = "Entry is workable but still needs confirmation from price action."
    else:
        entry_assessment = "Entry is either too aggressive or too remote from the likely path of price."

    if stop_too_tight:
        stop_assessment = "Stop is structurally placed but may still sit inside normal noise."
    else:
        stop_assessment = "Stop sits beyond invalidation with volatility buffer rather than inside noise."

    if tp_too_optimistic:
        take_profit_assessment = "Further targets look optimistic for the current hold window."
    elif rr1 >= 1.6:
        take_profit_assessment = "Targets are realistic and backed by structure/reachability."
    else:
        take_profit_assessment = "Targets are modest; upside exists but the reward/risk is only average."

    action = "WAIT"
    setup_type = trend_state
    confidence = 0.45
    key_risk = "Weak follow-through"

    if days_to_earnings is not None and int(days_to_earnings) <= config.earnings_hard_block_days:
        action = "AVOID"
        confidence = 0.82
        key_risk = "Earnings event is too close to trust the setup."
        reasons.append("Earnings proximity is too high for a normal swing entry.")
    elif trend_state in {"uptrend", "pullback_in_uptrend"} and rr1 >= config.min_reward_risk_for_buy and entry_quality >= 6.2 and selloff_state != "heavy_distribution":
        action = "BUY" if not entry_requires_confirmation else "WAIT"
        confidence = 0.62 if action == "WAIT" else 0.74
        key_risk = "Needs confirmation" if action == "WAIT" else "Normal pullback volatility"
    elif trend_state in {"downtrend", "weak_breakdown_risk"} or selloff_state == "heavy_distribution":
        action = "AVOID"
        confidence = 0.76
        key_risk = "Distribution / breakdown risk remains elevated."
    elif composite >= 6.0 and rr1 >= config.min_reward_risk_for_wait:
        action = "WAIT"
        confidence = 0.58
        key_risk = "Setup is close but not clean enough yet."
    else:
        action = "WAIT"
        confidence = 0.5
        key_risk = "Reward/risk or trend quality is not convincing enough yet."

    if reversal_state == "confirmed_bounce":
        reasons.append("Bounce volume is supportive rather than fading.")
    elif reversal_state == "weak_bounce":
        reasons.append("Bounce volume is weak, so reversal credibility is limited.")

    if days_to_earnings is not None:
        reasons.append(f"Earnings are {int(days_to_earnings)} days away, which changes timing risk.")

    confirmation_needed = payload.get("confirmation_trigger") or "Need a stable close above short-term resistance."
    consensus_view = (
        "Constructive pullback with disciplined levels."
        if action == "BUY"
        else "Watchlist setup; wait for confirmation."
        if action == "WAIT"
        else "Avoid forcing the trade while structure is compromised or event risk is high."
    )

    return LLMReviewResult(
        llm_action=action,
        setup_type=setup_type,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        consensus_view=consensus_view,
        entry_assessment=entry_assessment,
        stop_assessment=stop_assessment,
        take_profit_assessment=take_profit_assessment,
        rationale=reasons[:4],
        confirmation_needed=str(confirmation_needed),
        key_risk=key_risk,
        risk_tuning_reason=(
            f"trend={trend_state}; rr1={rr1:.2f}; entry_q={entry_quality:.2f}; "
            f"selloff={selloff_state}; reversal={reversal_state}; composite={composite:.2f}"
        ),
        llm_quality_score=round(max(0.0, min(10.0, composite + (1.2 if action == "BUY" else -0.6 if action == "AVOID" else 0.0))), 3),
    )


def review_setup(*, payload: dict, config: PlanningConfig, provider: str | None = None, model: str | None = None, style: str | None = None) -> dict:
    """Provider-isolated reasoning hook.

    The current code uses a deterministic fallback so API routes stay reliable without a live LLM.
    A future provider implementation can call `build_llm_prompt()` and `parse_llm_response()` here.
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
