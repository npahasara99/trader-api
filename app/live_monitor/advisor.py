"""Structured, event-triggered advisory layer with deterministic hard gates."""

from __future__ import annotations

from typing import Any, Callable

from .enums import AdvisoryDecision, MonitorState


PROMPT_VERSION = "live-advisor-v1"


def build_advisory_packet(
    *,
    baseline: dict,
    evaluation: dict,
    historical_profile: dict,
    similar_cases: list[dict],
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "market_context": {
            "market_regime": baseline.get("market_regime"),
            "sector": baseline.get("sector"),
            "sector_condition": baseline.get("sector_condition"),
            "relative_strength": baseline.get("relative_strength"),
        },
        "stock_context": {
            key: baseline.get(key)
            for key in (
                "ticker", "current_price", "broader_structure", "setup_type", "execution_structure",
                "support_levels", "resistance_levels", "atr", "atr_pct", "rsi", "ema20", "ema50",
                "ema100", "ema200", "volume_context",
            )
        },
        "confirmation_evidence": evaluation,
        "deterministic_trade_plan": evaluation.get("manual_order_plan"),
        "historical_stock_profile": historical_profile,
        "similar_historical_cases": similar_cases,
    }


def _validate_provider_output(output: dict) -> dict:
    decision = str(output.get("decision") or "WAIT").upper()
    if decision not in {item.value for item in AdvisoryDecision}:
        raise ValueError("LLM advisory decision must be APPROVE, WAIT, or REJECT")
    confidence = max(0.0, min(1.0, float(output.get("confidence") or 0.0)))
    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "reason_summary": str(output.get("reason_summary") or "No reason supplied."),
        "positive_factors": list(output.get("positive_factors") or []),
        "risk_factors": list(output.get("risk_factors") or []),
        "historical_context_used": list(output.get("historical_context_used") or []),
        "preferred_stop_candidate": output.get("preferred_stop_candidate"),
        "preferred_target_structure": output.get("preferred_target_structure"),
        "manual_order_comment": str(output.get("manual_order_comment") or "Manual execution only."),
    }


def review_advisory_packet(packet: dict, provider: Callable[[dict], dict] | None = None) -> dict[str, Any]:
    """Review only meaningful events; no market facts or price levels are invented."""
    evidence = packet.get("confirmation_evidence") or {}
    blockers = list(evidence.get("hard_blockers") or [])
    state = str(evidence.get("state") or "")
    if blockers:
        decision = "REJECT" if any(item in blockers for item in ("setup_invalidated", "maximum_chase_exceeded")) else "WAIT"
        return {
            "status": "available",
            "model": "hard-safety-gate",
            "hard_blockers": blockers,
            **_validate_provider_output({
                "decision": decision,
                "confidence": 1.0,
                "reason_summary": f"Hard deterministic gate: {', '.join(blockers)}.",
                "risk_factors": blockers,
                "manual_order_comment": "No order plan is approved while a hard blocker is active.",
            }),
        }
    if provider is not None:
        try:
            return {"status": "available", "model": "configured-provider", "hard_blockers": [], **_validate_provider_output(provider(packet))}
        except Exception as exc:
            return {
                "status": "unavailable",
                "model": "configured-provider",
                "hard_blockers": [],
                "decision": "WAIT",
                "confidence": 0.0,
                "reason_summary": f"LLM review unavailable: {type(exc).__name__}: {exc}",
                "positive_factors": [],
                "risk_factors": ["llm_unavailable"],
                "historical_context_used": [],
                "preferred_stop_candidate": None,
                "preferred_target_structure": None,
                "manual_order_comment": "Use the deterministic plan for manual review only.",
            }

    approved = state in {MonitorState.APPROVED, MonitorState.STRONGLY_CONFIRMED}
    historical = packet.get("historical_stock_profile") or {}
    return {
        "status": "available",
        "model": "deterministic-advisory-fallback",
        "hard_blockers": [],
        **_validate_provider_output({
            "decision": "APPROVE" if approved else "WAIT",
            "confidence": min(0.85, max(0.45, float(evidence.get("live_confirmation_score") or 0.0) / 10.0)),
            "reason_summary": "Deterministic confirmation passed; review the grounded manual order candidates." if approved else "Confirmation is incomplete; continue monitoring.",
            "positive_factors": [name for name, item in (evidence.get("confirmation_components") or {}).items() if item.get("passed")],
            "risk_factors": [name for name, item in (evidence.get("confirmation_components") or {}).items() if not item.get("passed")],
            "historical_context_used": [f"evidence_strength={historical.get('evidence_strength', 'INSUFFICIENT')}"] if historical else [],
            "preferred_stop_candidate": (evidence.get("manual_order_plan") or {}).get("suggested_stop"),
            "preferred_target_structure": "deterministic_candidates_only",
            "manual_order_comment": "Manual execution only; do not chase beyond the deterministic maximum.",
        }),
    }

