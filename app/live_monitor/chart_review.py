"""Structured multimodal chart review with deterministic validation fallback."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Callable

from .chart_levels import LEVEL_NAMES, derive_chart_level_candidates, number, validate_chart_levels
from .config import LiveMonitorConfig


CHART_STRUCTURE_PROMPT_VERSION = "chart-structure-review-v1"
CONFIRMED_TRADE_PROMPT_VERSION = "confirmed-trade-review-v1"


CHART_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chart_assessment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "broader_structure": {"type": "string"},
                "setup_type": {"type": "string"},
                "execution_structure": {"type": "string"},
                "setup_quality": {"type": "string"},
                "setup_stale": {"type": "boolean"},
            },
            "required": ["broader_structure", "setup_type", "execution_structure", "setup_quality", "setup_stale"],
        },
        "levels": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "support_zone": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "low": {"type": ["number", "null"]},
                        "high": {"type": ["number", "null"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["low", "high", "reason"],
                },
                **{
                    name: {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"price": {"type": ["number", "null"]}, "reason": {"type": "string"}},
                        "required": ["price", "reason"],
                    }
                    for name in ("near_confirmation", "primary_entry_trigger", "strong_confirmation", "major_trend_repair", "structural_invalidation")
                },
            },
            "required": ["support_zone", "near_confirmation", "primary_entry_trigger", "strong_confirmation", "major_trend_repair", "structural_invalidation"],
        },
        "targets": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"price": {"type": ["number", "null"]}, "reason": {"type": "string"}},
                    "required": ["price", "reason"],
                }
                for name in ("tp1", "tp2", "tp3")
            },
            "required": ["tp1", "tp2", "tp3"],
        },
        "planner_comparison": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "agrees_with_primary_trigger": {"type": "boolean"},
                "planner_trigger_issue": {"type": "string"},
                "recommended_action": {"type": "string"},
            },
            "required": ["agrees_with_primary_trigger", "planner_trigger_issue", "recommended_action"],
        },
        "decision": {"type": "string", "enum": ["APPROVE_LEVELS", "MODIFY_LEVELS", "KEEP_PLANNER", "MANUAL_REVIEW"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "positive_factors": {"type": "array", "items": {"type": "string"}},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "reason_summary": {"type": "string"},
    },
    "required": ["chart_assessment", "levels", "targets", "planner_comparison", "decision", "confidence", "positive_factors", "risk_factors", "reason_summary"],
}


CONFIRMED_TRADE_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "WAIT", "REJECT"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "chart_structure": {
            "type": "object",
            "additionalProperties": False,
            "properties": {name: {"type": "string"} for name in ("broader", "current_setup", "execution")},
            "required": ["broader", "current_setup", "execution"],
        },
        "confirmation_assessment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {name: {"type": "string"} for name in ("price", "volume", "retest", "breakout_quality")},
            "required": ["price", "volume", "retest", "breakout_quality"],
        },
        "trade_geometry": {
            "type": "object",
            "additionalProperties": False,
            "properties": {name: {"type": "string"} for name in ("entry_quality", "stop_quality", "target_quality", "rr_quality")},
            "required": ["entry_quality", "stop_quality", "target_quality", "rr_quality"],
        },
        "historical_assessment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"evidence_strength": {"type": "string"}, "summary": {"type": "string"}},
            "required": ["evidence_strength", "summary"],
        },
        "positive_factors": {"type": "array", "items": {"type": "string"}},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "reason_summary": {"type": "string"},
    },
    "required": ["decision", "confidence", "chart_structure", "confirmation_assessment", "trade_geometry", "historical_assessment", "positive_factors", "risk_factors", "reason_summary"],
}


def _flatten_levels(review: dict) -> dict[str, float | None]:
    level_payload = review.get("levels") or {}
    targets = review.get("targets") or {}
    output: dict[str, float | None] = {}
    for name in LEVEL_NAMES:
        if name == "invalidation_level":
            raw = level_payload.get("structural_invalidation") or level_payload.get(name)
        elif name == "optional_support_level":
            support_zone = level_payload.get("support_zone") or {}
            raw = support_zone.get("high") or support_zone.get("low") or level_payload.get(name)
        else:
            raw = level_payload.get(name) if name in level_payload else targets.get(name)
        output[name] = number(raw)
    return output


def _image_content(paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
    return content


def openai_chart_provider(packet: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Call the current OpenAI Responses API without exposing application secrets."""
    from openai import OpenAI

    review_type = packet.get("review_type") or "CHART_STRUCTURE_REVIEW"
    system_prompt = (
        "You review short-duration swing-trade charts. Exact numeric OHLCV values in the supplied JSON are authoritative. "
        "Use images for structural interpretation only. Distinguish local confirmation from major trend repair. "
        "Do not invent unsupported levels and do not force a trade."
        if review_type == "CHART_STRUCTURE_REVIEW"
        else
        "The deterministic monitor found possible price and volume confirmation. Review chart structure, confirmation quality, "
        "trade geometry, and supplied history. Return conservative structured evidence; hard deterministic failures cannot be overridden."
    )
    safe_packet = {key: value for key, value in packet.items() if key != "image_paths"}
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": "Authoritative structured market context:\n" + json.dumps(safe_packet, default=str, separators=(",", ":"))}
    ]
    content.extend(_image_content(packet.get("image_paths") or []))
    schema = CONFIRMED_TRADE_REVIEW_SCHEMA if review_type == "CONFIRMED_TRADE_REVIEW" else CHART_REVIEW_SCHEMA
    schema_name = "confirmed_trade_review" if review_type == "CONFIRMED_TRADE_REVIEW" else "chart_structure_review"
    response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    parsed = json.loads(response.output_text)
    parsed["_response_model"] = getattr(response, "model", None) or model
    return parsed


def review_chart_packet(
    packet: dict[str, Any],
    *,
    provider: Callable[[dict[str, Any]], dict[str, Any]] | None,
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    review_type = str(packet.get("review_type") or "CHART_STRUCTURE_REVIEW").upper()
    if review_type == "CONFIRMED_TRADE_REVIEW":
        if provider is None and not os.getenv("OPENAI_API_KEY"):
            return {
                "status": "UNAVAILABLE",
                "model": None,
                "prompt_version": CONFIRMED_TRADE_PROMPT_VERSION,
                "output": {},
                "proposed_levels": {},
                "candidate_levels": {},
                "validated_levels": {},
                "validation": {"status": "SKIPPED", "reason": "multimodal_provider_unavailable"},
                "decision": "WAIT",
                "confidence": 0.0,
                "reason_summary": "Multimodal confirmation review is unavailable; the deterministic result remains available for manual review.",
            }
        try:
            call = provider or (lambda value: openai_chart_provider(value, model=config.chart_review_model))
            output = call(packet)
            response_model = output.pop("_response_model", None) or config.chart_review_model
            decision = str(output.get("decision") or "WAIT").upper()
            if decision not in {"APPROVE", "WAIT", "REJECT"}:
                raise ValueError("Confirmed trade review must return APPROVE, WAIT, or REJECT")
            hard_blockers = list((packet.get("latest_evaluation") or {}).get("hard_blockers") or [])
            if hard_blockers:
                decision = "REJECT" if any(item in hard_blockers for item in ("setup_invalidated", "maximum_chase_exceeded")) else "WAIT"
            return {
                "status": "AVAILABLE",
                "model": response_model,
                "prompt_version": CONFIRMED_TRADE_PROMPT_VERSION,
                "output": output,
                "proposed_levels": {},
                "candidate_levels": {},
                "validated_levels": {},
                "validation": {"status": "SKIPPED", "hard_blockers": hard_blockers},
                "decision": decision,
                "confidence": max(0.0, min(1.0, float(output.get("confidence") or 0.0))),
                "reason_summary": str(output.get("reason_summary") or ""),
            }
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "model": config.chart_review_model,
                "prompt_version": CONFIRMED_TRADE_PROMPT_VERSION,
                "output": {},
                "proposed_levels": {},
                "candidate_levels": {},
                "validated_levels": {},
                "validation": {"status": "SKIPPED", "reason": "provider_failure"},
                "decision": "WAIT",
                "confidence": 0.0,
                "reason_summary": f"Multimodal confirmation review unavailable: {type(exc).__name__}: {exc}",
                "provider_error": f"{type(exc).__name__}: {exc}",
            }
    planner_levels = packet.get("planner_levels") or {}
    current_price = float(packet["current_price"])
    atr = float(packet.get("atr") or current_price * 0.02)
    structure_bars = packet.get("structure_bars") or []
    execution_bars = packet.get("execution_bars") or []
    candidates = derive_chart_level_candidates(
        current_price=current_price,
        atr=atr,
        planner_levels=planner_levels,
        structure_bars=structure_bars,
        execution_bars=execution_bars,
        config=config,
    )
    status = "AVAILABLE"
    model = config.chart_review_model
    if provider is None and (not os.getenv("OPENAI_API_KEY") or packet.get("_deterministic_fallback") is True):
        raw_review = {
            "chart_assessment": {
                "broader_structure": str(packet.get("broader_structure") or "unknown"),
                "setup_type": str(packet.get("setup_type") or "unknown"),
                "execution_structure": str(packet.get("execution_structure") or "unknown"),
                "setup_quality": "deterministic_only",
                "setup_stale": bool(packet.get("stale_plan", {}).get("stale")),
            },
            "levels": {
                "support_zone": {
                    "low": candidates.get("optional_support_level"),
                    "high": candidates.get("optional_support_level"),
                    "reason": "Deterministic local support candidate",
                },
                **{
                    name: {"price": candidates.get(name), "reason": "Deterministic pivot/reaction candidate"}
                    for name in ("near_confirmation", "primary_entry_trigger", "strong_confirmation", "major_trend_repair")
                },
                "structural_invalidation": {
                    "price": candidates.get("invalidation_level"),
                    "reason": "Existing deterministic invalidation retained",
                },
            },
            "targets": {name: {"price": candidates.get(name), "reason": "Existing planner target retained"} for name in ("tp1", "tp2", "tp3")},
            "planner_comparison": {
                "agrees_with_primary_trigger": not candidates.get("planner_primary_reclassified_as_major_repair"),
                "planner_trigger_issue": "Planner primary appears to be major trend repair" if candidates.get("planner_primary_reclassified_as_major_repair") else "No material trigger conflict detected",
                "recommended_action": "MANUAL_REVIEW" if candidates.get("planner_primary_reclassified_as_major_repair") else "KEEP_PLANNER",
            },
            "decision": "MANUAL_REVIEW" if candidates.get("planner_primary_reclassified_as_major_repair") else "KEEP_PLANNER",
            "confidence": 0.0,
            "positive_factors": [],
            "risk_factors": ["Multimodal model unavailable; deterministic review only"],
            "reason_summary": "Chart candidates were derived deterministically; no multimodal model was called.",
        }
        status = "DETERMINISTIC_FALLBACK"
        model = None
    else:
        try:
            call = provider or (lambda value: openai_chart_provider(value, model=config.chart_review_model))
            raw_review = call(packet)
            response_model = raw_review.pop("_response_model", None) or config.chart_review_model
        except Exception as exc:
            fallback_packet = {**packet, "image_paths": [], "_deterministic_fallback": True}
            fallback = review_chart_packet(fallback_packet, provider=None, config=config)
            fallback["status"] = "UNAVAILABLE"
            fallback["provider_error"] = f"{type(exc).__name__}: {exc}"
            return fallback
    proposed = _flatten_levels(raw_review)
    validation = validate_chart_levels(
        current_price=current_price,
        atr=atr,
        proposed_levels=proposed,
        planner_levels=planner_levels,
        candidate_evidence=candidates,
        structure_bars=structure_bars + execution_bars,
        trigger_max_atr=config.trigger_max_atr,
        stop_max_atr=config.chart_max_stop_atr,
        target_max_atr=config.chart_max_target_atr,
    )
    return {
        "status": status if validation["status"] != "VALIDATION_FAILED" else "VALIDATION_FAILED",
        "model": response_model if status == "AVAILABLE" else model,
        "prompt_version": CHART_STRUCTURE_PROMPT_VERSION if packet.get("review_type") != "CONFIRMED_TRADE_REVIEW" else CONFIRMED_TRADE_PROMPT_VERSION,
        "output": raw_review,
        "proposed_levels": proposed,
        "candidate_levels": candidates,
        "validated_levels": validation.get("accepted_levels") or {},
        "validation": validation,
        "decision": raw_review.get("decision") or "MANUAL_REVIEW",
        "confidence": float(raw_review.get("confidence") or 0.0),
        "reason_summary": raw_review.get("reason_summary") or "",
    }


__all__ = [
    "CHART_REVIEW_SCHEMA",
    "CONFIRMED_TRADE_REVIEW_SCHEMA",
    "CHART_STRUCTURE_PROMPT_VERSION",
    "CONFIRMED_TRADE_PROMPT_VERSION",
    "openai_chart_provider",
    "review_chart_packet",
]
