# Trader API planning notes

## Structured swing-planning upgrade

The swing planner no longer derives `entry`, `stop`, and `take_profit` from naive fixed percentages.

Current planning flow:
- `app/planner.py` orchestrates deterministic market-structure analysis.
- `app/indicators.py` computes ATR, moving averages, returns, and volume context inputs.
- `app/structure.py` classifies trend state and recent pivots/breakout context.
- `app/zones.py` builds support/resistance zones from pivots, MAs, fibs, consolidation, gaps, and volume congestion.
- `app/entry_engine.py` generates immediate/pullback/deeper entry candidates and selects the preferred entry.
- `app/risk_engine.py` places stop loss and take-profit targets using structure plus ATR buffers and estimates hold time.
- `app/scoring.py` produces component scores and a composite score.
- `app/llm_reasoning.py` isolates the reasoning layer so an external LLM can review structured data without inventing price levels.

Backward compatibility:
- Existing routes still return `entry`, `stop`, and `take_profit`.
- `take_profit` now maps to the first realistic target (`take_profit_1`) for compatibility with logging/evaluation.
- Richer structured fields are returned alongside the legacy fields for ranking, logging, and UI use.

Fallback behavior:
- If live quote data is unavailable, planning still falls back to recent cached closes.
- If full OHLCV bars are unavailable, the planner degrades gracefully using close-only history, but richer bars improve accuracy materially.
