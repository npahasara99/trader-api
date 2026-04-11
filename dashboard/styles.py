"""Shared styling helpers for the Streamlit trader dashboard."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .dash-subtitle {
            color: #9ca3af;
            margin-top: -0.35rem;
            margin-bottom: 1.15rem;
        }
        .section-caption {
            color: #9ca3af;
            margin-bottom: 0.8rem;
        }
        .kpi-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.45);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            min-height: 96px;
        }
        .kpi-label {
            color: #94a3b8;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }
        .kpi-value.small {
            font-size: 1.25rem;
        }
        .watch-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.58), rgba(15, 23, 42, 0.32));
            border-radius: 16px;
            padding: 0.95rem 1rem 1rem 1rem;
            min-height: 230px;
        }
        .watch-card h4 {
            margin: 0;
            font-size: 1.45rem;
            line-height: 1.1;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.55rem 0 0.75rem 0;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.62rem;
            font-size: 0.73rem;
            font-weight: 600;
            border: 1px solid transparent;
            white-space: nowrap;
        }
        .badge.buy { background: rgba(34,197,94,0.14); color: #86efac; border-color: rgba(34,197,94,0.32); }
        .badge.wait { background: rgba(245,158,11,0.14); color: #fcd34d; border-color: rgba(245,158,11,0.32); }
        .badge.avoid { background: rgba(239,68,68,0.14); color: #fca5a5; border-color: rgba(239,68,68,0.32); }
        .badge.primary { background: rgba(59,130,246,0.16); color: #93c5fd; border-color: rgba(59,130,246,0.32); }
        .badge.secondary { background: rgba(148,163,184,0.14); color: #cbd5e1; border-color: rgba(148,163,184,0.28); }
        .badge.ready-soon { background: rgba(34,197,94,0.14); color: #86efac; border-color: rgba(34,197,94,0.32); }
        .badge.monitor { background: rgba(245,158,11,0.14); color: #fcd34d; border-color: rgba(245,158,11,0.32); }
        .badge.background { background: rgba(107,114,128,0.18); color: #d1d5db; border-color: rgba(107,114,128,0.28); }
        .badge.high { background: rgba(34,197,94,0.14); color: #86efac; border-color: rgba(34,197,94,0.32); }
        .badge.medium { background: rgba(59,130,246,0.14); color: #93c5fd; border-color: rgba(59,130,246,0.32); }
        .badge.low { background: rgba(249,115,22,0.14); color: #fdba74; border-color: rgba(249,115,22,0.28); }
        .badge.unsuitable { background: rgba(239,68,68,0.14); color: #fca5a5; border-color: rgba(239,68,68,0.32); }
        .badge.muted { background: rgba(71,85,105,0.18); color: #cbd5e1; border-color: rgba(71,85,105,0.28); }
        .mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem 1rem;
            margin-top: 0.75rem;
        }
        .mini-label {
            color: #94a3b8;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .mini-value {
            font-weight: 600;
            margin-top: 0.1rem;
        }
        .summary-note {
            color: #d1d5db;
            margin-top: 0.85rem;
            line-height: 1.4;
        }
        .detail-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.2rem 0 1rem 0;
        }
        .panel-title {
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .muted-note {
            color: #94a3b8;
        }
        .small-divider {
            margin: 0.75rem 0 0.25rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.14);
        }
        .runner-status-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem 1.1rem;
            padding: 0.55rem 0.8rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(15, 23, 42, 0.26);
            border-radius: 12px;
            margin-bottom: 0.85rem;
        }
        .runner-status-item {
            display: inline-flex;
            gap: 0.45rem;
            align-items: baseline;
            color: #cbd5e1;
        }
        .runner-status-label {
            color: #94a3b8;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .runner-status-value {
            font-size: 0.83rem;
            color: #e5e7eb;
        }
        .runner-section-card {
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(15, 23, 42, 0.28);
            border-radius: 14px;
            padding: 0.85rem 0.95rem 0.95rem 0.95rem;
            margin-bottom: 0.9rem;
        }
        .runner-section-title {
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .runner-section-subtitle {
            color: #94a3b8;
            margin-bottom: 0.75rem;
        }
        .runner-form-group-title {
            color: #cbd5e1;
            font-size: 0.84rem;
            font-weight: 700;
            margin: 0.2rem 0 0.55rem 0;
        }
        .runner-bucket-panel {
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(15, 23, 42, 0.28);
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
            min-height: 108px;
        }
        .runner-bucket-inner {
            padding: 0.2rem 0.15rem 0.3rem 0.15rem;
        }
        .runner-bucket-title {
            color: #cbd5e1;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .runner-bucket-count {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.35rem;
            height: 1.35rem;
            padding: 0 0.38rem;
            margin-left: 0.35rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.22);
        }
        .ticker-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .ticker-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.77rem;
            font-weight: 600;
            color: #dbeafe;
            background: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.25);
        }
        .ticker-chip.muted {
            color: #cbd5e1;
            background: rgba(71, 85, 105, 0.16);
            border: 1px solid rgba(71, 85, 105, 0.26);
            font-size: 0.72rem;
            padding: 0.24rem 0.62rem;
        }
        .runner-empty-note {
            color: #94a3b8;
            font-size: 0.88rem;
            padding: 0.15rem 0;
        }
        .runner-result-heading {
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .runner-result-subtitle {
            color: #94a3b8;
            margin-bottom: 0.85rem;
        }
        .runner-conclusion {
            color: #e5e7eb;
            margin: 0.25rem 0 0.95rem 0;
            line-height: 1.45;
        }
        .runner-result-gap {
            height: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
