"""Shared styling helpers for the Streamlit trader dashboard."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-primary: #eef4ff;
            --text-secondary: #9fb0c5;
            --panel-bg: rgba(11, 20, 34, 0.78);
            --panel-bg-strong: rgba(10, 18, 31, 0.92);
            --panel-border: rgba(148, 163, 184, 0.16);
            --panel-border-strong: rgba(96, 165, 250, 0.22);
            --shadow-soft: 0 16px 34px rgba(2, 8, 23, 0.22);
            --shadow-lift: 0 22px 46px rgba(2, 8, 23, 0.30);
            --transition-fast: 180ms ease;
            --transition-slow: 260ms ease;
        }

        html, body {
            color: var(--text-primary);
            background: #08111c;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(79, 70, 229, 0.10), transparent 26%),
                radial-gradient(circle at 88% 92%, rgba(20, 184, 166, 0.08), transparent 28%),
                linear-gradient(180deg, #08111c 0%, #09121e 50%, #0a121d 100%);
            color: var(--text-primary);
        }

        .block-container {
            padding-top: 1.45rem;
            padding-bottom: 2.7rem;
        }

        h1, h2, h3, h4, h5, h6,
        p, label, span, div {
            color: inherit;
        }

        .dash-subtitle {
            color: var(--text-secondary);
            margin-top: -0.35rem;
            margin-bottom: 1.05rem;
        }

        .section-caption {
            color: var(--text-secondary);
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(8, 15, 26, 0.54);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 0.34rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary);
            border-radius: 12px;
            padding: 0.54rem 0.94rem;
            border: 1px solid transparent;
            transition: background var(--transition-fast), border-color var(--transition-fast), transform var(--transition-fast), color var(--transition-fast);
        }

        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-1px);
            color: var(--text-primary);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(16, 29, 49, 0.95), rgba(10, 20, 36, 0.96));
            color: var(--text-primary);
            border-color: var(--panel-border-strong);
            box-shadow: 0 10px 22px rgba(2, 8, 23, 0.20), inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .kpi-card,
        .watch-card,
        .runner-section-card,
        .runner-bucket-panel {
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            border: 1px solid var(--panel-border);
            background: linear-gradient(180deg, rgba(16, 26, 43, 0.82), rgba(9, 17, 29, 0.76));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: var(--shadow-soft), inset 0 1px 0 rgba(255,255,255,0.025);
            transition: transform var(--transition-slow), box-shadow var(--transition-slow), border-color var(--transition-slow), background var(--transition-slow);
        }

        .kpi-card::before,
        .watch-card::before,
        .runner-section-card::before,
        .runner-bucket-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(135deg, rgba(255,255,255,0.045), transparent 24%, transparent 70%, rgba(96,165,250,0.03));
        }

        .kpi-card:hover,
        .watch-card:hover,
        .runner-bucket-panel:hover,
        .runner-section-card:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.24);
            box-shadow: var(--shadow-lift), inset 0 1px 0 rgba(255,255,255,0.03);
            background: linear-gradient(180deg, rgba(18, 30, 49, 0.86), rgba(10, 19, 33, 0.80));
        }

        .kpi-card {
            padding: 0.95rem 1rem;
            min-height: 102px;
        }

        .kpi-label {
            color: #93a4ba;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.34rem;
        }

        .kpi-value {
            color: #f8fbff;
            font-size: 1.62rem;
            font-weight: 700;
            line-height: 1.08;
            margin-bottom: 0.18rem;
        }

        .kpi-value.small {
            font-size: 1.24rem;
        }

        .watch-card {
            border-radius: 20px;
            padding: 1rem 1.05rem 1.05rem 1.05rem;
            min-height: 236px;
        }

        .watch-card h4 {
            margin: 0;
            font-size: 1.45rem;
            line-height: 1.08;
            letter-spacing: -0.01em;
            color: #fbfdff;
        }

        .badge-row,
        .detail-chip-grid,
        .ticker-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .badge-row {
            margin: 0.62rem 0 0.82rem 0;
        }

        .badge,
        .ticker-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            white-space: nowrap;
            transition: transform var(--transition-fast), box-shadow var(--transition-fast), filter var(--transition-fast), border-color var(--transition-fast);
        }

        .badge {
            padding: 0.28rem 0.72rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .badge:hover,
        .ticker-chip:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
        }

        .badge.buy { background: linear-gradient(180deg, rgba(22,163,74,0.24), rgba(22,163,74,0.16)); color: #a7f3d0; border-color: rgba(34,197,94,0.34); box-shadow: 0 10px 24px rgba(22,163,74,0.12); }
        .badge.wait { background: linear-gradient(180deg, rgba(245,158,11,0.22), rgba(180,83,9,0.16)); color: #fde68a; border-color: rgba(245,158,11,0.34); box-shadow: 0 10px 24px rgba(245,158,11,0.10); }
        .badge.avoid { background: linear-gradient(180deg, rgba(239,68,68,0.22), rgba(153,27,27,0.16)); color: #fecaca; border-color: rgba(248,113,113,0.34); box-shadow: 0 10px 24px rgba(239,68,68,0.10); }
        .badge.primary { background: linear-gradient(180deg, rgba(59,130,246,0.22), rgba(37,99,235,0.16)); color: #bfdbfe; border-color: rgba(96,165,250,0.34); box-shadow: 0 10px 24px rgba(59,130,246,0.11); }
        .badge.secondary { background: linear-gradient(180deg, rgba(100,116,139,0.20), rgba(51,65,85,0.16)); color: #dbe4ef; border-color: rgba(148,163,184,0.28); }
        .badge.ready-soon { background: linear-gradient(180deg, rgba(16,185,129,0.23), rgba(5,150,105,0.15)); color: #a7f3d0; border-color: rgba(45,212,191,0.34); box-shadow: 0 10px 24px rgba(16,185,129,0.11); }
        .badge.monitor { background: linear-gradient(180deg, rgba(245,158,11,0.20), rgba(161,98,7,0.15)); color: #fde68a; border-color: rgba(245,158,11,0.30); }
        .badge.background { background: linear-gradient(180deg, rgba(100,116,139,0.18), rgba(51,65,85,0.14)); color: #d6deea; border-color: rgba(148,163,184,0.22); }
        .badge.high { background: linear-gradient(180deg, rgba(34,197,94,0.22), rgba(21,128,61,0.16)); color: #bbf7d0; border-color: rgba(74,222,128,0.30); }
        .badge.medium { background: linear-gradient(180deg, rgba(59,130,246,0.22), rgba(29,78,216,0.15)); color: #bfdbfe; border-color: rgba(96,165,250,0.30); }
        .badge.low { background: linear-gradient(180deg, rgba(249,115,22,0.22), rgba(154,52,18,0.14)); color: #fed7aa; border-color: rgba(251,146,60,0.28); }
        .badge.unsuitable { background: linear-gradient(180deg, rgba(239,68,68,0.22), rgba(127,29,29,0.16)); color: #fecaca; border-color: rgba(248,113,113,0.32); }
        .badge.muted { background: linear-gradient(180deg, rgba(71,85,105,0.20), rgba(30,41,59,0.16)); color: #d1d9e5; border-color: rgba(100,116,139,0.24); }

        .ticker-chip {
            padding: 0.31rem 0.76rem;
            font-size: 0.76rem;
            font-weight: 700;
            color: #e0edff;
            background: linear-gradient(180deg, rgba(37,99,235,0.18), rgba(30,64,175,0.12));
            border: 1px solid rgba(96,165,250,0.24);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.025), 0 10px 22px rgba(30,64,175,0.08);
        }

        .ticker-chip.muted {
            color: #d0dae7;
            background: linear-gradient(180deg, rgba(51,65,85,0.22), rgba(30,41,59,0.16));
            border: 1px solid rgba(100,116,139,0.24);
            font-size: 0.72rem;
            padding: 0.25rem 0.64rem;
            box-shadow: none;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.62rem 1rem;
            margin-top: 0.82rem;
        }

        .mini-label {
            color: #90a1b8;
            font-size: 0.71rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .mini-value {
            font-weight: 650;
            margin-top: 0.12rem;
            color: #eef5ff;
        }

        .summary-note,
        .runner-conclusion {
            color: #d7e0eb;
            line-height: 1.45;
        }

        .summary-note {
            margin-top: 0.88rem;
        }

        .detail-chip-grid {
            margin: 0.24rem 0 1rem 0;
        }

        .panel-title {
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .muted-note,
        .runner-empty-note,
        .runner-result-subtitle,
        .runner-section-subtitle,
        .runner-status-label {
            color: var(--text-secondary);
        }

        .small-divider {
            margin: 0.75rem 0 0.25rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.14);
        }

        .runner-status-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem 1.15rem;
            padding: 0.65rem 0.9rem;
            border: 1px solid var(--panel-border);
            background: linear-gradient(180deg, rgba(13, 22, 39, 0.74), rgba(9, 17, 31, 0.58));
            border-radius: 16px;
            margin-bottom: 0.95rem;
            box-shadow: var(--shadow-soft);
        }

        .runner-status-item {
            display: inline-flex;
            gap: 0.48rem;
            align-items: baseline;
            color: #dbe7f5;
        }

        .runner-status-value {
            font-size: 0.83rem;
            color: #eef4fc;
        }

        .runner-section-card {
            padding: 0.95rem 1rem 1rem 1rem;
            margin-bottom: 1rem;
        }

        .runner-section-title,
        .runner-result-heading,
        .runner-bucket-title {
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        .runner-form-group-title {
            color: #d7e4f0;
            font-size: 0.84rem;
            font-weight: 700;
            margin: 0.2rem 0 0.6rem 0;
        }

        .runner-bucket-panel {
            padding: 0.82rem 0.9rem;
            min-height: 114px;
        }

        .runner-bucket-inner {
            padding: 0.28rem 0.2rem 0.34rem 0.2rem;
        }

        .runner-bucket-title {
            color: #e8eff8;
            margin-bottom: 0.82rem;
        }

        .runner-bucket-count {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.4rem;
            height: 1.4rem;
            padding: 0 0.38rem;
            margin-left: 0.38rem;
            border-radius: 999px;
            font-size: 0.71rem;
            font-weight: 700;
            color: #d8e2ef;
            background: linear-gradient(180deg, rgba(100,116,139,0.22), rgba(51,65,85,0.16));
            border: 1px solid rgba(148,163,184,0.22);
        }

        .runner-empty-note {
            font-size: 0.88rem;
            padding: 0.15rem 0;
        }

        .runner-result-heading {
            margin-bottom: 0.15rem;
        }

        .runner-result-subtitle {
            margin-bottom: 0.9rem;
        }

        .runner-conclusion {
            margin: 0.28rem 0 1rem 0;
        }

        .runner-result-gap {
            height: 1.05rem;
        }

        div.stButton > button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
            border-radius: 14px !important;
            border: 1px solid rgba(148,163,184,0.2) !important;
            background: linear-gradient(180deg, rgba(16, 28, 48, 0.95), rgba(10, 20, 36, 0.92)) !important;
            color: #edf4ff !important;
            box-shadow: 0 12px 24px rgba(2,8,23,0.22), inset 0 1px 0 rgba(255,255,255,0.03) !important;
            transition: transform var(--transition-fast), box-shadow var(--transition-fast), border-color var(--transition-fast), filter var(--transition-fast) !important;
        }

        div.stButton > button:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="baseButton-primary"]:hover {
            transform: translateY(-1px);
            border-color: rgba(96,165,250,0.28) !important;
            box-shadow: 0 16px 30px rgba(2,8,23,0.28), 0 0 0 1px rgba(96,165,250,0.08) inset !important;
            filter: brightness(1.03);
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(180deg, rgba(37,99,235,0.95), rgba(29,78,216,0.92)) !important;
            border-color: rgba(96,165,250,0.34) !important;
            color: white !important;
            box-shadow: 0 16px 30px rgba(30,64,175,0.28), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        }

        [data-testid="stForm"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stExpander"] > details {
            background: transparent;
        }

        [data-testid="stDataFrame"] {
            padding: 0.2rem;
            background: rgba(10, 18, 31, 0.68);
            border: 1px solid var(--panel-border);
            box-shadow: var(--shadow-soft);
        }

        .stSelectbox > div[data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stNumberInput input {
            background: rgba(10, 18, 31, 0.88) !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
            border-radius: 14px !important;
            color: #edf4ff !important;
            transition: border-color var(--transition-fast), box-shadow var(--transition-fast), background var(--transition-fast) !important;
        }

        .stSelectbox > div[data-baseweb="select"] > div:hover,
        .stTextInput > div > div > input:hover,
        .stTextArea textarea:hover,
        .stNumberInput input:hover {
            border-color: rgba(96,165,250,0.24) !important;
            background: rgba(12, 22, 37, 0.92) !important;
        }

        .stSelectbox > div[data-baseweb="select"] > div:focus-within,
        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {
            box-shadow: 0 0 0 1px rgba(96,165,250,0.22), 0 0 0 4px rgba(59,130,246,0.08) !important;
            border-color: rgba(96,165,250,0.34) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 15, 27, 0.96), rgba(6, 12, 22, 0.96));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
