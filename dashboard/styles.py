"""Shared styling helpers for the Streamlit trader dashboard."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-base: #06111f;
            --bg-panel: rgba(10, 21, 38, 0.72);
            --bg-panel-strong: rgba(13, 25, 44, 0.88);
            --bg-panel-soft: rgba(15, 23, 42, 0.44);
            --border-soft: rgba(148, 163, 184, 0.16);
            --border-strong: rgba(148, 163, 184, 0.22);
            --text-primary: #edf4ff;
            --text-secondary: #a8b5c7;
            --shadow-soft: 0 18px 40px rgba(2, 8, 23, 0.26);
            --shadow-lift: 0 22px 50px rgba(3, 10, 26, 0.36);
            --glow-blue: rgba(67, 97, 238, 0.18);
            --glow-teal: rgba(45, 212, 191, 0.14);
            --transition-fast: 180ms ease;
            --transition-slow: 260ms ease;
        }

        html, body, [class*="css"] {
            color: var(--text-primary);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.06), transparent 34%),
                radial-gradient(circle at bottom right, rgba(45, 212, 191, 0.05), transparent 36%),
                linear-gradient(180deg, #06111f 0%, #08121d 50%, #09131d 100%);
            position: relative;
            overflow-x: hidden;
        }

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            inset: auto;
            width: 34rem;
            height: 34rem;
            border-radius: 999px;
            filter: blur(88px);
            pointer-events: none;
            z-index: 0;
            opacity: 0.72;
        }

        .stApp::before {
            top: -10rem;
            left: -8rem;
            background: radial-gradient(circle, var(--glow-blue) 0%, rgba(67, 97, 238, 0.08) 36%, transparent 72%);
        }

        .stApp::after {
            right: -10rem;
            bottom: -12rem;
            background: radial-gradient(circle, var(--glow-teal) 0%, rgba(14, 165, 233, 0.06) 38%, transparent 74%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.75rem;
            position: relative;
            z-index: 1;
        }

        .dash-subtitle {
            color: var(--text-secondary);
            margin-top: -0.35rem;
            margin-bottom: 1.15rem;
        }

        .section-caption {
            color: var(--text-secondary);
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            background: rgba(7, 15, 27, 0.46);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 0.35rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.55rem 0.95rem;
            color: var(--text-secondary);
            transition: background var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast), transform var(--transition-fast);
            border: 1px solid transparent;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(17, 33, 58, 0.96), rgba(10, 23, 42, 0.96));
            color: var(--text-primary);
            border-color: rgba(96, 165, 250, 0.22);
            box-shadow: 0 12px 26px rgba(2, 8, 23, 0.24), inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .kpi-card,
        .watch-card,
        .runner-section-card,
        .runner-bucket-panel,
        [data-testid="stForm"],
        [data-testid="stExpander"],
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-soft);
            background: linear-gradient(180deg, rgba(15, 25, 43, 0.78), rgba(9, 17, 31, 0.72));
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: var(--shadow-soft), inset 0 1px 0 rgba(255,255,255,0.025);
            transition: transform var(--transition-slow), border-color var(--transition-slow), box-shadow var(--transition-slow), background var(--transition-slow);
        }

        .kpi-card {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            min-height: 102px;
            position: relative;
            overflow: hidden;
        }

        .kpi-card::before,
        .watch-card::before,
        .runner-section-card::before,
        .runner-bucket-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background: linear-gradient(135deg, rgba(255,255,255,0.045), transparent 26%, transparent 70%, rgba(59,130,246,0.035));
            pointer-events: none;
        }

        .kpi-card:hover,
        .watch-card:hover,
        .runner-bucket-panel:hover,
        [data-testid="stForm"]:hover,
        [data-testid="stExpander"]:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.24);
            box-shadow: var(--shadow-lift), inset 0 1px 0 rgba(255,255,255,0.035);
            background: linear-gradient(180deg, rgba(16, 28, 48, 0.82), rgba(10, 19, 34, 0.76));
        }

        .kpi-label {
            color: #93a4ba;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.36rem;
        }

        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.08;
            margin-bottom: 0.2rem;
            color: #f8fbff;
            text-shadow: 0 1px 0 rgba(255,255,255,0.02);
        }

        .kpi-value.small {
            font-size: 1.24rem;
        }

        .watch-card {
            border-radius: 20px;
            padding: 1rem 1.05rem 1.05rem 1.05rem;
            min-height: 238px;
            position: relative;
            overflow: hidden;
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
            transition: transform var(--transition-fast), box-shadow var(--transition-fast), border-color var(--transition-fast), background var(--transition-fast), filter var(--transition-fast);
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
        .badge.wait { background: linear-gradient(180deg, rgba(245,158,11,0.22), rgba(180,83,9,0.16)); color: #fde68a; border-color: rgba(245,158,11,0.34); box-shadow: 0 10px 24px rgba(245,158,11,0.11); }
        .badge.avoid { background: linear-gradient(180deg, rgba(239,68,68,0.22), rgba(153,27,27,0.16)); color: #fecaca; border-color: rgba(248,113,113,0.34); box-shadow: 0 10px 24px rgba(239,68,68,0.11); }
        .badge.primary { background: linear-gradient(180deg, rgba(59,130,246,0.23), rgba(37,99,235,0.16)); color: #bfdbfe; border-color: rgba(96,165,250,0.34); box-shadow: 0 10px 24px rgba(59,130,246,0.12); }
        .badge.secondary { background: linear-gradient(180deg, rgba(100,116,139,0.2), rgba(51,65,85,0.16)); color: #dbe4ef; border-color: rgba(148,163,184,0.28); }
        .badge.ready-soon { background: linear-gradient(180deg, rgba(16,185,129,0.23), rgba(5,150,105,0.15)); color: #a7f3d0; border-color: rgba(45,212,191,0.34); box-shadow: 0 10px 24px rgba(16,185,129,0.12); }
        .badge.monitor { background: linear-gradient(180deg, rgba(245,158,11,0.2), rgba(161,98,7,0.15)); color: #fde68a; border-color: rgba(245,158,11,0.3); }
        .badge.background { background: linear-gradient(180deg, rgba(100,116,139,0.18), rgba(51,65,85,0.14)); color: #d6deea; border-color: rgba(148,163,184,0.22); }
        .badge.high { background: linear-gradient(180deg, rgba(34,197,94,0.22), rgba(21,128,61,0.16)); color: #bbf7d0; border-color: rgba(74,222,128,0.3); }
        .badge.medium { background: linear-gradient(180deg, rgba(59,130,246,0.22), rgba(29,78,216,0.15)); color: #bfdbfe; border-color: rgba(96,165,250,0.3); }
        .badge.low { background: linear-gradient(180deg, rgba(249,115,22,0.22), rgba(154,52,18,0.14)); color: #fed7aa; border-color: rgba(251,146,60,0.28); }
        .badge.unsuitable { background: linear-gradient(180deg, rgba(239,68,68,0.22), rgba(127,29,29,0.16)); color: #fecaca; border-color: rgba(248,113,113,0.32); }
        .badge.muted { background: linear-gradient(180deg, rgba(71,85,105,0.2), rgba(30,41,59,0.16)); color: #d1d9e5; border-color: rgba(100,116,139,0.24); }

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
            border: 1px solid var(--border-soft);
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
            border-radius: 18px;
            padding: 0.95rem 1rem 1rem 1rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
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
            border-radius: 18px;
            padding: 0.82rem 0.9rem;
            min-height: 114px;
            position: relative;
            overflow: hidden;
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
            padding: 0.25rem;
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
