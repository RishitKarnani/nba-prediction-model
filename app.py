"""
NBA Player Performance & Probability Engine — Interactive Testing Dashboard
===========================================================================
Interactive Streamlit application for scenario testing, probability estimation,
and outcome distribution visualization.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.loader import get_common_players, get_player_logs, get_team_logs
from models.inference import PlayerProbabilityEngine
from models.probability import LineProbabilityResult, OutcomeDistribution
from models.residuals import TARGETS, canonical_stat_name

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Player Performance & Probability Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e9ecef;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-stat-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #495057;
        margin-bottom: 2px;
    }
    .metric-point-pred {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1a73e8;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #6c757d;
        margin-top: 4px;
    }
    .prob-box {
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        text-align: center;
        font-weight: bold;
    }
    .prob-over {
        background-color: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
    }
    .prob-under {
        background-color: #fce8e6;
        color: #c5221f;
        border: 1px solid #fad2cf;
    }
    .prob-push {
        background-color: #fef7e0;
        color: #b06000;
        border: 1px solid #feefc3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached Resource Loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Probability Engine & Champion Models...")
def get_cached_engine() -> PlayerProbabilityEngine:
    return PlayerProbabilityEngine()


@st.cache_data(show_spinner="Loading NBA rosters and teams...")
def get_players_and_teams() -> Tuple[List[str], Dict[str, str], List[str]]:
    """Return sorted player display list, teams dict, and team abbrev list."""
    # 1. Teams
    teams_df = get_team_logs()
    team_map: Dict[str, str] = {}
    for _, row in (
        teams_df[["TEAM_ABBREVIATION", "TEAM_NAME"]]
        .drop_duplicates()
        .iterrows()
    ):
        abbrev = str(row["TEAM_ABBREVIATION"])
        name = str(row["TEAM_NAME"])
        team_map[abbrev] = f"{abbrev} — {name}"

    team_options = sorted(list(team_map.keys()))

    # 2. Players (prioritize recent active rotation players)
    try:
        p_logs = get_player_logs(seasons=["2024-25", "2025-26", "2026-27"])
        recent_counts = p_logs["PLAYER_NAME"].value_counts()
        recent_players = recent_counts[recent_counts >= 10].index.tolist()
    except Exception:
        c_players = get_common_players()
        recent_players = sorted(c_players["DISPLAY_FIRST_LAST"].dropna().unique().tolist())

    # Star players to feature at top of list
    featured_stars = [
        "LeBron James",
        "Stephen Curry",
        "Nikola Jokić",
        "Luka Dončić",
        "Giannis Antetokounmpo",
        "Jayson Tatum",
        "Kevin Durant",
        "Anthony Edwards",
        "Shai Gilgeous-Alexander",
        "Joel Embiid",
        "Damian Lillard",
        "Anthony Davis",
        "Trae Young",
        "Donovan Mitchell",
    ]

    all_other = sorted([p for p in recent_players if p not in featured_stars])
    final_players = [p for p in featured_stars if p in recent_players or True] + all_other

    return final_players, team_map, team_options


# ── Natural Language Query Parser ─────────────────────────────────────────────
def parse_natural_language_query(query: str, player_name: str) -> Optional[Tuple[str, float]]:
    """
    Parse a simple natural language question like:
    - 'Will Curry score more than 25.5 points?'
    - 'Can LeBron get over 7.5 assists?'
    - 'Probability of over 2.5 3pm?'
    Returns (stat, line) or None.
    """
    q = query.lower()
    stat_found: Optional[str] = None

    stat_aliases = {
        "PTS": ["point", "points", "pts", "score", "scoring"],
        "REB": ["rebound", "rebounds", "reb", "board", "boards"],
        "AST": ["assist", "assists", "ast", "dime", "dimes"],
        "3PM": ["three", "threes", "3pm", "3-pointer", "3pt", "3pts", "triple"],
        "STL": ["steal", "steals", "stl"],
        "BLK": ["block", "blocks", "blk"],
        "TOV": ["turnover", "turnovers", "tov"],
    }

    for s, words in stat_aliases.items():
        if any(w in q for w in words):
            stat_found = s
            break

    # Look for a number (e.g., 25.5 or 25)
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", q)
    if stat_found and match:
        val = float(match.group(1))
        return stat_found, val

    return None


# ── Main Application UI ───────────────────────────────────────────────────────
def main():
    engine = get_cached_engine()
    all_players, team_map, team_options = get_players_and_teams()

    st.title("🏀 NBA Player Performance & Probability Engine")
    st.markdown(
        "**Version 1.0 Production System** — LightGBM Minutes Conditioning & Role-Conditional Residual Bootstrap"
    )

    # ── Sidebar: Matchup Setup ────────────────────────────────────────────────
    st.sidebar.header("🎯 Matchup Selector")

    selected_player = st.sidebar.selectbox(
        "NBA Player",
        options=all_players,
        index=0,
        help="Type or select any NBA player to analyze.",
    )

    col_side1, col_side2 = st.sidebar.columns(2)
    with col_side1:
        default_opp_idx = team_options.index("GSW") if "GSW" in team_options else 0
        selected_opp = st.selectbox(
            "Opponent",
            options=team_options,
            index=default_opp_idx,
            format_func=lambda x: team_map.get(x, x),
        )

    with col_side2:
        venue = st.selectbox("Venue", options=["Home (vs.)", "Away (@)"], index=0)
        is_home = 1 if "Home" in venue else 0

    target_date = st.sidebar.date_input(
        "Matchup Date",
        value=datetime.date(2026, 11, 1),
        help="Date of the target game (ensures leak-free historical feature calculation strictly before this date).",
    )

    # Advanced Settings in Expander
    with st.sidebar.expander("⚙️ Advanced Parameters"):
        n_sims = st.slider("Monte Carlo Draws", min_value=2000, max_value=20000, value=10000, step=2000)
        sim_seed = st.number_input("Random Seed", value=42, step=1)

    # ── Run Prediction Pipeline ───────────────────────────────────────────────
    try:
        with st.spinner(f"Simulating {n_sims:,} games for {selected_player}..."):
            result = engine.predict_player(
                player=selected_player,
                opponent=selected_opp,
                is_home=is_home,
                game_date=str(target_date),
                n_simulations=n_sims,
                seed=sim_seed,
            )
    except Exception as e:
        st.error(f"Error generating projection for **{selected_player}**: {e}")
        st.info("Tip: Try adjusting the game date or selecting another active player.")
        return

    meta = result.get("metadata", {})
    pred_min = result["predicted_minutes"]
    is_starter = result["is_starter_context"]
    role_label = "Starter" if is_starter else "Bench"
    pred_stats = result["predicted_stats"]
    distributions: Dict[str, OutcomeDistribution] = result["distributions"]

    # ── Header Matchup Card ───────────────────────────────────────────────────
    st.markdown("---")
    head_col1, head_col2, head_col3 = st.columns([3, 2, 2])
    with head_col1:
        venue_symbol = "vs." if is_home else "@"
        st.subheader(f"**{meta.get('player_name', selected_player)}** ({meta.get('team_abbreviation', 'NBA')})")
        st.markdown(f"**Matchup:** {meta.get('team_abbreviation', '')} {venue_symbol} **{meta.get('opponent_abbrev', selected_opp)}**")
        st.caption(f"📅 Game Date: **{meta.get('game_date', str(target_date))}** | Prior games in database: {meta.get('n_prior_games', 'N/A')}")

    with head_col2:
        st.metric(
            label="Predicted Playing Time",
            value=f"{pred_min:.1f} MIN",
            delta=f"Role: {role_label}",
            delta_color="normal" if is_starter else "off",
        )

    with head_col3:
        pts_pred = pred_stats.get("PTS", 0.0)
        pts_sim_mean = distributions["PTS"].distribution_mean
        st.metric(
            label="Points Forecast (PTS)",
            value=f"{pts_pred:.1f}",
            delta=f"Sim Mean: {pts_sim_mean:.1f}",
        )

    # ── Section 1: 7 Statistical Point Projections ─────────────────────────────
    st.markdown("### 📊 7-Statistic Projections (LightGBM Median vs. Empirical Mean)")
    cols = st.columns(7)

    stat_descriptions = {
        "PTS": "Points",
        "REB": "Rebounds",
        "AST": "Assists",
        "3PM": "3-Pointers",
        "STL": "Steals",
        "BLK": "Blocks",
        "TOV": "Turnovers",
    }

    for idx, stat in enumerate(TARGETS):
        dist = distributions[stat]
        p_val = pred_stats.get(stat, 0.0)
        s_mean = dist.distribution_mean
        s_std = dist.distribution_std
        lo_90, hi_90 = dist.interval(0.90)

        with cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-stat-name">{stat}</div>
                    <div style="font-size:0.75rem; color:#888;">{stat_descriptions.get(stat, '')}</div>
                    <div class="metric-point-pred">{p_val:.1f}</div>
                    <div class="metric-subtext">Mean: <b>{s_mean:.2f}</b> (±{s_std:.1f})</div>
                    <div class="metric-subtext" style="color:#1a73e8;">90% CI: <b>[{int(lo_90)}, {int(hi_90)}]</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Section 2: Interactive Probability Engine & Line Tester ────────────────
    st.markdown("### 🎲 Interactive Probability & Line Testing Engine")
    st.write(
        "Enter or slide any threshold line below to calculate real-time **Over**, **Under**, and **Push** probabilities from 10,000 Monte Carlo draws."
    )

    calc_col1, calc_col2 = st.columns([1, 2])

    with calc_col1:
        chosen_stat = st.selectbox(
            "Select Target Statistic",
            options=TARGETS,
            index=0,
            format_func=lambda s: f"{s} ({stat_descriptions.get(s, '')})",
        )
        curr_dist = distributions[chosen_stat]
        curr_point = pred_stats.get(chosen_stat, 0.0)

        # Quick default suggestion around point prediction
        default_line = round(curr_point + 0.5, 1)

        chosen_line = st.number_input(
            f"Set {chosen_stat} Line",
            min_value=0.0,
            max_value=100.0,
            value=float(default_line),
            step=0.5,
            format="%.1f",
            help="Supports standard half-point lines (24.5) and integer lines (25.0).",
        )

        # Quick preset buttons
        st.write("Quick Presets:")
        preset_cols = st.columns(4)
        presets = [
            round(curr_point - 2.5, 1),
            round(curr_point - 0.5, 1),
            round(curr_point + 0.5, 1),
            round(curr_point + 2.5, 1),
        ]
        presets = [max(0.5, p) for p in presets]
        for b_idx, p_val in enumerate(presets):
            if preset_cols[b_idx].button(f"{p_val:.1f}", key=f"btn_preset_{b_idx}"):
                chosen_line = p_val

    # Calculate probability result
    line_res: LineProbabilityResult = curr_dist.probability_for_line(chosen_line)
    p_over = line_res.prob_over
    p_under = line_res.prob_under
    p_push = line_res.prob_push
    is_int_line = line_res.is_integer_line

    with calc_col2:
        st.markdown(f"#### Probability Outcome for **{chosen_stat} Line: {chosen_line:.1f}**")

        res_cols = st.columns(3 if is_int_line else 2)
        with res_cols[0]:
            st.markdown(
                f"""
                <div class="prob-box prob-over">
                    <div style="font-size:0.85rem;">OVER {chosen_line:.1f}</div>
                    <div style="font-size:1.8rem;">{p_over * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_cols[1]:
            st.markdown(
                f"""
                <div class="prob-box prob-under">
                    <div style="font-size:0.85rem;">UNDER {chosen_line:.1f}</div>
                    <div style="font-size:1.8rem;">{p_under * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if is_int_line:
            with res_cols[2]:
                st.markdown(
                    f"""
                    <div class="prob-box prob-push">
                        <div style="font-size:0.85rem;">EXACT PUSH ({chosen_line:.0f})</div>
                        <div style="font-size:1.8rem;">{p_push * 100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Context badge
        st.caption(
            f"Simulation Model: **{line_res.method_used}** | "
            f"Peer Context: **{line_res.fallback_tier}** ({line_res.n_support:,} comparable historical games) | "
            f"Calibration: `{line_res.calibration_status}`"
        )

    # ── Section 3: Distribution Chart Visualization ───────────────────────────
    st.markdown("---")
    st.markdown(f"### 📈 Simulated Outcome Distribution: {selected_player} ({chosen_stat})")

    outcomes = curr_dist.outcomes
    max_val = int(np.percentile(outcomes, 99.5)) + 4
    bins = np.arange(-0.5, max_val + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    # Color bars based on over/under
    n_counts, bin_edges = np.histogram(outcomes, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    colors = []
    for c in bin_centers:
        int_c = int(round(c))
        if is_int_line and int_c == int(round(chosen_line)):
            colors.append("#fbbc04")  # Push
        elif int_c > chosen_line:
            colors.append("#34a853")  # Over
        else:
            colors.append("#ea4335")  # Under

    ax.bar(bin_centers, n_counts / len(outcomes) * 100, width=0.85, color=colors, alpha=0.85, edgecolor="#333", linewidth=0.5)

    # Vertical markers
    ax.axvline(curr_point, color="#1a73e8", linestyle="--", linewidth=2, label=f"LightGBM Median ({curr_point:.1f})")
    ax.axvline(curr_dist.distribution_mean, color="#8e24aa", linestyle="-.", linewidth=2, label=f"Simulated Mean ({curr_dist.distribution_mean:.1f})")
    ax.axvline(chosen_line, color="#000000", linestyle="-", linewidth=2.5, label=f"Threshold Line ({chosen_line:.1f})")

    ax.set_title(f"{meta.get('player_name', selected_player)} — Simulated {chosen_stat} Outcome Frequency ({n_sims:,} Draws)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Actual {chosen_stat} in Game", fontsize=10)
    ax.set_ylabel("Probability Frequency (%)", fontsize=10)
    ax.set_xlim(-0.5, max_val + 0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Section 4: Multi-Stat Common Lines Table ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Multi-Stat Quick Consensus Table")
    st.caption("Standard lines evaluated simultaneously across all 7 statistics for this matchup.")

    summary_rows = []
    for stat in TARGETS:
        d = distributions[stat]
        p_pt = pred_stats[stat]
        line_half = round(p_pt + 0.5, 1)
        res_h = d.probability_for_line(line_half)

        line_alt = round(p_pt + 2.5, 1)
        res_alt = d.probability_for_line(line_alt)

        lo, hi = d.interval(0.90)
        summary_rows.append(
            {
                "Statistic": stat,
                "Point Pred": f"{p_pt:.2f}",
                "Sim Mean": f"{d.distribution_mean:.2f}",
                "Std Dev": f"{d.distribution_std:.2f}",
                "90% Range": f"[{int(lo)}, {int(hi)}]",
                f"Line ({line_half})": f"{line_half:.1f}",
                "Over %": f"{res_h.prob_over * 100:.1f}%",
                "Under %": f"{res_h.prob_under * 100:.1f}%",
                f"High Line ({line_alt})": f"{line_alt:.1f}",
                "High Over %": f"{res_alt.prob_over * 100:.1f}%",
                "Support": f"{d.n_support:,} games",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Section 5: Natural Language Question Box ───────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Ask a Question About This Matchup")
    user_q = st.text_input(
        "Type your question:",
        placeholder=f"e.g., 'Will {meta.get('player_name', selected_player).split()[-1]} score more than 24.5 points?'",
    )

    if user_q:
        parsed = parse_natural_language_query(user_q, meta.get("player_name", selected_player))
        if parsed:
            q_stat, q_line = parsed
            q_dist = distributions.get(q_stat)
            if q_dist:
                q_res = q_dist.probability_for_line(q_line)
                st.success(
                    f"**Analysis for '{user_q}':**\n\n"
                    f"- **Target Stat:** `{q_stat}` | **Line:** `{q_line}`\n"
                    f"- **Probability OVER {q_line}:** **{q_res.prob_over * 100:.2f}%**\n"
                    f"- **Probability UNDER {q_line}:** **{q_res.prob_under * 100:.2f}%**\n"
                    + (f"- **Probability of PUSH ({int(q_line)}):** **{q_res.prob_push * 100:.2f}%**\n" if q_res.is_integer_line else "")
                    + f"- **Forecast Context:** Expected playing time is **{pred_min:.1f} minutes** ({role_label}). LightGBM median is **{pred_stats[q_stat]:.1f} {q_stat}**."
                )
        else:
            st.warning("Could not identify both a target stat (e.g. points, assists) and a threshold number in your question. Try e.g. 'Will he score over 20.5 points?'")


if __name__ == "__main__":
    main()
