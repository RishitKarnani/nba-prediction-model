"""
NBA Player Performance & Probability Engine — 2-Day Matchup Forecast UI
========================================================================
Streamlined interface for today's and tomorrow's upcoming NBA games.
Automatically detects home/away teams, active rosters, and generates
leak-free minutes, stat forecasts, and threshold probabilities.
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
from data.schedule import _normalize_date_str, get_daily_schedule, get_team_recent_rotation
from models.inference import PlayerProbabilityEngine
from models.probability import LineProbabilityResult, OutcomeDistribution
from models.residuals import TARGETS, canonical_stat_name
from models.team_inference import TeamPredictionEngine

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
    .matchup-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .matchup-teams {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .matchup-meta {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 14px 10px;
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


@st.cache_resource(show_spinner="Loading Team Prediction Engine...")
def get_cached_team_engine() -> TeamPredictionEngine:
    return TeamPredictionEngine(apply_form=True)


@st.cache_data(show_spinner=False)
def get_final_matchup_prediction(
    home_team_abbrev: str,
    away_team_abbrev: str,
    injuries_home: tuple = (),
    injuries_away: tuple = (),
) -> Dict[str, Any]:
    """Fetch the 500-simulation average result for this exact matchup with injuries factored in."""
    team_engine = get_cached_team_engine()
    return team_engine.predict_game(
        home_team=home_team_abbrev,
        away_team=away_team_abbrev,
        n_simulations=500,
        injuries_home=list(injuries_home),
        injuries_away=list(injuries_away),
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schedule_for_date(date_str: str) -> List[Dict]:
    return get_daily_schedule(date_str)


# ── Natural Language Query Parser ─────────────────────────────────────────────
def parse_natural_language_query(query: str, player_name: str) -> Optional[Tuple[str, float]]:
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

    match = re.search(r"\b(\d+(?:\.\d+)?)\b", q)
    if stat_found and match:
        val = float(match.group(1))
        return stat_found, val

    return None


# ── Main Application UI ───────────────────────────────────────────────────────
def main():
    engine = get_cached_engine()

    st.title("🏀 NBA Player Performance & Probability Engine")
    st.caption("Live 2-Day Upcoming Matchup Predictions & Threshold Probability Estimator")

    # ── Sidebar: 2-Day Matchup Selection Workflow ─────────────────────────────
    st.sidebar.header("📅 Select Matchup")

    today_dt = datetime.date.today()
    yesterday_dt = today_dt - datetime.timedelta(days=1)
    tomorrow_dt = today_dt + datetime.timedelta(days=1)

    # 3-Day Slate Selector (Yesterday, Today, Tomorrow)
    day_options = {
        f"Yesterday ({yesterday_dt.strftime('%b %d')})": yesterday_dt,
        f"Today ({today_dt.strftime('%b %d')})": today_dt,
        f"Tomorrow ({tomorrow_dt.strftime('%b %d')})": tomorrow_dt,
    }

    selected_day_label = st.sidebar.radio(
        "Matchup Slate",
        options=list(day_options.keys()),
        index=1,
    )
    target_date = day_options[selected_day_label]

    # Optional Custom Date expander if user ever needs to check another date
    with st.sidebar.expander("🔍 Or check another date"):
        custom_input = st.text_input("Enter date (e.g. 19th nov, 19/11/26, 2026-11-19)", placeholder="e.g. 19th nov")
        if custom_input.strip():
            try:
                norm_str = _normalize_date_str(custom_input.strip())
                target_date = datetime.date.fromisoformat(norm_str)
                st.sidebar.caption(f"Showing games for: **{target_date.strftime('%a, %b %d, %Y')}**")
            except Exception:
                st.sidebar.error("Invalid date format. Use '19th nov', DD/MM/YY, or YYYY-MM-DD.")

    # Fetch official NBA schedule for the chosen date
    date_str = target_date.strftime("%Y-%m-%d")
    scheduled_games = fetch_schedule_for_date(date_str)

    if not scheduled_games:
        st.warning(f"No scheduled NBA games found for **{target_date.strftime('%a, %b %d, %Y')}**.")
        st.info("The NBA might not have games scheduled for this date (e.g. off-day or break). Select another day from the sidebar.")
        return

    st.sidebar.markdown("---")

    # ── Step 1: Select Game ───────────────────────────────────────────────────
    game_labels = [g["label"] for g in scheduled_games]
    chosen_game_idx = st.sidebar.selectbox(
        f"Select Game ({len(scheduled_games)} scheduled on {target_date.strftime('%b %d')})",
        options=range(len(scheduled_games)),
        format_func=lambda i: game_labels[i],
    )
    game_info = scheduled_games[chosen_game_idx]

    away_team_abbrev = game_info["away_team_abbrev"]
    away_team_name = game_info["away_team_name"]
    home_team_abbrev = game_info["home_team_abbrev"]
    home_team_name = game_info["home_team_name"]
    game_status = game_info["game_status"]

    # ── Step 2: Select Team (Auto Home / Away) ────────────────────────────────
    st.sidebar.markdown("**Select Team:**")
    team_choice = st.sidebar.radio(
        "Select Team",
        options=[
            f"✈️ Away: {away_team_name} ({away_team_abbrev})",
            f"🏠 Home: {home_team_name} ({home_team_abbrev})",
        ],
        index=0,
        label_visibility="collapsed",
    )

    is_away_selected = team_choice.startswith("✈️")
    if is_away_selected:
        focus_team = away_team_abbrev
        focus_team_name = away_team_name
        opponent_team = home_team_abbrev
        opponent_team_name = home_team_name
        is_home = 0
        venue_str = f"@ {home_team_name} ({home_team_abbrev})"
    else:
        focus_team = home_team_abbrev
        focus_team_name = home_team_name
        opponent_team = away_team_abbrev
        opponent_team_name = away_team_name
        is_home = 1
        venue_str = f"vs. {away_team_name} ({away_team_abbrev})"

    # ── Step 3: Select Player from That Team's Rotation ───────────────────────
    rotation = get_team_recent_rotation(focus_team)

    if rotation:
        player_choices = [p["player_name"] for p in rotation]
        player_labels = {p["player_name"]: p["display"] for p in rotation}
        selected_player = st.sidebar.selectbox(
            f"Select Player ({focus_team_name})",
            options=player_choices,
            format_func=lambda p: player_labels.get(p, p),
            index=0,
            help="Rotation players sorted by average playing time.",
        )
    else:
        # Fallback to general player logs for that team
        p_logs = get_player_logs()
        team_players = sorted(p_logs[p_logs["TEAM_ABBREVIATION"] == focus_team]["PLAYER_NAME"].unique().tolist())
        if not team_players:
            team_players = sorted(p_logs["PLAYER_NAME"].unique().tolist())
        selected_player = st.sidebar.selectbox(f"Select Player ({focus_team})", options=team_players, index=0)

    # ── Step 4: Injury Report (Optional) ──────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("🏥 Injury Report (Mark Players OUT)", expanded=False):
        st.caption("Enter comma-separated names of players who are OUT. The engine will automatically redistribute their minutes and adjust team offensive ratings.")
        inj_away_raw = st.text_input(f"❌ {away_team_abbrev} OUT Players", placeholder="e.g. Anthony Edwards", key="inj_away")
        inj_home_raw = st.text_input(f"❌ {home_team_abbrev} OUT Players", placeholder="e.g. Giannis Antetokounmpo", key="inj_home")

    injuries_away = tuple([p.strip() for p in inj_away_raw.split(",") if p.strip()])
    injuries_home = tuple([p.strip() for p in inj_home_raw.split(",") if p.strip()])

    # Simulation Settings in Sidebar Expander
    with st.sidebar.expander("⚙️ Simulation Settings"):
        n_sims = st.slider("Monte Carlo Draws", min_value=2000, max_value=20000, value=10000, step=2000)
        sim_seed = st.number_input("Random Seed", value=42, step=1)

    # ── Run Prediction Pipeline ───────────────────────────────────────────────
    try:
        with st.spinner(f"Simulating {n_sims:,} games for {selected_player}..."):
            result = engine.predict_player(
                player=selected_player,
                opponent=opponent_team,
                is_home=is_home,
                game_date=date_str,
                n_simulations=n_sims,
                seed=sim_seed,
            )
    except Exception as e:
        st.error(f"Error generating projection for **{selected_player}**: {e}")
        st.info("Tip: If this player has no historical games before this date, try another rotation player.")
        return

    meta = result.get("metadata", {})
    pred_min = result["predicted_minutes"]
    is_starter = result["is_starter_context"]
    role_label = "Starter" if is_starter else "Bench"
    pred_stats = result["predicted_stats"]
    distributions: Dict[str, OutcomeDistribution] = result["distributions"]

    # ── Main View: Matchup Header Banner ──────────────────────────────────────
    st.markdown(
        f"""
        <div class="matchup-banner">
            <div class="matchup-teams">
                {away_team_name} ({away_team_abbrev}) &nbsp;@&nbsp; {home_team_name} ({home_team_abbrev})
            </div>
            <div class="matchup-meta">
                📅 <b>{target_date.strftime('%A, %b %d, %Y')}</b> &nbsp;|&nbsp; 
                ⏰ <b>{game_status}</b> &nbsp;|&nbsp; 
                Focus: <b>{meta.get('player_name', selected_player)}</b> ({focus_team}) {venue_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section: Team Win Probability & Margin of Win ─────────────────────────
    st.markdown("### 🏆 Team Win Probability & Projected Margin")
    try:
        team_pred_res = get_final_matchup_prediction(
            home_team_abbrev=home_team_abbrev,
            away_team_abbrev=away_team_abbrev,
            injuries_home=injuries_home,
            injuries_away=injuries_away,
        )
        tp = team_pred_res["prediction"]
        winner_name = tp["predicted_winner"]
        winner_abbr = tp["winner_abbreviation"]
        h_prob = tp["home_win_probability"]
        a_prob = tp["away_win_probability"]
        pred_margin = tp["predicted_margin"]
        pred_spread = tp["implied_spread"]
        score_h = tp["predicted_score"].get(home_team_abbrev, 0.0)
        score_a = tp["predicted_score"].get(away_team_abbrev, 0.0)

        t_col1, t_col2, t_col3 = st.columns([2, 2, 2])
        with t_col1:
            st.metric(
                label=f"✈️ {away_team_name} ({away_team_abbrev})",
                value=f"{a_prob*100:.1f}% Win Prob",
                delta=f"Proj: {score_a:.1f} PTS",
                delta_color="normal" if a_prob >= 0.5 else "off",
            )
        with t_col2:
            st.metric(
                label="Predicted Winner & Margin",
                value=f"{winner_abbr} {pred_margin}",
                delta=f"Winner: {winner_name} ({pred_spread})",
            )
        with t_col3:
            st.metric(
                label=f"🏠 {home_team_name} ({home_team_abbrev})",
                value=f"{h_prob*100:.1f}% Win Prob",
                delta=f"Proj: {score_h:.1f} PTS",
                delta_color="normal" if h_prob >= 0.5 else "off",
            )

        st.progress(
            h_prob,
            text=f"Win Probability: {away_team_abbrev} {a_prob*100:.1f}%  vs  {home_team_abbrev} {h_prob*100:.1f}%",
        )

        # Show injury alert badges if injuries are active
        all_injuries = list(injuries_away) + list(injuries_home)
        if all_injuries:
            st.caption(f"🏥 **Injury adjustments factored in:** {', '.join(all_injuries)} marked OUT (minutes & points redistributed)")

        # Alert if the currently inspected player is injured
        if any(selected_player.lower() in inj.lower() or inj.lower() in selected_player.lower() for inj in all_injuries):
            st.warning(f"⚠️ **{selected_player}** is listed as **OUT** in the Injury Report for this game.")
    except Exception as e:
        st.warning(f"Team outcome model could not be loaded: {e}")

    st.markdown("---")

    # Top KPI Row
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric(
            label="Predicted Playing Time",
            value=f"{pred_min:.1f} MIN",
            delta=f"Role: {role_label}",
            delta_color="normal" if is_starter else "off",
        )
    with kpi_col2:
        pts_pred = pred_stats.get("PTS", 0.0)
        pts_mean = distributions["PTS"].distribution_mean
        st.metric(
            label="Points Forecast (PTS)",
            value=f"{pts_pred:.1f}",
            delta=f"Sim Mean: {pts_mean:.1f}",
        )
    with kpi_col3:
        reb_pred = pred_stats.get("REB", 0.0)
        ast_pred = pred_stats.get("AST", 0.0)
        st.metric(
            label="Rebounds & Assists",
            value=f"{reb_pred:.1f} REB",
            delta=f"{ast_pred:.1f} AST",
        )

    st.markdown("---")

    # ── Section 1: 7-Statistic Forecast Cards ─────────────────────────────────
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

    # ── Section 2: Interactive Probability & Threshold Testing Engine ────────
    st.markdown("### 🎲 Interactive Probability & Threshold Testing")
    st.caption("Calculate exact Over, Under, and Push probabilities for any line based on 10,000 empirical draws.")

    calc_col1, calc_col2 = st.columns([1, 2])

    with calc_col1:
        chosen_stat = st.selectbox(
            "Select Statistic to Test",
            options=TARGETS,
            index=0,
            format_func=lambda s: f"{s} ({stat_descriptions.get(s, '')})",
        )
        curr_dist = distributions[chosen_stat]
        curr_point = pred_stats.get(chosen_stat, 0.0)

        default_line = round(curr_point + 0.5, 1)

        chosen_line = st.number_input(
            f"Set {chosen_stat} Threshold Line",
            min_value=0.0,
            max_value=100.0,
            value=float(default_line),
            step=0.5,
            format="%.1f",
            help="Supports standard half-point lines (e.g. 24.5) and integer push lines (e.g. 25.0).",
        )

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

    line_res: LineProbabilityResult = curr_dist.probability_for_line(chosen_line)
    p_over = line_res.prob_over
    p_under = line_res.prob_under
    p_push = line_res.prob_push
    is_int_line = line_res.is_integer_line

    with calc_col2:
        st.markdown(f"#### Probability for **{chosen_stat} Line: {chosen_line:.1f}**")

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

        st.caption(
            f"Peer Context: **{line_res.fallback_tier}** ({line_res.n_support:,} comparable games) | "
            f"Calibration: `{line_res.calibration_status}`"
        )

    # ── Section 3: Distribution Chart ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📈 Simulated Outcome Distribution: {meta.get('player_name', selected_player)} ({chosen_stat})")

    outcomes = curr_dist.outcomes
    max_val = int(np.percentile(outcomes, 99.5)) + 4
    bins = np.arange(-0.5, max_val + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(10, 4.0))

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

    ax.axvline(curr_point, color="#1a73e8", linestyle="--", linewidth=2, label=f"LightGBM Median ({curr_point:.1f})")
    ax.axvline(curr_dist.distribution_mean, color="#8e24aa", linestyle="-.", linewidth=2, label=f"Simulated Mean ({curr_dist.distribution_mean:.1f})")
    ax.axvline(chosen_line, color="#000000", linestyle="-", linewidth=2.5, label=f"Line ({chosen_line:.1f})")

    ax.set_title(f"{meta.get('player_name', selected_player)} — {chosen_stat} Probability Distribution ({n_sims:,} Draws)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Actual {chosen_stat}", fontsize=10)
    ax.set_ylabel("Probability (%)", fontsize=10)
    ax.set_xlim(-0.5, max_val + 0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Section 4: Multi-Stat Quick Consensus Table ──────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Multi-Stat Quick Consensus Table")
    st.caption("Common lines evaluated simultaneously across all 7 statistics.")

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

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Section 5: Ask a Question Box ─────────────────────────────────────────
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
            st.warning("Could not identify both a target stat (e.g. points, assists) and a threshold number. Try: 'Will he score over 20.5 points?'")


if __name__ == "__main__":
    main()
