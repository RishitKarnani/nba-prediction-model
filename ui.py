"""
ui.py — NBA Prediction Model · Streamlit UI
Run with:  streamlit run ui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Prediction Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Team data ─────────────────────────────────────────────────────────────────
TEAMS = {
    "ATL": "Atlanta Hawks",   "BOS": "Boston Celtics",    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets","CHI": "Chicago Bulls",    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks","DEN": "Denver Nuggets",    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors","HOU": "Houston Rockets","IND": "Indiana Pacers",
    "LAC": "LA Clippers",    "LAL": "Los Angeles Lakers","MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",     "MIL": "Milwaukee Bucks",   "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans","NYK": "New York Knicks","OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",  "PHI": "Philadelphia 76ers","PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings","SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors","UTA": "Utah Jazz",         "WAS": "Washington Wizards",
}

TEAM_EMOJI = {
    "ATL": "🦅","BOS": "🍀","BKN": "🌐","CHA": "🐝","CHI": "🐂","CLE": "⚔️",
    "DAL": "🤠","DEN": "🏔️","DET": "🏎️","GSW": "🌉","HOU": "🚀","IND": "🏎️",
    "LAC": "⛵","LAL": "👑","MEM": "🐻","MIA": "🌴","MIL": "🦌","MIN": "🐺",
    "NOP": "⚜️","NYK": "🗽","OKC": "⚡","ORL": "✨","PHI": "🔔","PHX": "🌵",
    "POR": "🌲","SAC": "💜","SAS": "🤠","TOR": "🦖","UTA": "🏔️","WAS": "🧙",
}

TEAM_OPTIONS = [f"{abbr} — {name}" for abbr, name in sorted(TEAMS.items(), key=lambda x: x[1])]

def abbr_from_option(opt: str) -> str:
    return opt.split(" — ")[0]

def tier_icon(tier: str) -> str:
    if "HOT" in tier:  return "🔥"
    if "COLD" in tier: return "❄️"
    return "〰️"


# ── Cached engine loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading prediction engine (first run takes ~10s)…")
def load_engine():
    from models.team_inference import TeamPredictionEngine
    return TeamPredictionEngine(apply_form=True)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Winner card */
.winner-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    border: 1px solid #3730a3;
    margin-bottom: 24px;
}
.winner-name { font-size: 2.4rem; font-weight: 800; color: #fbbf24; }
.score-line  { font-size: 3rem; font-weight: 700; color: #f1f5f9; letter-spacing: 4px; margin: 10px 0; }
.spread-line { color: #94a3b8; font-size: 1rem; }
/* Team header */
.team-header {
    background: #0f172a;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #1e3a5f;
    margin-bottom: 8px;
}
/* Stat chip */
.chip {
    display: inline-block;
    background: #1e293b;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 2px 3px;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 NBA Predictor")
    st.caption("Tactical Simulation + LightGBM")
    st.divider()

    st.markdown("### 🏟️ Matchup")
    home_opt = st.selectbox(
        "🏠 Home Team", TEAM_OPTIONS,
        index=TEAM_OPTIONS.index("LAL — Los Angeles Lakers"),
    )
    away_opt = st.selectbox(
        "✈️ Away Team", TEAM_OPTIONS,
        index=TEAM_OPTIONS.index("BOS — Boston Celtics"),
    )
    home_abbr = abbr_from_option(home_opt)
    away_abbr = abbr_from_option(away_opt)

    st.divider()
    st.markdown("### 🏥 Injuries")
    st.caption("Comma-separated names of OUT players")
    inj_home_raw = st.text_input(
        f"❌ {home_abbr} injuries",
        placeholder="e.g. Luka Doncic, Anthony Davis",
    )
    inj_away_raw = st.text_input(
        f"❌ {away_abbr} injuries",
        placeholder="e.g. Stephen Curry",
    )

    st.divider()
    st.markdown("### ⚙️ Options")
    apply_form = st.toggle("📈 Apply Form Factor", value=True,
                           help="50% Last-5 · 30% Last-10 · 20% Season")
    n_sims = st.select_slider("🎲 Simulations", options=[10, 20, 50, 100], value=10)

    st.divider()
    predict_btn = st.button("🔮  Predict Game", type="primary", use_container_width=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("🏀 NBA Game Prediction Model")
st.caption("Monte Carlo simulation · LightGBM lineup minutes · Form factor · Injury engine")

if home_abbr == away_abbr:
    st.error("⚠️ Select two different teams.")
    st.stop()

# Parse injuries
inj_home = [n.strip() for n in inj_home_raw.split(",") if n.strip()]
inj_away = [n.strip() for n in inj_away_raw.split(",") if n.strip()]

# ── Matchup header ─────────────────────────────────────────────────────────────
col_a, col_vs, col_b = st.columns([3, 1, 3])
with col_a:
    st.markdown(f"### {TEAM_EMOJI.get(away_abbr,'🏀')}  {TEAMS[away_abbr]}")
    st.caption(f"Away · **{away_abbr}**")
    for p in inj_away:
        st.markdown(f"<span style='color:#ef4444'>❌ {p}</span>", unsafe_allow_html=True)
with col_vs:
    st.markdown("<div style='text-align:center;padding-top:16px;font-size:1.8rem;color:#475569'>@</div>",
                unsafe_allow_html=True)
with col_b:
    st.markdown(f"### {TEAM_EMOJI.get(home_abbr,'🏀')}  {TEAMS[home_abbr]}")
    st.caption(f"Home · **{home_abbr}**")
    for p in inj_home:
        st.markdown(f"<span style='color:#ef4444'>❌ {p}</span>", unsafe_allow_html=True)

st.divider()

# ── RUN PREDICTION ────────────────────────────────────────────────────────────
if predict_btn:
    engine = load_engine()

    with st.spinner(f"Running {n_sims} simulations…"):
        t0 = time.time()
        try:
            result = engine.predict_game(
                home_team=home_abbr,
                away_team=away_abbr,
                n_simulations=n_sims,
                injuries_home=inj_home,
                injuries_away=inj_away,
            )
            # Reapply form flag in result if engine was cached with different setting
            st.session_state["result"]  = result
            st.session_state["elapsed"] = round(time.time() - t0, 2)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# ── DISPLAY RESULT ────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:70px 20px;color:#475569">
        <div style="font-size:5rem">🏀</div>
        <h2 style="color:#94a3b8">Select a matchup and click <em>Predict Game</em></h2>
    </div>""", unsafe_allow_html=True)
    # Feature pill row
    p1, p2, p3, p4 = st.columns(4)
    p1.info("**📈 Form Factor**\n50% L5 · 30% L10 · 20% Season")
    p2.success("**🏥 Injury Engine**\nAuto-redistributes minutes")
    p3.warning("**🎯 Tactical Matrix**\n5 Off vs 5 Def coverages")
    p4.error("**🤖 LightGBM**\n66.78% backtest accuracy")
    st.stop()

result  = st.session_state["result"]
elapsed = st.session_state.get("elapsed", "?")
pred    = result["prediction"]
h, a    = result["home_team"], result["away_team"]

winner_abbr  = pred["winner_abbreviation"]
winner_name  = pred["predicted_winner"]
home_prob    = pred["home_win_probability"]
away_prob    = pred["away_win_probability"]
score_h      = pred["predicted_score"][h]
score_a      = pred["predicted_score"][a]

# ── Winner banner ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="winner-card">
  <div style="color:#94a3b8;font-size:0.8rem;letter-spacing:3px;text-transform:uppercase">PREDICTED WINNER</div>
  <div class="winner-name">{TEAM_EMOJI.get(winner_abbr,'🏆')} {winner_name}</div>
  <div class="score-line">{h} {score_h:.0f} — {score_a:.0f} {a}</div>
  <div class="spread-line">Spread: <strong style="color:#fbbf24">{pred['implied_spread']}</strong>
    &nbsp;·&nbsp; Margin: <strong style="color:#fbbf24">{pred['predicted_margin']}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Win probability ────────────────────────────────────────────────────────────
st.markdown("#### 📊 Win Probability")
c1, c2 = st.columns(2)
with c1:
    st.metric(f"🏠 {h}  (Home)", f"{home_prob*100:.1f}%")
    st.progress(home_prob)
with c2:
    st.metric(f"✈️ {a}  (Away)", f"{away_prob*100:.1f}%")
    st.progress(away_prob)

st.caption(f"⏱ {elapsed}s · {n_sims} simulations · Form: {'✅ ON' if result.get('form_applied') else '❌ OFF'}")
st.divider()

# ── Detail tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Team Form", "🏥 Injuries", "🎯 Tactics", "👥 Lineup"])

# ── TAB 1 · Form ──────────────────────────────────────────────────────────────
with tab1:
    form_data = result.get("form", {})
    if not any(form_data.values()):
        st.info("Form data not available. Make sure Form Factor toggle is ON.")
    else:
        for team_abbr, lbl in [(h, "HOME"), (a, "AWAY")]:
            f = form_data.get(team_abbr, {})
            if not f:
                continue
            icon = tier_icon(f.get("tier", ""))
            with st.expander(f"{icon} **{TEAMS[team_abbr]}** ({team_abbr}) · {lbl}", expanded=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Last 5 Net", f"{f.get('roll5_net', 0):+.1f}")
                m2.metric("Last 10 Net", f"{f.get('roll10_net', 0):+.1f}")
                m3.metric("Season Net", f"{f.get('season_net', 0):+.1f}")
                m4.metric("Form-Weighted ✅", f"{f.get('form_net', 0):+.1f}")

                x1, x2 = st.columns(2)
                x1.metric("Form Win %", f"{f.get('form_win_pct', 0)*100:.1f}%")
                x2.metric("Ortg Nudge Applied", f"{f.get('ortg_adjustment', 0):+.1f} pts")

# ── TAB 2 · Injuries ──────────────────────────────────────────────────────────
with tab2:
    injuries   = result.get("injuries", {})
    any_injury = False

    for team_abbr, lbl in [(h, "HOME"), (a, "AWAY")]:
        inj_sum = injuries.get(team_abbr, {})
        recs    = inj_sum.get("injuries", [])
        if not recs:
            continue
        any_injury = True
        st.subheader(f"⛔ {TEAMS[team_abbr]} ({lbl})")
        for rec in recs:
            r1, r2, r3 = st.columns(3)
            r1.metric("Player OUT", rec["player_name"])
            r2.metric("Minutes Vacated", f"{rec['vacated_minutes']:.1f} min")
            r3.metric("Pts Contribution Lost", f"{rec['pts_contribution']:.1f} pts/game")

        st.markdown("**Net Impact:**")
        i1, i2, i3 = st.columns(3)
        i1.metric("Total Pts Lost",     f"{inj_sum.get('total_pts_lost',0):.1f}")
        i2.metric("Truly Unreplaced",   f"{inj_sum.get('unreplaced_pts',0):.1f}")
        i3.metric("Ortg Δ",             f"{inj_sum.get('ortg_adjustment',0):+.1f}",
                  delta_color="inverse")
        st.caption(f"Minutes sanity: {inj_sum.get('minutes_sanity_check',0):.1f} / 240.0 ✓")
        st.divider()

    if not any_injury:
        st.info("No injuries entered. Add player names in the sidebar under **Injuries**.")

# ── TAB 3 · Tactics ───────────────────────────────────────────────────────────
with tab3:
    for team_abbr, lbl, tactics_key in [(h, "HOME", "home_team"), (a, "AWAY", "away_team")]:
        tactics = result["tactical_gameplan"][tactics_key]
        with st.expander(f"🎯 **{TEAMS[team_abbr]}** ({lbl})", expanded=True):
            tc1, tc2 = st.columns(2)
            tc1.success(f"**Top Offensive Play:**\n\n{tactics['best_offensive_action']}")
            tc2.info(f"**Top Defensive Scheme:**\n\n{tactics['best_defensive_action']}")

            st.markdown("**Offensive Weights:**")
            for action, weight in tactics["adjusted_offensive_weights"].items():
                cols = st.columns([4, 1])
                cols[0].markdown(f"<small>{action}</small>", unsafe_allow_html=True)
                cols[1].markdown(f"**{weight*100:.1f}%**")
                st.progress(weight)

            st.markdown("**Defensive Weights:**")
            for action, weight in tactics["adjusted_defensive_weights"].items():
                cols = st.columns([4, 1])
                cols[0].markdown(f"<small>{action}</small>", unsafe_allow_html=True)
                cols[1].markdown(f"**{weight*100:.1f}%**")
                st.progress(weight)

# ── TAB 4 · Lineup ────────────────────────────────────────────────────────────
with tab4:
    import pandas as pd
    lineup_data = result["lineup_predicted_minutes"]

    for team_abbr, lbl, key in [(h, "HOME", "home_team"), (a, "AWAY", "away_team")]:
        st.subheader(f"👥 {TEAMS[team_abbr]} ({lbl}) — Predicted Minutes")
        players = lineup_data[key]
        rows = []
        for p in players:
            is_out = p.get("status") == "OUT"
            rows.append({
                "Player":       ("❌ " if is_out else "✅ ") + p["player_name"],
                "Role":         p.get("role", "?"),
                "Min Allocated":p["allocated_minutes"],
                "Min Share":    f"{p.get('minute_ratio',0)*100:.1f}%",
                "PPG":          p.get("historical_pts_per_game", 0),
                "Status":       "OUT" if is_out else "Active",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Player":        st.column_config.TextColumn(width="large"),
                "Min Allocated": st.column_config.NumberColumn(format="%.1f min"),
                "PPG":           st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(f"Total players shown: {len(players)}")
        st.divider()
