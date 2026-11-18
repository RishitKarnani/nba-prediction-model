"""
src/tactical_registry.py
────────────────────────
Registry of all 30 NBA teams with their 5 Offensive Actions and 5 Defensive Coverages.
Maps each action to a tactical domain archetype and implements the Tactical Counter Matrix
to calculate expected Points Per 100 Possessions (PPP) deviations in head-to-head matchups.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ── 30-TEAM TACTICAL ACTION REGISTRY ─────────────────────────────────────────

TEAM_TACTICAL_ACTIONS: Dict[str, Dict[str, Any]] = {
    # ── EASTERN CONFERENCE ────────────────────────────────────────────────────
    "ATL": {
        "team_name": "Atlanta Hawks",
        "conference": "Eastern",
        "offensive_actions": [
            "High Spread PnR",
            "Early Transition Push",
            "Lob / Vertical Roll-Man Finishes",
            "Spain PnR (Back-screen on the big)",
            "Corner Spot-Up Kickouts",
        ],
        "defensive_actions": [
            "Drop Coverage (protecting rim)",
            "Weak-Side Low-Man Tags",
            "Aggressive Passing-Lane Stunts",
            "ICE on Side Ball Screens",
            "2–3 Zone Changeups",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Transition_Early",
            "Glass_Rim_Cuts",
            "PnR_Complex",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Help_Rotations",
            "Ball_Pressure_Trap",
            "ICE_Sideline",
            "Zone_Hybrid",
        ],
    },
    "BOS": {
        "team_name": "Boston Celtics",
        "conference": "Eastern",
        "offensive_actions": [
            "5-Out Drive & Kick",
            "Guard-Guard Inverted PnR",
            "Post-Up Mismatch Hunting",
            "Ghost Screens (slipping before contact)",
            "High-Volume Above-the-Break 3s",
        ],
        "defensive_actions": [
            "Switch 1–5 (with center rim help)",
            '"X-Out" Scramble Rotations',
            "Single-Coverage Post Fronting",
            "Top-Locking Shooters off Screens",
            "Late-Clock Isolation Funneling",
        ],
        "action_archetypes_off": [
            "Drive_Kick_5Out",
            "PnR_Complex",
            "Post_Elbow_Hub",
            "PnR_Complex",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Help_Rotations",
            "Help_Rotations",
            "Ball_Pressure_Trap",
            "ICE_Sideline",
        ],
    },
    "BKN": {
        "team_name": "Brooklyn Nets",
        "conference": "Eastern",
        "offensive_actions": [
            "Spread PnR Ball Handler",
            "Horns Set Entries",
            "DHOs into Perimeter Re-Screens",
            "Weak-Side Stagger Screens",
            "Transition Wing Lanes",
        ],
        "defensive_actions": [
            "Switch 1–4",
            "Aggressive Blitz / Trap on Stars",
            "Peel Switching on Drives",
            "High Drop with Length",
            "Full-Court Ball Pressure",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "PnR_Complex",
            "Motion_DHO",
            "Motion_DHO",
            "Transition_Early",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Drop_Coverage",
            "Ball_Pressure_Trap",
        ],
    },
    "CHA": {
        "team_name": "Charlotte Hornets",
        "conference": "Eastern",
        "offensive_actions": [
            "Early Drag Screens in Transition",
            "High Ball Screens (30+ feet out)",
            "Elbow DHOs",
            "Baseline Lob Cuts",
            "Pull-Up 3s off Screens",
        ],
        "defensive_actions": [
            "Conservative Drop Coverage",
            "Gap Help / Digging on Drives",
            "Weak-Side Zone-Up Rotations",
            "ICE Side Screens",
            "Occasional 3–2 Zone",
        ],
        "action_archetypes_off": [
            "Transition_Early",
            "PnR_Spread",
            "Motion_DHO",
            "Glass_Rim_Cuts",
            "PnR_Spread",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Help_Rotations",
            "Help_Rotations",
            "ICE_Sideline",
            "Zone_Hybrid",
        ],
    },
    "CHI": {
        "team_name": "Chicago Bulls",
        "conference": "Eastern",
        "offensive_actions": [
            '"Chicago" Action (Pindown + DHO)',
            "High-Tempo Secondary Break",
            "Midrange Isolation / Elbow Touches",
            "Short-Roll Playmaking",
            "Flare Screens for Shooters",
        ],
        "defensive_actions": [
            "Up-to-Touch / High Drop",
            "Point-of-Attack Ball Pressure",
            "Aggressive Dig-Downs on Post/Drives",
            "Scramble Closeouts",
            "Transition Stop-the-Ball Walls",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Transition_Early",
            "Isolation_Mismatch",
            "PnR_Spread",
            "Motion_DHO",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Ball_Pressure_Trap",
            "Help_Rotations",
            "Help_Rotations",
            "Help_Rotations",
        ],
    },
    "CLE": {
        "team_name": "Cleveland Cavaliers",
        "conference": "Eastern",
        "offensive_actions": [
            "Surgical Spread PnR",
            "Double-Big High-Low Sets",
            "Spain PnR",
            "Quick-Hit Pocket Passes to Rollers",
            "Weak-Side Hammer Screens",
        ],
        "defensive_actions": [
            "Elite Rim-Protecting Drop",
            "Two-Big Rim Deterrence / Tagging",
            "Guard Over-the-Screen Chase",
            "Late-Switch on Perimeter Screens",
            "Paint-Packing Gap Help",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Post_Elbow_Hub",
            "PnR_Complex",
            "PnR_Spread",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Drop_Coverage",
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Help_Rotations",
        ],
    },
    "DET": {
        "team_name": "Detroit Pistons",
        "conference": "Eastern",
        "offensive_actions": [
            "High Pick-and-Roll Creation",
            "Post-Up & Elbow Hub Touches",
            "DHOs into Downhill Drives",
            "Short-Roll Kickouts",
            "Offensive Rebound Putbacks",
        ],
        "defensive_actions": [
            "Physical Point-of-Attack Denial",
            "High-Wall / Drop Hybrid",
            "Aggressive Weak-Side Rim Rotations",
            "ICE Side Ball Screens",
            "Switch 1–4",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Post_Elbow_Hub",
            "Motion_DHO",
            "PnR_Spread",
            "Glass_Rim_Cuts",
        ],
        "action_archetypes_def": [
            "Ball_Pressure_Trap",
            "Drop_Coverage",
            "Help_Rotations",
            "ICE_Sideline",
            "Switch_Perimeter",
        ],
    },
    "IND": {
        "team_name": "Indiana Pacers",
        "conference": "Eastern",
        "offensive_actions": [
            "Ultra-Fast Early Transition",
            "Flow / Random Motion DHOs",
            "Spread PnR with Shooting Bigs",
            "45-Cuts & Backdoor Slips",
            "Quick-Trigger Spot-Up 3s",
        ],
        "defensive_actions": [
            "Full-Court Pickup / Ball Pressure",
            "At-the-Level Hedge & Recover",
            "Aggressive Stunts to Force Turnovers",
            "Scramble Closeouts",
            "Next / Peel Switching",
        ],
        "action_archetypes_off": [
            "Transition_Early",
            "Motion_DHO",
            "PnR_Complex",
            "Motion_DHO",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Ball_Pressure_Trap",
            "Ball_Pressure_Trap",
            "Ball_Pressure_Trap",
            "Help_Rotations",
            "Switch_Perimeter",
        ],
    },
    "MIA": {
        "team_name": "Miami Heat",
        "conference": "Eastern",
        "offensive_actions": [
            '"Zoom" & Elbow DHOs',
            "Delay Action (5-Out at top of key)",
            "Split Cuts off Post/Elbow Touches",
            "Flare & Exit Screens",
            "Post-Up Mismatch Iso",
        ],
        "defensive_actions": [
            "Switch 1–5",
            "Signature 2–3 / Matchup Zone",
            "Blitz / Trap Primary Ball Handlers",
            '"Scram" Switching (big out of mismatch)',
            "Aggressive Baseline Traps",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Motion_DHO",
            "Motion_DHO",
            "Motion_DHO",
            "Post_Elbow_Hub",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Zone_Hybrid",
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
        ],
    },
    "MIL": {
        "team_name": "Milwaukee Bucks",
        "conference": "Eastern",
        "offensive_actions": [
            "Transition Hit-Ahead & Euro-Step Drives",
            "Inverted Guard-Big PnR",
            "Post-Up / Elbow Hub Actions",
            "Wide Pindowns & DHOs",
            "5-Out Spot-Up Spacing",
        ],
        "defensive_actions": [
            "Deep Drop Coverage",
            "Weak-Side Help / Rim Protection",
            "No-Middle / Baseline Funneling",
            "Late-Clock Switch 1–4",
            "Box-Out & Rebound Finishes",
        ],
        "action_archetypes_off": [
            "Transition_Early",
            "PnR_Complex",
            "Post_Elbow_Hub",
            "Motion_DHO",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Help_Rotations",
            "ICE_Sideline",
            "Switch_Perimeter",
            "Help_Rotations",
        ],
    },
    "NYK": {
        "team_name": "New York Knicks",
        "conference": "Eastern",
        "offensive_actions": [
            "High Spread PnR",
            "Isolation Mismatch Hunting",
            "Corner-Crash Offensive Rebounding",
            "Short-Roll / Floater Pocket Plays",
            "Horns Twist (double ball screens)",
        ],
        "defensive_actions": [
            "Switch 1–4 / Wing Interchangeability",
            "Conservative Drop with Rim Protector",
            "ICE Side Screens",
            "Top-Locking Off-Ball Shooters",
            "Heavy Strong-Side Help / Stunts",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Isolation_Mismatch",
            "Glass_Rim_Cuts",
            "PnR_Spread",
            "PnR_Complex",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Drop_Coverage",
            "ICE_Sideline",
            "Ball_Pressure_Trap",
            "Help_Rotations",
        ],
    },
    "ORL": {
        "team_name": "Orlando Magic",
        "conference": "Eastern",
        "offensive_actions": [
            "DHOs into Downhill Wing Drives",
            "Post-Ups for Big Wings",
            "High PnR with Short-Roll Reads",
            "Baseline Cuts & Offensive Glass Crashes",
            "Horns Flare Sets",
        ],
        "defensive_actions": [
            "Switch 1–5 Length",
            "Aggressive Passing-Lane Deflections",
            "High-Show / Hedge on Ball Screens",
            "Peel Switching on Drives",
            "Transition Wall-Up Defense",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Post_Elbow_Hub",
            "PnR_Spread",
            "Glass_Rim_Cuts",
            "PnR_Complex",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Help_Rotations",
        ],
    },
    "PHI": {
        "team_name": "Philadelphia 76ers",
        "conference": "Eastern",
        "offensive_actions": [
            "Elbow / Nail Post-Up Isolation",
            "Two-Man DHO / Pitch Game",
            "Spread PnR (Pop & Roll)",
            "Clear-Out Wing Isolations",
            "Weak-Side Flare Screens",
        ],
        "defensive_actions": [
            "Drop Coverage (protecting paint)",
            "Aggressive Dig-Downs / Stunts",
            "Top-Locking Perimeter Shooters",
            "Late-Shot-Clock Switching",
            "2–3 Zone Changeups",
        ],
        "action_archetypes_off": [
            "Post_Elbow_Hub",
            "Motion_DHO",
            "PnR_Spread",
            "Isolation_Mismatch",
            "Motion_DHO",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Help_Rotations",
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Zone_Hybrid",
        ],
    },
    "TOR": {
        "team_name": "Toronto Raptors",
        "conference": "Eastern",
        "offensive_actions": [
            "Early Transition Pitch-Aheads",
            '"Delay" 5-Out DHOs',
            "High-Post / Elbow Split Cuts",
            "Baseline & 45-Cuts",
            "Crash-the-Glass Second Chances",
        ],
        "defensive_actions": [
            "Switch Everything (Positionless)",
            "Aggressive Blitz / Trap on Screens",
            '"X-Out" Long Closeout Rotations',
            "Full-Court Press / Ball Pressure",
            "Passing-Lane Gamble / Deflections",
        ],
        "action_archetypes_off": [
            "Transition_Early",
            "Motion_DHO",
            "Motion_DHO",
            "Motion_DHO",
            "Glass_Rim_Cuts",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Help_Rotations",
            "Ball_Pressure_Trap",
            "Ball_Pressure_Trap",
        ],
    },
    "WAS": {
        "team_name": "Washington Wizards",
        "conference": "Eastern",
        "offensive_actions": [
            "High-Pace Transition Push",
            "Spread PnR Ball Handler",
            "Quick DHOs into Drives",
            "Early Shot-Clock Spot-Up 3s",
            "Weak-Side Stagger Screens",
        ],
        "defensive_actions": [
            "Drop Coverage",
            "Switch 1–4",
            "Gap Help / Stunting at Drivers",
            "ICE Side Ball Screens",
            "Scramble Closeouts",
        ],
        "action_archetypes_off": [
            "Transition_Early",
            "PnR_Spread",
            "Motion_DHO",
            "Drive_Kick_5Out",
            "Motion_DHO",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Switch_Perimeter",
            "Help_Rotations",
            "ICE_Sideline",
            "Help_Rotations",
        ],
    },

    # ── WESTERN CONFERENCE ────────────────────────────────────────────────────
    "DAL": {
        "team_name": "Dallas Mavericks",
        "conference": "Western",
        "offensive_actions": [
            "High Spread PnR (1-on-1 Mismatch)",
            "Lob / Vertical Spacer Dive Plays",
            "Isolation Clear-Outs",
            "Corner Skip-Pass Spot-Ups",
            "Horns Out / Double-Drag Screens",
        ],
        "defensive_actions": [
            "Drop Coverage with Rim Protectors",
            "Weak-Side Low-Man Tags",
            "Switch 1–4 on Perimeter",
            "Funnel Drivers to Shot Blockers",
            "Late-Clock Blitzes",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Glass_Rim_Cuts",
            "Isolation_Mismatch",
            "Drive_Kick_5Out",
            "PnR_Complex",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Help_Rotations",
            "Switch_Perimeter",
            "Drop_Coverage",
            "Ball_Pressure_Trap",
        ],
    },
    "DEN": {
        "team_name": "Denver Nuggets",
        "conference": "Western",
        "offensive_actions": [
            "Two-Man Elbow DHO / Get Game",
            "Post-Up Hub & Backdoor Cuts",
            "Dunker-Spot Baseline Seals",
            "High PnR (Short-Roll Playmaking)",
            '"Sombor Shuffle" / Late-Clock Iso',
        ],
        "defensive_actions": [
            "Up-to-Touch / High Hedge",
            '"Scram" Switching (keeping big out of iso)',
            "Weak-Side Zone-Up Help",
            "ICE Side Screens",
            "Stunt & Recover on Perimeter",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Post_Elbow_Hub",
            "Glass_Rim_Cuts",
            "PnR_Spread",
            "Isolation_Mismatch",
        ],
        "action_archetypes_def": [
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Help_Rotations",
            "ICE_Sideline",
            "Help_Rotations",
        ],
    },
    "GSW": {
        "team_name": "Golden State Warriors",
        "conference": "Western",
        "offensive_actions": [
            "Split-Cut Action off Post/Elbow",
            "Relocate & Flare Screens off DHOs",
            "Step-Up / Slip Ball Screens",
            '"Elevator" & Stagger Double Screens',
            "Early Transition 3-Point Pitch-Aheads",
        ],
        "defensive_actions": [
            "Switch 1–4",
            "Up-to-Touch / Show & Recover",
            "Weak-Side Help / Box-Out Discipline",
            "Peel Switching on Drives",
            "Box-and-1 / Triangle-and-2 Junk Zones",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Motion_DHO",
            "PnR_Complex",
            "Motion_DHO",
            "Transition_Early",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Help_Rotations",
            "Switch_Perimeter",
            "Zone_Hybrid",
        ],
    },
    "HOU": {
        "team_name": "Houston Rockets",
        "conference": "Western",
        "offensive_actions": [
            "High PnR with Short-Roll Hub",
            "Post-Up & Elbow Playmaking",
            "Aggressive Offensive Rebound Crashes",
            '"Zoom" DHO Actions',
            "Transition Rim Runs",
        ],
        "defensive_actions": [
            "Switch 1–5 Physicality",
            "Aggressive Ball Pressure / Denials",
            "High-Show / Hedge on Screens",
            "Weak-Side Low-Man Rim Help",
            "ICE Side Screens",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Post_Elbow_Hub",
            "Glass_Rim_Cuts",
            "Motion_DHO",
            "Transition_Early",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Ball_Pressure_Trap",
            "Help_Rotations",
            "ICE_Sideline",
        ],
    },
    "LAC": {
        "team_name": "LA Clippers",
        "conference": "Western",
        "offensive_actions": [
            "Wing / Mid-Post Isolation",
            "Spread PnR Ball Handler",
            "Horns Twist / Double Ball Screens",
            "Weak-Side Stagger Screens",
            "Spot-Up Kickouts",
        ],
        "defensive_actions": [
            "Switch 1–4",
            "Drop Coverage with Bigs",
            "ICE Side Ball Screens",
            "Heavy Gap Dig-Downs",
            "Late-Clock Traps",
        ],
        "action_archetypes_off": [
            "Isolation_Mismatch",
            "PnR_Spread",
            "PnR_Complex",
            "Motion_DHO",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Drop_Coverage",
            "ICE_Sideline",
            "Help_Rotations",
            "Ball_Pressure_Trap",
        ],
    },
    "LAL": {
        "team_name": "Los Angeles Lakers",
        "conference": "Western",
        "offensive_actions": [
            "Spread PnR Mismatch Hunting",
            "High-Low / Elbow Post Entries",
            '"Chicago" (Pindown + DHO) Sets',
            "Transition Rim Pressure",
            "Spain PnR / Baseline Cuts",
        ],
        "defensive_actions": [
            "Drop Coverage (funneling to rim)",
            "Switch 1–4",
            "Weak-Side Zone-Up Help",
            "2–3 Zone Changeups",
            "Low-Man Tagging on Rollers",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Post_Elbow_Hub",
            "Motion_DHO",
            "Transition_Early",
            "PnR_Complex",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Switch_Perimeter",
            "Help_Rotations",
            "Zone_Hybrid",
            "Help_Rotations",
        ],
    },
    "MEM": {
        "team_name": "Memphis Grizzlies",
        "conference": "Western",
        "offensive_actions": [
            "High-Pace Transition Push",
            '"Delay" 5-Out Motion & DHOs',
            "Spread PnR with Hard Rim Rolls",
            "45-Cuts & Backdoor Slips",
            "Offensive Rebound Putbacks",
        ],
        "defensive_actions": [
            "Drop Coverage with Roaming Help-Side Big",
            "Aggressive Passing-Lane Stunts",
            "ICE Side Screens",
            "Guard Over-the-Screen Chase",
            "Transition Wall-Up",
        ],
        "action_archetypes_off": [
            "Transition_Early",
            "Motion_DHO",
            "PnR_Spread",
            "Motion_DHO",
            "Glass_Rim_Cuts",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Ball_Pressure_Trap",
            "ICE_Sideline",
            "Ball_Pressure_Trap",
            "Help_Rotations",
        ],
    },
    "MIN": {
        "team_name": "Minnesota Timberwolves",
        "conference": "Western",
        "offensive_actions": [
            "High PnR with Lob Threat",
            "Double-Drag Transition Screens",
            "Elbow / Post-Up Entries",
            "Wing Isolation Drives",
            "Corner Spot-Up Spacing",
        ],
        "defensive_actions": [
            "Elite Deep Drop (funneling to rim)",
            "Point-of-Attack Screen Navigation",
            "Two-Big Rim Protection / Tagging",
            "Switch 1–4 with Length",
            'Weak-Side "X-Out" Closeouts',
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "PnR_Complex",
            "Post_Elbow_Hub",
            "Isolation_Mismatch",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Ball_Pressure_Trap",
            "Drop_Coverage",
            "Switch_Perimeter",
            "Help_Rotations",
        ],
    },
    "NOP": {
        "team_name": "New Orleans Pelicans",
        "conference": "Western",
        "offensive_actions": [
            "Point-Forward Isolation / Bull-Drives",
            "Inverted Guard-Big PnR",
            "DHOs into Downhill Attacks",
            "Dunker-Spot Cuts & Putbacks",
            "Corner Kickout 3s",
        ],
        "defensive_actions": [
            "Switch 1–5 Length",
            "Aggressive Ball-Pressure / Denial",
            "Passing-Lane Deflections",
            "Peel Switching on Drives",
            "High Hedge & Recover",
        ],
        "action_archetypes_off": [
            "Isolation_Mismatch",
            "PnR_Complex",
            "Motion_DHO",
            "Glass_Rim_Cuts",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Ball_Pressure_Trap",
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
        ],
    },
    "OKC": {
        "team_name": "Oklahoma City Thunder",
        "conference": "Western",
        "offensive_actions": [
            "5-Out Clear-Side Isolation",
            "Guard-Guard / Inverted PnR",
            "Ghost Screens (slipping into space)",
            "Drive-and-Kick 0.5-Second Passing",
            "Turnovers-to-Transition Buckets",
        ],
        "defensive_actions": [
            "Switch Everything / Swarming Help",
            "Aggressive Gap Dig-Downs & Strips",
            '"Scram" & Peel Switching',
            "Drop with Rim-Protecting Big",
            'High-Speed "X-Out" Closeouts',
        ],
        "action_archetypes_off": [
            "Isolation_Mismatch",
            "PnR_Complex",
            "PnR_Complex",
            "Drive_Kick_5Out",
            "Transition_Early",
        ],
        "action_archetypes_def": [
            "Switch_Perimeter",
            "Help_Rotations",
            "Switch_Perimeter",
            "Drop_Coverage",
            "Help_Rotations",
        ],
    },
    "PHX": {
        "team_name": "Phoenix Suns",
        "conference": "Western",
        "offensive_actions": [
            "Spread PnR (Midrange Pull-Up Reads)",
            '"Spain" & Stack PnR Actions',
            "Elbow Isolation / Clear-Outs",
            "Weak-Side Flare & Exit Screens",
            "Corner Spot-Up Spacing",
        ],
        "defensive_actions": [
            "Drop Coverage",
            "Switch 1–4",
            "Weak-Side Low-Man Tags",
            "ICE Side Ball Screens",
            "Scramble Closeouts",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "PnR_Complex",
            "Post_Elbow_Hub",
            "Motion_DHO",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Switch_Perimeter",
            "Help_Rotations",
            "ICE_Sideline",
            "Help_Rotations",
        ],
    },
    "POR": {
        "team_name": "Portland Trail Blazers",
        "conference": "Western",
        "offensive_actions": [
            "High Spread PnR Ball Handler",
            "Transition Hit-Ahead Drives",
            "Wing Isolation / Step-Backs",
            "DHOs into Re-Screens",
            "Offensive Glass Putbacks",
        ],
        "defensive_actions": [
            "Drop Coverage",
            "Switch 1–4 with Athletic Wings",
            "Full-Court Ball Pressure",
            "ICE Side Screens",
            "Aggressive Passing-Lane Stunts",
        ],
        "action_archetypes_off": [
            "PnR_Spread",
            "Transition_Early",
            "Isolation_Mismatch",
            "Motion_DHO",
            "Glass_Rim_Cuts",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "ICE_Sideline",
            "Ball_Pressure_Trap",
        ],
    },
    "SAC": {
        "team_name": "Sacramento Kings",
        "conference": "Western",
        "offensive_actions": [
            'High-Post / Elbow DHOs ("Zoom")',
            "Split Cuts & Backdoor Slips",
            "Early Transition Pitch-Aheads",
            "Spread PnR (Short-Roll Playmaking)",
            "Relocate & Drift Corner 3s",
        ],
        "defensive_actions": [
            "Up-to-Touch / High Show on Screens",
            "Aggressive Dig-Downs on Drives",
            "Switch 1–4",
            "Weak-Side Low-Man Tags",
            "Scramble Closeouts",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Motion_DHO",
            "Transition_Early",
            "PnR_Spread",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Ball_Pressure_Trap",
            "Help_Rotations",
            "Switch_Perimeter",
            "Help_Rotations",
            "Help_Rotations",
        ],
    },
    "SAS": {
        "team_name": "San Antonio Spurs",
        "conference": "Western",
        "offensive_actions": [
            "High PnR with Vertical/Pick-and-Pop Big",
            '"Loop" & Motion DHOs',
            "High-Low Post Entries",
            "Weak-Side Stagger & Flare Screens",
            "Transition Lob Finishes",
        ],
        "defensive_actions": [
            "Drop Coverage (elite rim deterrence)",
            "Switch 1–4 on Perimeter",
            "Aggressive Guard Chase Over Screens",
            "Late-Clock Zone Changeups",
            "Weak-Side Tag & Recover",
        ],
        "action_archetypes_off": [
            "PnR_Complex",
            "Motion_DHO",
            "Post_Elbow_Hub",
            "Motion_DHO",
            "Transition_Early",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Switch_Perimeter",
            "Ball_Pressure_Trap",
            "Zone_Hybrid",
            "Help_Rotations",
        ],
    },
    "UTA": {
        "team_name": "Utah Jazz",
        "conference": "Western",
        "offensive_actions": [
            "5-Out Motion & DHOs",
            "Weak-Side Stagger & Flare Screens",
            "High PnR with Pick-and-Pop Bigs",
            "Baseline & 45-Cuts",
            "High-Volume Spot-Up 3s",
        ],
        "defensive_actions": [
            "Drop Coverage (funneling to rim)",
            "Weak-Side Zone-Up Help",
            "ICE Side Ball Screens",
            "Switch 1–4",
            "2–3 Zone Changeups",
        ],
        "action_archetypes_off": [
            "Motion_DHO",
            "Motion_DHO",
            "PnR_Complex",
            "Motion_DHO",
            "Drive_Kick_5Out",
        ],
        "action_archetypes_def": [
            "Drop_Coverage",
            "Help_Rotations",
            "ICE_Sideline",
            "Switch_Perimeter",
            "Zone_Hybrid",
        ],
    },
}

# ── TACTICAL SYNERGY / COUNTER EFFICIENCY MATRIX ─────────────────────────────
# Net Offensive Efficiency impact (Points Per 100 Possessions deviation)
# when an offensive archetype meets an opponent's defensive scheme.
TACTICAL_COUNTER_MATRIX: Dict[Tuple[str, str], float] = {
    # PnR_Spread (High spread ball screens)
    ("PnR_Spread", "Drop_Coverage"): -1.5,
    ("PnR_Spread", "Switch_Perimeter"): 2.0,
    ("PnR_Spread", "Ball_Pressure_Trap"): -2.5,
    ("PnR_Spread", "Help_Rotations"): 0.5,
    ("PnR_Spread", "ICE_Sideline"): -3.0,
    ("PnR_Spread", "Zone_Hybrid"): 1.0,

    # PnR_Complex (Spain PnR, Ghost screens, Inverted PnR, Horns)
    ("PnR_Complex", "Drop_Coverage"): 5.5,     # Backscreen / slip destroys drop big
    ("PnR_Complex", "Switch_Perimeter"): 0.5,
    ("PnR_Complex", "Ball_Pressure_Trap"): 3.0,
    ("PnR_Complex", "Help_Rotations"): 2.0,
    ("PnR_Complex", "ICE_Sideline"): 1.5,
    ("PnR_Complex", "Zone_Hybrid"): 2.5,

    # Motion_DHO (Split cuts, Zoom DHO, Chicago action, Delay 5-out)
    ("Motion_DHO", "Drop_Coverage"): 3.5,     # Pulls drop big into curling 3s
    ("Motion_DHO", "Switch_Perimeter"): -3.0,  # Switches easily blow up handoffs
    ("Motion_DHO", "Ball_Pressure_Trap"): -1.0,
    ("Motion_DHO", "Help_Rotations"): 1.5,
    ("Motion_DHO", "ICE_Sideline"): 0.5,
    ("Motion_DHO", "Zone_Hybrid"): 4.0,       # Motion shreds zone shifts

    # Drive_Kick_5Out (5-Out, Corner spot-ups, high volume above-the-break 3s)
    ("Drive_Kick_5Out", "Drop_Coverage"): 6.0,  # Center stuck in paint, wide open 3s
    ("Drive_Kick_5Out", "Switch_Perimeter"): -1.0,
    ("Drive_Kick_5Out", "Ball_Pressure_Trap"): 1.5,
    ("Drive_Kick_5Out", "Help_Rotations"): -2.0, # Good X-out closeouts suppress 3s
    ("Drive_Kick_5Out", "ICE_Sideline"): 2.0,
    ("Drive_Kick_5Out", "Zone_Hybrid"): 3.5,

    # Post_Elbow_Hub (High-low, Elbow hubs, mismatch post-ups)
    ("Post_Elbow_Hub", "Drop_Coverage"): 2.0,
    ("Post_Elbow_Hub", "Switch_Perimeter"): 5.0, # Punishes small guard switches
    ("Post_Elbow_Hub", "Ball_Pressure_Trap"): -3.5, # Dig-downs force turnovers
    ("Post_Elbow_Hub", "Help_Rotations"): -2.5,
    ("Post_Elbow_Hub", "ICE_Sideline"): 1.0,
    ("Post_Elbow_Hub", "Zone_Hybrid"): 1.5,

    # Transition_Early (Hit-aheads, early push, fast secondary break)
    ("Transition_Early", "Drop_Coverage"): 4.0,  # Defense not set in drop
    ("Transition_Early", "Switch_Perimeter"): 1.0,
    ("Transition_Early", "Ball_Pressure_Trap"): -4.0, # Full-court pressure halts break
    ("Transition_Early", "Help_Rotations"): -2.5,
    ("Transition_Early", "ICE_Sideline"): 1.0,
    ("Transition_Early", "Zone_Hybrid"): 3.0,

    # Glass_Rim_Cuts (Lob finishes, dunker spot, offensive glass putbacks)
    ("Glass_Rim_Cuts", "Drop_Coverage"): -3.0,  # Drop big controls paint & rebounds
    ("Glass_Rim_Cuts", "Switch_Perimeter"): 2.5, # Rebound mismatches on glass
    ("Glass_Rim_Cuts", "Ball_Pressure_Trap"): 3.5, # Cuts behind traps
    ("Glass_Rim_Cuts", "Help_Rotations"): -1.5,
    ("Glass_Rim_Cuts", "ICE_Sideline"): -1.0,
    ("Glass_Rim_Cuts", "Zone_Hybrid"): 2.0,

    # Isolation_Mismatch (1-on-1 clear outs, bull drives)
    ("Isolation_Mismatch", "Drop_Coverage"): 1.0,
    ("Isolation_Mismatch", "Switch_Perimeter"): -2.5,
    ("Isolation_Mismatch", "Ball_Pressure_Trap"): -4.5, # Double teams smother iso
    ("Isolation_Mismatch", "Help_Rotations"): -2.0,
    ("Isolation_Mismatch", "ICE_Sideline"): 0.5,
    ("Isolation_Mismatch", "Zone_Hybrid"): -3.5,
}


def get_action_delta(off_archetype: str, def_archetype: str) -> float:
    """Return the Points Per 100 Possessions delta for an offensive vs defensive archetype."""
    return TACTICAL_COUNTER_MATRIX.get((off_archetype, def_archetype), 0.0)


def evaluate_tactical_matchup(
    team_a_abbrev: str,
    team_b_abbrev: str,
    weights_a_off: List[float],
    weights_b_def: List[float],
) -> float:
    """
    Calculate the net Tactical Offensive Rating delta (per 100 possessions)
    for Team A attacking Team B with given action probability distributions.
    """
    a_tactics = TEAM_TACTICAL_ACTIONS[team_a_abbrev]
    b_tactics = TEAM_TACTICAL_ACTIONS[team_b_abbrev]

    a_arch_off = a_tactics["action_archetypes_off"]
    b_arch_def = b_tactics["action_archetypes_def"]

    net_delta = 0.0
    for i, w_off in enumerate(weights_a_off):
        for j, w_def in enumerate(weights_b_def):
            synergy = get_action_delta(a_arch_off[i], b_arch_def[j])
            net_delta += w_off * w_def * synergy
    return net_delta
