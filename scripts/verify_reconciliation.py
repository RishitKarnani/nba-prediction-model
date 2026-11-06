import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from database.connection import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    sql_teams = dict(conn.execute(text("SELECT SEASON, COUNT(*) FROM dbo.stg_team_game_logs GROUP BY SEASON ORDER BY SEASON")).fetchall())
    sql_players = dict(conn.execute(text("SELECT SEASON, COUNT(*) FROM dbo.stg_player_game_logs GROUP BY SEASON ORDER BY SEASON")).fetchall())

raw_dir = "data/raw"
seasons = sorted(list(sql_teams.keys()))

print(f"{'Season':<10} | {'Pq Teams':<10} | {'SQL Teams':<10} | {'Team Status':<12} | {'Pq Players':<10} | {'SQL Players':<11} | {'Player Status':<12}")
print("-" * 90)

all_match = True
for s in seasons:
    t_pq = len(pd.read_parquet(os.path.join(raw_dir, "team_logs", f"team_logs_{s}.parquet")))
    p_pq = len(pd.read_parquet(os.path.join(raw_dir, "player_logs", f"player_logs_{s}.parquet")))
    t_sql = sql_teams.get(s, 0)
    p_sql = sql_players.get(s, 0)
    
    t_match = t_pq == t_sql
    p_match = p_pq == p_sql
    if not (t_match and p_match):
        all_match = False
        
    t_status = "EXACT MATCH" if t_match else "MISMATCH"
    p_status = "EXACT MATCH" if p_match else "MISMATCH"
    print(f"{s:<10} | {t_pq:<10,d} | {t_sql:<10,d} | {t_status:<12} | {p_pq:<10,d} | {p_sql:<11,d} | {p_status:<12}")

print("-" * 90)
total_team_pq = sum(len(pd.read_parquet(os.path.join(raw_dir, "team_logs", f"team_logs_{s}.parquet"))) for s in seasons)
total_team_sql = sum(sql_teams.values())
total_player_pq = sum(len(pd.read_parquet(os.path.join(raw_dir, "player_logs", f"player_logs_{s}.parquet"))) for s in seasons)
total_player_sql = sum(sql_players.values())
print(f"{'TOTAL':<10} | {total_team_pq:<10,d} | {total_team_sql:<10,d} | {'100% MATCH' if total_team_pq == total_team_sql else 'MISMATCH'} | {total_player_pq:<10,d} | {total_player_sql:<11,d} | {'100% MATCH' if total_player_pq == total_player_sql else 'MISMATCH'}")
