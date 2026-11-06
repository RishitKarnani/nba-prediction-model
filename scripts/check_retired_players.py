import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    query = text("""
        SELECT 
            PLAYER_NAME, 
            COUNT(*) AS total_games,
            MIN(SEASON) AS first_season,
            MAX(SEASON) AS last_season,
            SUM(PTS) AS total_pts,
            ROUND(AVG(PTS), 1) AS ppg
        FROM dbo.stg_player_game_logs
        WHERE PLAYER_NAME IN (
            'Kobe Bryant', 'Tim Duncan', 'Dirk Nowitzki', 
            'Dwyane Wade', 'Vince Carter', 'Carmelo Anthony', 
            'Paul Pierce', 'Kevin Garnett', 'Manu Ginobili', 
            'LaMarcus Aldridge', 'Amar''e Stoudemire', 'Deron Williams'
        )
        GROUP BY PLAYER_NAME
        ORDER BY MAX(SEASON) DESC, COUNT(*) DESC;
    """)
    rows = conn.execute(query).fetchall()
    
    print(f"{'Player Name':<20} | {'Games':<6} | {'First Season':<12} | {'Last Season':<12} | {'Total PTS':<10} | {'PPG':<6}")
    print("-" * 80)
    for r in rows:
        print(f"{r.PLAYER_NAME:<20} | {r.total_games:<6} | {r.first_season:<12} | {r.last_season:<12} | {r.total_pts:<10,.0f} | {r.ppg:<6.1f}")

    # Also check total distinct players in DB vs currently active in 2026-27
    total_distinct = conn.execute(text("SELECT COUNT(DISTINCT PLAYER_ID) FROM dbo.stg_player_game_logs")).scalar()
    active_2026 = conn.execute(text("SELECT COUNT(DISTINCT PLAYER_ID) FROM dbo.stg_player_game_logs WHERE SEASON = '2026-27'")).scalar()
    retired_or_inactive = total_distinct - active_2026
    print("-" * 80)
    print(f"Total Unique Players Across All Seasons (2014-2026): {total_distinct:,}")
    print(f"Players Active in Ongoing 2026-27 Season: {active_2026:,}")
    print(f"Players Who Played Between 2014-2026 but Not Active in 2026-27 (Retired / Overseas): {retired_or_inactive:,}")
