import duckdb, pathlib, pandas as pd
import os

PARQUET = 'data/features/statcast_2024.parquet'   # pick any season file you already have
paths = f"['{PARQUET}']"

LAG_SQL = f"""
SELECT game_date, pitcher, at_bat_number, pitch_number,
       pitch_type_can,
       LAG(pitch_type_can,1) OVER w AS prev_pt1,
       LAG(pitch_type_can,2) OVER w AS prev_pt2,
       release_speed - LAG(release_speed,1) OVER w AS dvelo1
FROM parquet_scan({paths})
WINDOW w AS (
  PARTITION BY pitcher, game_pk
  ORDER BY at_bat_number, pitch_number
)
LIMIT 10
"""

df = duckdb.query(LAG_SQL).df()
print(df.head())

FEATURES_DIR = 'data/features'
lag_cols = ['prev_pt1', 'prev_pt2']

for fname in sorted(os.listdir(FEATURES_DIR)):
    if fname.startswith('statcast_') and fname.endswith('.parquet'):
        fpath = os.path.join(FEATURES_DIR, fname)
        try:
            df = pd.read_parquet(fpath, columns=lag_cols)
            has_cols = all(col in df.columns for col in lag_cols)
            print(f"{fname}: {'OK' if has_cols else 'MISSING'}")
        except Exception as e:
            print(f"{fname}: ERROR ({e})") 