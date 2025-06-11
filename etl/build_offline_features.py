#!/usr/bin/env python
"""
Historic Statcast feature builder
---------------------------------
• Pitcher arsenal (9 canonical + OTHER) with V/Spin/Whiff + USAGE_TD + USAGE_30
• Batter success vs pitch-type (XWOBA_TD / XWOBA_30)
• Count-state whiff/contact rates (30-day)
• 7-day recent-form block (whiff-rate, velo, hit-rate)
• Pitcher L/R platoon splits (30-day)
• Batter season-form (K_PCT_TD / 30D)
• Raw context cols, park-factor hook, float32 + Z-std Parquet
Usage:
    python etl/build_offline_features.py 2021
"""

import sys
import os
import time
import logging
import duckdb as db
import pandas as pd
import numpy as np
from pybaseball import statcast, playerid_lookup
from pybaseball.cache import enable

enable()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ETL")

# -------- constants & helpers -------- #
def canon(pt: str) -> str:
    """Canonical pitch types - consistent with historical features"""
    CANONICAL_MAPPING = {
        'FF': 'FF',    # Four-seam fastball
        'SI': 'SI',    # Sinker  
        'SL': 'SL',    # Slider
        'CH': 'CH',    # Changeup
        'CU': 'CU',    # Curveball
        'FC': 'FC',    # Cutter
        'KC': 'KC',    # Knuckle curve
        'FS': 'FS',    # Splitter
        'ST': 'ST',    # Sweeper (modern slider variant)
        'SV': 'SL',    # Slurve → map to Slider
        'FA': 'FF',    # Fastball → map to Four-seam
        'FT': 'SI',    # Two-seam → map to Sinker
        'CS': 'CU',    # Slow curve → map to Curveball
        'SW': 'ST',    # Sweeper alias
    }
    return CANONICAL_MAPPING.get(pt, "OTHER")


def fetch_statcast(start: str, end: str, retries=3, delay=5):
    """Download Statcast with simple retry logic."""
    for i in range(1, retries + 1):
        try:
            log.info(f"Statcast {start} → {end}  (try {i}/{retries})")
            df = statcast(start, end, verbose=True)
            if df is None or df.empty:
                raise ValueError("empty frame")
            return df
        except Exception as e:
            if i == retries:
                raise
            log.warning(f"{e} – retrying in {delay}s")
            time.sleep(delay)


# -------------- main ------------------ #
def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
    start, end = f"{year}-01-01", f"{year}-12-31"

    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)

    # ---- player ID x-walk (once) ---- #
    xwalk_pq = "data/reference/player_xwalk.parquet"
    if not os.path.exists(xwalk_pq):
        log.info("Building MLBAM ⇄ Fangraphs x-walk …")
        playerid_lookup("all", "all")[
            ["key_mlbam", "key_fangraphs"]
        ].drop_duplicates().rename(
            columns={"key_mlbam": "mlbam", "key_fangraphs": "fg"}
        ).to_parquet(
            xwalk_pq
        )
    xwalk = pd.read_parquet(xwalk_pq)

    # ---- raw Statcast pull ---- #
    raw = fetch_statcast(start, end)
    keep = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "inning",
        "inning_topbot",
        "batter",
        "pitcher",
        "stand",
        "p_throws",
        "home_team",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "home_score",
        "away_score",
        "balls",
        "strikes",
        "pitch_type",
        "pitch_name",
        "release_speed",
        "effective_speed",
        "release_extension",
        "release_spin_rate",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "spin_axis",
        "sz_top",
        "sz_bot",
        "zone",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "events",
        "description",
        "delta_home_win_exp",
        "delta_run_exp",
        "home_win_exp",
        "bat_win_exp",
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "hc_x",
        "hc_y",
        "estimated_woba_using_speedangle",
    ]
    raw = raw[[c for c in keep if c in raw.columns]].copy()
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["pitch_type_can"] = raw["pitch_type"].apply(canon)

    raw = (
        raw.merge(xwalk.rename(columns={"mlbam": "batter"}), on="batter", how="left")
        .rename(columns={"fg": "batter_fg"})
        .merge(xwalk.rename(columns={"mlbam": "pitcher"}), on="pitcher", how="left")
        .rename(columns={"fg": "pitcher_fg"})
    )

    # ---- DuckDB session ---- #
    con = db.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.register("p", raw)

    # ---------- batter K% (overall form) ---------- #
    bat_daily = con.execute(
        """
        SELECT
          batter    AS mlbam,
          game_date,
          COUNT(*)                           AS PA,
          SUM(description = 'swinging_strike') AS SO
        FROM p
        GROUP BY batter, game_date
    """
    ).df()
    con.register("bat", bat_daily)
    bat_daily = con.execute(
        """
        SELECT *,
               SUM(PA) OVER w  AS PA_CUM,
               SUM(SO) OVER w  AS SO_CUM,
               SUM(PA) OVER w30 AS PA_30,
               SUM(SO) OVER w30 AS SO_30
        FROM bat
        WINDOW
          w   AS (PARTITION BY mlbam ORDER BY game_date),
          w30 AS (PARTITION BY mlbam ORDER BY game_date
                   RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW)
    """
    ).df()
    bat_daily["K_PCT_TD"] = bat_daily["SO_CUM"] / bat_daily["PA_CUM"].replace(0, np.nan)
    bat_daily["K_PCT_30D"] = bat_daily["SO_30"] / bat_daily["PA_30"].replace(0, np.nan)

    # ---------- pitcher arsenal + USAGE ---------- #
    arsenal = con.execute(
        """
        WITH base AS (
          SELECT
            pitcher AS mlbam,
            pitch_type_can AS pt,
            game_date,
            COUNT(*)                        AS CNT,
            AVG(release_speed)              AS V,
            AVG(release_spin_rate)          AS SPIN,
            SUM(description = 'swinging_strike')::FLOAT / COUNT(*) AS WHIFF
          FROM p
          GROUP BY pitcher, pt, game_date
        )
        SELECT *,
               SUM(CNT)  OVER w   AS CNT_TD,
               SUM(CNT)  OVER w30 AS CNT_30,
               AVG(V)    OVER w   AS V_TD,
               AVG(SPIN) OVER w   AS SPIN_TD,
               AVG(WHIFF)OVER w30 AS WHIFF_30
        FROM base
        WINDOW
          w   AS (PARTITION BY mlbam, pt ORDER BY game_date),
          w30 AS (PARTITION BY mlbam, pt ORDER BY game_date
                   RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW)
    """
    ).df()
    # USAGE ratios via SQL (no pandas transform)
    arsenal = con.execute(
        """
        SELECT *,
               CNT_TD ::FLOAT / SUM(CNT_TD) OVER (PARTITION BY mlbam, game_date)  AS USAGE_TD,
               CNT_30 ::FLOAT / SUM(CNT_30) OVER (PARTITION BY mlbam, game_date)  AS USAGE_30
        FROM arsenal
    """
    ).df()

    wide_pit = arsenal.pivot_table(
        index=["mlbam", "game_date"],
        columns="pt",
        values=["V_TD", "SPIN_TD", "WHIFF_30", "USAGE_TD", "USAGE_30"],
        aggfunc="first",
    ).sort_index()
    wide_pit.columns = [f"{m}_{pt}" for m, pt in wide_pit.columns]
    wide_pit.reset_index(inplace=True)

    # ---------- count-state table ---------- #
    count_metrics = con.execute(
        """
        WITH base AS (
          SELECT
            pitcher AS mlbam,
            game_date,
            CASE
              WHEN balls <= 1 AND strikes <= 1 THEN 'EVEN'
              WHEN balls > strikes THEN 'BEHIND'
              ELSE 'AHEAD' END              AS state,
            COUNT(*)                        AS CNT,
            SUM(description = 'swinging_strike')::FLOAT / COUNT(*)  AS WHIFF,
            SUM(events IN ('single','double','triple','home_run'))::FLOAT / COUNT(*) AS CONTACT
          FROM p
          GROUP BY pitcher, game_date, state
        )
        SELECT *,
               AVG(WHIFF)   OVER w30 AS WHIFF_30,
               AVG(CONTACT) OVER w30 AS CONTACT_30
        FROM base
        WINDOW
          w30 AS (PARTITION BY mlbam, state ORDER BY game_date
                   RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW)
    """
    ).df()
    wide_count = count_metrics.pivot_table(
        index=["mlbam", "game_date"],
        columns="state",
        values=["WHIFF_30", "CONTACT_30"],
        aggfunc="first",
    ).sort_index()
    wide_count.columns = [f"{m}_{s}" for m, s in wide_count.columns]
    wide_count.reset_index(inplace=True)

    # ---------- recent 7-day form ---------- #
    recent_form = con.execute(
        """
        WITH base AS (
        SELECT
            pitcher AS mlbam,
          game_date,
            COUNT(*)                                             AS PITCHES,
            SUM(description = 'swinging_strike')::FLOAT / COUNT(*) AS WHIFF,
            AVG(release_speed)                                   AS V,
            SUM(events IN ('single','double','triple','home_run'))::FLOAT / COUNT(*) AS HIT_RATE
        FROM p
        GROUP BY pitcher, game_date
        )
        SELECT mlbam, game_date,
               AVG(WHIFF)    OVER w7 AS WHIFF_7D,
               AVG(V)        OVER w7 AS VELO_7D,
               AVG(HIT_RATE) OVER w7 AS HIT_7D
        FROM base
        WINDOW
          w7 AS (PARTITION BY mlbam ORDER BY game_date
                  RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
    """
    ).df()

    # ---------- batter xwOBA vs pitch-type ---------- #
    hitter_split = con.execute(
        """
        WITH base AS (
          SELECT
            batter AS mlbam,
            pitch_type_can AS pt,
            game_date,
            AVG(estimated_woba_using_speedangle) AS XWOBA
          FROM p
          GROUP BY batter, pt, game_date
        )
        SELECT *,
               AVG(XWOBA) OVER w   AS XWOBA_TD,
               AVG(XWOBA) OVER w30 AS XWOBA_30
        FROM base
        WINDOW
          w   AS (PARTITION BY mlbam, pt ORDER BY game_date),
          w30 AS (PARTITION BY mlbam, pt ORDER BY game_date
                   RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW)
        """
    ).df()
    wide_hit = hitter_split.pivot_table(
        index=["mlbam", "game_date"],
        columns="pt",
        values=["XWOBA_TD", "XWOBA_30"],
        aggfunc="first",
    ).sort_index()
    wide_hit.columns = [f"{m}_{pt}" for m, pt in wide_hit.columns]
    wide_hit.reset_index(inplace=True)

    # ---------- pitcher platoon splits ---------- #
    platoon = con.execute(
        """
        WITH base AS (
          SELECT
            pitcher AS mlbam,
            game_date,
            stand    AS SIDE,
            COUNT(*) AS CNT,
            AVG(estimated_woba_using_speedangle)                                    AS XWOBA,
            SUM(description = 'swinging_strike')::FLOAT / COUNT(*)                  AS WHIFF,
            SUM(events IN ('single','double','triple','home_run'))::FLOAT / COUNT(*) AS HIT_RATE
            FROM p
          GROUP BY pitcher, game_date, SIDE
        )
        SELECT *,
               AVG(WHIFF)    OVER w30 AS WHIFF_30,
               AVG(HIT_RATE) OVER w30 AS HIT_30,
               AVG(XWOBA)    OVER w30 AS XWOBA_30
        FROM base
        WINDOW
          w30 AS (PARTITION BY mlbam, SIDE ORDER BY game_date
                  RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW)
        """
    ).df()
    wide_platoon = platoon.pivot_table(
        index=["mlbam", "game_date"],
        columns="SIDE",
        values=["WHIFF_30", "HIT_30", "XWOBA_30"],
        aggfunc="first",
    ).sort_index()
    wide_platoon.columns = [f"{m}_VS_{side}" for m, side in wide_platoon.columns]
    wide_platoon.reset_index(inplace=True)

    # ---------- merge everything ---------- #
    data = (
        raw.merge(
            bat_daily[["mlbam", "game_date", "K_PCT_TD", "K_PCT_30D"]],
            left_on=["batter", "game_date"],
            right_on=["mlbam", "game_date"],
            how="left",
        )
        .drop(columns="mlbam")
        .merge(
            wide_pit,
            left_on=["pitcher", "game_date"],
            right_on=["mlbam", "game_date"],
            how="left",
        )
        .drop(columns="mlbam")
        .merge(
            wide_hit,
            left_on=["batter", "game_date"],
            right_on=["mlbam", "game_date"],
            how="left",
        )
        .drop(columns="mlbam")
        .merge(
            wide_count,
            left_on=["pitcher", "game_date"],
            right_on=["mlbam", "game_date"],
            how="left",
        )
        .drop(columns="mlbam")
        .merge(
            recent_form,
            left_on=["pitcher", "game_date"],
            right_on=["mlbam", "game_date"],
            how="left",
        )
        .drop(columns="mlbam")
        .merge(
            wide_platoon,
            left_on=["pitcher", "game_date"],
            right_on=["mlbam", "game_date"],
            how="left",
        )
        .drop(columns="mlbam")
    )

    # ---------- park-factor hook ---------- #
    try:
        parks = pd.read_csv("data/reference/park_factors.csv")  # cols: home_team, PF
        data = data.merge(parks, on="home_team", how="left")
    except FileNotFoundError:
        log.warning("No park_factors.csv – skipping.")

    # ---------- optimise & write ---------- #
    float_cols = data.select_dtypes("float64").columns
    data[float_cols] = data[float_cols].astype("float32")
    for c in ["pitch_type", "pitch_type_can", "stand", "p_throws", "inning_topbot"]:
        if c in data.columns:
            data[c] = data[c].astype("category")

    out = f"data/features/statcast_{year}.parquet"
    data.to_parquet(out, compression="zstd")
    log.info(f"✔  {len(data):,} rows → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
