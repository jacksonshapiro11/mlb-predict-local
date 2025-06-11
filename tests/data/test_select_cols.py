import polars as pl
from backend.scripts.csv2parquet import select_columns


def test_select_columns():
    # Create a test DataFrame with extra columns
    df = pl.DataFrame(
        {
            "game_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "pitch_type": ["FF", "SL", "CH"],
            "events": ["hit", "out", "walk"],
            "release_speed": [95.0, 85.0, 80.0],
            "release_spin_rate": [2200, 2400, 1800],
            "hc_x": [100.0, 200.0, 300.0],
            "hc_y": [400.0, 500.0, 600.0],
            "stand": ["R", "L", "R"],
            "p_throws": ["R", "L", "R"],
            "home_team": ["NYY", "BOS", "NYY"],
            "extra_col1": ["a", "b", "c"],  # Extra column
            "extra_col2": [1, 2, 3],  # Extra column
        }
    )

    # Select columns
    result = select_columns(df)

    # Assert we have exactly 10 columns
    assert len(result.columns) == 10

    # Assert all required columns are present
    required_columns = [
        "game_date",
        "pitch_type",
        "events",
        "release_speed",
        "release_spin_rate",
        "hc_x",
        "hc_y",
        "stand",
        "p_throws",
        "home_team",
    ]
    assert all(col in result.columns for col in required_columns)

    # Assert no extra columns are present
    assert not any(col not in required_columns for col in result.columns)
