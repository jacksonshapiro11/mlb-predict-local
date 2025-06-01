import os
import polars as pl
from pathlib import Path
import subprocess

def test_write_parquet(tmp_path):
    # Create a test CSV file
    csv_data = """game_date,pitch_type,events,release_speed,release_spin_rate,hc_x,hc_y,stand,p_throws,home_team,extra_col
2020-01-01,FF,hit,95.0,2200,100.0,400.0,R,R,NYY,extra1
2020-01-02,SL,out,85.0,2400,200.0,500.0,L,L,BOS,extra2
2020-01-03,CH,walk,80.0,1800,300.0,600.0,R,R,NYY,extra3"""
    
    # Create input and output directories
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    
    # Write test CSV
    csv_file = in_dir / "statcast_2020.csv"
    csv_file.write_text(csv_data)
    
    # Run the script
    result = subprocess.run(
        ["python3", "backend/scripts/csv2parquet.py", 
         f"--in_dir={in_dir}", f"--out_dir={out_dir}"],
        capture_output=True,
        text=True
    )
    
    # Assert script ran successfully
    assert result.returncode == 0
    
    # Check if Parquet file exists
    parquet_file = out_dir / "2020" / "part_0.parquet"
    assert parquet_file.exists()
    
    # Read back the Parquet file
    df = pl.read_parquet(parquet_file)
    
    # Assert row count and schema
    assert len(df) == 3
    assert len(df.columns) == 10
    required_columns = [
        'game_date', 'pitch_type', 'events', 'release_speed',
        'release_spin_rate', 'hc_x', 'hc_y', 'stand', 'p_throws', 'home_team'
    ]
    assert all(col in df.columns for col in required_columns) 