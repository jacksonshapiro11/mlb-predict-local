import subprocess
import os


def test_csv2parquet_cli():
    # Create test directories
    os.makedirs("out", exist_ok=True)

    # Run the script
    result = subprocess.run(
        ["python3", "backend/scripts/csv2parquet.py", "--in_dir=.", "--out_dir=./out"],
        capture_output=True,
        text=True,
    )

    # Assert exit code is 0 and output is "OK"
    assert result.returncode == 0
    assert result.stdout.strip() == "OK"
