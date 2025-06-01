import os
import subprocess
import sys
import pytest
from unittest.mock import patch

def test_downloader_script():
    # Test years
    start_year = 2020
    end_year = 2021
    
    # Expected curl commands
    expected_commands = [
        f"curl -L -o backend/data/raw/statcast_{year}.csv https://baseballsavant-downloads.s3.amazonaws.com/dataverse/statcast_{year}.csv"
        for year in range(start_year, end_year + 1)
    ]
    
    # Mock subprocess.run to capture commands
    captured_commands = []
    
    def mock_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[0] == "curl":
            captured_commands.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0)
    
    # Run the script with mocked subprocess
    with patch("subprocess.run", side_effect=mock_run):
        result = subprocess.run(
            ["./backend/scripts/get_dataverse_csvs.sh", str(start_year), str(end_year)],
            capture_output=True,
            text=True
        )
        
        # Verify script executed successfully
        assert result.returncode == 0
        
        # Verify curl commands were constructed correctly
        assert len(captured_commands) == len(expected_commands)
        for cmd, expected in zip(captured_commands, expected_commands):
            assert cmd == expected

def test_downloader_bad_args():
    # Test with invalid number of arguments
    result = subprocess.run(
        ["./backend/scripts/get_dataverse_csvs.sh", "2020"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "Usage:" in result.stdout
    
    # Test with invalid year format
    result = subprocess.run(
        ["./backend/scripts/get_dataverse_csvs.sh", "20", "2021"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "Years must be 4-digit numbers" in result.stdout
    
    # Test with start year > end year
    result = subprocess.run(
        ["./backend/scripts/get_dataverse_csvs.sh", "2022", "2021"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "START_YEAR must be less than or equal to END_YEAR" in result.stdout

@pytest.mark.parametrize("years,expected", [
    (("2020", "2021"), [
        'curl -L -o backend/data/raw/statcast_2020.csv https://baseballsavant-downloads.s3.amazonaws.com/dataverse/statcast_2020.csv',
        'curl -L -o backend/data/raw/statcast_2021.csv https://baseballsavant-downloads.s3.amazonaws.com/dataverse/statcast_2021.csv',
    ]),
])
def test_downloader(monkeypatch, tmp_path, years, expected):
    # Setup raw dir
    raw_dir = tmp_path / "backend/data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "backend/scripts/get_dataverse_csvs.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    # Write a minimal script for test
    script.write_text("""#!/bin/bash\necho $@\n""")
    os.chmod(script, 0o755)
    calls = []
    def fake_run(cmd, *a, **k):
        if isinstance(cmd, list) and cmd[0] == "curl":
            calls.append(" ".join(cmd))
        class Result: returncode = 0
        return Result()
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Simulate missing files
    for y in range(int(years[0]), int(years[1])+1):
        f = raw_dir / f"statcast_{y}.csv"
        if y == int(years[0]):
            continue  # Simulate missing file
        f.write_text('dummy')  # Simulate existing file
    # Run
    for y in range(int(years[0]), int(years[1])+1):
        if not (raw_dir / f"statcast_{y}.csv").exists():
            subprocess.run([
                "curl", "-L", "-o", str(raw_dir / f"statcast_{y}.csv"),
                f"https://baseballsavant-downloads.s3.amazonaws.com/dataverse/statcast_{y}.csv"
            ])
    # Check
    assert calls == expected 