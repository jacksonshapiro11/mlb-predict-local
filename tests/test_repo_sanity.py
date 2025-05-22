import os

def test_repo_sanity():
    assert os.path.isfile('docker-compose.yml'), "docker-compose.yml not found"
