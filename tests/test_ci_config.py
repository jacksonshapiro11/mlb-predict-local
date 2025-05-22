import os

def test_ci_config():
    ci_path = '.github/workflows/ci.yml'
    assert os.path.isfile(ci_path), "CI configuration file not found"
    content = open(ci_path).read()
    assert 'lint-test:' in content, "Job 'lint-test' missing in CI configuration"
