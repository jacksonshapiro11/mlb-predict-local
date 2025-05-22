import black
import ruff

def test_tools_import():
    assert hasattr(black, "__version__")
    assert hasattr(ruff, "__version__")
