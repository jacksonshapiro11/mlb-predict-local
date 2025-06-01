import importlib


def test_tools_import():
    importlib.import_module("black")
    importlib.import_module("ruff")
