import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_list_modules_returns_stable_sorted_records():
    # Arrange
    registry = importlib.import_module("QES.registry")

    # Act
    modules = registry.list_modules(include_submodules=True)

    # Assert
    assert modules, "Expected non-empty module inventory"
    assert [m["name"] for m in modules] == sorted(m["name"] for m in modules)
    for entry in modules:
        assert set(entry) == {"name", "path", "description"}
        assert entry["path"].startswith("QES.")


def test_describe_module_curated_and_unknown_paths():
    # Arrange
    registry = importlib.import_module("QES.registry")

    # Act
    algebra_desc = registry.describe_module("Algebra")
    unknown_desc = registry.describe_module("does.not.exist")

    # Assert
    assert isinstance(algebra_desc, str)
    assert algebra_desc.strip() != ""
    assert unknown_desc == "No description available."
