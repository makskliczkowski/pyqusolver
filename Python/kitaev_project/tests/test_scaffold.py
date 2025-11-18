"""
Smoke tests for the scaffold components.
"""
from pathlib import Path
import sys

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _ROOT / "Python"
for candidate in (_ROOT, _PYTHON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from kitaev_project.io import KitaevResultWriter
from kitaev_project.models import ModelConfig
from kitaev_project.nqs.types import TrainingArtifact


def test_model_config_summary():
    cfg = ModelConfig(lx=2, ly=2, lz=1)
    summary = cfg.summary()
    assert summary["lx"] == 2
    assert summary["K"]["z"] == 1.0


def test_writer_creates_file(tmp_path: Path):
    artifact = TrainingArtifact(
        ansatz_name="test",
        stage="ground_state",
        hdf5_path="",
        metrics={"foo": 1.0},
        observables={"energy": 0.0},
        network_state={"net": "spec"},
    )
    writer = KitaevResultWriter(path=tmp_path / "results.h5")
    out_path = writer.write_training_run(artifact)
    assert Path(out_path).exists()
