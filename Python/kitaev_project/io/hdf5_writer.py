"""
HDF5 writer adhering to the shared schema.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from .hdf5_schema import KitaevHDF5Schema


class KitaevResultWriter:
    def __init__(self, path: Optional[Path] = None, schema: Optional[KitaevHDF5Schema] = None):
        self.path = Path(path or "kitaev_results.h5")
        self.schema = schema or KitaevHDF5Schema()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _ensure_group(self, h5file: h5py.File, group_path: str) -> h5py.Group:
        return h5file.require_group(group_path)

    def _write_dict(self, group: h5py.Group, prefix: str, payload: Dict):
        target = group.require_group(prefix)
        for key, value in payload.items():
            if isinstance(value, dict):
                self._write_dict(target, key, value)
            else:
                data = np.array(value)
                if key in target:
                    del target[key]
                target.create_dataset(key, data=data)

    # ------------------------------------------------------------------

    def write_training_run(self, artifact) -> str:
        """
        Append a training artifact to the HDF5 store.
        """
        with h5py.File(self.path, "a") as h5:
            grp = self._ensure_group(h5, self.schema.training_dataset(artifact.stage))
            grp.attrs["ansatz"] = artifact.ansatz_name
            grp.attrs["timestamp"] = datetime.utcnow().isoformat()
            self._write_dict(grp, "metrics", artifact.metrics)
            self._write_dict(grp, "observables", artifact.observables)
            self._write_dict(grp, "network_state", artifact.network_state)
            self._write_dict(grp, "extras", artifact.extras)
        return str(self.path)

    def write_ed_result(self, label: str, energies: np.ndarray, metadata: Dict[str, object]):
        with h5py.File(self.path, "a") as h5:
            grp = self._ensure_group(h5, self.schema.ed_dataset(label))
            grp.attrs["timestamp"] = datetime.utcnow().isoformat()
            if "energies" in grp:
                del grp["energies"]
            grp.create_dataset("energies", data=np.asarray(energies))
            self._write_dict(grp, "metadata", metadata)

    def write_metadata(self, payload: Dict[str, object]):
        with h5py.File(self.path, "a") as h5:
            grp = self._ensure_group(h5, self.schema.metadata_group)
            self._write_dict(grp, "global", payload)
