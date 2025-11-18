"""
Declarative description of the HDF5 layout used across the project.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KitaevHDF5Schema:
    """
    Simple namespace describing where different artefacts live inside the HDF5 file.
    """

    root_group: str = "/"
    training_group: str = "/nqs/training"
    excited_group: str = "/nqs/excited"
    ed_group: str = "/ed"
    metadata_group: str = "/metadata"

    def training_dataset(self, stage: str) -> str:
        return f"{self.training_group}/{stage}"

    def excited_dataset(self, label: str) -> str:
        return f"{self.excited_group}/{label}"

    def ed_dataset(self, label: str) -> str:
        return f"{self.ed_group}/{label}"
