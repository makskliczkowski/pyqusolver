"""
Result containers for NQS time-domain and frequency-domain spectral workflows.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class NQSTDVPRecord:
    r"""TDVP trajectory sampled on a user-provided time grid."""

    times: np.ndarray
    param_history: np.ndarray
    global_phase: np.ndarray
    mean_energy: np.ndarray
    std_energy: np.ndarray
    sigma2: np.ndarray
    r_hat: np.ndarray
    num_samples: int
    num_chains: int
    shapes: Optional[list[Any]] = None
    sizes: Optional[list[int]] = None
    is_complex: Optional[list[bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NQSCorrelatorResult:
    r"""Time-domain dynamical correlator evaluated from NQS states."""

    times: np.ndarray
    correlator: np.ndarray
    trajectory: Optional[NQSTDVPRecord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NQSSpectralResult:
    """Frequency-domain spectral estimate derived from an NQS correlator."""

    times: np.ndarray
    correlator: np.ndarray
    frequencies: np.ndarray
    spectrum: np.ndarray
    spectrum_complex: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NQSSpectralMapResult:
    """Momentum-resolved spectral map."""

    times: np.ndarray
    correlator: np.ndarray
    frequencies: np.ndarray
    spectrum: np.ndarray
    spectrum_complex: np.ndarray
    k_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSSpectralResult",
    "NQSTDVPRecord",
]
