"""
Driver-facing callbacks and loggers for NQS training.

The goal is to provide a small, explicit control surface similar to the one
used by NetKet drivers without changing the existing TDVP/VMC core.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np


class StopTraining(RuntimeError):
	"""Signal graceful early termination of the training loop."""

class AbstractLog:
	"""Minimal logger interface for variational drivers."""

	def __call__(self, step: int, item: Dict[str, Any], trainer: Any = None) -> None:
		raise NotImplementedError

	def flush(self, trainer: Any = None) -> None:
		return None

	def serialize(self, path: Union[str, Path]) -> None:
		raise NotImplementedError

@dataclass
class RuntimeLog(AbstractLog):
	"""In-memory training logger."""

	data: Dict[str, List[Any]] = field(default_factory=dict)

	def __call__(self, step: int, item: Dict[str, Any], trainer: Any = None) -> None:
		record = {"step": int(step)}
		record.update(item)
		for key, value in record.items():
			self.data.setdefault(key, []).append(value)

	def flush(self, trainer: Any = None) -> None:
		return None

	def serialize(self, path: Union[str, Path]) -> None:
		with Path(path).open("w", encoding="utf-8") as handle:
			json.dump(_to_jsonable(self.data), handle, indent=2)

@dataclass
class JsonLog(RuntimeLog):
	"""Runtime log that periodically flushes to JSON."""

	output_prefix	: Union[str, Path] = "nqs_run"
	write_every		: int = 50

	def __post_init__(self) -> None:
		self.output_prefix = Path(self.output_prefix)
		self.output_prefix.parent.mkdir(parents=True, exist_ok=True)

	def __call__(self, step: int, item: Dict[str, Any], trainer: Any = None) -> None:
		super().__call__(step, item, trainer=trainer)
		if self.write_every > 0 and ((step + 1) % self.write_every == 0):
			self.flush(trainer=trainer)

	def flush(self, trainer: Any = None) -> None:
		self.serialize(self.output_prefix.with_suffix(".json"))

@dataclass
class InvalidLossStopping:
	"""Stop if the monitored quantity is invalid for a given patience."""

	monitor: str = "mean"
	patience: int = 0
	_invalid_steps: int = 0

	def __call__(self, step: int, log_data: Dict[str, Any], trainer: Any) -> bool:
		value = log_data.get(self.monitor, None)
		try:
			valid = value is not None and math.isfinite(float(value))
		except Exception:
			valid = False
		if valid:
			self._invalid_steps = 0
			return True
		self._invalid_steps += 1
		return self._invalid_steps <= self.patience


@dataclass
class ConvergenceStopping:
	"""Stop when the monitored quantity stays below a threshold."""

	threshold: float
	monitor: str = "std"
	patience: int = 0
	_below_steps: int = 0

	def __call__(self, step: int, log_data: Dict[str, Any], trainer: Any) -> bool:
		value = log_data.get(self.monitor, None)
		try:
			below = value is not None and float(value) <= float(self.threshold)
		except Exception:
			below = False
		if below:
			self._below_steps += 1
		else:
			self._below_steps = 0
		return self._below_steps <= self.patience


@dataclass
class TimeoutStopping:
	"""Stop after a wall-clock timeout."""

	timeout: float
	_t0: Optional[float] = None

	def __call__(self, step: int, log_data: Dict[str, Any], trainer: Any) -> bool:
		if self._t0 is None:
			self._t0 = time.time()
		return (time.time() - self._t0) < float(self.timeout)


def normalize_loggers(out: Optional[Union[str, AbstractLog, Iterable[AbstractLog]]]) -> List[AbstractLog]:
	"""Normalize logger specifications to concrete logger objects."""
	if out is None:
		return []
	if isinstance(out, (str, Path)):
		return [JsonLog(output_prefix=out)]
	if isinstance(out, AbstractLog):
		return [out]
	if isinstance(out, Iterable):
		loggers = list(out)
		for logger in loggers:
			if not isinstance(logger, AbstractLog):
				raise TypeError("All loggers passed to `out` must implement AbstractLog.")
		return loggers
	raise TypeError("`out` must be None, a path-like prefix, a logger, or an iterable of loggers.")


def normalize_callbacks(
	callback: Optional[Union[Callable[[int, Dict[str, Any], Any], bool], Iterable[Callable[[int, Dict[str, Any], Any], bool]]]]
) -> List[Callable[[int, Dict[str, Any], Any], bool]]:
	"""Normalize callback specifications to a flat callback list."""
	if callback is None:
		return []
	if callable(callback):
		return [callback]
	if isinstance(callback, Iterable):
		callbacks = list(callback)
		for cb in callbacks:
			if not callable(cb):
				raise TypeError("All callbacks must be callable.")
		return callbacks
	raise TypeError("`callback` must be None, a callable, or an iterable of callables.")


def _to_jsonable(value: Any) -> Any:
	"""Convert nested driver data to JSON-serializable objects."""
	if isinstance(value, dict):
		return {str(k): _to_jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(v) for v in value]
	if isinstance(value, complex):
		return {"real": float(np.real(value)), "imag": float(np.imag(value))}
	if isinstance(value, np.ndarray):
		return _to_jsonable(value.tolist())
	if isinstance(value, np.generic):
		return _to_jsonable(value.item())
	return value


__all__ = [
	"AbstractLog",
	"RuntimeLog",
	"JsonLog",
	"StopTraining",
	"InvalidLossStopping",
	"ConvergenceStopping",
	"TimeoutStopping",
	"normalize_loggers",
	"normalize_callbacks",
]
