import pytest

try:
	from QES.Solver.MonteCarlo.sampler import resolve_state_defaults
except ImportError:
	pytest.skip("Required sampler module is not available.", allow_module_level=True)


def test_resolve_state_defaults_accepts_hyphenated_binary_alias():
	spin, mode_repr = resolve_state_defaults(
		state_representation="binary-01",
		spin=None,
		mode_repr=None,
		fallback_mode_repr=0.5,
	)

	assert spin is False
	assert mode_repr == pytest.approx(1.0)
