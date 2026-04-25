import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from QES.general_python.algebra import solvers


def _wide_operator():
	"""Return O with N_params > N_samples, matching CNN/GCNN SR regime."""
	return jnp.array(
		[
			[1.0 + 0.2j, 2.0 - 0.1j, 0.5 + 0.3j, -1.0 + 0.4j, 0.7 - 0.2j],
			[0.3 - 0.5j, -1.5 + 0.2j, 2.0 + 0.1j, 0.2 - 0.7j, -0.8 + 0.6j],
		],
		dtype=jnp.complex64,
	)


def _loss_vector():
	return jnp.array([0.4 - 0.1j, -0.2 + 0.7j], dtype=jnp.complex64)


def _gram_solver(shift):
	solver = solvers.choose_solver(
		"minres_qlp",
		is_gram=True,
		sigma=shift,
		backend="jax",
		maxiter=200,
	)
	return solver.get_solver_func(jnp, use_fisher=True, sigma=shift)


def test_matrix_free_standard_sr_uses_sample_normalization_for_wide_operator():
	O       = _wide_operator()
	loss    = _loss_vector()
	n_s     = O.shape[0]
	shift   = 0.2
	s       = O
	s_p     = O.T.conj()
	b       = s_p @ loss / n_s
	solve   = _gram_solver(shift)

	result = solve(
		s=s,
		s_p=s_p,
		b=b,
		x0=None,
		tol=1e-8,
		maxiter=200,
		precond_apply=None,
		sigma=shift,
		normalization=n_s,
	)
	expected = jnp.linalg.solve(
		s_p @ s / n_s + shift * jnp.eye(O.shape[1], dtype=O.dtype),
		b,
	)

	assert result.converged
	np.testing.assert_allclose(np.asarray(result.x), np.asarray(expected), rtol=2e-5, atol=2e-6)


def test_matrix_free_minsr_uses_sample_normalization_for_transposed_operator():
	O       = _wide_operator()
	loss    = _loss_vector()
	n_s     = O.shape[0]
	shift   = 0.2
	s       = O.T.conj()
	s_p     = O
	b       = loss / n_s
	solve   = _gram_solver(shift)

	result = solve(
		s=s,
		s_p=s_p,
		b=b,
		x0=None,
		tol=1e-8,
		maxiter=200,
		precond_apply=None,
		sigma=shift,
		normalization=n_s,
	)
	expected = jnp.linalg.solve(
		s_p @ s / n_s + shift * jnp.eye(O.shape[0], dtype=O.dtype),
		b,
	)

	assert result.converged
	np.testing.assert_allclose(np.asarray(result.x), np.asarray(expected), rtol=2e-5, atol=2e-6)
