"""Regression tests for tdvp matrix free regressions."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from QES.general_python.algebra import solvers
from QES.general_python.algebra.solvers.cg import CgSolver, CgSolverScipy
from QES.general_python.algebra.solvers.minres import MinresSolver, MinresSolverScipy
from QES.general_python.algebra.solvers.minres_qlp import MinresQLPSolver
from QES.general_python.algebra.solvers.minsr import SpectralExactSolver
from QES.general_python.algebra.solvers.pseudoinverse import PseudoInverseSolver
from QES.NQS.src.tdvp import TDVP


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
	"""Create vector."""
	return jnp.array([0.4 - 0.1j, -0.2 + 0.7j], dtype=jnp.complex64)


def _gram_solver(shift):
	"""Build Gram solver."""
	solver = solvers.choose_solver(
		"minres_qlp",
		is_gram=True,
		sigma=shift,
		backend="jax",
		maxiter=200,
	)
	return solver.get_solver_func(jnp, use_fisher=True, sigma=shift)


def test_matrix_free_standard_sr_uses_sample_normalization_for_wide_operator():
	"""Verify test matrix free standard sr uses sample normalization for wide operator."""
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
	"""Verify test matrix free minsr uses sample normalization for transposed operator."""
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


@pytest.mark.parametrize(
	"solver_cls",
	[
		CgSolver,
		CgSolverScipy,
		MinresSolver,
		MinresSolverScipy,
		MinresQLPSolver,
		PseudoInverseSolver,
		SpectralExactSolver,
	],
)
def test_jax_fisher_solvers_use_runtime_normalization_for_minsr_shape(solver_cls):
	"""Verify Fisher solvers use Ns normalization when s has transposed MinSR shape."""
	O       = _wide_operator()
	loss    = _loss_vector()
	n_s     = O.shape[0]
	shift   = 0.35
	s       = O.T.conj()
	s_p     = O
	b       = loss / n_s
	solve   = solver_cls.get_solver_func(
		jnp,
		use_fisher=True,
		sigma=shift,
	)

	result = solve(
		s=s,
		s_p=s_p,
		b=b,
		x0=None,
		tol=1e-8,
		maxiter=256,
		precond_apply=None,
		sigma=shift,
		normalization=n_s,
	)
	expected = jnp.linalg.solve(
		s_p @ s / n_s + shift * jnp.eye(O.shape[0], dtype=O.dtype),
		b,
	)

	np.testing.assert_allclose(np.asarray(result.x), np.asarray(expected), rtol=2e-4, atol=2e-5)


def test_tdvp_accepts_only_canonical_solver_form_names():
	"""Verify TDVP solver-form parsing accepts canonical names only."""
	tdvp = TDVP(backend="jax", sr_lin_solver="cg", sr_lin_solver_t="gram")

	assert tdvp._normalize_solver_form("gram") == solvers.SolverForm.GRAM
	assert tdvp._normalize_solver_form("matvec") == solvers.SolverForm.MATVEC
	assert tdvp._normalize_solver_form("matrix") == solvers.SolverForm.MATRIX

	with pytest.raises(ValueError):
		tdvp._normalize_solver_form("full")
	with pytest.raises(ValueError):
		tdvp._normalize_solver_form("mat")


def test_tdvp_reuses_cached_jit_wrappers():
	"""Verify TDVP reuses shared JIT wrappers for SR helper functions."""
	tdvp_a = TDVP(backend="jax", sr_lin_solver="cg", sr_lin_solver_t="gram")
	tdvp_b = TDVP(backend="jax", sr_lin_solver="cg", sr_lin_solver_t="gram")

	assert tdvp_a._gradient_fn_j is tdvp_b._gradient_fn_j
	assert tdvp_a._loss_c_fn_j is tdvp_b._loss_c_fn_j
	assert tdvp_a._deriv_c_fn_j is tdvp_b._deriv_c_fn_j


def test_tdvp_rejects_minsr_without_sr():
	"""Verify MinSR cannot silently use the parameter-space non-SR gradient path."""
	with pytest.raises(ValueError, match="MinSR requires use_sr=True"):
		TDVP(backend="jax", use_sr=False, use_minsr=True)
	tdvp = TDVP(backend="jax", use_sr=False, use_minsr=False)
	with pytest.raises(ValueError, match="MinSR requires use_sr=True"):
		tdvp.set_useminsr(True)


def test_tdvp_projects_batched_minsr_with_reverse_matvec():
	"""Verify matrix-free MinSR projects sample-space coefficients to parameter space."""

	class FakeBatchedJacobian:
		def __init__(self, operator):
			self.operator = operator

		def compute_weighted_sum(self, weights):
			return self.operator.T.conj() @ weights

		def rmv(self, vector):
			return self.operator.T.conj() @ vector

	O = _wide_operator()
	sample_coeffs = jnp.array([0.2 + 0.3j, -0.4 + 0.1j], dtype=jnp.complex64)
	tdvp = TDVP(backend="jax", use_sr=True, use_minsr=True, sr_lin_solver="cg", sr_lin_solver_t="matvec")
	solution = solvers.SolverResult(x=sample_coeffs, converged=True, iterations=3, residual_norm=1e-7)

	projected = tdvp._project_minsr_result(solution, FakeBatchedJacobian(O))

	np.testing.assert_allclose(
		np.asarray(projected.x),
		np.asarray(O.T.conj() @ sample_coeffs),
		rtol=2e-6,
		atol=2e-6,
	)
	assert projected.converged
	assert projected.iterations == solution.iterations


def test_minsr_real_parameter_complex_output_uses_stacked_real_system():
	"""Verify MinSR follows Chen-Heyl real-parameter complex-output equation."""
	O = jnp.array(
		[
			[1.0 + 0.5j, -0.2 + 0.7j],
			[0.3 - 0.4j, 0.8 + 0.1j],
			[-0.6 + 0.2j, 0.5 - 0.3j],
		],
		dtype=jnp.complex64,
	)
	loss = jnp.array([0.4 + 0.2j, -0.1 + 0.5j, 0.3 - 0.6j], dtype=jnp.complex64)
	n_s = O.shape[0]
	tdvp = TDVP(
		backend="jax",
		use_sr=True,
		use_minsr=True,
		rhs_prefactor=1.0,
		sr_lin_solver=None,
		sr_diag_shift=0.0,
		sr_pinv_tol=1e-10,
	)

	solution, _ = tdvp.solve(loss, O, params_are_complex=False)
	tdvp_gram = TDVP(
		backend="jax",
		use_sr=True,
		use_minsr=True,
		rhs_prefactor=1.0,
		sr_lin_solver="pseudo_inverse",
		sr_lin_solver_t="gram",
		sr_diag_shift=0.0,
		sr_pinv_tol=1e-10,
	)
	solution_gram, _ = tdvp_gram.solve(loss, O, params_are_complex=False)

	O_c = O - jnp.mean(O, axis=0)
	loss_c = loss - jnp.mean(loss)
	O_real = jnp.concatenate([jnp.real(O_c), jnp.imag(O_c)], axis=0)
	loss_real = jnp.concatenate([jnp.real(loss_c), jnp.imag(loss_c)], axis=0)
	O_real_np = np.asarray(O_real, dtype=np.float64)
	loss_real_np = np.asarray(loss_real, dtype=np.float64)
	T = O_real_np @ O_real_np.T / n_s
	expected = O_real_np.T @ (np.linalg.pinv(T, rcond=1e-10) @ (loss_real_np / n_s))

	np.testing.assert_allclose(np.asarray(solution.x), np.asarray(expected), rtol=2e-5, atol=2e-6)
	np.testing.assert_allclose(np.asarray(solution_gram.x), np.asarray(expected), rtol=2e-5, atol=2e-6)
	assert not np.iscomplexobj(np.asarray(solution.x))
