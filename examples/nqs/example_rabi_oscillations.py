"""
Example: single-spin Rabi / Larmor oscillations with ED and NQS-TDVP
====================================================================

This example is the cleanest real-time sanity check for the NQS TDVP stack.
It evolves a single spin-1/2 in a longitudinal field

    H = h_z S^z,

starting from the equal-amplitude ``|+x>`` state represented exactly by a
zero-parameter RBM. The observable

    <S^x(t)>

then performs an exactly solvable two-level oscillation. Because the initial
state and Hamiltonian are both elementary, disagreement with ED immediately
signals a broken time convention or TDVP prefactor rather than a variational
limitation of the ansatz.

The plot is saved under ``Python/tmp/nqs``.
"""

import os
import sys

import numpy as np

current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import jax
import jax.numpy as jnp

from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
from QES.Algebra.Operator.impl.operators_spin import sig_x
from QES.Algebra.Properties.time_evo import time_evo_block
from QES.Algebra.hilbert import HilbertSpace
from QES.NQS.nqs import NQS
from QES.NQS.src.nqs_spectral import (
    _NQSParamView,
    _enumerate_basis_states,
    _exact_wavefunction_vector,
    _materialize_trajectory_params,
)
from QES.general_python.common.plot import Plotter, configure_style
from QES.general_python.lattices import SquareLattice
from QES.general_python.ml.net_impl.networks.net_rbm import RBM


def _output_dir():
    """
    Save plots under the Python-library local tmp directory.
    """

    repo_root = current_dir if os.path.isdir(os.path.join(current_dir, "Python")) else os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    outdir = os.path.join(repo_root, "Python", "tmp", "nqs")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def build_single_spin_rabi_nqs(seed: int = 0):
    r"""
    Build the one-spin qubit used for the Rabi / Larmor benchmark.

    The zeroed RBM gives a constant log-amplitude and therefore the exact
    equal-amplitude superposition over the one-spin basis. Under ``H = h_z S^z``
    this state precesses with a closed-form oscillation in ``<S^x(t)>``.
    """

    lattice = SquareLattice(dim=1, lx=1, bc="obc")
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=0.0,
        hx=0.0,
        hz=1.0,
        dtype=np.complex128,
    )

    net = RBM(
        input_shape=(hilbert.ns,),
        n_hidden=hilbert.ns,
        seed=seed,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
    )
    zero_params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), net.get_params())
    net.set_params(zero_params)

    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=32,
        backend="jax",
        dtype=np.complex128,
        symmetrize=False,
        verbose=False,
        seed=seed,
        s_numsamples=128,
        s_numchains=16,
        s_therm_steps=1,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)
    return model, nqs


def exact_trajectory_observable(nqs, trajectory, operator) -> np.ndarray:
    r"""
    Evaluate ``<O(t)>`` exactly along the TDVP trajectory.

    For this one-spin problem the full Hilbert space is tiny, so exact-sum
    evaluation removes all Monte Carlo estimator noise and shows the actual
    projected dynamics produced by TDVP.
    """

    basis_states = _enumerate_basis_states(nqs)
    op_matrix = operator.compute_matrix(
        hilbert_1=nqs.hilbert,
        matrix_type="dense",
        use_numpy=True,
    )

    values = []
    for params_t, phase_t in zip(trajectory.param_history, trajectory.global_phase):
        state_view = _NQSParamView(
            nqs,
            _materialize_trajectory_params(nqs, trajectory, params_t),
            global_phase=complex(phase_t),
        )
        psi_t = _exact_wavefunction_vector(state_view, basis_states=basis_states)
        psi_t /= np.sqrt(np.vdot(psi_t, psi_t))
        values.append(np.vdot(psi_t, op_matrix @ psi_t))
    return np.asarray(values, dtype=np.complex128)


def exact_ed_observable(model, times, operator) -> np.ndarray:
    r"""
    Compute the exact ED reference for the same ``<S^x(t)>`` observable.
    """

    model.diagonalize()
    eig_vec = np.asarray(model._eig_vec)
    eig_val = np.asarray(model._eig_val)
    psi0 = np.ones(eig_vec.shape[0], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    overlaps = eig_vec.conj().T @ psi0
    psi_t = time_evo_block(eig_vec, eig_val, overlaps, times)
    op_matrix = operator.compute_matrix(
        hilbert_1=model.hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    return np.einsum("it,ij,jt->t", np.conj(psi_t), op_matrix, psi_t)


def _save_rabi_plot(times, ed_values, nqs_values, outdir):
    fig, axes = Plotter.get_subplots(
        nrows=2,
        ncols=1,
        sizex=6.1,
        sizey=5.0,
        share_x=True,
    )

    Plotter.plot(axes[0], times, np.real(ed_values), color="black", linewidth=1.7, label="ED")
    Plotter.plot(axes[0], times, np.real(nqs_values), color="C0", linewidth=1.4, label="NQS-TDVP")
    Plotter.set_ax_params(
        axes[0],
        ylabel=r"$\langle S^x(t) \rangle$",
        title="Single-spin Rabi / Larmor benchmark",
    )
    Plotter.grid(axes[0], alpha=0.25)
    Plotter.set_legend(axes[0], loc="best")

    Plotter.plot(
        axes[1],
        times,
        np.abs(nqs_values - ed_values),
        color="C3",
        linewidth=1.4,
    )
    Plotter.set_ax_params(
        axes[1],
        xlabel=r"$t$",
        ylabel=r"$|\Delta S^x(t)|$",
    )
    Plotter.grid(axes[1], alpha=0.25)

    Plotter.save_fig(outdir, "single_spin_rabi_oscillation_compare.png", format="png", dpi=220, adjust=False, fig=fig)
    return os.path.join(outdir, "single_spin_rabi_oscillation_compare.png")


def main():
    configure_style("publication", font_size=10, dpi=140)

    outdir = _output_dir()
    times = np.linspace(0.0, 2.0 * np.pi, 65)

    model, nqs = build_single_spin_rabi_nqs(seed=0)
    observable = sig_x(ns=model.hilbert.ns, sites=[0])
    trajectory = nqs.time_evolve(
        times,
        num_samples=128,
        num_chains=16,
        n_batch=32,
        diag_shift=1e-8,
        sr_maxiter=64,
        ode_solver="RK4",
        n_substeps=12,
        restore=True,
    )

    nqs_values = exact_trajectory_observable(nqs, trajectory, observable)
    ed_values = exact_ed_observable(model, times, observable)
    plot_path = _save_rabi_plot(times, ed_values, nqs_values, outdir)

    print("=" * 72)
    print("Single-spin ED vs NQS-TDVP Rabi / Larmor oscillations")
    print("=" * 72)
    print(f"Times: {times[0]:.2f} .. {times[-1]:.2f} with {times.size} points")
    print(f"Max |<Sx>_nqs - <Sx>_ed|: {np.max(np.abs(nqs_values - ed_values)):.6e}")
    print(f"Saved plot: {plot_path}")
    print()
    print("t        ED            NQS")
    for t, ed_val, nqs_val in zip(times[::8], ed_values[::8], nqs_values[::8]):
        print(f"{t:6.3f}   {np.real(ed_val): .6f}   {np.real(nqs_val): .6f}")


if __name__ == "__main__":
    main()
