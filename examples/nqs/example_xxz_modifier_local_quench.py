"""
Example: Heisenberg local-quench dynamics with an NQS ansatz modifier
=====================================================================

This example benchmarks the physically meaningful modifier workflow for a
spin-1/2 antiferromagnetic Heisenberg chain. The base NQS is first trained
toward the ground state of the XXZ model at the isotropic point. A local spin
flip operator

    S^x_0

is then applied as an ansatz modifier, producing the probe state

    |phi_0> = S^x_0 |psi_gs>.

That probe state is evolved in real time in two ways:

    1. exactly, by ED of the Heisenberg Hamiltonian,
    2. approximately, by TDVP evolution inside the modifier-NQS manifold.

The benchmark observable is the local-quench autocorrelator

    C_x(t) = <psi_gs| S^x_0(t) S^x_0(0) |psi_gs>
           = <phi_0|phi(t)>,

evaluated on the same physical time grid for ED and NQS. This is the cleanest
test that the ansatz modifier is not just syntactically wired, but produces the
correct probe-state dynamics.
"""

import os
import sys

import numpy as np

current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import jax.numpy as jnp

from QES.Algebra.Model.Interacting.Spin.xxz import XXZ
from QES.Algebra.Operator.impl.operators_spin import sig_x
from QES.Algebra.Properties.time_evo import time_evo_block
from QES.Algebra.hilbert import HilbertSpace
from QES.NQS.nqs import NQS
from QES.general_python.common.plot import Plotter, configure_style
from QES.general_python.lattices import SquareLattice
from QES.general_python.ml.net_impl.networks.net_rbm import RBM


def _output_dir():
    """
    Save figures under the Python-library local tmp directory.
    """

    repo_root = current_dir if os.path.isdir(os.path.join(current_dir, "Python")) else os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    outdir = os.path.join(repo_root, "Python", "tmp", "nqs")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def build_heisenberg_quench_nqs(ns: int = 4, seed: int = 7):
    """
    Construct a small periodic Heisenberg chain and an RBM NQS in the
    ``S^z_tot = 0`` sector.

    The Heisenberg point is represented through the package XXZ convention
    ``jxy=-1`` and ``delta=1``. Exchange updates and fixed magnetization are
    used so the sampler remains inside the physically correct spin sector during
    training and TDVP evolution.
    """

    lattice = SquareLattice(dim=1, lx=ns, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice)
    model = XXZ(
        lattice=lattice,
        hilbert_space=hilbert,
        jxy=-1.0,
        delta=1.0,
        hx=0.0,
        hz=0.0,
        dtype=np.complex128,
    )

    net = RBM(
        input_shape=(hilbert.ns,),
        alpha=4.0,
        seed=seed,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        in_activation=True,
        visible_bias=False,
    )

    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=256,
        backend="jax",
        dtype=np.complex128,
        verbose=False,
        seed=seed,
        s_numsamples=512,
        s_numchains=16,
        s_therm_steps=8,
        s_sweep_steps=1,
        s_upd_fun="EXCHANGE",
        s_initstate="RND_FIXED",
        magnetization=0,
    )
    return model, hilbert, nqs


def _exact_local_quench_correlator(model, hilbert, probe, times):
    """
    Compute the exact local-quench correlator

        C_x(t) = <gs| S^x_0(t) S^x_0(0) |gs>

    by acting with the dense probe matrix on the ED ground state and evolving
    that probe state with the full Hamiltonian spectrum.
    """

    model.diagonalize()
    eig_vec = np.asarray(model._eig_vec, dtype=np.complex128)
    eig_val = np.asarray(model._eig_val, dtype=np.float64)
    psi_gs = np.asarray(eig_vec[:, 0], dtype=np.complex128)
    probe_matrix = probe.compute_matrix(
        hilbert_1=hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    phi0 = np.asarray(probe_matrix @ psi_gs, dtype=np.complex128)
    overlaps = eig_vec.conj().T @ phi0
    phi_t = time_evo_block(eig_vec, eig_val, overlaps, times)
    correlator = np.einsum("i,it->t", np.conj(phi0), phi_t)
    correlator = correlator * np.exp(1.0j * float(np.real(eig_val[0])) * (times - times[0]))
    return correlator, float(np.real(eig_val[0]))


def _save_quench_plot(times, exact_corr, nqs_corr, outdir):
    """
    Save the ED versus TDVP local-quench correlator on a common time grid.

    Agreement in this time-domain quantity is the primary validation target.
    Any frequency-domain response should only be trusted once this curve is
    already under control.
    """

    fig, axes = Plotter.get_subplots(
        nrows=2,
        ncols=1,
        sizex=6.0,
        sizey=5.2,
        share_x=True,
    )
    Plotter.plot(axes[0], times, np.real(exact_corr), color="black", label="ED", linewidth=1.6)
    Plotter.plot(axes[0], times, np.real(nqs_corr), color="C0", label="NQS-TDVP", linewidth=1.4)
    Plotter.set_ax_params(
        axes[0],
        ylabel=r"$\mathrm{Re}\,C_x(t)$",
        title=r"Heisenberg local quench: $C_x(t)=\langle gs|S^x_0(t)S^x_0(0)|gs\rangle$",
    )
    Plotter.grid(axes[0], alpha=0.25)
    Plotter.set_legend(axes[0], loc="best")

    Plotter.plot(axes[1], times, np.imag(exact_corr), color="black", label="ED", linewidth=1.6)
    Plotter.plot(axes[1], times, np.imag(nqs_corr), color="C1", label="NQS-TDVP", linewidth=1.4)
    Plotter.set_ax_params(
        axes[1],
        xlabel=r"$t$",
        ylabel=r"$\mathrm{Im}\,C_x(t)$",
    )
    Plotter.grid(axes[1], alpha=0.25)

    Plotter.save_fig(
        outdir,
        "xxz_modifier_local_quench_compare.png",
        format="png",
        dpi=220,
        adjust=False,
        fig=fig,
    )
    return os.path.join(outdir, "xxz_modifier_local_quench_compare.png")


def main():
    """
    Train a small Heisenberg-chain RBM, generate the local ``S^x_0|gs>`` probe
    state through an ansatz modifier, and compare its ED and TDVP dynamics.
    """

    configure_style("publication", font_size=10, dpi=140)

    outdir = _output_dir()
    # A short real-time window is the honest regime for this lightweight
    # 4-site TDVP benchmark: it shows the modifier dynamics tracking ED before
    # longer-time variational drift becomes dominant.
    times = np.linspace(0.0, 0.20, 9)

    model, hilbert, nqs = build_heisenberg_quench_nqs(ns=4, seed=7)
    nqs.train(
        n_epochs=60,
        phases="default",
        lr=0.01,
        checkpoint_every=500,
        diag_shift=1e-3,
        sr_maxiter=128,
        exact_predictions=model.eig_vals,
    )
    model.diagonalize()

    probe = sig_x(ns=hilbert.ns, sites=[0])
    probe_nqs = nqs.spawn_like(modifier=probe, verbose=False)
    trajectory = probe_nqs.time_evolve(
        times,
        num_samples=512,
        num_chains=16,
        n_batch=256,
        diag_shift=1e-3,
        sr_maxiter=128,
        ode_solver="RK4",
        max_dt=0.01,
        restore=True,
    )
    nqs_corr = nqs.compute_dynamical_correlator(
        times,
        ket_probe_operator=probe,
        bra_probe_operator=probe,
        trajectory=trajectory,
        reference_energy=float(np.real(model._eig_val[0])),
        exact_sum=True,
    )

    exact_corr, exact_energy = _exact_local_quench_correlator(model, hilbert, probe, times)
    plot_path = _save_quench_plot(times, exact_corr, nqs_corr.correlator, outdir)

    print("=" * 72)
    print("Heisenberg local-quench benchmark with an NQS ansatz modifier")
    print("=" * 72)
    print(f"Exact ground-state energy: {exact_energy:.8f}")
    print(f"Max |C_nqs(t)-C_exact(t)|: {np.max(np.abs(nqs_corr.correlator - exact_corr)):.6e}")
    print(f"Saved comparison plot:     {plot_path}")
    print()
    print("t        exact(Re)     nqs(Re)       exact(Im)     nqs(Im)")
    for t, c_ex, c_nqs in zip(times, exact_corr, nqs_corr.correlator):
        print(
            f"{t:5.2f}    {np.real(c_ex): .6f}   {np.real(c_nqs): .6f}"
            f"   {np.imag(c_ex): .6f}   {np.imag(c_nqs): .6f}"
        )


if __name__ == "__main__":
    main()
