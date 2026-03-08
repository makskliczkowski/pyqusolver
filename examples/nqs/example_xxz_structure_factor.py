r"""
Example: 1D Heisenberg dynamics and finite-size S(q, omega)
===========================================================

This example benchmarks the antiferromagnetic spin-1/2 Heisenberg chain using
an RBM NQS in the physically relevant ``S^z_tot = 0`` sector. It focuses the
NQS-vs-ED comparison on the antiferromagnetic ``q = \pi`` line,

    C_q(t) = <gs| S_{-q}^z(t) S_q^z(0) |gs>,

because a full momentum-resolved NQS map would require an independent TDVP
trajectory for each momentum. The example therefore does two complementary
things:

1. compares the ``q = \pi`` correlator and spectral line between ED and
   NQS-TDVP on the same finite-time FFT convention,
2. builds exact finite-time and exact Lehmann k-path maps to validate the
   existing spectral plotting helpers on the same physics problem.

For the small finite chain this remains a finite-size benchmark rather than a
thermodynamic two-spinon continuum, but the exact Lehmann map is the correct
many-body reference and the NQS comparison is made on a numerically fair,
same-grid spectral estimator.
"""

import os
import sys

import numpy as np

current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import jax.numpy as jnp

from QES.Algebra.Model.Interacting.Spin.xxz import XXZ
from QES.Algebra.Operator.impl.operators_spin import sig_k
from QES.Algebra.hilbert import HilbertSpace
from QES.NQS.nqs import NQS
from QES.general_python.common.plot import Plotter, configure_style
from QES.general_python.common.plotters.plot_helpers import plot_spectral_function
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


def build_heisenberg_chain(ns: int = 6, seed: int = 0):
    """
    Build a periodic spin-1/2 Heisenberg chain and a symmetry-aware RBM NQS.

    The XXZ convention used here is the antiferromagnetic Heisenberg point
    J_xy = -1, Delta = 1 in the local sign convention of the package. The
    sampler is restricted to the ``S^z_tot = 0`` sector through exchange moves
    and a fixed-magnetization initialization, which is essential for a stable
    Heisenberg benchmark.
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
        s_numsamples=256,
        s_numchains=16,
        s_therm_steps=8,
        s_sweep_steps=1,
        s_upd_fun="EXCHANGE",
        s_initstate="RND_FIXED",
        magnetization=0,
    )
    return model, hilbert, lattice, nqs


def _last_energy(stats):
    if hasattr(stats, "history") and "Energy" in stats.history and len(stats.history["Energy"]) > 0:
        return stats.history["Energy"][-1]
    return None


def _build_spin_probes(lattice, hilbert):
    """
    Build the S_q^z and S_-q^z operator families on the sampled lattice momenta.

    The 1D chain uses the first Cartesian momentum component of ``lattice.kvectors``.
    The bra probes are chosen as S_-q^z so the resulting correlator matches the
    physical dynamical structure factor.
    """

    k_values = np.asarray(lattice.kvectors, dtype=float)
    q_values = np.asarray(k_values[:, 0], dtype=float)
    ket_probes = [sig_k(float(q), lattice=lattice, ns=hilbert.ns) for q in q_values]
    bra_probes = [sig_k(float(-q), lattice=lattice, ns=hilbert.ns) for q in q_values]
    labels = [rf"$q={q / np.pi:.2f}\pi$" for q in q_values]
    return k_values, q_values, ket_probes, bra_probes, labels


def _exact_time_correlator_map(model, hilbert, ket_probes, bra_probes, times):
    r"""
    Evaluate the exact finite-time correlator map

        C_q(t) = <gs| S_{-q}^z(t) S_q^z(0) |gs>

    for the supplied probe family.

    This is the correct ED reference for comparing with NQS-TDVP on the same
    time mesh before any Fourier transform is applied.
    """

    eig_vec = np.asarray(model._eig_vec, dtype=np.complex128)
    eig_val = np.asarray(model._eig_val, dtype=np.float64)
    psi0 = np.asarray(eig_vec[:, 0], dtype=np.complex128)

    from QES.Algebra.Properties.time_evo import time_evo_block

    corr_rows = []
    for ket_probe, bra_probe in zip(ket_probes, bra_probes):
        ket_matrix = ket_probe.compute_matrix(
            hilbert_1=hilbert,
            matrix_type="dense",
            use_numpy=True,
        )
        bra_matrix = bra_probe.compute_matrix(
            hilbert_1=hilbert,
            matrix_type="dense",
            use_numpy=True,
        )
        psi_probe = ket_matrix @ psi0
        bra_probe_state = bra_matrix @ psi0
        overlaps_probe = eig_vec.conj().T @ psi_probe
        psi_probe_t = time_evo_block(eig_vec, eig_val, overlaps_probe, times)
        corr_rows.append(np.einsum("i,it->t", np.conj(bra_probe_state), psi_probe_t))
    return np.asarray(corr_rows, dtype=np.complex128)


def _exact_structure_factor_map(model, hilbert, ket_probes, omega, *, eta):
    """
    Compute the broadened exact Lehmann reference S(q, omega).

    Unlike the finite-time FFT comparison, this is the direct many-body
    spectral function of the Hamiltonian and therefore the right exact physics
    benchmark for the finite chain.
    """

    exact_rows = []
    for probe in ket_probes:
        probe_matrix = probe.compute_matrix(
            hilbert_1=hilbert,
            matrix_type="dense",
            use_numpy=True,
        )
        spectrum = model.spectral.dynamic_structure_factor(
            omega,
            probe_matrix,
            state_idx=0,
            eta=eta,
            use_lanczos=False,
        )
        exact_rows.append(np.asarray(spectrum, dtype=np.float64))
    return np.asarray(exact_rows, dtype=np.float64)


def _save_time_trace_plot(times, exact_corr, nqs_corr, q_value, outdir):
    """
    Save the time-domain Heisenberg correlator at one momentum.

    This is the primary dynamical observable that should agree before the FFT to
    ``S(q, omega)`` is trusted.
    """

    fig, axes = Plotter.get_subplots(
        nrows=2,
        ncols=1,
        sizex=6.0,
        sizey=5.1,
        share_x=True,
    )
    Plotter.plot(axes[0], times, np.real(exact_corr), color="black", label="ED", linewidth=1.6)
    Plotter.plot(axes[0], times, np.real(nqs_corr), color="C0", label="NQS-TDVP", linewidth=1.4)
    Plotter.set_ax_params(
        axes[0],
        ylabel=r"$\mathrm{Re}\,C_q(t)$",
        title=rf"Heisenberg chain correlator at $q={q_value / np.pi:.2f}\pi$",
    )
    Plotter.grid(axes[0], alpha=0.25)
    Plotter.set_legend(axes[0], loc="best")

    Plotter.plot(axes[1], times, np.imag(exact_corr), color="black", label="ED", linewidth=1.6)
    Plotter.plot(axes[1], times, np.imag(nqs_corr), color="C1", label="NQS-TDVP", linewidth=1.4)
    Plotter.set_ax_params(
        axes[1],
        xlabel=r"$t$",
        ylabel=r"$\mathrm{Im}\,C_q(t)$",
    )
    Plotter.grid(axes[1], alpha=0.25)
    Plotter.save_fig(
        outdir,
        "xxz_sqw_qpi_time_compare.png",
        format="png",
        dpi=220,
        adjust=False,
        fig=fig,
    )
    return os.path.join(outdir, "xxz_sqw_qpi_time_compare.png")


def _finite_time_spectrum_map(nqs, times, correlator_map, *, eta, window=None):
    """
    Apply the same finite-time Fourier pipeline to each correlator channel.

    This keeps the ED and NQS spectra on identical numerical footing.
    """

    rows = []
    frequencies = None
    for corr in np.asarray(correlator_map):
        result = nqs.spectrum_from_correlator(
            times,
            corr,
            eta=eta,
            window=window,
            positive_frequencies_only=True,
        )
        if frequencies is None:
            frequencies = np.asarray(result.frequencies, dtype=np.float64)
        rows.append(np.asarray(result.spectrum, dtype=np.float64))
    return frequencies, np.asarray(rows, dtype=np.float64)


def _save_line_plot(omega, exact_line, exact_lehmann_line, nqs_line, q_value, outdir):
    """
    Save the momentum-resolved line cut through S(q, omega).

    Three curves are shown:
    - exact Lehmann spectrum of the finite chain,
    - exact finite-time FFT on the same time grid as NQS,
    - NQS-TDVP finite-time FFT.
    """

    fig, ax = Plotter.get_subplots(
        nrows=1,
        ncols=1,
        sizex=5.6,
        sizey=3.4,
        single_if_1=True,
    )
    Plotter.plot(ax, omega, exact_lehmann_line, color="0.65", label="ED Lehmann", linewidth=1.2)
    Plotter.plot(ax, omega, exact_line, color="black", label="ED finite-time", linewidth=1.6)
    Plotter.plot(ax, omega, nqs_line, color="C0", label="NQS-TDVP", linewidth=1.4)
    Plotter.set_ax_params(
        ax,
        xlabel=r"$\omega$",
        ylabel=rf"$S(q={q_value / np.pi:.2f}\pi, \omega)$",
        title="Finite-size Heisenberg structure factor at fixed momentum",
    )
    Plotter.set_tickparams(ax, labelsize=9)
    Plotter.grid(ax, alpha=0.25)
    Plotter.set_legend(ax, loc="best")
    Plotter.save_fig(outdir, "xxz_sqw_qpi_compare.png", format="png", dpi=220, adjust=False, fig=fig)
    return os.path.join(outdir, "xxz_sqw_qpi_compare.png")


def _save_exact_kpath_plot(omega, exact_fft_map, exact_lehmann_map, k_values, lattice, outdir):
    """
    Save exact k-path maps for both finite-time FFT and broadened Lehmann data.

    This validates the spectral plotting helper on the Heisenberg benchmark and
    makes the difference between finite-time resolution and the exact broadened
    many-body line shape explicit.
    """

    common_vmax = max(float(np.nanmax(exact_fft_map)), float(np.nanmax(exact_lehmann_map)))
    fig, axes = Plotter.get_subplots(
        nrows=1,
        ncols=2,
        sizex=9.4,
        sizey=3.4,
    )

    plot_spectral_function(
        ax=axes[0],
        fig=fig,
        omega=omega,
        intensity=exact_fft_map,
        k_values=k_values,
        lattice=lattice,
        mode="kpath",
        path_labels=["Gamma", "X", "Gamma"],
        sampled_only=True,
        colorbar=False,
        vmin=0.0,
        vmax=common_vmax,
        title="ED finite-time FFT",
        intensity_label=r"$S(q,\omega)$",
        colorbar_label=r"$S(q,\omega)$",
    )
    plot_spectral_function(
        ax=axes[1],
        fig=fig,
        omega=omega,
        intensity=exact_lehmann_map,
        k_values=k_values,
        lattice=lattice,
        mode="kpath",
        path_labels=["Gamma", "X", "Gamma"],
        sampled_only=True,
        colorbar=False,
        vmin=0.0,
        vmax=common_vmax,
        title="ED Lehmann",
        intensity_label=r"$S(q,\omega)$",
        colorbar_label=r"$S(q,\omega)$",
    )
    Plotter.save_fig(outdir, "xxz_sqw_kpath_exact_compare.png", format="png", dpi=220, adjust=False, fig=fig)
    return os.path.join(outdir, "xxz_sqw_kpath_exact_compare.png")


def main():
    configure_style("publication", font_size=10, dpi=140)

    ns = 4
    eta = 0.4
    times = np.linspace(0.0, 8.0, 161)
    outdir = _output_dir()

    model, hilbert, lattice, nqs = build_heisenberg_chain(ns=ns, seed=7)
    model.diagonalize()

    print("=" * 72)
    print("1D Heisenberg chain: time-domain correlator and finite-size S(q, omega)")
    print("=" * 72)
    print(f"System size: L={ns}, eta={eta}")
    print("Training RBM ground state...")

    stats = nqs.train(
        n_epochs=160,
        phases="default",
        lr=0.01,
        checkpoint_every=250,
        diag_shift=1e-3,
        sr_maxiter=128,
        exact_predictions=model.eig_vals,
    )

    k_values, q_values, ket_probes, bra_probes, labels = _build_spin_probes(lattice, hilbert)
    q_pi_idx = int(np.argmin(np.abs(q_values - np.pi)))
    q_pi = float(q_values[q_pi_idx])
    ket_probe = ket_probes[q_pi_idx]
    bra_probe = bra_probes[q_pi_idx]

    sqw_nqs = nqs.dynamic_structure_factor(
        times,
        ket_probe_operator=ket_probe,
        bra_probe_operator=bra_probe,
        exact_sum=True,
        num_samples=128,
        num_chains=8,
        n_batch=128,
        diag_shift=1e-3,
        sr_maxiter=128,
        ode_solver="RK4",
        n_substeps=2,
        eta=eta,
        window=None,
        restore=True,
    )

    exact_corr_map = _exact_time_correlator_map(
        model,
        hilbert,
        ket_probes,
        bra_probes,
        times,
    )
    omega, sqw_exact = _finite_time_spectrum_map(
        nqs,
        times,
        exact_corr_map,
        eta=eta,
        window=None,
    )

    sqw_exact_lehmann = _exact_structure_factor_map(
        model,
        hilbert,
        ket_probes,
        omega,
        eta=eta,
    )

    sqw_exact = np.real(np.asarray(sqw_exact, dtype=np.float64))
    sqw_exact_lehmann = np.real(np.asarray(sqw_exact_lehmann, dtype=np.float64))
    sqw_nqs_line = np.real(np.asarray(sqw_nqs.spectrum, dtype=np.float64))
    exact_time = np.asarray(exact_corr_map[q_pi_idx], dtype=np.complex128)
    exact_line = sqw_exact[q_pi_idx]
    exact_lehmann_line = sqw_exact_lehmann[q_pi_idx]
    time_plot = _save_time_trace_plot(times, exact_time, sqw_nqs.correlator, q_pi, outdir)
    line_plot = _save_line_plot(omega, exact_line, exact_lehmann_line, sqw_nqs_line, q_pi, outdir)
    kpath_plot = _save_exact_kpath_plot(omega, sqw_exact, sqw_exact_lehmann, k_values, lattice, outdir)

    peak_nqs = omega[int(np.argmax(sqw_nqs_line))]
    peak_exact = omega[int(np.argmax(exact_line))]
    peak_lehmann = omega[int(np.argmax(exact_lehmann_line))]
    overlap = (
        np.corrcoef(sqw_nqs_line, exact_line)[0, 1]
        if np.std(sqw_nqs_line) > 0 and np.std(exact_line) > 0
        else float("nan")
    )
    time_error = float(np.max(np.abs(np.asarray(sqw_nqs.correlator) - exact_time)))
    lehmann_overlap = (
        np.corrcoef(exact_line, exact_lehmann_line)[0, 1]
        if np.std(exact_line) > 0 and np.std(exact_lehmann_line) > 0
        else float("nan")
    )

    final_energy = _last_energy(stats)
    if final_energy is not None:
        print(f"NQS final energy:   {final_energy:.8f}")
        print(f"Energy error:       {abs(final_energy - model._eig_val[0]):.8f}")
    print(f"Exact ground energy: {model._eig_val[0]:.8f}")
    print(f"Time-domain max error at q=pi: {time_error:.6e}")
    print(f"Peak position NQS:   {peak_nqs:.6f}")
    print(f"Peak position ED finite-time: {peak_exact:.6f}")
    print(f"Peak position ED Lehmann:     {peak_lehmann:.6f}")
    print(f"Line-shape correlation at q=pi (NQS vs ED finite-time): {overlap:.6f}")
    print(f"Line-shape correlation at q=pi (ED finite-time vs Lehmann): {lehmann_overlap:.6f}")
    print(f"Saved q=pi time plot: {time_plot}")
    print(f"Saved q=pi line plot: {line_plot}")
    print(f"Saved exact k-path plot: {kpath_plot}")
    print()
    print("Spectral map shapes:")
    print(f"  omega.shape        = {omega.shape}")
    print(f"  S_exact_fft.shape  = {sqw_exact.shape}")
    print(f"  S_exact_mb.shape   = {sqw_exact_lehmann.shape}")
    print(f"  S_nqs_qpi.shape    = {sqw_nqs_line.shape}")


if __name__ == "__main__":
    main()
