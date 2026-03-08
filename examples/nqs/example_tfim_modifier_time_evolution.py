r"""
Example: physical ansatz modifier sanity check for dynamical response
=====================================================================

This example validates the modifier-based TDVP plumbing in the cleanest regime
where the current variational manifold should be exact: a zero-Hamiltonian
transverse-field Ising model. Starting from the uniform reference state, it
inserts a momentum-space spin probe

    |\phi_0\rangle = S^z_q |\psi_0\rangle,

and then evolves that modified state. The benchmark quantity is the modifier
state overlap

    C_M(t) = \langle \phi_0 | \phi(t) \rangle.

Because the Hamiltonian is identically zero, the exact answer is time
independent. The example uses the exact small-system summation path
(``exact_sum=True``) so the result is a deterministic regression benchmark for
the NQS modifier/TDVP plumbing rather than a Monte Carlo noise test.
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
from QES.Algebra.Operator.impl.operators_spin import sig_k
from QES.Algebra.hilbert import HilbertSpace
from QES.NQS.nqs import NQS
from QES.general_python.common.plot import Plotter, configure_style
from QES.general_python.lattices import SquareLattice
from QES.general_python.ml.net_impl.networks.net_rbm import RBM


def _output_dir():
    """
    Return the repository-local output directory for NQS example figures.
    """

    repo_root = current_dir if os.path.isdir(os.path.join(current_dir, "Python")) else os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    outdir = os.path.join(repo_root, "Python", "tmp", "nqs")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def build_static_uniform_state(seed: int = 0):
    """
    Build a zero-Hamiltonian TFIM and an exact uniform RBM reference state.

    The equal-amplitude wavefunction is the ``|+x>`` product state in the
    computational basis. With all couplings set to zero, both the reference
    state and any probe state are stationary, which makes the dynamical
    structure factor an exact consistency check for the modifier plumbing.
    """

    lattice = SquareLattice(dim=1, lx=4, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=0.0,
        hx=0.0,
        hz=0.0,
        dtype=np.complex128,
    )

    net = RBM(
        input_shape=(hilbert.ns,),
        n_hidden=2 * hilbert.ns,
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
        batch_size=128,
        backend="jax",
        dtype=np.complex128,
        symmetrize=False,
        verbose=False,
        seed=seed,
        s_numsamples=256,
        s_numchains=8,
        s_therm_steps=4,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)

    psi0 = np.ones(2 ** hilbert.ns, dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    return model, hilbert, lattice, nqs, psi0


def _save_time_plot(times, exact_corr, nqs_corr, outdir):
    """
    Save the exact-sum modifier correlator benchmark.
    """

    fig, axes = Plotter.get_subplots(
        nrows=2,
        ncols=1,
        sizex=6.0,
        sizey=5.0,
        share_x=True,
    )
    Plotter.plot(axes[0], times, np.real(exact_corr), color="black", label="ED", linewidth=1.6)
    Plotter.plot(axes[0], times, np.real(nqs_corr), color="C3", label="NQS exact-sum", linewidth=1.4)
    Plotter.set_ax_params(
        axes[0],
        ylabel=r"$\mathrm{Re}\,C_q(t)$",
        title=r"Modifier benchmark for $C_M(t)=\langle \phi_0 | \phi(t) \rangle$",
    )
    Plotter.grid(axes[0], alpha=0.25)
    Plotter.set_legend(axes[0], loc="best")

    Plotter.plot(axes[1], times, np.imag(exact_corr), color="black", label="ED", linewidth=1.6)
    Plotter.plot(axes[1], times, np.imag(nqs_corr), color="C1", label="NQS exact-sum", linewidth=1.4)
    Plotter.set_ax_params(
        axes[1],
        xlabel=r"$t$",
        ylabel=r"$\mathrm{Im}\,C_q(t)$",
    )
    Plotter.grid(axes[1], alpha=0.25)
    Plotter.save_fig(
        outdir,
        "tfim_modifier_time_evolution_compare.png",
        format="png",
        dpi=220,
        adjust=False,
        fig=fig,
    )
    return os.path.join(outdir, "tfim_modifier_time_evolution_compare.png")


def _save_spectrum_plot(freq_exact, spec_exact, freq_nqs, spec_nqs, outdir):
    """
    Save the broadened zero-frequency spectrum of the modifier overlap.
    """

    fig, ax = Plotter.get_subplots(nrows=1, ncols=1, sizex=5.8, sizey=3.5, single_if_1=True)
    Plotter.plot(ax, freq_exact, spec_exact, color="black", label="ED", linewidth=1.6)
    Plotter.plot(ax, freq_nqs, spec_nqs, color="C0", label="NQS exact-sum", linewidth=1.4)
    Plotter.set_ax_params(
        ax,
        xlabel=r"$\omega$",
        ylabel=r"$S(q,\omega)$",
        title=r"Broadened zero-H modifier spectrum",
    )
    Plotter.grid(ax, alpha=0.25)
    Plotter.set_legend(ax, loc="best")
    Plotter.save_fig(
        outdir,
        "tfim_modifier_spectral_compare.png",
        format="png",
        dpi=220,
        adjust=False,
        fig=fig,
    )
    return os.path.join(outdir, "tfim_modifier_spectral_compare.png")


def main():
    """
    Run the exact-summation sanity benchmark for modifier-based dynamics.
    """

    configure_style("publication", font_size=10, dpi=140)

    outdir = _output_dir()
    times = np.linspace(0.0, 1.0, 17)
    q_value = np.pi
    eta = 0.05

    model, hilbert, lattice, nqs, _psi0 = build_static_uniform_state(seed=0)
    probe = sig_k(q_value, lattice=lattice, ns=hilbert.ns)

    probe_nqs = nqs.spawn_like(
        modifier=probe,
        directory=outdir,
        verbose=False,
    )
    trajectory = probe_nqs.time_evolve(
        times,
        num_samples=256,
        num_chains=8,
        n_batch=128,
        diag_shift=1e-3,
        sr_maxiter=64,
        ode_solver="RK4",
        restore=True,
    )
    physical_nqs = nqs.compute_dynamical_correlator(
        times,
        ket_probe_operator=probe,
        bra_probe_operator=probe,
        trajectory=trajectory,
        exact_sum=True,
    )
    spectral_nqs = nqs.dynamic_structure_factor(
        times,
        ket_probe_operator=probe,
        bra_probe_operator=probe,
        trajectory=trajectory,
        exact_sum=True,
        eta=eta,
        window="hann",
    )

    from QES.NQS.src.nqs_spectral import _enumerate_basis_states, _exact_wavefunction_vector

    basis_states = _enumerate_basis_states(probe_nqs)
    phi0 = _exact_wavefunction_vector(probe_nqs, basis_states=basis_states)
    exact_corr = np.full_like(times, np.vdot(phi0, phi0), dtype=np.complex128)

    exact_spectrum = nqs.spectrum_from_correlator(
        times,
        exact_corr,
        eta=eta,
        window="hann",
    )
    freq_exact = exact_spectrum.frequencies
    spec_exact = exact_spectrum.spectrum

    time_plot = _save_time_plot(times, exact_corr, physical_nqs.correlator, outdir)
    spectrum_plot = _save_spectrum_plot(
        freq_exact,
        spec_exact,
        spectral_nqs.frequencies,
        spectral_nqs.spectrum,
        outdir,
    )

    print("=" * 72)
    print("TFIM zero-H sanity check: modifier-state dynamics")
    print("=" * 72)
    print(f"Momentum probe: q = {q_value / np.pi:.2f} pi")
    print(f"Max |C_nqs(t) - C_exact(t)|: {np.max(np.abs(physical_nqs.correlator - exact_corr)):.6e}")
    print(f"Saved correlator plot: {time_plot}")
    print(f"Saved spectrum plot:  {spectrum_plot}")
    print()
    print("t        exact(Re)     nqs(Re)       exact(Im)     nqs(Im)")
    for t, c_ex, c_nqs in zip(times, exact_corr, physical_nqs.correlator):
        print(
            f"{t:5.2f}    {np.real(c_ex): .6f}   {np.real(c_nqs): .6f}"
            f"   {np.imag(c_ex): .6f}   {np.imag(c_nqs): .6f}"
        )


if __name__ == "__main__":
    main()
