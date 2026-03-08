r"""
Example: exact-sum NQS entropy benchmarks across regions and Renyi order
=======================================================================

This example makes the entropy benchmark more reliable by comparing NQS
exact-summation entropies against ED for:

1. an exactly encoded two-spin cat-like RBM state, and
2. a compact two-spin TFIM ground state learned by short VMC/TDVP training.

For both states, it evaluates multiple subsystems and multiple Renyi indices
``q`` so the output checks site dependence, subsystem-size dependence, and the
Renyi-order dependence of the entanglement spectrum.
"""

import os
import sys

import numpy as np

current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import jax.numpy as jnp

from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
from QES.Algebra.hilbert import HilbertSpace
from QES.NQS.nqs import NQS
from QES.NQS.src.nqs_entropy import compute_ed_entanglement_entropy, compute_renyi_entropy
from QES.general_python.common.plot import Plotter, configure_style
from QES.general_python.lattices import SquareLattice
from QES.general_python.ml.net_impl.networks.net_rbm import RBM


def _output_dir():
    repo_root = current_dir if os.path.isdir(os.path.join(current_dir, "Python")) else os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    outdir = os.path.join(repo_root, "Python", "tmp", "nqs")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def build_exact_cat_state_nqs():
    """
    Build a two-spin RBM state with exactly known nontrivial entanglement.

    The encoded state is proportional to

        2 |00> + |11>,

    in the native NQS wrapper basis. It provides a deterministic benchmark for
    exact-sum Renyi entropies across several subsystem choices and Renyi orders.
    """

    lattice = SquareLattice(dim=1, lx=2, bc="obc")
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
        input_shape=(2,),
        n_hidden=2,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        seed=0,
    )
    params = net.get_params()
    params["visible_bias"] = jnp.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128)
    params["VisibleToHidden"]["kernel"] = jnp.array(
        [[1j * np.pi / 2, 1j * np.pi], [1j * np.pi, 1j * np.pi / 2]],
        dtype=jnp.complex128,
    )
    params["VisibleToHidden"]["bias"] = jnp.zeros((2,), dtype=jnp.complex128)
    net.set_params(params)

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
        seed=0,
        s_numsamples=1024,
        s_numchains=8,
        s_therm_steps=4,
        s_sweep_steps=1,
    )
    nqs.set_params(params)

    psi_exact = np.array([2.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    psi_exact /= np.linalg.norm(psi_exact)
    return hilbert, nqs, psi_exact


def build_trained_tfim_nqs():
    """
    Train a compact RBM on a two-site interacting TFIM ground state.

    This provides an entropy benchmark on an actually optimized NQS rather than
    only on a hand-constructed exact state.
    """

    lattice = SquareLattice(dim=1, lx=2, bc="obc")
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=1.0,
        hx=0.7,
        hz=0.0,
        dtype=np.complex128,
    )
    model.diagonalize()

    net = RBM(
        input_shape=(2,),
        n_hidden=2,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        seed=0,
    )
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
        seed=0,
        s_numsamples=32,
        s_numchains=4,
        s_therm_steps=4,
        s_sweep_steps=1,
    )

    stats = nqs.train(
        n_epochs=20,
        checkpoint_every=1000,
        lr=0.03,
        diag_shift=1e-3,
        n_batch=32,
        num_samples=32,
        num_chains=4,
        num_thermal=4,
        num_sweep=1,
        ode_solver="RK4",
        phases="default",
        use_pbar=False,
        exact_predictions=model.eig_vals,
    )
    psi_exact = np.asarray(model._eig_vec)[:, 0]
    return hilbert, nqs, psi_exact, stats


def _region_label(region):
    region = np.asarray(region, dtype=int).reshape(-1)
    if region.size == 0:
        return "[]"
    return "[" + ",".join(str(int(x)) for x in region) + "]"


def _entropy_grids(nqs, psi_exact, hilbert, regions, q_values):
    nqs_grid = np.zeros((len(regions), len(q_values)), dtype=np.float64)
    ed_grid = np.zeros_like(nqs_grid)

    for i, region in enumerate(regions):
        ed = compute_ed_entanglement_entropy(
            psi_exact,
            np.asarray(region, dtype=int),
            hilbert.ns,
            q_values=list(q_values),
            n_states=1,
        )
        for j, q in enumerate(q_values):
            nqs_grid[i, j] = compute_renyi_entropy(nqs, region=region, q=q, exact_sum=True)
            ed_grid[i, j] = float(ed[f"renyi_{q}"][0])

    return nqs_grid, ed_grid


def _draw_heatmap(ax, values, row_labels, col_labels, title, cmap):
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), [str(int(q)) for q in col_labels])
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    Plotter.set_ax_params(
        ax,
        xlabel=r"Renyi index $q$",
        ylabel="Subsystem A",
        title=title,
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if abs(values[i, j]) > 0.25 * np.max(np.abs(values) + 1e-12) else "black",
                fontsize=8,
            )
    return im


def _save_entropy_plot(
    exact_nqs,
    exact_ed,
    trained_nqs,
    trained_ed,
    regions,
    q_values,
    outdir,
):
    row_labels = [_region_label(region) for region in regions]
    fig, axes = Plotter.get_subplots(nrows=2, ncols=2, sizex=10.6, sizey=7.2)

    im00 = _draw_heatmap(
        axes[0, 0],
        exact_nqs,
        row_labels,
        q_values,
        r"Exact RBM: $S_q(A)$ from NQS exact-sum",
        cmap="viridis",
    )
    im01 = _draw_heatmap(
        axes[0, 1],
        np.abs(exact_nqs - exact_ed),
        row_labels,
        q_values,
        r"Exact RBM: $|S_q^{\mathrm{NQS}}-S_q^{\mathrm{ED}}|$",
        cmap="magma",
    )
    im10 = _draw_heatmap(
        axes[1, 0],
        trained_nqs,
        row_labels,
        q_values,
        r"Trained TFIM: $S_q(A)$ from NQS exact-sum",
        cmap="viridis",
    )
    im11 = _draw_heatmap(
        axes[1, 1],
        np.abs(trained_nqs - trained_ed),
        row_labels,
        q_values,
        r"Trained TFIM: $|S_q^{\mathrm{NQS}}-S_q^{\mathrm{ED}}|$",
        cmap="magma",
    )

    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    Plotter.save_fig(
        outdir,
        "nqs_entropy_compare.png",
        format="png",
        dpi=220,
        adjust=False,
        fig=fig,
    )
    return os.path.join(outdir, "nqs_entropy_compare.png")


def _print_table(title, nqs_grid, ed_grid, regions, q_values):
    print("=" * 72)
    print(title)
    print("=" * 72)
    for i, region in enumerate(regions):
        label = _region_label(region)
        print(f"Region A = {label}")
        for j, q in enumerate(q_values):
            diff = abs(nqs_grid[i, j] - ed_grid[i, j])
            print(
                f"  q={int(q)}: NQS={nqs_grid[i, j]: .12f}  "
                f"ED={ed_grid[i, j]: .12f}  |d|={diff:.3e}"
            )
        print()


def main():
    """
    Evaluate exact-sum NQS entropies for several cuts and Renyi orders, and
    compare them against ED for both an exactly encoded state and a trained NQS.
    """

    configure_style("publication", font_size=10, dpi=140)

    outdir = _output_dir()
    q_values = (2, 3, 4)
    regions = ([0], [1], [0, 1])

    hilbert_exact, nqs_exact, psi_exact = build_exact_cat_state_nqs()
    exact_nqs, exact_ed = _entropy_grids(nqs_exact, psi_exact, hilbert_exact, regions, q_values)

    hilbert_train, nqs_train, psi_train, stats = build_trained_tfim_nqs()
    trained_nqs, trained_ed = _entropy_grids(nqs_train, psi_train, hilbert_train, regions, q_values)

    plot_path = _save_entropy_plot(
        exact_nqs,
        exact_ed,
        trained_nqs,
        trained_ed,
        regions,
        q_values,
        outdir,
    )

    _print_table("Exact cat-state RBM entropy benchmark", exact_nqs, exact_ed, regions, q_values)
    _print_table("Trained TFIM entropy benchmark", trained_nqs, trained_ed, regions, q_values)

    print(f"Trained TFIM final energy: {float(stats.history[-1]):.12f}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
