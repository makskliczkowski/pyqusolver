#!/usr/bin/env python3
"""
NQS Demo: Variational Neural Quantum States for Frustrated Magnets

This script demonstrates how to use the QES NQS module to find ground states
and compute observables for various spin models on different lattices.

Features:
- Lattice selection (square, honeycomb, hexagonal, triangular)
- Model selection (Kitaev, Heisenberg, TFIM, XXZ, J1J2)
- Ansatz selection (RBM, CNN, ResNet, GCNN, AR, etc.)
- ED comparison for small system sizes
- Spin-spin correlation functions
- Rényi entanglement entropy estimation
- Weight loading/saving
- Training history and observable visualization

Usage:
    python demo_nqs.py --lattice honeycomb --lx 4 --ly 3 --model kitaev --ansatz rbm --train
    python demo_nqs.py --test  # Quick test on small system
"""

from __future__ import annotations
import  os
import  sys
import  argparse
from    pathlib import Path
from    typing  import List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))

try:
    import  jax
    import  jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    from QES.general_python.lattices            import Lattice
    from QES.general_python.common.flog         import get_global_logger
except ImportError as e:
    raise ImportError(f"Required modules from QES.general_python could not be imported. Please ensure the module is available.") from e

try:
    from QES.NQS                                import NQS, NQSPhysicsConfig, NQSSolverConfig, NQSTrainConfig
    # Computation...
    from QES.NQS.src.nqs_entropy                import compute_renyi_entropy, compute_ed_entanglement_entropy, bipartition_cuts
except ImportError as e:
    raise ImportError(f"Required modules from QES.NQS could not be imported. Please ensure the module is available.") from e

import  numpy as np
import  matplotlib.pyplot as plt

logger = get_global_logger()

# ----------------------------------------------------------------------------
#! Helpers
# ----------------------------------------------------------------------------

def _plot_training(history: List[float], ed_stats: Optional[Any], args: argparse.Namespace):
    ''' Helper function to plot training history with ED comparison. '''
    fig, ax         = plt.subplots(figsize=(8, 4))
    history         = np.array(history)
    ax.plot(history, label='NQS Energy', lw=2)
    
    if ed_stats and ed_stats.has_exact:
        ax.axhline(ed_stats.exact_gs, color='red', linestyle='--', label='ED Ground State', alpha=0.7)
        
        # Plot exact gap if available
        if len(ed_stats.exact_predictions) > 1:
            for i in range(1, min(5, len(ed_stats.exact_predictions))):
                ax.axhline(ed_stats.exact_predictions[i], color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Energy')
    ax.set_title(f'Training History: {args.model} {args.lx}x{args.ly}')
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(args.save_dir) / "demo_training_history.png", dpi=200)
    logger.info(f"Saved training plot to {args.save_dir}/demo_training_history.png", lvl=2)

def compute_observables(nqs: Optional[NQS], hamil: Optional[Any], lattice: Lattice, s_cfg: NQSSolverConfig, out_dir: Path):
    """Unified computation of observables for both NQS and ED."""
    
    results     = {}
    num_sites   = lattice.ns
    
    # NQS Observables
    if nqs is not None:
        logger.info(f"Computing NQS observables...", lvl=1, color='green')
        # Sample states from trained model
        (_, _), (states, log_psi), probabilities = nqs.sample(num_samples=s_cfg.n_samples)
        
        # Magnetization
        if nqs.model.name.lower() in ('kitaev', 'heisenberg', 'tfim', 'xxz'):
            sz_op               = nqs.model.operators.sig_z(ns=num_sites, type_act='local')
            sz_vals             = nqs.compute_observable(functions=sz_op.jax, states=states, ansatze=log_psi, probabilities=probabilities)
            results['nqs_mag']  = sz_vals
            logger.info(f"NQS Average Magnetization <Sz>: {np.mean(sz_vals.mean):.6f}", lvl=2)

        # Correlations
        correlators = ['zz', 'xx']
        for corr in correlators:
            corr_kernels    = nqs.model.correlators(correlators=[corr], type_acting='correlation', compute=False)
            kernel_jax      = corr_kernels[corr]['i,j'].jax
            corr_mat        = np.zeros((num_sites, num_sites))
            for i in range(num_sites):
                for j in range(i, num_sites):
                    res             = nqs.compute_observable(functions=kernel_jax, states=states, ansatze=log_psi, probabilities=probabilities, args=(i, j))
                    corr_mat[i, j]  = np.real(res.mean)
                    corr_mat[j, i]  = corr_mat[i, j]
            results[f'nqs_corr_{corr}'] = corr_mat

        # Entropy
        cuts = bipartition_cuts(lattice, cut_type="half_x")
        for label, region in cuts.items():
            s2, s2_err = compute_renyi_entropy(nqs, region=region, q=2, num_samples=s_cfg.n_samples, return_error=True)
            results[f'nqs_entropy_{label}'] = (s2, s2_err)
            logger.info(f"NQS Rényi-2 Entropy ({label}): {s2:.6f} +/- {s2_err:.6f}", lvl=2)

    # 2. ED Observables
    if hamil is not None and hamil.eig_vec is not None:
        logger.info(f"Computing ED observables for comparison...", lvl=1, color='cyan')
        # We reuse the logic from impurity_solver but simplified for demo
        from QES.NQS import EDDataset
        ed_ds = EDDataset(num_states=1, losses=hamil.eig_val, model_type=str(hamil), lattice_type=lattice.typek)
        corr, mag = ed_ds.get_operators(hamil, nstates_to_store=1)
        results['ed_corr'] = corr
        results['ed_mag'] = mag
        
        # Exact Entropy
        cuts = bipartition_cuts(lattice, cut_type="all")
        for label, region in cuts.items():
            if len(region) == 0 or len(region) >= num_sites: continue
            ee = compute_ed_entanglement_entropy(hamil.eig_vec, region, num_sites, q_values=[2])
            results[f'ed_entropy_{label}'] = ee['renyi_2'][0]
            logger.info(f"ED Rényi-2 Entropy ({label}): {ee['renyi_2'][0]:.6f}", lvl=2)

    return results

# ----------------------------------------------------------------------------
#! Main demo function
# ----------------------------------------------------------------------------

def run_demo(args: argparse.Namespace):
    """Main demo workflow."""
    
    # 1. Setup Configuration
    p_cfg = NQSPhysicsConfig(
        model_type      = args.model,
        lattice_type    = args.lattice,
        lx              = args.lx,
        ly              = args.ly,
        bc              = args.bc,
        hx              = args.hx,
        hy              = 0.0,
        hz              = args.hz,
    )
    
    if args.impurity:
        for imp in args.impurity:
            p_cfg.impurities.append((int(imp[0]), imp[1], imp[2], imp[3]))

    if args.model.lower() == 'kitaev':
        p_cfg.args['kxy']       = args.K
        p_cfg.args['kz']        = args.K
        p_cfg.args['gamma_xy']  = args.Gamma
        p_cfg.args['gamma_z']   = args.Gamma
    elif args.model.lower() in ('heisenberg', 'tfim', 'xxz', 'j1j2'):
        p_cfg.args['J']         = args.J
        if args.model.lower() == 'xxz': p_cfg.args['jz'] = args.Jz
        if args.model.lower() == 'j1j2':
            p_cfg.args['J1'] = args.J
            p_cfg.args['J2'] = args.J2

    s_cfg = NQSSolverConfig(
        ansatz          = args.ansatz,
        n_chains        = args.num_chains,
        n_samples       = args.num_samples,
        lr              = args.lr,
        epochs          = args.epochs,
        dtype           = "complex128" if args.complex else "float64",
        backend         = args.backend,
        optimizer       = args.optimizer,
        early_stopping  = args.early_stopping,
        patience        = args.patience,
    )

    # 2. Setup Physical Model
    hamil, hilbert, lattice = p_cfg.make_hamiltonian(dtype=np.complex128 if args.complex else np.float64)
    num_sites               = lattice.ns
    logger.title(f"NQS: {args.model.upper()} on {args.lattice.upper()} ({args.lx}x{args.ly}, Ns={num_sites})")
    
    # 3. Setup NQS Network
    net = s_cfg.make_net(p_cfg, alpha=args.alpha)
    
    # 4. Initialize NQS
    psi = NQS(
        logansatz       =   net,
        model           =   hamil,
        hilbert         =   hilbert,
        backend         =   args.backend,
        dtype           =   jnp.complex128 if args.complex else jnp.float64,
        directory       =   args.save_dir,
        seed            =   args.seed,
        s_numchains     =   args.num_chains,
        s_numsamples    =   args.num_samples,
        s_therm_steps   =   args.num_therm,
        s_sweep_steps   =   args.num_sweep,
        s_upd_fun       =   args.sampler_rule,
        optimizer       =   s_cfg.optimizer,
        early_stopper   =   s_cfg.early_stopping,
    )
    
    if args.load: psi.load_weights(args.load)
        
    # Exact Diagonalization
    ed_stats = None
    if args.ed or (num_sites <= args.max_ed_size):
        logger.info(f"Performing Exact Diagonalization...", lvl=1, color='cyan')
        ed_stats = psi.get_exact(k=6, verbose=True)
        
    # 5. Training
    if args.train:
        train_cfg = NQSTrainConfig.from_solver(
            s_cfg,
            n_epochs            = args.epochs,
            checkpoint_every    = args.checkpoint_every,
            phases              = "kitaev" if args.model.lower() == "kitaev" else "default",
            lr_scheduler        = args.lr_scheduler,
            diag_shift          = args.diag_shift,
            use_sr              = not args.no_sr,
            patience            = args.patience if args.early_stopping else None,
        )
        stats = psi.train(
            **train_cfg.to_train_kwargs(
                exact_predictions=ed_stats.exact_predictions if ed_stats is not None else None,
                use_pbar=True,
            )
        )
        if args.plot: _plot_training(stats.history, ed_stats, args)

    # 6. Compute Observables (NQS & ED)
    if args.compute_observables or args.compute_correlations:
        results = compute_observables(psi, hamil if ed_stats else None, lattice, s_cfg, Path(args.save_dir))
        
        # Save results to NPZ
        np.savez(Path(args.save_dir) / "demo_results.npz", **results)

# ----------------------------------------------------------------------------
#! CLI Arguments
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NQS Demo: Variational Ground State Search")
    parser.add_argument("--lattice",    type=str,   default="honeycomb")
    parser.add_argument("--lx",         type=int,   default=4)
    parser.add_argument("--ly",         type=int,   default=3)
    parser.add_argument("--bc",         type=str,   default="pbc")
    parser.add_argument("--model",      type=str,   default="kitaev")
    parser.add_argument("-J",           type=float, default=1.0)
    parser.add_argument("-K",           type=float, default=1.0)
    parser.add_argument("--Gamma",      type=float, default=0.0)
    parser.add_argument("--Jz",         type=float, default=1.0)
    parser.add_argument("--J2",         type=float, default=0.0)
    parser.add_argument("--hx",         type=float, default=0.0)
    parser.add_argument("--hz",         type=float, default=0.0)
    parser.add_argument("--impurity",   type=float, nargs=4, action="append")
    parser.add_argument("--ansatz",     type=str,   default="rbm")
    parser.add_argument("--alpha",      type=float, default=None)
    parser.add_argument("--complex",    action="store_true")
    parser.add_argument("--accuracy",   type=str,   default="medium")
    parser.add_argument("--backend",    type=str,   default="jax")
    parser.add_argument("--num-chains",  type=int,   default=16)
    parser.add_argument("--num-samples", type=int,   default=1000)
    parser.add_argument("--num-therm",   type=int,   default=100)
    parser.add_argument("--num-sweep",   type=int,   default=10)
    parser.add_argument("--sampler-rule",type=str,   default="LOCAL")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--lr-scheduler",type=str,  default="cosine")
    parser.add_argument("--optimizer",  type=str,   default=None, help="Optax optimizer: adam, sgd, adamw")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience",   type=int,   default=50, help="Early stopping patience")
    parser.add_argument("--no-sr",      action="store_true")
    parser.add_argument("--diag-shift", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--ed",         action="store_true")
    parser.add_argument("--max-ed-size",type=int,   default=16)
    parser.add_argument("--load",       type=str,   default=None)
    parser.add_argument("--save-dir",   type=str,   default="./demo_output")
    parser.add_argument("--compute-observables", action="store_true")
    parser.add_argument("--compute-correlations", action="store_true")
    parser.add_argument("--compute-entropy",      action="store_true")
    parser.add_argument("--num-samples-entropy",  type=int, default=4096)
    parser.add_argument("--plot",       action="store_true")
    parser.add_argument("--show",       action="store_true")
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        args.lx, args.ly, args.lattice, args.model = 2, 2, "honeycomb", "kitaev"
        args.epochs, args.train, args.ed, args.compute_observables = 50, True, True, True
        args.complex, args.plot = True, True
    os.makedirs(args.save_dir, exist_ok=True)
    run_demo(args)
