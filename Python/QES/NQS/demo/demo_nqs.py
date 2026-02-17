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

import  argparse
import  copy
import  os
import  sys
import  time
from    pathlib import Path
from    typing  import List, Optional, Tuple, Union, Dict, Any

import  numpy as np
import  matplotlib.pyplot as plt

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

from QES.general_python.lattices            import choose_lattice
from QES.Algebra.Model.Interacting.Spin     import HeisenbergKitaev, TransverseFieldIsing, XXZ, J1J2Model
from QES.NQS.nqs                            import NQS
from QES.NQS.src.nqs_network_integration    import NetworkFactory, estimate_network_params
from QES.NQS.src.nqs_entropy                import compute_renyi_entropy, bipartition_cuts_honeycomb
from QES.general_python.common.flog         import get_global_logger
from QES.general_python.common.plot         import Plotter

logger = get_global_logger()

# ----------------------------------------------------------------------------
#! Helpers
# ----------------------------------------------------------------------------

def make_hamiltonian(args):
    """Factory for Hamiltonian creation from CLI args."""
    lattice = choose_lattice(
        typek   = args.lattice,
        lx      = args.lx,
        ly      = args.ly,
        bc      = args.bc,
    )
    
    model_type = args.model.lower()
    
    # Process impurities
    impurities = []
    if args.impurity:
        for imp in args.impurity:
            # site, phi, theta, amplitude
            impurities.append((int(imp[0]), imp[1], imp[2], imp[3]))

    if model_type == 'kitaev':
        # Heisenberg-Kitaev model
        hamil = HeisenbergKitaev(
            lattice =   lattice,
            K       =   args.K,
            J       =   args.J,
            hx      =   args.hx,
            hz      =   args.hz,
            impurities = impurities,
            dtype   =   np.complex128 if args.complex else np.float64
        )
    elif model_type == 'heisenberg':
        hamil = HeisenbergKitaev(
            lattice =   lattice,
            K       =   0.0,
            J       =   args.J,
            hx      =   args.hx,
            hz      =   args.hz,
            impurities = impurities,
            dtype   =   np.float64
        )
    elif model_type == 'tfim':
        hamil = TransverseFieldIsing(
            lattice =   lattice,
            j       =   args.J,
            hx      =   args.hx,
            hz      =   args.hz,
        )
    elif model_type == 'xxz':
        hamil = XXZ(
            lattice =   lattice,
            jxy     =   args.J,
            jz      =   args.Jz,
            hx      =   args.hx,
            hz      =   args.hz,
        )
    elif model_type == 'j1j2':
        hamil = J1J2Model(
            lattice =   lattice,
            J1      =   args.J,
            J2      =   args.J2,
            impurities = impurities,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    return hamil, lattice

def make_net(args, num_sites):
    """Factory for network creation from CLI args."""
    
    # Use SOTA parameter estimation
    sota_cfg = estimate_network_params(
        net_type        = args.ansatz,
        num_sites       = num_sites,
        lattice_dims    = (args.lx, args.ly),
        lattice_type    = args.lattice,
        model_type      = args.model,
        target_accuracy = args.accuracy,
        dtype           = 'complex128' if args.complex else 'float64',
    )
    
    # User overrides
    factory_kwargs = sota_cfg.to_factory_kwargs()
    if args.alpha: factory_kwargs['alpha'] = args.alpha
    
    # Check for GCNN symmetry perms
    if args.ansatz in ('gcnn', 'eqgcnn'):
        # For GCNN we often need to provide symmetry perms explicitly if not using from_lattice
        # But NetworkFactory.create handles it via kwargs if provided
        pass

    net = NetworkFactory.create(
        network_type    = args.ansatz,
        **factory_kwargs
    )
    
    logger.info(f"Created {args.ansatz} network with {net.nparams} parameters.", lvl=1, color='blue')
    logger.info(f"SOTA config: {sota_cfg.description}", lvl=2)
    
    return net

# ----------------------------------------------------------------------------
#! Main Execution
# ----------------------------------------------------------------------------

def run_demo(args):
    """Main demo workflow."""
    
    # 1. Setup Physical Model
    hamil, lattice = make_hamiltonian(args)
    num_sites = lattice.ns
    logger.title(f"NQS Demo: {args.model.upper()} on {args.lattice.upper()} ({args.lx}x{args.ly}, Ns={num_sites})")
    
    # 2. Setup NQS
    net = make_net(args, num_sites)
    
    # Sampler config
    sample_config = {
        's_numchains'   : args.num_chains,
        's_numsamples'  : args.num_samples,
        's_therm_steps' : args.num_therm,
        's_sweep_steps' : args.num_sweep,
        's_upd_fun'     : args.sampler_rule,
    }
    
    psi = NQS(
        logansatz   =   net,
        model       =   hamil,
        backend     =   args.backend,
        dtype       =   jnp.complex128 if args.complex else jnp.float64,
        directory   =   args.save_dir,
        seed        =   args.seed,
        **sample_config
    )
    
    # 3. Load Weights if specified
    if args.load:
        logger.info(f"Loading weights from {args.load}...", lvl=1, color='green')
        psi.load_weights(args.load)
        
    # 4. Exact Diagonalization (for comparison)
    ed_stats = None
    if args.ed or (num_sites <= args.max_ed_size):
        logger.info(f"Performing Exact Diagonalization (Lanczos) for comparison...", lvl=1, color='cyan')
        ed_stats = psi.get_exact(k=6, verbose=True)
        
    # 5. Training
    if args.train:
        logger.info(f"Starting Variational Ground State Search (SR)...", lvl=1, color='green')
        train_config = {
            'n_epochs'          : args.epochs,
            'checkpoint_every'  : args.checkpoint_every,
            'lr'                : args.lr,
            'lr_scheduler'      : args.lr_scheduler,
            'use_sr'            : not args.no_sr,
            'diag_shift'        : args.diag_shift,
            'use_pbar'          : True,
        }
        stats = psi.train(**train_config)
        
        # Plot training history
        if args.plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            history = np.array(stats.history)
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
            plt.savefig(Path(args.save_dir) / "training_history.png", dpi=200)
            logger.info(f"Saved training plot to {args.save_dir}/training_history.png", lvl=2)

    # 6. Compute Observables
    if args.compute_observables or args.compute_correlations:
        logger.info(f"Computing ground state observables...", lvl=1, color='green')
        
        # Sample states from trained model
        (_, _), (states, log_psi), _ = psi.sample(num_samples=2000)
        
        # Magnetization
        if args.model.lower() in ('kitaev', 'heisenberg', 'tfim', 'xxz'):
            sz_op = hamil.operators.sig_z(ns=num_sites, type_act='local')
            sz_vals = psi.compute_observable(functions=sz_op.jax, states=states, ansatze=log_psi)
            logger.info(f"Average Magnetization <Sz>: {np.mean(sz_vals.mean):.6f} +/- {np.mean(sz_vals.error_of_mean):.6f}", lvl=2)

        # Spin-Spin Correlations
        if args.compute_correlations:
            logger.info("Computing spin-spin correlation matrix <Sz_i Sz_j>...", lvl=2)
            # Use correlator kernels
            corr_kernels = hamil.operators.correlators(
                correlators = ['zz'],
                type_acting = 'global',
                compute     = False
            )
            zz_kernel = corr_kernels['zz']['i,j'].jax
            
            # Compute for all pairs (Warning: O(N^2) calls)
            corr_mat = np.zeros((num_sites, num_sites))
            for i in range(num_sites):
                for j in range(i, num_sites):
                    res = psi.compute_observable(functions=zz_kernel, states=states, ansatze=log_psi, args=(i, j))
                    corr_mat[i, j] = np.real(res.mean)
                    corr_mat[j, i] = corr_mat[i, j]
            
            if args.plot:
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-0.25, vmax=0.25)
                plt.colorbar(im, label=r'$\langle S^z_i S^z_j \rangle$')
                ax.set_title(f'Spin-Spin Correlations ({args.model})')
                ax.set_xlabel('Site $j$')
                ax.set_ylabel('Site $i$')
                plt.tight_layout()
                plt.savefig(Path(args.save_dir) / "correlations.png", dpi=200)
                logger.info(f"Saved correlation plot to {args.save_dir}/correlations.png", lvl=2)

        # Plaquette Operators (for Kitaev Honeycomb)
        if args.model.lower() == 'kitaev' and args.lattice == 'honeycomb':
            logger.info("Computing Kitaev plaquette operators W_p...", lvl=2)
            try:
                # W_p = sigma^x_1 sigma^y_2 sigma^z_3 sigma^x_4 sigma^y_5 sigma^z_6
                # We need to get hexagons from lattice
                plaquettes = lattice.get_plaquettes()
                if plaquettes:
                    w_p_values = []
                    for p_sites in plaquettes:
                        # Construct product operator
                        # This is a bit involved to construct on the fly, 
                        # but we can use the kernel approach if we have a W_p kernel.
                        # For now, let's just log that we found them.
                        pass
                    logger.info(f"Identified {len(plaquettes)} plaquettes on the lattice.", lvl=3)
            except Exception as e:
                logger.warning(f"Could not compute plaquettes: {e}", lvl=3)

    # 7. Entanglement Entropy
    if args.compute_entropy:
        logger.info(f"Estimating Rényi entanglement entropy (replica method)...", lvl=1, color='green')
        
        # Determine bipartition cuts
        if args.lattice == 'honeycomb':
            cuts = bipartition_cuts_honeycomb(lattice, cut_type="half_x")
        else:
            # Default to half-system cut for other lattices
            cuts = {"half": np.arange(num_sites // 2)}
            
        for label, region in cuts.items():
            s2, s2_err = compute_renyi_entropy(
                psi, region=region, q=2, num_samples=args.num_samples_entropy, return_error=True
            )
            logger.info(f"Rényi-2 Entropy ({label}, size={len(region)}): {s2:.6f} +/- {s2_err:.6f}", lvl=2)

    if args.show:
        plt.show()

# ----------------------------------------------------------------------------
#! CLI Arguments
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NQS Demo: Variational Ground State Search")
    
    # Lattice & Model
    parser.add_argument("--lattice",    type=str,   default="honeycomb", help="Lattice type")
    parser.add_argument("--lx",         type=int,   default=4,           help="Lattice Lx")
    parser.add_argument("--ly",         type=int,   default=3,           help="Lattice Ly")
    parser.add_argument("--bc",         type=str,   default="pbc",       help="Boundary conditions (pbc/obc)")
    parser.add_argument("--model",      type=str,   default="kitaev",    help="Model type (kitaev/heisenberg/tfim/xxz/j1j2)")
    
    # Model Params
    parser.add_argument("-J",           type=float, default=1.0,         help="Heisenberg coupling J")
    parser.add_argument("-K",           type=float, default=1.0,         help="Kitaev coupling K")
    parser.add_argument("--Jz",         type=float, default=1.0,         help="Ising coupling Jz (for XXZ)")
    parser.add_argument("--J2",         type=float, default=0.0,         help="NNN coupling J2 (for J1J2)")
    parser.add_argument("--hx",         type=float, default=0.0,         help="Field hx")
    parser.add_argument("--hz",         type=float, default=0.0,         help="Field hz")
    parser.add_argument("--impurity",   type=float, nargs=4, action="append", help="Add impurity: site phi theta amplitude")
    
    # NQS Setup
    parser.add_argument("--ansatz",     type=str,   default="rbm",       help="Ansatz type (rbm/cnn/resnet/eqgcnn/ar)")
    parser.add_argument("--alpha",      type=float, default=None,        help="RBM hidden density (hidden = alpha * sites)")
    parser.add_argument("--complex",    action="store_true",             help="Use complex-valued NQS")
    parser.add_argument("--accuracy",   type=str,   default="medium",    help="SOTA parameter tier (fast/medium/high)")
    parser.add_argument("--backend",    type=str,   default="jax",       help="Computation backend (jax/numpy)")
    
    # Sampler
    parser.add_argument("--num-chains",  type=int,   default=16,         help="Number of MCMC chains")
    parser.add_argument("--num-samples", type=int,   default=1000,       help="Samples per chain per epoch")
    parser.add_argument("--num-therm",   type=int,   default=100,        help="Thermalization steps")
    parser.add_argument("--num-sweep",   type=int,   default=10,         help="Sweep steps between samples")
    parser.add_argument("--sampler-rule",type=str,   default="LOCAL",    help="Update rule (LOCAL/EXCHANGE/WORM)")
    
    # Training
    parser.add_argument("--train",      action="store_true",             help="Run training loop")
    parser.add_argument("--epochs",     type=int,   default=300,         help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=0.01,        help="Learning rate")
    parser.add_argument("--lr-scheduler",type=str,  default="cosine",    help="LR scheduler type")
    parser.add_argument("--no-sr",      action="store_true",             help="Use plain SGD instead of SR")
    parser.add_argument("--diag-shift", type=float, default=1e-3,        help="SR regularization diagonal shift")
    parser.add_argument("--checkpoint-every", type=int, default=50,      help="Save weights every N epochs")
    
    # Actions & Utilities
    parser.add_argument("--test",       action="store_true",             help="Run small test scenario")
    parser.add_argument("--ed",         action="store_true",             help="Force Exact Diagonalization")
    parser.add_argument("--max-ed-size",type=int,   default=16,          help="Auto-ED if num_sites <= this")
    parser.add_argument("--load",       type=str,   default=None,        help="Path to weights file to load")
    parser.add_argument("--save-dir",   type=str,   default="./demo_output", help="Output directory")
    parser.add_argument("--compute-observables", action="store_true",    help="Compute magnetization and energy")
    parser.add_argument("--compute-correlations", action="store_true",   help="Compute spin correlation matrix")
    parser.add_argument("--compute-entropy",      action="store_true",   help="Estimate Rényi-2 entropy")
    parser.add_argument("--num-samples-entropy",  type=int, default=4096, help="Samples for entropy estimation")
    parser.add_argument("--plot",       action="store_true",             help="Save plots")
    parser.add_argument("--show",       action="store_true",             help="Show plots interactively")
    parser.add_argument("--seed",       type=int,   default=42,          help="Random seed")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        # Override for small test
        args.lx = 2
        args.ly = 2
        args.lattice = "honeycomb"
        args.model = "kitaev"
        args.epochs = 50
        args.train = True
        args.ed = True
        args.compute_observables = True
        args.compute_correlations = True
        args.compute_entropy = True
        args.complex = True
        args.plot = True
        
    os.makedirs(args.save_dir, exist_ok=True)
    run_demo(args)
