import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from QES.Algebra.Model.Interacting.Fermionic.hubbard import HubbardModel
from QES.general_python.lattices.square import SquareLattice
from QES.pydqmc.dqmc_solver import DQMCSolver
from QES.Solver.MonteCarlo.montecarlo import McsTrain

def test_and_plot_gs():
    # 1. Setup small 4x4 Lattice
    L = 4
    lat = SquareLattice(lx=L, ly=L, dim=2, bc="pbc")
    
    # 2. Parameters
    t = 1.0
    U = 2.0  # Reduced for stability
    beta = 2.0
    M = 40 
    
    hamil = HubbardModel(lattice=lat, t=t, U=U)
    
    # 3. Solver
    solver = DQMCSolver(model=hamil, beta=beta, M=M, num_chains=2, n_stable=5)
    
    print(f"Running DQMC for {L}x{L} Hubbard model...")
    print(f"Beta={beta}, U={U}, M={M}")
    
    # 4. Short Simulation
    train_params = McsTrain(mcth=50, mcsam=100, mc_corr=2)
    solver.train(train_params, verbose=True)
    
    # 5. Extract Equal-Time Green's Function G(0)
    G_up = np.array(solver.sampler.Gs_avg[0][0])
    
    # 6. Extract Unequal-Time Green's Function G(tau, 0)
    unequal_gs = solver.sampler.compute_unequal_time_Gs()
    G_tau = np.array(unequal_gs[0, 0])
    
    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    # Plot 1: Spatial Green's Function G(i, 0) heatmap
    G_spatial = G_up[0].reshape(L, L)
    im = axes[0].imshow(G_spatial, cmap='RdBu_r', origin='lower')
    axes[0].set_title(f"Spatial Green's Function $G_{{i,0}}$ (Equal-time)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0])
    
    # Plot 2: Imaginary Time Decay G(tau, 0)
    G_local_tau = np.mean(np.diagonal(G_tau, axis1=-2, axis2=-1), axis=-1)
    taus = np.linspace(0, beta, M)
    axes[1].plot(taus, G_local_tau, 'o-', markersize=3, label="Local $G(\\tau)$")
    axes[1].set_yscale('log')
    axes[1].set_title("Imaginary Time Decay (Local)")
    axes[1].set_xlabel("$\\tau$")
    axes[1].set_ylabel("$\\langle c_i(\\tau) c_i^\\dagger(0) \\rangle$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plot_path = "dqmc_gs_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Simple validation
    obs = solver.measure_observables()
    print(f"Final Obs: Energy={obs['energy']:.4f}, Density={obs['density']:.4f}")

if __name__ == "__main__":
    test_and_plot_gs()
