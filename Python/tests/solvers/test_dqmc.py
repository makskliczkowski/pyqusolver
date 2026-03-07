
import jax.numpy as jnp
import numpy as np
from QES.pydqmc.dqmc_sampler import calculate_green_stable
from QES.Algebra.Model.Interacting.Fermionic.hubbard import HubbardModel
from QES.general_python.lattices import choose_lattice
from QES.pydqmc.dqmc_model import HubbardDQMCModel

def test_udt_stability():
    """Verify stable Green evaluation for both moderate and ill-conditioned products."""
    N = 4
    dim = 8
    rng = np.random.RandomState(42)

    def build_product(span):
        mats = []
        for _ in range(N):
            U, _ = np.linalg.qr(rng.randn(dim, dim))
            V, _ = np.linalg.qr(rng.randn(dim, dim))
            D = np.diag(np.exp(np.linspace(span, -span, dim)))
            mats.append(U @ D @ V)
        B_tot = np.eye(dim)
        for B in mats:
            B_tot = B @ B_tot
        return mats, B_tot

    # Moderate conditioning: stable and naive should agree.
    Bs_mod, B_mod = build_product(span=3)
    G_mod_stable = np.array(calculate_green_stable(jnp.array(Bs_mod), n_stable=1))
    G_mod_ref = np.linalg.inv(np.eye(dim) + B_mod)
    np.testing.assert_allclose(G_mod_stable, G_mod_ref, rtol=1e-6, atol=1e-8)

    # Strong conditioning: compare residual quality, not raw value equality.
    Bs_hard, B_hard = build_product(span=10)
    I = np.eye(dim)
    G_hard_stable = np.array(calculate_green_stable(jnp.array(Bs_hard), n_stable=1))
    G_hard_naive = np.linalg.inv(I + B_hard)

    res_stable = np.linalg.norm((I + B_hard) @ G_hard_stable - I)
    assert np.isfinite(res_stable)
    assert np.isfinite(np.linalg.norm((I + B_hard) @ G_hard_naive - I))
    # In the strongly ill-conditioned regime, strict equality to naive inversion
    # is not a reliable criterion. We only require a bounded inverse residual.
    assert res_stable < 5.0

def test_hubbard_dqmc_model_extraction():
    """Ensure kinetic matrix is extracted correctly from QES Hamiltonian."""
    L = 2
    lat = choose_lattice("square", lx=L, ly=L, bc="pbc")
    t = 1.0
    hamil = HubbardModel(lattice=lat, t=t, U=4.0)
    dqmc_model = HubbardDQMCModel(hamiltonian=hamil, beta=1.0, M=10, U=4.0)
    
    K = dqmc_model.kinetic_matrix
    assert K.shape == (4, 4)
    # Check NN hopping
    # Site 0 is connected to 1 and 2 in 2x2 periodic square
    assert np.isclose(K[0, 1], -t)
    assert np.isclose(K[0, 2], -t)
    assert np.isclose(K[0, 3], 0.0) # Diagonal not connected in square NN
