import numpy as np

from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev

def _to_dense(mat):
    ''' Utility function to convert a sparse matrix to a dense array, if it has a `toarray` method, otherwise just convert to a NumPy array.'''
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)

def test_heisenberg_kitaev_forward_flux_build_is_hermitian():
    ''' Test that the Hamiltonian built with forward hopping and flux is Hermitian. 
    This is a nontrivial test because the flux-induced shifts from twisted boundary conditions 
    can lead to complex hopping terms, and we want to ensure that the implementation correctly maintains Hermiticity in this case.'''
    
    lat = HoneycombLattice(
        dim=2,
        lx=2,
        ly=2,
        bc="pbc",
        flux={"x": 0.0, "y": 2.0 * np.pi / 3.0},
    )
    
    hamil = HeisenbergKitaev(
        lattice=lat,
        J=1.0,
        K=(0.0, 0.0, 0.0),
        Gamma=None,
        hx=0.0,
        hy=0.0,
        hz=0.0,
        use_forward=True,
        dtype=np.complex128,
    )
    hamil.build(verbose=False)

    H               = _to_dense(hamil.hamil)
    hermitian_err   = np.linalg.norm(H - H.conj().T)
    assert hermitian_err < 1e-10

# ------------------------------
#! EOF
# ------------------------------