import unittest
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys

from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.NQS.nqs import NQS

class TestNQSInvariants(unittest.TestCase):
    def setUp(self):
        # Setup a small system
        self.lattice = HoneycombLattice(dim=2, lx=1, ly=1) # 2 sites
        self.hilbert = HilbertSpace(lattice=self.lattice)
        self.hamil = HeisenbergKitaev(
            hilbert_space=self.hilbert,
            lattice=self.lattice,
            K=1.0, J=0.5,
            backend='jax'
        )
        self.nqs = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=42
        )

    def test_nqs_properties(self):
        """Test basic NQS properties."""
        self.assertEqual(self.nqs.nvisible, 2)
        self.assertEqual(self.nqs.backend_str, 'jax')

    def test_nqs_output_shape(self):
        """Test NQS output shape for a batch of states."""
        n_states = 10
        key = jax.random.PRNGKey(0)
        states = jax.random.choice(key, jnp.array([-1, 1]), shape=(n_states, self.lattice.ns))

        # Cast states to float if network expects it (default precision policy might expect float)
        states = states.astype(jnp.float64)

        log_psi = self.nqs(states)
        self.assertEqual(log_psi.shape, (n_states,))

        # Check dtype
        self.assertTrue(jnp.iscomplexobj(log_psi) or jnp.issubdtype(log_psi.dtype, jnp.floating))

    def test_probability_positive(self):
        """Test that sampled probabilities are positive."""
        _, _, probs = self.nqs.sample(num_samples=100)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(np.isfinite(probs)))

if __name__ == '__main__':
    unittest.main()
