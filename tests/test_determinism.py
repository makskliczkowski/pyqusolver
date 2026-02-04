import unittest
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys

# Ensure pythonpath
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Python'))

from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.NQS.nqs import NQS

class TestNQSDeterminism(unittest.TestCase):
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

    def test_sampling_determinism(self):
        """Test that sampling is deterministic given a seed."""
        # Run 1
        nqs1 = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=42
        )
        nqs1.sampler.set_numchains(4)
        nqs1.sampler.set_numsamples(10)
        _, (samples1, _), _ = nqs1.sample(reset=True)

        # Run 2
        nqs2 = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=42
        )
        nqs2.sampler.set_numchains(4)
        nqs2.sampler.set_numsamples(10)
        _, (samples2, _), _ = nqs2.sample(reset=True)

        np.testing.assert_array_equal(samples1, samples2, err_msg="Sampling not deterministic across runs with same seed")

    def test_sampling_randomness(self):
        """Test that sampling changes with different seeds."""
        # Run 1
        nqs1 = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=42
        )
        nqs1.sampler.set_numchains(4)
        nqs1.sampler.set_numsamples(10)
        _, (samples1, _), _ = nqs1.sample(reset=True)

        # Run 2
        nqs2 = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=43
        )
        nqs2.sampler.set_numchains(4)
        nqs2.sampler.set_numsamples(10)
        _, (samples2, _), _ = nqs2.sample(reset=True)

        self.assertFalse(np.array_equal(samples1, samples2), "Sampling should be different for different seeds")

    def test_optimization_step_determinism(self):
        """Test that optimization step is deterministic."""

        nqs1 = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=100
        )
        # Force initialization
        nqs1.init_network()
        params1 = nqs1.get_params(unravel=True)

        # Take a step
        res1 = nqs1.step(batch_size=4, num_samples=10, num_chains=4)
        grad1 = res1.grad_flat

        nqs2 = NQS(
            logansatz='rbm',
            model=self.hamil,
            backend='jax',
            hilbert=self.hilbert,
            sampler='vmc',
            seed=100
        )
        nqs2.init_network()
        params2 = nqs2.get_params(unravel=True)

        # Initial params must be same
        np.testing.assert_allclose(params1, params2, err_msg="Initial params not deterministic")

        res2 = nqs2.step(batch_size=4, num_samples=10, num_chains=4)
        grad2 = res2.grad_flat

        np.testing.assert_allclose(grad1, grad2, err_msg="Gradients not deterministic")

if __name__ == '__main__':
    unittest.main()
