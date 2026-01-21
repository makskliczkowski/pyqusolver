
import pytest
import numpy as np
from unittest.mock import MagicMock

from QES.NQS.src.nqs_train import NQSTrainer, NQS, TDVP, TDVPAutoTuner, NQSTimeModes
from QES.Solver.MonteCarlo.vmc import VMCSampler

class DummyNQS(NQS):
    def __init__(self):
        # Bypass NQS.__init__
        self._backend = np
        self._backend_str = 'numpy'
        self._verbose = False
        self._dtype = np.float64
        self._sampler = MagicMock()
        self._sampler.accepted_ratio = 0.5
        self._sampler.diagnose.return_value = {'r_hat': 1.05, 'ess': 500}
        self._net = MagicMock()
        self._precision_policy = MagicMock()
        self._precision_policy.accum_real_dtype = np.float64
        self._precision_policy.accum_complex_dtype = np.complex128

        # Mocking required methods
        self.wrap_single_step_jax = MagicMock(return_value=lambda *args, **kwargs: (np.zeros(10), 0.0, (MagicMock(mean_energy=1.0, std_energy=0.1, theta0_dot=None), ([], [], False))))
        self.get_params = MagicMock(return_value=np.zeros(10))
        self.set_params = MagicMock()
        self.sample = MagicMock(return_value=((None, None), (np.zeros((10,10)), np.zeros(10)), np.zeros(10)))

        self.defdir = "."
        self.ckpt_manager = MagicMock()
        self._exact_info = None
        self._modifier_wrapper = None
        self._seed = 123
        self._shape = (10,)
        self._logger = MagicMock()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def backend(self):
        return self._backend

    @property
    def backend_str(self):
        return self._backend_str

    @property
    def verbose(self):
        return self._verbose

def test_trainer_initialization_with_autotuner():
    nqs = DummyNQS()
    trainer = NQSTrainer(nqs, auto_tune=True)
    assert trainer.auto_tuner is not None
    assert isinstance(trainer.auto_tuner, TDVPAutoTuner)

def test_trainer_step_autotuner_logic():
    nqs = DummyNQS()
    trainer = NQSTrainer(nqs, auto_tune=True, n_batch=100)

    # Run one step of training
    # Mocking ode_solver step to return dummy data
    trainer.ode_solver = MagicMock()
    trainer.ode_solver.step.return_value = (np.zeros(10), 0.1, (MagicMock(mean_energy=1.0, std_energy=0.1, sr_converged=True, theta0_dot=None), ([], [], False)))

    # Disable JIT for testing with Mocks by wrapping the nojit function
    def dummy_step(*args, **kwargs):
        # The nojit function expects lower_states
        if 'lower_states' not in kwargs:
            kwargs['lower_states'] = None
        return trainer._step_nojit(*args, **kwargs)

    trainer._step_jit = dummy_step

    stats = trainer.train(n_epochs=1, use_pbar=False)

    # Check if diagnose was called
    nqs.sampler.diagnose.assert_called()

    # Check if auto_tuner update was called (implicitly by checking if params changed or mocked)
    # But hard to check internal state without better mocks.
    # At least ensure no crash.
    assert len(stats.history) == 1
