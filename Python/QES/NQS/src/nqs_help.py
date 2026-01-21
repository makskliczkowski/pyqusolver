"""
Help module for NQS.
"""
from QES.general_python.ml.networks import NetworkFactory

def nqs_help(self, topic: str = "general"):
    """
    Prints usage information and physics background for NQS features.

    Parameters
    ----------
    topic : str
        The topic to query. Options:
        - 'general':
            Overview of the NQS object.
        - 'modifier':
            Details on state modifiers (Projectors, Symmetries).
        - 'sampling':
            Info on VMC vs Autoregressive sampling.
        - 'network':
            Details about the loaded ansatz.
        - 'networks':
            Overview of available ansatz architectures.
        - 'usage':
            Example workflows and common operations.
        - 'train':
            Training with the train() method (LR scheduling, SR, etc.).
        - 'checkpoints':
            Saving and loading model weights and checkpoints.
    """
    topic   = topic.lower().strip()
    msg     = ""
    border  = "-" * 60

    if topic == "general":
        msg = f"""
            {border}
            NQS Solver Help: General
            {border}
            This object represents a variational quantum state |Psi_theta>.

            Current Configuration:
            - Backend: {self._backend_str}
            - Ansatz: {type(self._net).__name__}
            - Params: {self.num_params}
            - Sampler: {type(self._sampler).__name__} (Batch: {self._batch_size}, mu: {self._sampler._mu})

            Key Methods:
            - train(...): Optimize parameters via VMC/TDVP.
            - sample(...): Generate configurations s ~ |Psi(s)|^{self._sampler._mu} (default mu=2 - squared amplitude).
            - evaluate(s): Compute log(Psi(s)).
            - set_modifier(O): Transform ansatz to O|Psi>.
            """

    elif topic == "modifier":
        msg = f"""
            {border}
            NQS Solver Help: State Modifiers
            {border}
            Modifiers allow you to apply an operator O to the ansatz:
            |Psi_new> = O |Psi_old>

            This is done on-the-fly during evaluation. The NQS computes:
            log <s|Psi_new> = log ( sum_k <s|O|s'_k> * <s'_k|Psi_old> )

            Usage:
                op = Operator(...) # e.g., Symmetry projector or S^z
                nqs.set_modifier(op)

            Performance Modes (Auto-Detected):
            1. Single-Branch (M=1):
            Operator maps s -> s'. Very fast.
            Used for:
                Quantum Numbers (S^z, N), Basis Rotations. Depends on operator.

            2. Multi-Branch (M>1):
            Operator maps s -> sum_k w_k |s'_k>. Slower (evaluates net M times).
            Used for:
                Symmetries (Sum over group), Hamiltonian action.

            Current Status:
            - Modified: {self.modified}
            - Modifier: {self._modifier}
            """

    elif topic == "sampling":
        msg = f"""
            {border}
            NQS Solver Help: Sampling
            {border}
            Current Sampler: {type(self._sampler).__name__}

            1. MCMC (VMCSampler):
            - Uses Metropolis-Hastings.
            - Good for general RBMs/CNNs.
            - Suffers from autocorrelation time (tau).

            2. Autoregressive (ARSampler):
            - Generates samples sequentially (s1 -> s2 -> ...).
            - Zero autocorrelation (iid samples).
            - Requires 'ar' or 'made' network architecture.
            - Exact likelihoods P(s) available.

            Update Rules (VMC):
            - "LOCAL" (default):
                - Single spin flips.
            - "EXCHANGE":
                - Neighbor swaps (conserves N, Sz).
            - "GLOBAL":
                - Pattern/Plaquette flips (reduces autocorrelation).
            - "MULTI_FLIP":
                - Flips N random sites.
            - "SUBPLAQUETTE":
                - Flips sub-sequences of a pattern (e.g. edges of a plaquette).

            Change rule:
                psi = NQS(..., upd_fun="EXCHANGE", hilbert=h)
                # or during training
                psi.train(..., upd_fun="GLOBAL", update_kwargs={{'patterns': [...]}})
            """

    elif topic == "usage":
        msg = f"""
            {border}
            NQS Solver Help: General usage
            {border}
            Current Ansatz: {type(self._net).__name__}

            1. Initialization
            psi     = NQS(logansatz='ar', model=hamil, sampler='ARSampler')
            # or
            psi     = NQS(logansatz=custom_net, sampler=custom_sampler)
            # or VMC
            psi     = NQS(logansatz='rbm', model=hamil, sampler='MCSampler', backend='jax', s_numsamples=5000)

            2. Training (via NQSTrainer)
            trainer = NQSTrainer(psi, lin_solver='jax_cg', ...)
            stats   = trainer.train(n_epochs=100)

            3. Sampling & Observables
            # Get raw samples and log-amplitudes
            (_, _), (configs, log_psi), weights = psi.sample(num_samples=1000)

            # Compute Expectation Values (e.g. Energy)
            E_stats = psi.compute_energy(configs)
            print(f"Energy: {{E_stats.mean:.4f}} +/- {{E_stats.error:.4f}}")

            4. I/O Operations
            psi.save_weights("checkpoint.h5")
            psi.load_weights("checkpoint.h5")

            5. Dynamic Settings (What you can change)
            psi.batch_size = 2048  # Adjust batch size for evaluation
            psi.net = new_net      # Swap architecture (resets optimizer)
            psi.sampler = 'vmc'    # Switch sampling strategy

            6. Compute observables, for example for Hamiltonian for spin-1/2
            lat = Lattice(...)
            mod = Hamiltonian(...)
            sig_x = mod.operators.sig_x(lattice=lat, sites=[0])
            obs_x = psi.compute_observable(sig_x, num_samples=1000)
            print(f"<Sx_0> = {{obs_x.mean:.4f}} +/- {{obs_x.error:.4f}}")
            """

    elif topic == "train":
        msg = f"""
            {border}
            NQS Solver Help: Training with train()
            {border}
            The train() method provides a convenient way to train the NQS
            without manually creating an NQSTrainer instance.

            Basic Usage:
                stats = psi.train(n_epochs=300)

            Key Parameters:
            - n_epochs: Number of training epochs (default: 300)
            - checkpoint_every: Save checkpoint every N epochs (default: 50)
            - override: If True (default), create new trainer; if False, reuse existing

            Learning Rate Scheduling:
                # Constant LR
                stats = psi.train(lr=1e-3)

                # Cosine annealing (recommended)
                stats = psi.train(lr=1e-2, lr_scheduler='cosine', min_lr=1e-5)

                # Exponential decay (gamma^epoch)
                stats = psi.train(lr=1e-2, lr_scheduler='exponential', lr_decay=0.99)

            Stochastic Reconfiguration:
                # Standard SR
                stats = psi.train(use_sr=True, diag_shift=1e-4)

                # MinSR (memory efficient for large networks)
                stats = psi.train(use_minsr=True)

                # Plain gradient descent
                stats = psi.train(use_sr=False)

            Comparison with Exact Diagonalization:
                hamil.diagonalize()
                stats = psi.train(exact_predictions=hamil.eigenvalues, exact_method='lanczos')

            Continuing Training (reuse optimizer state):
                stats = psi.train(n_epochs=100)         # First run
                stats = psi.train(n_epochs=100, override=False)  # Continue

            Accessing the Trainer:
                psi.train(n_epochs=100)
                trainer = psi.trainer  # Access underlying NQSTrainer

            Current Status:
            - Trainer exists: {self._trainer is not None}
            - Trainer type: {type(self._trainer).__name__ if self._trainer else 'None'}
            """

    elif topic == "network":
        info = NetworkFactory.net_help()
        if self._net is not None:
            net_type    = self._net.name if hasattr(self._net, 'name') else type(self._net).__name__
            msg         = f"""
                        {border}
                        NQS Solver Help: Network Details
                        {border}
                        Current Network Type: {net_type}

                        {info.get(net_type, "No additional info available for this network type.")}
                        """
    elif topic == "networks":
        info = NetworkFactory.net_help()
        msg  = f"""
            {border}
            NQS Solver Help: Available Networks
            {border}
            The following ansatz architectures are available:

            {chr(10).join([f"- {k}: {v.splitlines()[0]}" for k,v in info.items()])}

            For detailed information on each architecture, use:
                psi.help(topic='network')
            """

    elif topic == "checkpoints":
        msg = f"""
            {border}
            NQS Solver Help: Checkpoints & I/O
            {border}
            Save and load model weights for resuming training or inference.

            Manual Save/Load (Weights Only):
                # Save current network parameters
                psi.save_weights("my_model.h5")

                # Load weights (network architecture must match)
                psi.load_weights("my_model.h5")

                # With custom path
                psi.save_weights("/path/to/checkpoints/epoch_100.h5")

            Automatic Checkpoints During Training:
                stats = psi.train(
                    n_epochs=300,
                    checkpoint_every=50,       # Save every 50 epochs
                    save_path="./checkpoints"  # Directory for auto-saves
                )
                # Creates: ./checkpoints/epoch_50.h5, epoch_100.h5, ...

            Resume Training from Checkpoint:
                # Load weights from previous run
                psi.load_weights("./checkpoints/epoch_200.h5")

                # Continue training (new trainer)
                stats = psi.train(n_epochs=100)

                # Or continue with same optimizer state
                stats = psi.train(n_epochs=100, override=False)

            File Format:
                - HDF5 (.h5) format for weights
                - Contains: network parameters, metadata
                - Compatible with JAX/Flax serialization

            Tips:
                - Always save after successful training
                - Use descriptive filenames (e.g., 'gs_N16_E-5.234.h5')
                - Keep checkpoint_every reasonable (50-100 epochs)
            """

    else:
        msg = f"Unknown topic '{topic}'. Try: 'general', 'modifier', 'sampling', 'usage', 'train', 'checkpoints'."

    print(msg)
