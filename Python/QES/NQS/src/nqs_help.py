"""
Help module for NQS. Provides the `nqs_help()` method for user guidance on NQS features, usage, and best practices.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Version     : 2.0
License     : MIT
--------------------------------
"""

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
        - 'entropy':
            Entanglement entropy estimators and ED benchmark path.
        - 'spectral':
            TDVP response, FFT spectra, and ED/Lanczos comparison path.
        - 'checkpoints':
            Saving and loading model weights and checkpoints.
    """
    topic = topic.lower().strip()
    msg = ""
    border = "-" * 60

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
            - compute_energy(...), measure(...): Monte Carlo observables.
            - compute_renyi_entropies(...): Batched replica entropy estimates.
            - time_evolve(...), dynamic_structure_factor(...): TDVP response.
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

            Recommended config-first workflow:
                from QES.NQS import NQS, NQSPhysicsConfig, NQSSolverConfig, NQSTrainConfig

                # Accelerated path:
                # pip install "QES[jax]"
                # or:
                # pip install "QES[all]"

                p_cfg = NQSPhysicsConfig(model_type='tfim', lattice_type='chain', lx=16, bc='obc')
                s_cfg = NQSSolverConfig(ansatz='rbm', backend='jax')

                model, hilbert, lattice = p_cfg.make_hamiltonian()
                net = s_cfg.make_net(p_cfg)
                psi = NQS(logansatz=net, model=model, hilbert=hilbert, backend=s_cfg.backend)

                train_cfg = NQSTrainConfig.from_solver(s_cfg, n_epochs=200)
                stats = psi.train(**train_cfg.to_train_kwargs())

            Compact expert path:
                psi = NQS(logansatz='rbm', model=hamil, hilbert=hilbert, backend='jax')
                stats = psi.train(n_epochs=200, lr=1e-2)

            Sampling and measurement:
                energy = psi.compute_energy(num_samples=1000)
                obs = psi.measure({{'sx0': sig_x0.jax, 'sz0': sig_z0.jax}}, num_samples=1000)

            Dynamic settings:
                psi.batch_size = 2048
                psi.net = new_net
                psi.sampler = 'vmc'

            Extra help:
                psi.help('train')
                psi.help('entropy')
                psi.help('spectral')
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

    elif topic == "entropy":
        msg = f"""
            {border}
            NQS Solver Help: Entanglement entropy
            {border}
            Production NQS path:
            - compute_renyi2(region, ...): single Renyi-2 estimate.
            - compute_renyi_entropies(regions, q_values=[2, 3], ...): preferred batched path.
            - compute_topological_entropy(...): lattice-cut wrapper for topological entropy.

            Recommended many-cut usage:
                cuts = lattice.get_entropy_cuts(cut_type='all')
                valid = {{k: r for k, r in cuts.items() if 0 < len(r) < lattice.ns}}
                entropy = psi.compute_renyi_entropies(
                    valid,
                    q_values=[2, 3],
                    num_samples=2000,
                    return_error=True,
                )

            Why the batched path matters:
            - draws replica samples once for the largest requested q,
            - reuses original log amplitudes across q values,
            - batches swapped evaluations across regions,
            - avoids one expensive sampler/evaluator loop per cut.

            ED benchmark path:
                from QES.NQS.src.nqs_entropy import compute_ed_entanglement_entropy
                ed = compute_ed_entanglement_entropy(
                    hamil.eig_vec,
                    region,
                    lattice.ns,
                    q_values=[2, 3],
                    method='svd',
                )

            The ED helper is only for exact eigenvectors. The NQS estimator uses
            the replica trick with the active sampler convention.
            """

    elif topic == "spectral":
        msg = f"""
            {border}
            NQS Solver Help: Spectral and time-domain response
            {border}
            Production NQS path:
            - time_evolve(times, ...): real-time TDVP trajectory.
            - compute_transition_correlator(trajectory, ...): overlap/transition correlator.
            - compute_dynamical_correlator(times, ket_probe_operator=..., ...): probe-state correlator.
            - dynamic_structure_factor(times, probe_operator=..., ...): probe correlator + FFT.
            - spectrum_from_correlator(times, correlator, ...): FFT postprocessing only.
            - dynamic_structure_factor_kspace(...): probe or momentum resolved map.

            Typical NQS usage:
                times = np.linspace(0.0, 10.0, 256)
                dsf = psi.dynamic_structure_factor(
                    times,
                    probe_operator=sq_op,
                    num_samples=2000,
                    eta=0.05,
                    window='hann',
                )

            ED/Lanczos comparison path:
                from QES.general_python.physics.spectral.spectral_backend import (
                    operator_spectral_function_multi_omega,
                    integrated_spectral_weight,
                    find_spectral_peaks,
                )

                hamil.diagonalize()
                ed = operator_spectral_function_multi_omega(
                    omega_grid,
                    hamil.eig_val,
                    hamil.eig_vec,
                    operator_matrix,
                    eta=0.05,
                )

            The NQS spectral package does not reimplement the general ED
            backend. Internal exact basis-summation helpers are only for tiny
            variational-state regression tests.
            """

    elif topic == "network":
        from QES.NQS.src.network import NetworkFactory
        info = NetworkFactory.net_help()
        if self._net is not None:
            net_type = self._net.name if hasattr(self._net, "name") else type(self._net).__name__
            msg = f"""
                        {border}
                        NQS Solver Help: Network Details
                        {border}
                        Current Network Type: {net_type}

                        {info.get(net_type, "No additional info available for this network type.")}
                        """
    elif topic == "networks":
        from QES.NQS.src.network import NetworkFactory
        info = NetworkFactory.net_help()
        msg = f"""
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
            Persistence levels:

            1. Weights only
                psi.save_weights("my_model.h5")
                psi.load_weights("my_model.h5")

            2. Training checkpoints
                stats = psi.train(
                    n_epochs=300,
                    checkpoint_every=50,
                    save_path="./checkpoints"
                )

            3. Full config-driven reconstruction
                from QES.NQS import load_nqs
                bundle = load_nqs(physics_config, solver_config, checkpoint_step='latest')
                psi = bundle.nqs

            Use weights-only loading when the architecture/config is already known.
            Use config-driven loading when the construction path itself should be reproducible.
            """

    else:
        msg = (
            f"Unknown topic '{topic}'. Try: "
            "'general', 'modifier', 'sampling', 'network', 'networks', "
            "'usage', 'train', 'entropy', 'spectral', 'checkpoints'."
        )

    print(msg)
