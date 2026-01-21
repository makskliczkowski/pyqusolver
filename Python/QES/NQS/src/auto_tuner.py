"""
TDVP Auto-Tuner.
Adaptive controller for TDVP hyperparameters.
"""

import  numpy as np
from    dataclasses import dataclass, field
from    typing import Optional, Dict, List, Any, Tuple

@dataclass
class AutoTunerConfig:
    """Configuration for TDVP Auto-Tuner."""
    # Learning Rate / Step Size
    lr_initial              : float     = 1e-2
    lr_min                  : float     = 1e-5
    lr_max                  : float     = 0.5
    lr_growth_factor        : float     = 1.05
    lr_shrink_factor        : float     = 0.5

    # Diagonal Shift
    diag_initial            : float     = 1e-3
    diag_min                : float     = 1e-6
    diag_max                : float     = 1.0
    diag_growth_factor      : float     = 2.0
    diag_shrink_factor      : float     = 0.9
    # Sampler Length
    samples_initial         : int       = 1000
    samples_min             : int       = 100
    samples_max             : int       = 100000
    samples_growth_factor   : float     = 1.5

    # Targets
    target_ess_fraction     : float     = 0.1 # ESS should be at least 10% of total samples
    target_rhat             : float     = 1.1
    target_acceptance_min   : float     = 0.3
    target_acceptance_max   : float     = 0.6 # Not strictly enforced, just info

    # Heuristics
    energy_window           : int       = 5
    variance_tolerance      : float     = 10.0 # Factor of allowed variance increase
    
class TDVPAutoTuner:
    """
    Adaptive controller for TDVP hyperparameters.

    Controls:
    - Learning rate (dt)
    - Diagonal shift (regularization)
    - Sampler length (n_samples)

    Based on online metrics:
    - Energy variance
    - Acceptance rate
    - ESS / R-hat
    - SR solver status
    """

    def __init__(self, config: AutoTunerConfig = None, logger=None):
        self.config     = config or AutoTunerConfig()
        self.logger     = logger

        # State
        self.lr         = self.config.lr_initial
        self.diag_shift = self.config.diag_initial
        self.n_samples  = self.config.samples_initial

        self.history    = {
                            'energy'    : [],
                            'variance'  : [],
                            'lr'        : [],
                            'diag'      : [],
                            'samples'   : []
                        }

    def _log(self, msg, lvl=1, color='white'):
        if self.logger:
            self.logger.info(f"[AutoTuner] {msg}", lvl=lvl, color=color)

    def update(self, step_info, diagnostics: Dict[str, float], current_energy: float) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Update hyperparameters based on current step info and diagnostics.

        Parameters:
            step_info: TDVPStepInfo object (contains mean_energy, std_energy, etc.)
            diagnostics: Dict from sampler.diagnose()
            current_energy: Current energy (mean_energy)

        Returns:
            params: Dict with new 'lr', 'diag_shift', 'n_samples'
            accepted: Bool, whether the step seems acceptable (convergence gate)
            warnings: List of warning strings
        """

        warnings    = []
        accepted    = True

        # Sampler Diagnostics Check
        ess         = diagnostics.get('ess', float('inf'))
        r_hat       = diagnostics.get('r_hat', 1.0)

        # Check R-hat (Multi-chain convergence)
        if r_hat > self.config.target_rhat:
            warnings.append(f"Poor convergence (R-hat={r_hat:.3f} > {self.config.target_rhat}). Increasing samples.")
            
            self.n_samples = int(self.n_samples * self.config.samples_growth_factor)
            
            # Potentially reject if very bad?
            if r_hat > 1.2:
                accepted = False
                warnings.append("Step rejected due to poor chain convergence.")

        # Check ESS
        target_ess  = self.config.target_ess_fraction * self.n_samples
        
        if ess < target_ess:
            warnings.append(f"Low ESS ({ess:.1f} < {target_ess:.1f}). Increasing samples.")
            self.n_samples = int(self.n_samples * self.config.samples_growth_factor)

        # Clamp samples
        self.n_samples  = min(max(self.n_samples, self.config.samples_min), self.config.samples_max)

        # Diagonal Shift (SR Stability)
        sr_converged    = getattr(step_info, 'sr_converged', True)

        if not sr_converged:
            warnings.append("SR solver failed to converge. Increasing diagonal shift.")
            self.diag_shift *= self.config.diag_growth_factor
        else:
            self.diag_shift *= self.config.diag_shrink_factor

        self.diag_shift = min(max(self.diag_shift, self.config.diag_min), self.config.diag_max)

        # Learning Rate / Step Size (Energy Stability)
        # -----------------------------------------------
        # Heuristic: If energy increased significantly (for ground state search), reduce LR.
        # If energy is decreasing or stable, increase LR slowly.

        prev_energy = self.history['energy'][-1] if self.history['energy'] else None

        # Determine trend
        if prev_energy is not None:
            energy_diff = current_energy - prev_energy

            # If energy shot up (assuming minimization)
            # Threshold: > 3 * standard deviation of previous estimate?
            # Or just raw value if we expect monotonic decrease (imaginary time)

            # Using prev variance as scale
            prev_std    = self.history['variance'][-1] if self.history['variance'] else 1.0

            if energy_diff > 0.0 and energy_diff > 3 * prev_std:
                
                warnings.append(f"Energy spike (+{energy_diff:.2e}). Reducing LR.")
                
                self.lr *= self.config.lr_shrink_factor
                
                # Reject step if spike is massive?
                if energy_diff > 10 * prev_std:
                    accepted = False
                    warnings.append("Step rejected due to massive energy spike.")
            else:
                # Energy decreased or stable noise
                self.lr *= self.config.lr_growth_factor

        self.lr = min(max(self.lr, self.config.lr_min), self.config.lr_max)

        # Update history
        self.history['lr'].append(self.lr)
        self.history['diag'].append(self.diag_shift)
        self.history['samples'].append(self.n_samples)
        self.history['energy'].append(float(current_energy))
        self.history['variance'].append(float(step_info.std_energy))

        return {
            'lr'            : self.lr,
            'diag_shift'    : self.diag_shift,
            'n_samples'     : self.n_samples
        }, accepted, warnings
        
# -----------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------
