'''

'''

from typing import List

# we need 
# 1) creating a function to evaluate log(Ψ(s))_w' / current log(\psi(s))_w for a batch of states s
# this is already done in nqs_networks.py but we need to enclose the network parameters in 
# the function signature. 
# 2) 

class NQSLowerStates:
    
    def __init__(self,
            lower_nqs_instances     : List[NQS],                # List of NQS objects for lower states
            lower_betas             : List[float],              # Penalty terms beta_i
            parent_nqs              : 'NQS'                 # Parent NQS object (the excited state being trained):

        if len(lower_nqs_instances) != len(lower_betas):
            raise ValueError("Number of lower NQS instances must match number of betas.")
        
        self._backend               = parent_nqs.backend
        self._isjax                 = parent_nqs.isjax
        
        # assert that all are the same backend
        if not all([nqs.backend == parent_nqs.backend for nqs in lower_nqs_instances]):
            raise ValueError("All lower NQS instances must have the same backend as the parent NQS.")
        
        self._parent_nqs                    = parent_nqs
        self._parent_apply_fn               = parent_nqs.ansatz
        self._parent_params                 = parent_nqs.get_params()                               # will likely be updated during training
        self._parent_evaluate               = parent_nqs.apply_f
        self._log_p_ratio_fn                = parent_nqs.log_prob_ratio        
        
        #! handle the lower states
        self._lower_nqs             = lower_nqs_instances
        self._lower_betas           = self._backend.array(lower_betas)
        self._num_lower_states      = len(lower_nqs_instances)
        self._is_set                = self._num_lower_states > 0

        if not self._is_set:
            return

        # Store apply functions and parameters for each lower state
        self._lower_apply_fns       = [nqs.ansatz for nqs in self._lower_nqs]               # this is static as it is not updated during training
        self._lower_params          = [nqs.get_params() for nqs in self._lower_nqs]         # this is static as it is not updated during training
        if self._isjax:
            self._lower_evaluate    = [nqs.apply_f for nqs in self._lower_nqs]     # this is static as it is not updated during training
        else:
            self._lower_evaluate    = [nqs.apply_f for nqs in self._lower_nqs]

        #! Placeholder for ratios - these would be computed during the excited state's MC sampling
        # Shape: (num_lower_states, num_samples_excited_state)
        self._ratios_psi_exc_div_psi_lower = None       # Psi_W / Psi_W_j (current excited / lower_j)
        self._ratios_psi_lower_div_psi_exc = None       # Psi_W_j / Psi_W (lower_j / current excited)