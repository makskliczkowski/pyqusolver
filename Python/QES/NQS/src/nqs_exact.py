"""
Exact Diagonalization and Ground Truth Utilities for NQS.

This module contains helper functions for computing or loading exact
ground truth information (e.g. exact diagonalization energies) for
comparison with NQS results.
"""

import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..nqs import NQS
    from .nqs_train import NQSTrainStats

# ----------------------------------------------------------------
#! NQS Exact Implementations
# ----------------------------------------------------------------


def get_exact_impl(nqs_instance: "NQS", **kwargs) -> Optional["NQSTrainStats"]:
    """
    Implementation of NQS.get_exact.
    Get exact predictions, e.g., ground state energy, from the model via diagonalization.

    Parameters
    ----------
    nqs_instance : NQS
        The NQS instance to compute exact information for.
    **kwargs : dict
        Additional keyword arguments forwarded to the diagonalization method.

    Note:
        Currently only supports 'wavefunction' physics type.

    #!TODO: Support other physics types (e.g., density matrices)
    """

    logfun = nqs_instance.log if nqs_instance._logger else lambda x, **kw: print(x)

    if nqs_instance._nqsproblem.typ == "wavefunction":
        from .nqs_train import NQSTrainStats

        stats = nqs_instance.trainer.stats if nqs_instance.trainer is not None else NQSTrainStats()

        # Perform exact diagonalization if not already done
        if not stats.has_exact:
            if nqs_instance.model.ns > kwargs.get("max_exact_size", 20):
                return stats  # Avoid expensive diagonalization for large systems
            
            if nqs_instance.model.eig_val is None:
                nqs_instance.model.diagonalize(
                    method      =   "lanczos",
                    k           =   kwargs.get("k", 6),
                    store_basis =   False,
                    verbose     =   kwargs.get("verbose", True),
                    use_scipy   =   kwargs.get("use_scipy", True),
                    tol         =   kwargs.get("tol", 1e-7),
                    maxiter     =   kwargs.get("max_iter", 200),
                )

            # Get predictions
            pred = nqs_instance.model.eigenvalues
            if stats is not None:
                stats.exact_predictions = pred

            nstate              = nqs_instance._nthstate
            nqs_instance.exact  = {
                                    "exact_predictions" : nqs_instance.model.eigenvalues,
                                    "exact_method"      : "scipy_lanczos",
                                    "exact_energy"      : float(pred) if np.ndim(pred) == 0 else pred[nstate],
                                }
            # Log results
            if nstate == 0:
                logfun(
                    f"Exact ground state energy: {nqs_instance.model.eig_val[0]:.6f}",
                    lvl=1,
                    color="green",
                )
            else:
                logfun(
                    f"Exact state[{nstate}] energy: {nqs_instance.model.eig_val[nstate]:.6f}",
                    lvl=1,
                    color="green",
                )
                logfun(
                    f"Lowest energies: {nqs_instance.model.eig_val[:max(nstate, 5)]}",
                    lvl=2,
                    color="green",
                )

        # Return stats with exact predictions
        return stats
    else:
        raise NotImplementedError("Exact is not implemented for other physics types yet...")


def load_exact_impl(nqs_instance: "NQS", filepath: str, *, key: str = "energy_values"):
    """
    Implementation of NQS.load_exact.
    Load exact information from a file.
    """

    logfun = nqs_instance.log if nqs_instance._logger else lambda x, **kw: print(x)

    if not os.path.isfile(filepath):
        logfun(f"Exact file {filepath} does not exist.", lvl=1, color="red")
        return

    # determine file extension
    extension = os.path.splitext(str(filepath))[1].lower()

    if extension == ".h5" or extension == ".hdf5":
        import h5py

        exact_values = []
        with h5py.File(filepath, "r") as f:
            if key in f:
                exact_values = f[key][:]
            else:
                raise KeyError(f"Key '{key}' not found in HDF5 file '{filepath}'.")

    elif extension == ".npy":
        exact_values = np.load(filepath)

    elif extension == ".txt" or extension == ".dat" or extension == ".csv":
        exact_values = np.loadtxt(filepath)
    else:
        # Fallback or error? Assuming numpy load might work or just raise
        exact_values = np.loadtxt(filepath)

    nqs_instance.exact = {
        "exact_predictions"     : np.array(exact_values),
        "exact_method"          : f"loaded_from_{os.path.basename(filepath)}",
        "exact_energy"          : (
                                    float(exact_values)
                                    if np.ndim(exact_values) == 0
                                    else exact_values[nqs_instance._nthstate]
                                ),
    }

    # Log results
    logfun(f"Loaded exact information from {filepath}.", lvl=1, color="green")


def set_exact_impl(nqs_instance: "NQS", info: Optional[dict]):
    """
    Implementation of NQS.set_exact.
    Set exact information directly.
    TODO: Validate the provided information. Implement other formats.
    """

    logfun = nqs_instance.log if nqs_instance._logger else lambda x, **kw: print(x)

    if isinstance(info, dict):
        nqs_instance._exact_info = info

    elif isinstance(info, np.ndarray):
        nqs_instance._exact_info = {
            "exact_predictions" : info,
            "exact_method"      : "provided_array",
            "exact_energy"      : float(info) if np.ndim(info) == 0 else info[nqs_instance._nthstate],
        }
        logfun(
            f"Exact information set from array: {nqs_instance._exact_info['exact_energy']}", lvl=1
        )
    else:
        raise ValueError("Exact information must be provided as a dictionary or numpy array.")


# ----------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------
