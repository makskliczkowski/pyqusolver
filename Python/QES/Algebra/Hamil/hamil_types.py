"""
file: Algebra/hamil_types.py
This module contains the enumeration of Hamiltonian models and related functions.
A Hamiltonian model is defined as a class of Hamiltonians that share a common structure.

Date: 2025-02-01
"""


try:
    from QES.general_python.algebra.utils import distinguish_type
except ImportError:
    raise ImportError(
        "Could not import 'distinguish_type' from 'QES.general_python.algebra.utils'. Please ensure the module is available."
    )

from enum import Enum, unique

###########################################################################
# ED limits (using hexadecimal where applicable)
UI_LIMITS_MAXFULLED             = 0x40000
UI_LIMITS_MAXPRINT              = 0x8
UI_LIMITS_SI_STATENUM           = 100
UI_LIMITS_MIDDLE_SPEC_STATENUM  = 200


# ############################ EXISTING MODELS ############################
@unique
class Hamiltonians(Enum):
    """
    Enumeration for different Hamiltonian models.
    """

    NONE                        = 0
    # interacting models:
    ISING_M                     = 1  # Quantum Ising model
    XYZ_M                       = 2  # XYZ model
    HEI_KIT_M                   = 3  # Heisenberg model with Kitaev interaction
    QSM_M                       = 4  # Quantum Summer Model (QSM)
    RP_M                        = 5  # Rosenzweig-Porter Model
    ULTRAMETRIC_M               = 6  # Ultrametric Model (UM)
    # quadratic (or noninteracting) models:
    FREE_FERMIONS_M             = 100  # Free Fermions Model
    AUBRY_ANDRE_M               = 101  # Aubry-André Model
    SYK2_M                      = 102  # Sachdev-Ye-Kitaev (SYK) Model
    ANDERSON_M                  = 103  # Anderson Model
    POWER_LAW_RANDOM_BANDED_M   = 104  # Power-law Random Banded Model


MY_MODELS_LIST = [
    Hamiltonians.ISING_M,
    Hamiltonians.XYZ_M,
    Hamiltonians.HEI_KIT_M,
    Hamiltonians.QSM_M,
    Hamiltonians.RP_M,
    Hamiltonians.ULTRAMETRIC_M,
    Hamiltonians.FREE_FERMIONS_M,
    Hamiltonians.AUBRY_ANDRE_M,
    Hamiltonians.SYK2_M,
    Hamiltonians.ANDERSON_M,
    Hamiltonians.POWER_LAW_RANDOM_BANDED_M,
    Hamiltonians.NONE,
]

# Maximum interacting model value defined as the value of ULTRAMETRIC_M.
MY_MODELS_MAX_INTERACTING = Hamiltonians.ULTRAMETRIC_M.value

################################################################################
#! TODO: Check the implementation of the following functions.
################################################################################


def check_noninteracting(model: Hamiltonians) -> bool:
    """
    Check if a model is noninteracting.

    Parameters
    ----------
    model : Hamiltonians
        Hamiltonian model enum value.

    Returns
    -------
    bool
        True when the model is treated as noninteracting by the legacy API.
    """
    return (model.value >= Hamiltonians.FREE_FERMIONS_M.value) or (model == Hamiltonians.RP_M)


def check_dense(model: Hamiltonians) -> bool:
    """
    Check if a model is dense.

    Parameters
    ----------
    model : Hamiltonians
        Hamiltonian model enum value.

    Returns
    -------
    bool
        True when the model is treated as dense by the legacy API.
    """
    return model in (
        Hamiltonians.POWER_LAW_RANDOM_BANDED_M,
        Hamiltonians.ULTRAMETRIC_M,
        Hamiltonians.RP_M,
    )

##################################################################################
#! END OF FILE
##################################################################################
