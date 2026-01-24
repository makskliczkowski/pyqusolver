"""
QES Solver Module
=================

Simulation engines and optimization backends.

This module provides the abstract interfaces and concrete implementations for
simulating quantum systems. It primarily houses the Monte Carlo engines used
by NQS.

Entry Points
------------
- :class:`MonteCarloSolver`: Base class for MC-based simulations.
- :class:`QES.Solver.MonteCarlo.sampler.Sampler`: Abstract interface for sampling.
- :class:`QES.Solver.MonteCarlo.vmc.VMCSampler`: Metropolis-Hastings sampler.

Flow
----
::

    Sampler (MCMC)
        |
        v
    MonteCarloSolver (Optimization Loop)
        |
        v
    Physics Results

Submodules
----------
- ``MonteCarlo``: Samplers and VMC logic.
- ``solver``: Abstract base classes.
"""

from .solver import Solver

__all__ = ["Solver"]

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = "Core solver interfaces and Monte Carlo-based solvers."

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
