r"""
Spinful onsite Hubbard model for DQMC-oriented workflows.

This class stores the single-particle hopping matrix together with the onsite
interaction metadata required by determinant quantum Monte Carlo:

.. math::

    H = -\sum_{\langle i,j \rangle,\sigma} t_{ij}
        \left(c^\dagger_{i\sigma} c_{j\sigma} + \mathrm{h.c.}\right)
        + U \sum_i n_{i\uparrow} n_{i\downarrow}
        - \sum_{i,\sigma} \mu_i n_{i\sigma}.

The quadratic part is identical for both spin species, so the model exposes one
spatial hopping matrix ``hamil_sp`` and keeps ``U``/``mu`` as metadata for the
DQMC adapter layer.
"""

from __future__ import annotations

from typing import Optional, Union, List

import numpy as np

from QES.Algebra.hamil_quadratic import QuadraticHamiltonian


class SpinfulHubbardModel(QuadraticHamiltonian):
    """
    Particle-conserving spinful onsite Hubbard model on a lattice.

    This is intentionally a quadratic-plus-metadata model:
    the stored matrix is the one-body hopping/chemical-potential part, while
    the onsite interaction ``U`` is carried separately for DQMC decoupling.
    """

    def __init__(
        self,
        lattice,
        *,
        t: Union[float, List[float]] = 1.0,
        U: Union[float, List[float]] = 1.0,
        mu: Union[float, List[float], None] = None,
        dtype: type = np.complex128,
        backend: str = "default",
        constant_offset: float = 0.0,
        **kwargs,
    ):
        if lattice is None:
            raise ValueError("SpinfulHubbardModel requires a lattice.")

        super().__init__(
            ns=getattr(lattice, "ns", None),
            particle_conserving=True,
            dtype=dtype,
            backend=backend,
            particles="fermions",
            is_sparse=False,
            constant_offset=constant_offset,
            lattice=lattice,
            hilbert_space=None,
            **kwargs,
        )

        self._name = "Spinful Hubbard Model"
        self._t = self._set_some_coupling(t)
        self._u = self._set_some_coupling(U)
        self._mu = self._set_some_coupling(mu) if mu is not None else None
        self._hamiltonian_quadratic(use_numpy=not self._is_jax)

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """
        Build the spatial hopping matrix ``K`` shared by spin-up and spin-down.

        Math:
            the DQMC factorization uses the same one-body kinetic matrix for
            both spin channels; only the HS potential differs between them.
        """
        xp = np if use_numpy else self._backend
        self._hamil_sp = xp.zeros((self._ns, self._ns), dtype=self._dtype)

        lat = self._lattice
        if lat is None:
            return self._hamil_sp

        for i in range(self._ns):
            if self._mu is not None and self._mu[i] != 0:
                self._hamil_sp[i, i] -= self._mu[i]

            for nidx in range(lat.get_nn_num(i)):
                j = lat.get_nn(i, num=nidx)
                if lat.wrong_nei(j):
                    continue
                j = int(j)
                wx, wy, wz = lat.bond_winding(i, j)
                phase = lat.boundary_phase_from_winding(wx, wy, wz)
                self._hamil_sp[i, j] += -phase * self._t[i]
        return self._hamil_sp

    def __repr__(self):
        return (
            f"SpinfulHubbardModel(Ns={self._ns}, "
            f"t={self._t[0] if hasattr(self._t, '__len__') else self._t}, "
            f"U={self._u[0] if hasattr(self._u, '__len__') else self._u}, "
            f"mu={self._mu[0] if (self._mu is not None and hasattr(self._mu, '__len__')) else self._mu})"
        )
