"""Quadratic interop regression tests for the maintained Python API."""

import sys
import types

import numpy as np
import pytest

from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
from QES.Algebra.interop import OpenFermionInterop, QiskitInterop


# -----------------------------------------------------------------------------
# Fake optional dependency shims

def _install_fake_qiskit(monkeypatch):
    """Install a minimal fake Qiskit Nature surface for round-trip testing."""
    class FakeQuadraticHamiltonian:
        def __init__(self, hermitian_part, antisymmetric_part=None, constant=0.0):
            self.hermitian_part = np.array(hermitian_part, dtype=np.complex128)
            self.antisymmetric_part = None if antisymmetric_part is None else np.array(antisymmetric_part, dtype=np.complex128)
            self.constant = float(constant)

        def second_q_op(self):
            return {
                "kind": "fake_second_q_op",
                "hermitian_part": self.hermitian_part.copy(),
                "antisymmetric_part": None if self.antisymmetric_part is None else self.antisymmetric_part.copy(),
                "constant": self.constant,
            }

    qiskit_nature = types.ModuleType("qiskit_nature")
    second_q = types.ModuleType("qiskit_nature.second_q")
    hamiltonians = types.ModuleType("qiskit_nature.second_q.hamiltonians")
    hamiltonians.QuadraticHamiltonian = FakeQuadraticHamiltonian
    second_q.hamiltonians = hamiltonians
    qiskit_nature.second_q = second_q

    monkeypatch.setitem(sys.modules, "qiskit_nature", qiskit_nature)
    monkeypatch.setitem(sys.modules, "qiskit_nature.second_q", second_q)
    monkeypatch.setitem(sys.modules, "qiskit_nature.second_q.hamiltonians", hamiltonians)
    monkeypatch.setattr(QiskitInterop, "is_qiskit_available", staticmethod(lambda: True))

    return FakeQuadraticHamiltonian

def _install_fake_openfermion(monkeypatch):
    """Install a minimal fake OpenFermion surface for round-trip testing."""
    class FakeFermionOperator:
        def __init__(self, label: str | None = None, coeff: complex = 1.0):
            self.terms = {}
            if label is None:
                return
            if isinstance(label, tuple):
                self.terms[tuple(label)] = complex(coeff)
                return
            if label == "" and coeff != 0.0:
                self.terms[tuple()] = complex(coeff)
            else:
                parsed = []
                for token in label.split():
                    if token.endswith("^"):
                        parsed.append((int(token[:-1]), 1))
                    else:
                        parsed.append((int(token), 0))
                self.terms[tuple(parsed)] = complex(coeff)

        def __iadd__(self, other):
            for term, coeff in other.terms.items():
                self.terms[term] = self.terms.get(term, 0.0) + coeff
            return self

        def __add__(self, other):
            out         = FakeFermionOperator()
            out.terms   = dict(self.terms)
            out        += other
            return out

        def __mul__(self, coeff):
            out         = FakeFermionOperator()
            out.terms   = {term: coeff * val for term, val in self.terms.items()}
            return out

        __rmul__ = __mul__

    openfermion = types.ModuleType("openfermion")
    openfermion.FermionOperator = FakeFermionOperator

    monkeypatch.setitem(sys.modules, "openfermion", openfermion)
    monkeypatch.setattr(OpenFermionInterop, "is_openfermion_available", staticmethod(lambda: True))

    return FakeFermionOperator

# -----------------------------------------------------------------------------
# Bogoliubov-transform behavior

def test_particle_conserving_bogoliubov_transform_matches_matrix_diagonalization():
    H = np.array(
        [
            [0.4, -1.0 + 0.2j, 0.0],
            [-1.0 - 0.2j, -0.1, -0.5],
            [0.0, -0.5, 0.2],
        ],
        dtype=np.complex128,
    )
    qh = QuadraticHamiltonian.from_hermitian_matrix(H, constant=0.7, dtype=np.complex128)

    W, orbital_energies, transformed_constant = qh.diagonalizing_bogoliubov_transform()
    evals, _ = np.linalg.eigh(H)

    assert W.shape == (3, 3)
    np.testing.assert_allclose(orbital_energies, evals, atol=1e-12)
    np.testing.assert_allclose(W @ W.conj().T, np.eye(3), atol=1e-12)
    assert transformed_constant == 0.7


def test_bdg_bogoliubov_transform_returns_positive_branch_and_shift():
    K = np.array(
        [
            [0.2, -1.0, 0.0],
            [-1.0, -0.3, -0.5],
            [0.0, -0.5, 0.1],
        ],
        dtype=np.complex128,
    )
    Delta = np.array(
        [
            [0.0, 0.15, 0.0],
            [-0.15, 0.0, 0.25],
            [0.0, -0.25, 0.0],
        ],
        dtype=np.complex128,
    )
    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.1,
        dtype=np.complex128,
    )

    W, orbital_energies, transformed_constant = qh.diagonalizing_bogoliubov_transform()

    assert W.shape == (3, 6)
    assert orbital_energies.shape == (3,)
    assert np.all(orbital_energies >= 0.0)
    np.testing.assert_allclose(W @ W.conj().T, np.eye(3), atol=1e-10)
    expected_constant = 0.1 - 0.5 * np.sum(orbital_energies) + 0.5 * np.trace(K).real
    np.testing.assert_allclose(transformed_constant, expected_constant, atol=1e-12)


def test_from_bdg_matrices_preserves_explicit_bdg_mode_even_for_zero_pairing():
    K = np.diag([0.1, -0.2, 0.3]).astype(np.complex128)
    Delta = np.zeros_like(K)
    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        dtype=np.complex128,
    )

    assert qh.conserves_particle_number() is False
    bdg = qh.build_bdg_matrix()
    assert bdg.shape == (6, 6)

# -----------------------------------------------------------------------------
# Optional-package interop behavior

def test_qiskit_interop_round_trip_uses_quadratic_hamiltonian_api(monkeypatch):
    FakeQuadraticHamiltonian = _install_fake_qiskit(monkeypatch)

    K = np.array([[0.1, -1.0], [-1.0, -0.2]], dtype=np.complex128)
    Delta = np.array([[0.0, 0.3], [-0.3, 0.0]], dtype=np.complex128)

    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.4,
        dtype=np.complex128,
    )

    qiskit_ham = qh.to_qiskit_hamiltonian()
    assert isinstance(qiskit_ham, FakeQuadraticHamiltonian)
    np.testing.assert_allclose(qiskit_ham.hermitian_part, K, atol=1e-12)
    np.testing.assert_allclose(qiskit_ham.antisymmetric_part, Delta, atol=1e-12)
    assert qiskit_ham.constant == 0.4

    op = QiskitInterop.to_qiskit_second_quantized_op(K, v_matrix=Delta, constant=0.4)
    assert op["kind"] == "fake_second_q_op"

    h_matrix, v_matrix, constant = QiskitInterop.from_qiskit_hamiltonian(qiskit_ham, num_spin_orbitals=2)
    np.testing.assert_allclose(h_matrix, K, atol=1e-12)
    np.testing.assert_allclose(v_matrix, Delta, atol=1e-12)
    assert constant == 0.4

    rebuilt = QuadraticHamiltonian.from_qiskit_hamiltonian(qiskit_ham, dtype=np.complex128)
    np.testing.assert_allclose(rebuilt.build_bdg_matrix(), qh.build_bdg_matrix(), atol=1e-12)


def test_openfermion_interop_round_trip_preserves_pairing(monkeypatch):
    _install_fake_openfermion(monkeypatch)

    K = np.array([[0.1, -1.0], [-1.0, -0.2]], dtype=np.complex128)
    Delta = np.array([[0.0, 0.3], [-0.3, 0.0]], dtype=np.complex128)

    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.4,
        dtype=np.complex128,
    )

    of_ham = qh.to_openfermion_hamiltonian()
    h_matrix, v_matrix, constant = OpenFermionInterop.from_openfermion_hamiltonian(of_ham, num_orbitals=2)
    np.testing.assert_allclose(h_matrix, K, atol=1e-12)
    np.testing.assert_allclose(v_matrix, Delta, atol=1e-12)
    assert constant == 0.4

    rebuilt = QuadraticHamiltonian.from_openfermion_hamiltonian(of_ham, num_orbitals=2, dtype=np.complex128)
    np.testing.assert_allclose(rebuilt.build_bdg_matrix(), qh.build_bdg_matrix(), atol=1e-12)


# -----------------------------------------------------------------------------
# Real optional-environment smoke tests

def test_optional_real_qiskit_env_round_trip_if_installed():
    if not QiskitInterop.is_qiskit_available():
        pytest.skip("Qiskit Nature not installed")

    K = np.array([[0.25, -1.0], [-1.0, -0.15]], dtype=np.complex128)
    Delta = np.array([[0.0, 0.2], [-0.2, 0.0]], dtype=np.complex128)
    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.3,
        dtype=np.complex128,
    )

    qiskit_ham = qh.to_qiskit_hamiltonian()
    rebuilt = QuadraticHamiltonian.from_qiskit_hamiltonian(qiskit_ham, dtype=np.complex128)
    np.testing.assert_allclose(rebuilt.build_bdg_matrix(), qh.build_bdg_matrix(), atol=1e-12)


def test_optional_real_openfermion_env_round_trip_if_installed():
    if not OpenFermionInterop.is_openfermion_available():
        pytest.skip("OpenFermion not installed")

    K = np.array([[0.25, -1.0], [-1.0, -0.15]], dtype=np.complex128)
    Delta = np.array([[0.0, 0.2], [-0.2, 0.0]], dtype=np.complex128)
    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.3,
        dtype=np.complex128,
    )

    of_ham = qh.to_openfermion_hamiltonian()
    rebuilt = QuadraticHamiltonian.from_openfermion_hamiltonian(of_ham, num_orbitals=2, dtype=np.complex128)
    np.testing.assert_allclose(rebuilt.build_bdg_matrix(), qh.build_bdg_matrix(), atol=1e-12)

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------