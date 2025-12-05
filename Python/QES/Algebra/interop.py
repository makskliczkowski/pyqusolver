"""
Qiskit/OpenFermion interoperability layer for QuadraticHamiltonian.

Provides converters to/from Qiskit Nature and OpenFermion formats.

--------------------------------------------------------------------------
File        : Algebra/interop.py
Author      : Maksymilian Kliczkowski
Date        : 2025-06-15
License     : MIT
Description : Interoperability utilities for Qiskit Nature and OpenFermion.
--------------------------------------------------------------------------
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

# --------------------------------------------------------------------------

class QiskitInterop:
    """Qiskit Nature interoperability utilities."""
    
    @staticmethod
    def is_qiskit_available() -> bool:
        """Check if Qiskit Nature is installed."""
        try:
            import qiskit_nature
            return True
        except ImportError:
            return False
    
    @staticmethod
    def to_qiskit_second_quantized_op(h_matrix          : np.ndarray,
                                    v_matrix            : Optional[np.ndarray]  = None,
                                    constant            : float                 = 0.0,
                                    num_spin_orbitals   : Optional[int]         = None):
        """Convert QuadraticHamiltonian to Qiskit SecondQuantizedOp.
        
        Parameters
        ----------
        h_matrix : np.ndarray
            One-body hopping matrix (N x N)
        v_matrix : np.ndarray, optional
            Pairing matrix for BdG (N x N)
        constant : float, optional
            Constant energy offset
        num_spin_orbitals : int, optional
            Total number of spin orbitals
            
        Returns
        -------
        qiskit_nature.second_q.operators.FermionicOp
            Qiskit fermionic operator
            
        Raises
        ------
        ImportError
            If Qiskit Nature not installed
        """
        if not QiskitInterop.is_qiskit_available():
            raise ImportError("Qiskit Nature required for this operation. "
                            "Install: pip install qiskit-nature")
        
        from qiskit_nature.second_q.operators import FermionicOp
        
        if num_spin_orbitals is None:
            num_spin_orbitals = 2 * h_matrix.shape[0]
        
        terms = {}
        
        # One-body terms
        for i in range(h_matrix.shape[0]):
            for j in range(h_matrix.shape[1]):
                if np.abs(h_matrix[i, j]) > 1e-12:
                    # Spin-up
                    label_up        = f"+_{i} -_{j}"
                    terms[label_up] = terms.get(label_up, 0) + h_matrix[i, j]
                    # Spin-down
                    label_dn        = f"+_{i + h_matrix.shape[0]} -_{j + h_matrix.shape[0]}"
                    terms[label_dn] = terms.get(label_dn, 0) + h_matrix[i, j]
        
        # Pairing terms (BdG)
        if v_matrix is not None:
            for i in range(v_matrix.shape[0]):
                for j in range(v_matrix.shape[1]):
                    if np.abs(v_matrix[i, j]) > 1e-12:
                        # Pairing: (c_i c_j + h.c.)
                        label_pair              = f"+_{i} +_{j}"
                        terms[label_pair]       = terms.get(label_pair, 0) + v_matrix[i, j]
                        label_pair_hc           = f"-_{j} -_{i}"
                        terms[label_pair_hc]    = terms.get(label_pair_hc, 0) + np.conj(v_matrix[i, j])
        
        # Constant term
        if np.abs(constant) > 1e-12:
            terms[''] = constant
        
        return FermionicOp(terms)
    
    @staticmethod
    def from_qiskit_hamiltonian(qiskit_ham, num_spin_orbitals: int) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Extract matrices from Qiskit QuadraticHamiltonian.
        
        Parameters
        ----------
        qiskit_ham
            Qiskit QuadraticHamiltonian object
        num_spin_orbitals : int
            Number of spatial orbitals (will be multiplied by 2 for spin)
            
        Returns
        -------
        h_matrix : np.ndarray
            One-body hopping matrix
        v_matrix : np.ndarray or None
            Pairing matrix (if present)
        constant : float
            Constant energy offset
            
        Raises
        ------
        ImportError
            If Qiskit Nature not installed
        """
        if not QiskitInterop.is_qiskit_available():
            raise ImportError("Qiskit Nature required for this operation.")
        
        # Extract matrices from Qiskit object
        h_matrix = np.array(qiskit_ham.normal_ordered().matrices[0])
        
        v_matrix = None
        if len(qiskit_ham.normal_ordered().matrices) > 1:
            v_matrix = np.array(qiskit_ham.normal_ordered().matrices[1])
        
        constant = float(qiskit_ham.normal_ordered().constant)
        
        return h_matrix, v_matrix, constant
    
    @staticmethod
    def to_qiskit_circuit(qiskit_op, num_qubits: int, initial_state: Optional[Dict] = None):
        """Convert fermionic operator to quantum circuit.
        
        Parameters
        ----------
        qiskit_op
            Qiskit FermionicOp or similar
        num_qubits : int
            Number of qubits
        initial_state : dict, optional
            Initial state specification
            
        Returns
        -------
        qiskit.QuantumCircuit
            Quantum circuit implementing the operator
            
        Raises
        ------
        ImportError
            If Qiskit not installed
        """
        if not QiskitInterop.is_qiskit_available():
            raise ImportError("Qiskit required for this operation.")
        
        from qiskit import QuantumCircuit
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        
        try:
            from qiskit.circuit.library import UnitaryGate
            
            mapper      = JordanWignerMapper()
            qubit_op    = mapper.map(qiskit_op)
            qc          = QuantumCircuit(num_qubits)
            
            if initial_state:
                # Prepare initial state
                for qubit, val in initial_state.items():
                    if val:
                        qc.x(qubit)
            
            return qc
        except Exception as e:
            print(f"Error converting to Qiskit circuit: {e}")
            raise

# --------------------------------------------------------------------------
#! GOOGLE
# --------------------------------------------------------------------------

class OpenFermionInterop:
    """OpenFermion interoperability utilities."""
    
    @staticmethod
    def is_openfermion_available() -> bool:
        """Check if OpenFermion is installed."""
        try:
            import openfermion
            return True
        except ImportError:
            return False
    
    @staticmethod
    def to_openfermion_hamiltonian(h_matrix : np.ndarray,
                                v_matrix    : Optional[np.ndarray] = None,
                                constant    : float = 0.0) -> 'FermionOperator':
        """Convert to OpenFermion FermionOperator.
        
        Parameters
        ----------
        h_matrix : np.ndarray
            One-body hopping matrix (N x N)
        v_matrix : np.ndarray, optional
            Pairing matrix for BdG (N x N)
        constant : float, optional
            Constant energy offset
            
        Returns
        -------
        openfermion.FermionOperator
            OpenFermion fermionic operator
            
        Raises
        ------
        ImportError
            If OpenFermion not installed
        """
        if not OpenFermionInterop.is_openfermion_available():
            raise ImportError("OpenFermion required. Install: pip install openfermion")
        
        from openfermion import FermionOperator
        
        H = FermionOperator()
        
        # One-body terms
        for i in range(h_matrix.shape[0]):
            for j in range(h_matrix.shape[1]):
                if np.abs(h_matrix[i, j]) > 1e-12:
                    H += h_matrix[i, j] * FermionOperator(f'{i}^ {j}')
        
        # Pairing terms (BdG)
        if v_matrix is not None:
            for i in range(v_matrix.shape[0]):
                for j in range(v_matrix.shape[1]):
                    if np.abs(v_matrix[i, j]) > 1e-12:
                        # Pair creation and annihilation
                        H += v_matrix[i, j] * FermionOperator(f'{i}^ {j}^')
                        H += np.conj(v_matrix[i, j]) * FermionOperator(f'{j} {i}')
        
        # Constant
        if np.abs(constant) > 1e-12:
            H += constant * FermionOperator()
        
        return H
    
    @staticmethod
    def from_openfermion_hamiltonian(of_ham, num_orbitals: int) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Extract matrices from OpenFermion Hamiltonian.
        
        Parameters
        ----------
        of_ham
            OpenFermion FermionOperator
        num_orbitals : int
            Number of spatial orbitals
            
        Returns
        -------
        h_matrix : np.ndarray
            One-body hopping matrix
        v_matrix : np.ndarray or None
            Pairing matrix (if present)
        constant : float
            Constant energy offset
        """
        if not OpenFermionInterop.is_openfermion_available():
            raise ImportError("OpenFermion required.")
        
        h_matrix = np.zeros((num_orbitals, num_orbitals), dtype=complex)
        v_matrix = None
        constant = 0.0
        
        for term, coeff in of_ham.terms.items():
            if len(term) == 0:
                # Constant term
                constant = float(np.real(coeff))
            elif len(term) == 2:
                # One-body or pairing term
                i, op_i = term[0]
                j, op_j = term[1]
                
                if op_i == 1 and op_j == 0:  # c_i^ c_j
                    h_matrix[i, j] += coeff
                elif op_i == 0 and op_j == 1:  # c_i c_j^
                    h_matrix[i, j] += coeff
        
        return h_matrix, v_matrix, constant

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------