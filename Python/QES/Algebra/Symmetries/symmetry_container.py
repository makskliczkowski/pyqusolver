"""
Symmetry container for managing symmetry operations in Hilbert spaces.

This module provides a general framework for handling symmetries in quantum many-body
systems - interacting and non-interacting alike. It supports various symmetry types
(e.g., translation, reflection, parity, etc.) through a unified interface.

Key Features
------------
1. **Uniform treatment**: 
    All symmetries (translation, reflection, parity, etc.) 
    are handled through the same interface.
2. **Automatic group construction**: 
    Builds full symmetry group from generators.
3. **Compatibility checking**: 
    Automatically determines which symmetries commute.
4. **Multiple backends**: 
    Supports integer, NumPy, and JAX state representations.
5. **Memory efficient**: 
    Computes representatives and normalizations on-the-fly.

----------------------------------------------------------------------------
File    : QES/Algebra/Symmetries/symmetry_container.py
Author  : Maksymilian Kliczkowski
Date    : 2025-10-28
Version : 1.2.0
----------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from itertools import combinations
from functools import lru_cache

# --------------------------------------------------------------------------

try:
    from QES.general_python.common.flog import get_global_logger
    from QES.general_python.lattices.lattice import Lattice, LatticeDirection
    from QES.Algebra.Symmetries.base import SymmetryOperator, SymmetryClass
    from QES.Algebra.Operator.operator import SymmetryGenerators
    from QES.Algebra.globals import GlobalSymmetry
    
    JAX_AVAILABLE = True
    try:
        import jax.numpy as jnp
    except ImportError:
        JAX_AVAILABLE = False
        jnp = np
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}")

#############################################################################
#! Constants
#############################################################################

_SYM_NORM_THRESHOLD = 1e-12
_INT_HUGE           = np.iinfo(np.int64).max

#############################################################################
#! Type Aliases
#############################################################################

StateInt            = int                                                       # Integer representation of state - normally 64-bit
StateArray          = np.ndarray                                                # or jnp.ndarray if JAX is used    
SymmetrySpec        = Tuple[SymmetryGenerators, Union[int, float, complex]]     # (Symmetry type, sector value)
GroupElement        = Tuple[SymmetryOperator, ...]                              # Tuple of operators to apply sequentially

#############################################################################
#! Symmetry Compatibility Checker
#############################################################################

class SymmetryCompatibility:
    """
    Determines which symmetry generators are compatible with each other.
    Uses the symmetries' own compatibility rules rather than hardcoded logic.
    
    This class coordinates compatibility checking but delegates the actual
    logic to the SymmetryOperator instances themselves.
    
    Examples
    --------
    >>> compat = SymmetryCompatibility(ns=4, nhl=2, lattice=lattice)
    >>> # Symmetries define their own compatibility
    >>> compat.check_compatibility(sym1, sym2)
    """
    
    def __init__(self, ns: int, nhl: int = 2, lattice: Optional[Lattice] = None, logger: Optional[Callable[[str], None]] = None):
        """
        Initialize compatibility checker.
        
        Parameters
        ----------
        ns : int
            Number of sites in the system
        nhl : int
            Local Hilbert space dimension
        lattice : Optional[Lattice]
            Lattice structure of the system
        logger : Optional[Callable[[str], None]]
            Logger function for debugging messages
        """
        self.ns         = ns
        self.nhl        = nhl
        self.lattice    = lattice
        self.logger     = logger or get_global_logger()

        # Cache of compatibility decisions
        self._compat_cache: Dict[Tuple, bool] = {}
    
    # -----------------------------------------------------
    #! Pairwise compatibility check
    # -----------------------------------------------------
    
    def check_pair_compatibility(
        self,
        op1     : SymmetryOperator,
        op2     : SymmetryOperator,
        spec1   : SymmetrySpec,
        spec2   : SymmetrySpec
        ) -> Tuple[bool, str]:
        """
        Check if two symmetry operators are compatible.
        
        Delegates to the symmetries' own commutes_with() methods.
        
        Parameters
        ----------
        op1, op2 : SymmetryOperator
            The symmetry operators - instances of SymmetryOperator
            with their own compatibility logic
        spec1, spec2 : SymmetrySpec
            The (generator_type, sector) specifications
        
        Returns
        -------
        compatible : bool
            Whether the operators can be used together
        reason : str
            Explanation of the decision
        """
        gen1, sector1   = spec1
        gen2, sector2   = spec2
        
        # Build context for compatibility check
        context         = {
                            'ns'        : self.ns,
                            'nhl'       : self.nhl,
                            'lattice'   : self.lattice
                        }
        
        # Check if op1 commutes with op2
        if op1.commutes_with(op2, **context):
            return True, "Compatible"
        
        # Check both sectors for "realness" - more specific reason
        if not op1.is_real_sector(**context):
            return False, f"{gen1.name} sector {sector1} incompatible with {gen2.name}"
        
        if not op2.is_real_sector(**context):
            return False, f"{gen2.name} sector {sector2} incompatible with {gen1.name}"
        
        return False, f"{gen1.name} and {gen2.name} do not commute"
    
    # -----------------------------------------------------
    
    def check_boundary_conditions(self, operator: SymmetryOperator) -> Tuple[bool, str]:
        """
        Check if lattice boundary conditions support this symmetry.
        
        Delegates to the operator's own check_boundary_conditions() method.
        
        Parameters
        ----------
        operator : SymmetryOperator
            The symmetry operator to check
        
        Returns
        -------
        valid : bool
            Whether boundary conditions are compatible
        reason : str
            Explanation
        """
        return operator.check_boundary_conditions(lattice=self.lattice, ns=self.ns, nhl=self.nhl)
    
    # -----------------------------------------------------

    def check_global_symmetry_effects(self, 
                                    operators   : List[Tuple[SymmetryOperator, SymmetrySpec]], 
                                    global_syms : List[GlobalSymmetry]) -> List[Tuple[SymmetryOperator, SymmetrySpec]]:
        """
        Remove or modify operators based on global symmetries.
        
        Delegates to each symmetry's is_compatible_with_global_symmetry() method.
        This is fully general and extensible - no hardcoded rules.

        Parameters
        ----------
        operators : List[Tuple[SymmetryOperator, SymmetrySpec]]
            List of (operator, (gen_type, sector)) tuples
        global_syms : List[GlobalSymmetry]
            Global symmetries to check against
        
        Returns
        -------
        filtered : List[Tuple[SymmetryOperator, SymmetrySpec]]
            Operators compatible with global symmetries
        """
        if not global_syms:
            return operators  # No global symmetries, no filtering needed
        
        filtered = []
        
        for op, spec in operators:
            gen, sector = spec
            keep        = True
            reason      = ""
            
            # Check compatibility with each global symmetry
            for gsym in global_syms:
                compatible, msg = op.is_compatible_with_global_symmetry(
                                    gsym, 
                                    ns      = self.ns, 
                                    nhl     = self.nhl, 
                                    lattice = self.lattice,
                                    sector  = sector
                                )
                if not compatible:
                    keep    = False
                    reason  = msg
                    break
            
            if keep:
                filtered.append((op, spec))
            elif reason:
                self.logger.info(reason)
        
        return filtered
    
    # -----------------------------------------------------
    #! Main compatibility checker
    # -----------------------------------------------------
    
    def get_compatible_operators(self,
            operators       : List[Tuple[SymmetryOperator, SymmetrySpec]],
            global_syms     : List[GlobalSymmetry]) -> List[Tuple[SymmetryOperator, SymmetrySpec]]:
        """
        Filter operators to only include compatible ones.
        
        Parameters
        ----------
        operators : List[Tuple[SymmetryOperator, SymmetrySpec]]
            Operators to filter
        global_syms : List[GlobalSymmetry]
            Global symmetries
        
        Returns
        -------
        compatible : List[Tuple[SymmetryOperator, SymmetrySpec]]
            Mutually compatible operators
        """
        
        # First filter based on global symmetries
        filtered    = self.check_global_symmetry_effects(operators, global_syms)
        
        # Now check pairwise compatibility
        result      = []
        
        for op, spec in filtered:
            gen, sector = spec
            
            # Check BC compatibility
            bc_valid, bc_reason = self.check_boundary_conditions(op)
            if not bc_valid:
                self.logger.warning(f"Removing {gen.name}: {bc_reason}")
                continue
            
            # Check pairwise compatibility with already accepted operators
            compatible  = True
            for prev_op, prev_spec in result:
                is_compat, reason = self.check_pair_compatibility(op, prev_op, spec, prev_spec)
                if not is_compat:
                    self.logger.warning(f"Removing {gen.name}: {reason}")
                    compatible = False
                    break
            
            if compatible:
                result.append((op, spec))
        
        return result # Return only compatible operators!

####################################################################################################
#! Symmetry Container
####################################################################################################

@dataclass
class SymmetryContainer:
    """
    TODO: Consider non-abelian symmetries in future.
    
    Container for all symmetry operations in a Hilbert space.
    
    This class provides a unified interface for:
    - Building symmetry groups from generators      - 'build_group()' 
    - Finding representative states                 - 'find_representative()'
    - Computing normalization factors               - 'compute_normalization()'
    - Acting with symmetries on states              - 'act_with_symmetry()'

    Architecture
    ------------
    The container separates global and local symmetries:
    
    - **Global symmetries**: Act as filters (e.g., U(1) particle conservation)
    see the theoretical description in the documentation...
        1. Check if a state satisfies the constraint
        2. Don't form groups or orbits
        3. Applied before representative finding

    - **Local symmetries**: Form groups and orbits (e.g., translation, reflection)
      1. Build full symmetry group from generators
      2. Find representative (minimal state in orbit)
      3. Compute normalization (stabilizer subgroup sum)

    How to Add New Symmetries
    --------------------------
    To add a new symmetry operator:
    
    1. **Create symmetry class** in QES/Algebra/Symmetries/
       ```python
       class MySymmetry(SymmetryOperator):
           def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
               # Transform integer state, return (new_state, phase)
               ...
           
           def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
               # Transform numpy state vector
               ...
           
           def apply_jax(self, state: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, complex]:
               # Transform JAX state vector (if JAX available)
               ...
       ```
    
    2. **Add to SymmetryGenerators enum** in operator.py
       ```python
       class SymmetryGenerators(Enum):
           MY_SYMMETRY = "my_symmetry"
       ```
    
    3. **Register compatibility rules** in SymmetryCompatibility.check_pair_compatibility()
       ```python
       if {gen1, gen2} == {SymmetryGenerators.MY_SYMMETRY, SymmetryGenerators.OTHER}:
           return False, "Reason why incompatible"
       ```
       
    4. **Register global compatibility effects** in SymmetryCompatibility.check_global_symmetry_effects()
    
    5. **Use in HilbertSpace**
       ```python
       hilbert = HilbertSpace(
           ns       = 10,
           sym_gen  = [
               (SymmetryGenerators.MY_SYMMETRY, sector_value),
               ...
           ]
       )
       ```
    
    The container automatically:
    - Checks compatibility with other symmetries        - pairwise checks
    - Builds the full symmetry group                    - combinations of generators
    - Finds representatives using your apply_int method - minimal state in orbit
    - Computes normalizations                           - stabilizer subgroup sums
    
    Examples
    --------
    Basic usage:
    
    >>> from QES.Algebra.Symmetries.translation import TranslationSymmetry
    >>> container = SymmetryContainer(ns=4, lattice=lattice)
    >>> container.add_generator(SymmetryGenerators.T, sector=0, operator=TranslationSymmetry(lattice))
    >>> container.build_group()
    >>> rep, phase = container.find_representative(5) # Find rep of state |0101>
    
    Parameters
    ----------
    ns : int
        Number of sites in the system
    lattice : Optional[Lattice]
        Lattice structure (needed for spatial symmetries)
    nhl : int
        Local Hilbert space dimension (default: 2 for spin-1/2)
    backend : str
        Backend for computations ('numpy', 'jax', or 'default')
    """
    
    ns                  : int
    lattice             : Optional[Lattice] = None
    nhl                 : int               = 2
    backend             : str               = 'default'

    # Storage
    generators          : List[Tuple[SymmetryOperator, SymmetrySpec]]   = field(default_factory=list)
    global_symmetries   : List[GlobalSymmetry]                          = field(default_factory=list)
    symmetry_group      : List[GroupElement]                            = field(default_factory=list)

    # Cached data
    _repr_map           : Optional[np.ndarray]                          = None
    _compatibility      : Optional[SymmetryCompatibility]               = None
    logger              : Optional[Callable[[str], None]]               = None
    
    # -----------------------------------------------------
    #! Initialization
    # -----------------------------------------------------

    def __post_init__(self):
        """Initialize compatibility checker and logger."""
        self._compatibility = SymmetryCompatibility(self.ns, self.nhl, self.lattice)
        self.logger         = get_global_logger() if self.logger is None else self.logger

    # -----------------------------------------------------
    #! Generator Management
    # -----------------------------------------------------

    def add_generator(
        self, 
        gen_type    : SymmetryGenerators, 
        sector      : Union[int, float, complex],
        operator    : SymmetryOperator) -> bool:
        """
        Add a symmetry generator to the container.
        
        Parameters
        ----------
        gen_type : SymmetryGenerators
            Type of symmetry
        sector : Union[int, float, complex]
            Sector value (quantum number)
        operator : SymmetryOperator
            Symmetry operator instance
        
        Returns
        -------
        added : bool
            Whether the generator was successfully added
        """
        spec = (gen_type, sector)
        
        # Check boundary conditions first
        bc_valid, bc_reason = self._compatibility.check_boundary_conditions(operator)
        if not bc_valid:
            self.logger.warning(f"Cannot add {gen_type.name}: {bc_reason}")
            return False
        
        # Check compatibility with existing generators
        for existing_op, existing_spec in self.generators:
            compat, reason = self._compatibility.check_pair_compatibility(operator, existing_op, spec, existing_spec)
            if not compat:
                self.logger.warning(f"Cannot add {gen_type.name}: {reason}")
                return False
        
        self.generators.append((operator, spec))
        self.logger.info(f"Added symmetry generator: {gen_type.name} = {sector}")
        return True
    
    # -----------------------------------------------------
    #! Global Symmetry Management
    # -----------------------------------------------------
    
    def add_global_symmetry(self, global_sym: GlobalSymmetry) -> None:
        """Add a global symmetry (filtering constraint)."""
        self.global_symmetries.append(global_sym)
        self.logger.info(f"Added global symmetry: {global_sym.name}")
    
    # -----------------------------------------------------
    #! Compatibility Filtering - built-in function
    # -----------------------------------------------------
    
    def build_group(self) -> None:
        """
        Build the full symmetry group from generators.
        
        Algorithm
        ---------
        1. Separate translations from other generators
        2. For non-translation generators, create all combinations
        3. Build translation group (product of cyclic groups for multi-D)
        4. Combine: full_group = non_translation_combos × translation_group
        
        For 2D with Tx and Ty:
            - Non-translation: {E, P, R, PR, ...}
            - Translation: {Tx^i Ty^j : i=0..Nx-1, j=0..Ny-1}
            - Full: each non-translation combo × each translation combo
        """
        if not self.generators:
            self.logger.info("No symmetry generators - empty group")
            self.symmetry_group = [()]  # Identity element
            return
        
        # Separate translations from other generators
        translations: Dict[str, Tuple[SymmetryOperator, SymmetrySpec]] = {}  # direction -> (op, spec)
        other_generators = []
        
        for op, spec in self.generators:
            gen_type = spec[0]
            # Check if this is any translation symmetry
            if gen_type == SymmetryGenerators.Translation_x:
                translations['x'] = (op, spec)
            elif gen_type == SymmetryGenerators.Translation_y:
                translations['y'] = (op, spec)
            elif gen_type == SymmetryGenerators.Translation_z:
                translations['z'] = (op, spec)
            else:
                other_generators.append((op, spec))
        
        # Build combinations of non-translation generators
        # Start with identity (empty tuple)
        base_elements: List[GroupElement] = [()]
        
        # Add all combinations using bitmask approach
        n_other = len(other_generators)
        for r in range(1, n_other + 1):
            for combo_indices in combinations(range(n_other), r):
                # Build operator tuple for this combination
                ops_tuple = tuple(other_generators[i][0] for i in combo_indices)
                base_elements.append(ops_tuple)
        
        # If no translations, we're done
        if not translations:
            self.symmetry_group = base_elements
            self.logger.info(f"Built symmetry group with {len(self.symmetry_group)} elements")
            return
        
        # Build translation group (product of cyclic groups)
        translation_elements = self._build_translation_group(translations)
        
        # Combine: each base element with each translation element
        full_group: List[GroupElement] = []
        for t_elem in translation_elements:
            for base_elem in base_elements:
                # Concatenate tuples: (translation ops) + (other ops)
                combined = t_elem + base_elem
                full_group.append(combined)
        
        self.symmetry_group = full_group
        self.logger.info(f"Built symmetry group with {len(self.symmetry_group)} elements "
                        f"({len(translation_elements)} translation × {len(base_elements)} base)")
    
    def _build_translation_group(
        self, 
        translations: Dict[str, Tuple[SymmetryOperator, SymmetrySpec]]
    ) -> List[GroupElement]:
        """
        Build the translation subgroup as a product of cyclic groups.
        
        For 1D (only Tx): {E, Tx, Tx^2, ..., Tx^(Nx-1)}
        For 2D (Tx, Ty): {Tx^i Ty^j : i=0..Nx-1, j=0..Ny-1}
        For 3D (Tx, Ty, Tz): {Tx^i Ty^j Tz^k : i,j,k over ranges}
        
        Parameters
        ----------
        translations : Dict[str, Tuple[SymmetryOperator, SymmetrySpec]]
            Dictionary mapping direction ('x', 'y', 'z') to (operator, spec)
        
        Returns
        -------
        translation_group : List[GroupElement]
            All combinations of translation powers
        """
        # Determine the period for each direction
        # For periodic BC, translation^N = identity, so we have N elements in cyclic group
        periods = {'x': self.ns, 'y': self.ns, 'z': self.ns}
        
        # If lattice has different sizes in different directions, use those
        if self.lattice is not None:
            if hasattr(self.lattice, 'lx') and 'x' in translations:
                periods['x'] = self.lattice.lx
            if hasattr(self.lattice, 'ly') and 'y' in translations:
                periods['y'] = self.lattice.ly
            if hasattr(self.lattice, 'lz') and 'z' in translations:
                periods['z'] = self.lattice.lz
        
        # Build all combinations
        translation_group: List[GroupElement] = []
        
        # Get sorted directions for consistent ordering
        directions = sorted(translations.keys())
        
        if len(directions) == 0:
            return [()]
        
        # Build product of cyclic groups
        # For each combination of powers (i, j, k, ...) create Tx^i Ty^j Tz^k ...
        ranges = [range(periods[d]) for d in directions]
        
        # Generate all combinations using itertools.product
        from itertools import product as cartesian_product
        
        for powers in cartesian_product(*ranges):
            # Build tuple of operators: Tx repeated i times, Ty repeated j times, etc.
            ops_tuple = ()
            for direction, power in zip(directions, powers):
                t_op, t_spec = translations[direction]
                # Add this translation operator 'power' times
                ops_tuple += tuple([t_op] * power)
            
            translation_group.append(ops_tuple)
        
        return translation_group
    
    # -----------------------------------------------------
    #! Core Functionality
    # -----------------------------------------------------
    
    def apply_group_element(self, element: GroupElement, state: StateInt) -> Tuple[StateInt, Union[complex, float]]:
        """
        Apply a group element (tuple of operators) to a state.
        
        Parameters
        ----------
        element : GroupElement
            Tuple of operators to apply sequentially
        state : StateInt
            Integer representation of state
        
        Returns
        -------
        new_state : StateInt
            Transformed state
        phase : complex
            Accumulated phase from symmetry eigenvalues
        """
        if len(element) == 0:
            # Identity element
            return state, 1.0
        
        current_state       = state
        accumulated_phase   = 1.0

        # Apply operators left to right
        for op in element:
            current_state, phase = op.apply_int(current_state, self.ns, nhl=self.nhl)
            accumulated_phase   *= phase
        
        return current_state, accumulated_phase

    def find_representative(self, state: StateInt, use_cache: bool = True) -> Tuple[StateInt, complex]:
        """
        Find the representative (minimal state) in the orbit of a given state.
        
        The representative is defined as the state with the smallest integer value
        that can be reached by applying symmetry operations.
        
        Parameters
        ----------
        state : StateInt
            State to find representative for
        use_cache : bool
            Whether to use cached representative map if available
        
        Returns
        -------
        representative : StateInt
            Minimal state in the orbit
        symmetry_eigenvalue : complex
            Phase accumulated when transforming state → representative
        
        Algorithm
        ---------
        1. Check cache if available
        2. Apply all group elements to the state
        3. Find the transformation that gives minimal state
        4. Return (min_state, phase_to_reach_it)
        """
        # Check cache
        if use_cache and self._repr_map is not None and state < len(self._repr_map):
            idx     = self._repr_map[state, 0]
            sym_eig = self._repr_map[state, 1]
            
            if idx != _INT_HUGE:
                return int(idx), sym_eig
        
        # No symmetries - state is its own representative
        if len(self.symmetry_group) == 0:
            return state, 1.0
        
        min_state = _INT_HUGE
        min_phase = 1.0
        
        # Try all group elements
        for element in self.symmetry_group:
            new_state, phase = self.apply_group_element(element, state)
            
            if new_state < min_state:
                min_state = new_state
                min_phase = phase
        
        min_phase = phase
        
        return min_state, min_phase
    
    def get_character(self, element: GroupElement) -> complex:
        """
        Compute the character (representation eigenvalue) for a group element.
        
        For translation T^n in momentum sector k: χ_k(T^n) = exp(2πi * k * n / L)
        For other symmetries: χ(g) = sector_value (usually ±1)
        
        Parameters
        ----------
        element : GroupElement
            Tuple of symmetry operators representing a group element
        
        Returns
        -------
        character : complex
            Character value for this element in the current representation
        """
        if len(element) == 0:
            return 1.0  # Identity element
        
        character = 1.0
        
        # Count how many times each generator appears
        from collections import Counter
        op_counts = Counter(element)
        
        for op in op_counts:
            count = op_counts[op]
            
            # Find the sector for this operator
            sector_value = None
            for gen_op, (gen_type, sector) in self.generators:
                if gen_op is op:
                    sector_value = sector
                    break
            
            if sector_value is None:
                continue
            
            # For translation: character is exp(2πi * k * power / L)
            if hasattr(op, 'symmetry_class') and op.symmetry_class == SymmetryClass.TRANSLATION:
                # Get the period (lattice size in this direction)
                period = self.ns  # Default
                if self.lattice is not None:
                    if hasattr(op, 'direction'):
                        from QES.general_python.lattices.lattice import LatticeDirection
                        if op.direction == LatticeDirection.X or str(op.direction) == 'x':
                            period = getattr(self.lattice, 'lx', self.ns)
                        elif op.direction == LatticeDirection.Y or str(op.direction) == 'y':
                            period = getattr(self.lattice, 'ly', self.ns)
                        elif op.direction == LatticeDirection.Z or str(op.direction) == 'z':
                            period = getattr(self.lattice, 'lz', self.ns)
                
                # Character for T^count in momentum sector k
                character *= np.exp(2j * np.pi * sector_value * count / period)
            
            # For discrete symmetries (reflection, parity): character is sector_value^count
            else:
                # Usually sector is ±1, so character is (±1)^count
                character *= sector_value ** count
        
        return character
    
    def compute_normalization(self, state: StateInt) -> complex:
        """
        Compute normalization factor for a representative state in the current sector.
        
        The normalization is computed using the projection formula:
        N = sqrt(sum_{g in G} χ_k(g)^* <state|g|state>)
        
        where χ_k(g) is the character of element g in irrep k.
        
        For a state to belong to momentum sector k, it must satisfy:
        |state_k> = (1/√N) sum_{g in G} χ_k(g)^* g|rep>
        
        Parameters
        ----------
        state : StateInt
            Representative state
        
        Returns
        -------
        norm : complex
            Normalization factor (0 if state not in this sector)
        
        Algorithm
        ---------
        1. Apply all group elements to state
        2. Sum characters * phases for elements that return to same state
        3. Return sqrt of sum
        
        Physical Interpretation
        -----------------------
        If norm = 0, the state does not belong to this momentum/symmetry sector.
        If norm > 0, the state is a valid representative for this sector.
        """
        if len(self.symmetry_group) == 0:
            return 1.0
        
        projection_sum = 0.0
        
        for element in self.symmetry_group:
            new_state, intrinsic_phase = self.apply_group_element(element, state)
            
            # Only states in the orbit contribute
            # For momentum sectors, we need the character even if state changes
            character = self.get_character(element)
            
            # The projection operator is P_k = (1/|G|) sum_g χ_k(g)^* g
            # We compute <state|P_k|state> = (1/|G|) sum_g χ_k(g)^* <state|g|state>
            # The overlap <state|g|state> = phase * δ_{state, g(state)}
            
            if new_state == state:
                # State is invariant under this group element
                # Contribution: χ_k(g)^* * phase
                projection_sum += np.conj(character) * intrinsic_phase
        
        # Normalization is sqrt of the projection sum
        norm = np.sqrt(abs(projection_sum))
        
        # Check if normalization is non-zero (state allowed in this sector)
        if abs(norm) < _SYM_NORM_THRESHOLD:
            return 0.0
        
        return norm
    
    def check_global_symmetries(self, state: StateInt) -> bool:
        """
        Check if a state satisfies all global symmetry constraints.
        
        Parameters
        ----------
        state : StateInt
            State to check
        
        Returns
        -------
        satisfies : bool
            Whether state satisfies all global symmetries
        """
        for global_sym in self.global_symmetries:
            if not global_sym(int(state)):
                return False
        return True


####################################################################################################
#! Utility Functions
####################################################################################################

def create_symmetry_container_from_specs(
    ns                  : int,
    generator_specs     : List[SymmetrySpec],
    global_syms         : List[GlobalSymmetry],
    lattice             : Optional[Lattice] = None,
    nhl                 : int = 2,
    backend             : str = 'default',
    build_group         : bool = True,
    build_repr_map      : bool = False
) -> SymmetryContainer:
    """
    Factory function to create and initialize a SymmetryContainer.
    
    Parameters
    ----------
    ns : int
        Number of sites
    generator_specs : List[SymmetrySpec]
        List of (generator_type, sector) tuples
    global_syms : List[GlobalSymmetry]
        List of global symmetries
    lattice : Optional[Lattice]
        Lattice structure
    nhl : int
        Local Hilbert space dimension
    backend : str
        Computation backend
    build_group : bool
        Whether to build symmetry group immediately
    build_repr_map : bool
        Whether to build representative map immediately
    
    Returns
    -------
    container : SymmetryContainer
        Initialized container
    """
    container = SymmetryContainer(ns=ns, lattice=lattice, nhl=nhl, backend=backend)
    
    # Add global symmetries
    for gsym in global_syms:
        container.add_global_symmetry(gsym)
    
    # Create operator instances first
    operators_with_specs: List[Tuple[SymmetryOperator, SymmetrySpec]] = []
    for gen_type, sector in generator_specs:
        operator = _create_symmetry_operator(gen_type, sector, lattice, ns, nhl)
        if operator is not None:
            operators_with_specs.append((operator, (gen_type, sector)))
    
    # Filter for compatibility
    compat      = SymmetryCompatibility(ns, nhl, lattice)
    filtered    = compat.get_compatible_operators(operators_with_specs, global_syms)
    
    # Add filtered operators to container
    for operator, (gen_type, sector) in filtered:
        container.add_generator(gen_type, sector, operator)
    
    # Build group if requested
    if build_group:
        container.build_group()
    
    # Build representative map if requested
    if build_repr_map:
        nh_full = nhl ** ns
        container.build_representative_map(nh_full)
    
    return container

# -----------------------------------------------------
#! Symmetry Operator Factory
# -----------------------------------------------------

def _create_symmetry_operator(
    gen_type        : SymmetryGenerators,
    sector          : Union[int, float, complex],
    lattice         : Optional[Lattice],
    ns              : int,
    nhl             : int
) -> Optional[SymmetryOperator]:
    """
    Factory to create symmetry operator instances.
    
    This function imports the appropriate symmetry class and instantiates it.
    """
    try:
        # Translation symmetries (different directions)
        if gen_type in (SymmetryGenerators.Translation_x, 
                       SymmetryGenerators.Translation_y, 
                       SymmetryGenerators.Translation_z):
            from QES.Algebra.Symmetries.translation import TranslationSymmetry
            # Extract direction from enum name
            direction_map = {
                SymmetryGenerators.Translation_x: 'x',
                SymmetryGenerators.Translation_y: 'y',
                SymmetryGenerators.Translation_z: 'z'
            }
            direction = direction_map[gen_type]
            return TranslationSymmetry(lattice=lattice, sector=sector, ns=ns, direction=direction)
        
        # Reflection
        elif gen_type == SymmetryGenerators.Reflection:
            from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
            return ReflectionSymmetry(lattice=lattice, sector=sector, ns=ns)
        
        # Parity symmetries
        elif gen_type == SymmetryGenerators.ParityX:
            from QES.Algebra.Symmetries.parity import ParitySymmetry
            return ParitySymmetry(axis='x', sector=sector, ns=ns, nhl=nhl)
        
        elif gen_type == SymmetryGenerators.ParityY:
            from QES.Algebra.Symmetries.parity import ParitySymmetry
            return ParitySymmetry(axis='y', sector=sector, ns=ns, nhl=nhl)
        
        elif gen_type == SymmetryGenerators.ParityZ:
            from QES.Algebra.Symmetries.parity import ParitySymmetry
            return ParitySymmetry(axis='z', sector=sector, ns=ns, nhl=nhl)
        
        else:
            get_global_logger().warning(f"Unknown symmetry type: {gen_type}")
            return None
    
    except ImportError as e:
        get_global_logger().error(f"Failed to import symmetry {gen_type}: {e}")
        return None

####################################################################################################
#! End of file
####################################################################################################
