# Changelog

All notable changes to the QES (Quantum Eigen Solver) package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Central `qes_globals` module providing singleton accessors for logger and backend manager.
- Singleton identity tests (`test_globals_singleton.py`).
- Lightweight import-time health test (`test_imports_lightweight.py`) for key modules.
- README documentation for global singleton usage and migration pattern.
- Initial package structure and setup (consolidated historical note).

### Changed

- Refactored all fragile/bare imports to use absolute or explicit relative imports for robustness.
- Refactored `general_python.algebra.utils` to obtain logger via `qes_globals` preventing duplicate initialization.
- Enhanced `QES.__init__` to re-export global accessor helpers and use new reseed wrappers.
- Improved package organization & dependency management (historical consolidation).

### Fixed

- Eliminated double initialization of logger and backend RNG state across multiple import paths.
- Fixed missing imports, type guards, and indentation errors from import refactor.
- Package installation and dependency issues (historical note).

### Deprecated

- Nothing yet

### Removed

- Nothing yet

### Security

- Nothing yet

## [0.1.0] - 2025-05-24

### Initial Release Summary

- Core quantum eigenvalue solving functionality
- Neural Quantum States (NQS) implementation
- Monte Carlo solver framework
- Algebra and operator modules
- General purpose utilities
- JAX and NumPy backend support
- Documentation and examples

[Unreleased]: https://github.com/makskliczkowski/QuantumEigenSolver/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/makskliczkowski/QuantumEigenSolver/releases/tag/v0.1.0

------------------------------------------------------------------------------------------------------------------------

<!-- Removed duplicate 0.1.0 section & redundant links to satisfy linters -->
