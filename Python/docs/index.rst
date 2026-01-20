.. QES documentation master file, created by
   sphinx-quickstart on Tue Nov  7 12:00:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Quantum EigenSolver (QES)
====================================

.. image:: https://img.shields.io/badge/license-CC--BY--4.0-blue.svg
    :target: https://github.com/makskliczkowski/QuantumEigenSolver/blob/master/LICENSE
    :alt: License

**Quantum EigenSolver (QES)** is a comprehensive Python framework designed for solving quantum many-body eigenvalue problems. It unifies Exact Diagonalization (ED) and Variational Monte Carlo (VMC) methods with Neural Quantum States (NQS), providing a modular and high-performance toolkit for quantum physics research.

Whether you are simulating spin systems, fermions, or implementing novel neural network ansatzes, QES offers the flexibility and speed you need, leveraging modern libraries like JAX and Numba.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   architecture
   physics_background
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/algebra
   api/nqs
   api/solver
   api/utilities

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
