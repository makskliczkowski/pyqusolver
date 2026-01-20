# Introduction

Welcome to **Quantum EigenSolver (QES)**! We are thrilled to have you here.

QES is a robust and modular Python framework designed to simulate quantum many-body systems. Whether you are a researcher exploring new phases of matter, a student learning about quantum algorithms, or a developer building high-performance quantum tools, QES provides the foundation you need.

## What is QES?

At its core, QES is a unified solver for quantum eigenvalue problems. It seamlessly bridges the gap between traditional methods like **Exact Diagonalization (ED)** and modern approaches like **Variational Monte Carlo (VMC)** with **Neural Quantum States (NQS)**.

We built QES with three principles in mind:

1.  **Performance**: Leveraging **JAX** for GPU acceleration and automatic differentiation, and **Numba** for high-speed CPU compilation, QES handles complex simulations efficiently.
2.  **Modularity**: The codebase is structured like a library of building blocks—Hamiltonians, Hilbert spaces, Operators, and Ansatzes—that you can mix and match or extend with your own custom implementations.
3.  **Usability**: We strive to make quantum simulation accessible, with clear APIs, helpful utilities, and comprehensive documentation.

## Key Features

*   **Versatile Solvers**:
    *   **Exact Diagonalization**: Solve for ground and excited states of spin and fermionic systems on arbitrary lattices.
    *   **Variational Monte Carlo**: Optimize neural network quantum states (RBM, CNN, Autoregressive) to find ground states of large systems beyond the reach of ED.
*   **Rich Physics Support**:
    *   Built-in support for **Spins**, **Fermions**, and **Bosons**.
    *   **Quadratic Hamiltonians** (free fermions/bosons) with ultra-fast analytical solvers.
    *   **Symmetry Handling**: Utilize translation, parity, and point-group symmetries to reduce computational cost.
*   **Modern Tech Stack**:
    *   **JAX Integration**: Enjoy seamless gradient computation and GPU support.
    *   **Flexible Backends**: Switch between NumPy and JAX backends to suit your hardware and debugging needs.

## Why Use QES?

If you are working on quantum many-body physics, you often face a choice: write your own code from scratch (which is error-prone and time-consuming) or use a "black box" package that is hard to customize.

QES aims to be the middle ground. It gives you the high-level tools to set up standard models quickly, while exposing the low-level components so you can tweak the math, try new ansatzes, or implement exotic Hamiltonians without fighting the framework.

## Next Steps

Ready to dive in?

*   Check out the :doc:`Installation Guide <installation>` to get set up.
*   Jump into the :doc:`Quick Start <quickstart>` to run your first simulation.
*   Explore :doc:`Examples <examples>` to see QES in action.

We hope QES empowers your research and development. Happy coding!
