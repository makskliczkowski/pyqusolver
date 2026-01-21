# Contributing to QES

We welcome contributions! Whether you are fixing a bug, adding a new physical model, or improving documentation, your help is appreciated.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/QuantumEigenSolver.git
    cd QuantumEigenSolver/Python
    ```
3.  **Install dependencies** in editable mode with dev tools:
    ```bash
    pip install -e ".[dev,all]"
    ```
4.  **Create a branch** for your feature:
    ```bash
    git checkout -b feature/my-new-feature
    ```

## Code Style

We aim for clean, readable, and "pythonic" code.
*   **Formatting**: We use `black` for code formatting.
*   **Type Hints**: We encourage using Python type hints for function arguments and return values.
*   **Docstrings**: Please write clear docstrings (NumPy style preferred) for all public classes and functions.

## Running Tests

Before submitting a Pull Request, please ensure all tests pass.

```bash
pytest
```

If you are adding a new feature, please add a corresponding test case in the `tests/` directory.

## Documentation

If you change the API or add new features, please update the documentation in `Python/docs/`. You can build the docs locally to check your changes:

```bash
cd Python/docs
make html
```

## Pull Request Process

1.  Push your branch to GitHub.
2.  Open a Pull Request against the `main` branch of the original repository.
3.  Describe your changes clearly in the PR description.
4.  Wait for a review! We'll do our best to provide feedback quickly.

Thank you for helping make QES better!
