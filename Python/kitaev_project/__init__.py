"""
Kitaev Honeycomb NQS/ED workflow package.

This namespace exposes the high-level builder and workflow helpers so downstream
code can simply do:

    from kitaev_project.workflows import pipelines

and orchestrate ground-state, excited-state, transfer-learning, and ED routines.
"""

__all__ = [
    "KitaevModelBuilder",
    "run_pipeline",
]


def __getattr__(name):
    if name == "KitaevModelBuilder":
        from .models.kitaev_model import KitaevModelBuilder

    raise AttributeError(name)
