"""
Tests for global singletons (logger, backend manager, RNG) ensuring
multiple import paths resolve to the same instances.

These are lightweight runtime identity tests (no heavy numerical work) so
they should be fast and safe for CI.

Only basic imports are done here; no heavy initialization.
"""

try:
    from importlib import reload
except ImportError:
    from imp import reload # type: ignore

def test_logger_singleton_identity():
    ''' Test that get_logger() returns the same instance across multiple calls. '''
    from QES.qes_globals import get_logger
    log1    = get_logger()
    log2    = get_logger()
    assert log1 is log2, "get_logger() returned different instances (expected singleton)."

def test_backend_manager_identity():
    ''' Test that get_backend_manager() returns the same instance across multiple calls. '''
    from QES.qes_globals import get_backend_manager
    mgr1    = get_backend_manager()
    mgr2    = get_backend_manager()
    assert mgr1 is mgr2, "Backend manager is not a singleton."

def test_rng_identity_after_reseed():
    ''' Test that get_numpy_rng() returns the same RNG state after reseed and reload. '''
    from QES.qes_globals import get_backend_manager, reseed_all, get_numpy_rng
    reseed_all(123)
    mgr     = get_backend_manager()
    rng1    = get_numpy_rng()
    # Force a reload of qes_globals to ensure lazy re-link still yields same manager
    import QES.qes_globals as qg
    reload(qg)
    rng2    = qg.get_numpy_rng()
    # After reload the underlying manager object should still be the same instance
    assert mgr is qg.get_backend_manager(), "Reload broke backend manager singleton."
    
    # The RNG object will be a generator; identity may change if reseed is called again intentionally.
    assert rng1.bit_generator.state == rng2.bit_generator.state, "RNG state diverged unexpectedly without reseed."

# ----------------------------------------------------------------------------------------------------
#! End of test_globals_singleton.py
# ----------------------------------------------------------------------------------------------------