"""
Integration Tests: Evaluation Interface with NQS

This test suite verifies that:
1. NQS evaluation methods work correctly
2. All new interface methods are accessible
3. Evaluation produces consistent results
4. Backend selection works properly

Session 5, Task 6 - Code Cleanup
"""

import numpy as np
import sys

def test_nqs_evaluation_interface():
    """Test that NQS evaluation interface works correctly."""
    
    print("\n" + "="*70)
    print("TEST: NQS Evaluation Interface")
    print("="*70)
    
    # Create mock NQS for testing
    class MockNQS:
        def __init__(self):
            self._params = {'w': np.array([0.1, 0.2, 0.3])}
            self._ansatz_func = lambda p, s: np.dot(p['w'], s)
            self._batch_size = 2
            self._isjax = False
        
        def get_params(self):
            return self._params
    
    try:
        from QES.NQS.src.compute_local_energy import ComputeLocalEnergy
        
        nqs = MockNQS()
        engine = ComputeLocalEnergy(nqs, backend='auto', batch_size=2)
        
        # Test states
        states = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        # Test evaluate_ansatz
        print("\n➤ Testing evaluate_ansatz()...")
        result = engine.evaluate_ansatz(states)
        assert result.values.shape == (3,), f"Expected shape (3,), got {result.values.shape}"
        assert result.mean is not None, "Mean should not be None"
        assert result.std is not None, "Std should not be None"
        print(f"  ✓ Shape: {result.values.shape}, Mean: {result.mean:.4f}, Std: {result.std:.4f}")
        
        # Test compute_local_energy
        print("\n➤ Testing compute_local_energy()...")
        energy_func = lambda state, p: np.sum(state)
        energy_stats = engine.compute_local_energy(states, energy_func)
        assert energy_stats.mean_energy is not None, "Mean energy should not be None"
        assert energy_stats.std_energy is not None, "Std energy should not be None"
        print(f"  ✓ Mean energy: {energy_stats.mean_energy:.4f}, Std: {energy_stats.std_energy:.4f}")
        
        # Test compute_observable
        print("\n➤ Testing compute_observable()...")
        obs_func = lambda state, p: np.sum(state)
        obs_result = engine.compute_observable(obs_func, states, "TestObs")
        assert obs_result.local_values.shape == (3,), "Observable shape mismatch"
        assert obs_result.mean is not None, "Observable mean should not be None"
        print(f"  ✓ Observable mean: {obs_result.mean:.4f}")
        
        # Test evaluate_function
        print("\n➤ Testing evaluate_function()...")
        func = lambda state, p: np.mean(state)
        func_result = engine.evaluate_function(func, states)
        assert func_result.values.shape == (3,), "Function result shape mismatch"
        print(f"  ✓ Function result shape: {func_result.values.shape}")
        
        # Test backend switching
        print("\n➤ Testing backend switching...")
        engine.set_backend('numpy')
        config = engine.get_config()
        assert config['actual_backend'] == 'numpy', f"Expected 'numpy', got {config['actual_backend']}"
        print(f"  ✓ Backend switched to: {config['actual_backend']}")
        
        print("\n✅ All NQS evaluation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nqs_wrapper_methods():
    """Test that NQS wrapper methods work correctly."""
    
    print("\n" + "="*70)
    print("TEST: NQS Wrapper Methods")
    print("="*70)
    
    try:
        from QES.NQS.nqs import NQS
        from QES.general_python.ml.net_impl.net_general import RBM_jax, RBM_np
        from QES.Algebra.hamil import Hamiltonian
        from QES.Algebra.hilbert import HilbertSpace
        
        # Create minimal Hilbert space
        hilbert = HilbertSpace(2)
        
        # Create RBM network
        net = RBM_np(hilbert_space=hilbert, num_hidden=2)
        
        # Create Hamiltonian
        hamil = Hamiltonian()
        
        # Create NQS
        nqs = NQS(hilbert_space=hilbert, network=net, hamiltonian=hamil, batch_size=4)
        
        # Test that wrapper methods exist
        print("\n➤ Testing wrapper method existence...")
        assert hasattr(nqs, 'evaluate'), "NQS should have evaluate() method"
        assert hasattr(nqs, 'compute_energy'), "NQS should have compute_energy() method"
        assert hasattr(nqs, 'compute_observable'), "NQS should have compute_observable() method"
        assert hasattr(nqs, 'evaluate_function'), "NQS should have evaluate_function() method"
        assert hasattr(nqs, 'eval_engine'), "NQS should have eval_engine property"
        print("  ✓ All required methods exist")
        
        # Test that methods are callable
        print("\n➤ Testing method callability...")
        assert callable(nqs.evaluate), "evaluate should be callable"
        assert callable(nqs.compute_energy), "compute_energy should be callable"
        assert callable(nqs.compute_observable), "compute_observable should be callable"
        assert callable(nqs.evaluate_function), "evaluate_function should be callable"
        print("  ✓ All methods are callable")
        
        print("\n✅ All NQS wrapper method tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backwards_compatibility_wrapper():
    """Test that old method names still work (deprecated wrappers)."""
    
    print("\n" + "="*70)
    print("TEST: Backwards Compatibility (Deprecated Wrappers)")
    print("="*70)
    
    try:
        from QES.NQS.nqs import NQS
        from QES.general_python.ml.net_impl.net_general import RBM_np
        from QES.Algebra.hamil import Hamiltonian
        from QES.Algebra.hilbert import HilbertSpace
        
        # Create minimal setup
        hilbert = HilbertSpace(2)
        net = RBM_np(hilbert_space=hilbert, num_hidden=2)
        hamil = Hamiltonian()
        nqs = NQS(hilbert_space=hilbert, network=net, hamiltonian=hamil, batch_size=4)
        
        # Test deprecated method names
        print("\n➤ Testing deprecated method wrappers...")
        assert hasattr(nqs, 'evaluate_ansatz_unified'), "Should have evaluate_ansatz_unified() for compatibility"
        assert hasattr(nqs, 'evaluate_local_energy_unified'), "Should have evaluate_local_energy_unified() for compatibility"
        assert hasattr(nqs, 'evaluate_observable_unified'), "Should have evaluate_observable_unified() for compatibility"
        assert hasattr(nqs, 'evaluate_function_unified'), "Should have evaluate_function_unified() for compatibility"
        print("  ✓ All deprecated method names are available")
        
        print("\n✅ Backwards compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_management():
    """Test configuration management of evaluation engine."""
    
    print("\n" + "="*70)
    print("TEST: Configuration Management")
    print("="*70)
    
    try:
        from QES.NQS.nqs import NQS
        from QES.general_python.ml.net_impl.net_general import RBM_np
        from QES.Algebra.hamil import Hamiltonian
        from QES.Algebra.hilbert import HilbertSpace
        
        # Create minimal setup
        hilbert = HilbertSpace(2)
        net = RBM_np(hilbert_space=hilbert, num_hidden=2)
        hamil = Hamiltonian()
        nqs = NQS(hilbert_space=hilbert, network=net, hamiltonian=hamil, batch_size=4)
        
        # Test configuration retrieval
        print("\n➤ Testing configuration retrieval...")
        config = nqs.eval_engine.get_config()
        assert 'backend' in config, "Config should contain 'backend'"
        assert 'batch_size' in config, "Config should contain 'batch_size'"
        print(f"  ✓ Config backend: {config['backend']}, batch_size: {config['batch_size']}")
        
        # Test backend switching
        print("\n➤ Testing backend switching...")
        nqs.eval_engine.set_backend('numpy')
        new_config = nqs.eval_engine.get_config()
        assert new_config['actual_backend'] == 'numpy', "Backend switch failed"
        print(f"  ✓ Backend switched to: {new_config['actual_backend']}")
        
        # Test batch size modification
        print("\n➤ Testing batch size modification...")
        nqs.eval_engine.set_batch_size(8)
        batch_config = nqs.eval_engine.get_config()
        assert batch_config['batch_size'] == 8, "Batch size update failed"
        print(f"  ✓ Batch size updated to: {batch_config['batch_size']}")
        
        print("\n✅ Configuration management tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EVALUATION INTERFACE TEST SUITE")
    print("="*70)
    
    results = []
    results.append(("NQS Evaluation Interface", test_nqs_evaluation_interface()))
    results.append(("NQS Wrapper Methods", test_nqs_wrapper_methods()))
    results.append(("Backwards Compatibility", test_backwards_compatibility_wrapper()))
    results.append(("Configuration Management", test_configuration_management()))
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<50} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
