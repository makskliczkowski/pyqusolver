"""
Quick Smoke Test: Evaluation Interface Method Names

Verifies that the new evaluation interface methods exist and are callable.

Session 5, Task 6 - Code Cleanup
"""

import numpy as np
import sys

def test_method_existence():
    """Test that all evaluation methods exist and are callable."""
    
    print("\n" + "="*70)
    print("TEST: Evaluation Method Existence and Callability")
    print("="*70)
    
    try:
        from QES.NQS.nqs import NQS
        
        # Check that NQS class has all the required methods
        print("\n➤ Checking NQS class for required methods...")
        
        required_methods = {
            'evaluate': 'Core ansatz evaluation method',
            'ansatz': 'Backwards compatible wrapper for evaluate()',
            'compute_energy': 'Compute local energy values',
            'compute_observable': 'Compute observable values',
            'evaluate_function': 'General function evaluation',
        }
        
        required_properties = {
            'eval_engine': 'Access to evaluation engine',
        }
        
        # Check methods
        all_present = True
        for method_name, description in required_methods.items():
            if hasattr(NQS, method_name):
                method = getattr(NQS, method_name)
                if callable(method) or isinstance(method, property):
                    print(f"  ✓ {method_name:.<30} ({description})")
                else:
                    print(f"  ❌ {method_name:.<30} exists but not callable")
                    all_present = False
            else:
                print(f"  ❌ {method_name:.<30} NOT FOUND")
                all_present = False
        
        # Check properties
        for prop_name, description in required_properties.items():
            if hasattr(NQS, prop_name):
                prop = getattr(NQS, prop_name)
                if isinstance(prop, property):
                    print(f"  ✓ {prop_name:.<30} ({description})")
                else:
                    print(f"  ❌ {prop_name:.<30} exists but not a property")
                    all_present = False
            else:
                print(f"  ❌ {prop_name:.<30} NOT FOUND")
                all_present = False
        
        if not all_present:
            raise AssertionError("Some required methods/properties are missing")
        
        # Check deprecated methods exist for backwards compatibility
        print("\n➤ Checking deprecated method wrappers...")
        deprecated_methods = {
            'evaluate_ansatz_unified': 'Wrapper for evaluate()',
            'evaluate_local_energy_unified': 'Wrapper for compute_energy()',
            'evaluate_observable_unified': 'Wrapper for compute_observable()',
            'evaluate_function_unified': 'Wrapper for evaluate_function()',
        }
        
        for method_name, description in deprecated_methods.items():
            if hasattr(NQS, method_name):
                method = getattr(NQS, method_name)
                if callable(method) or isinstance(method, property):
                    print(f"  ✓ {method_name:.<30} ({description})")
                else:
                    print(f"  ❌ {method_name:.<30} exists but not callable")
                    all_present = False
            else:
                print(f"  ❌ {method_name:.<30} NOT FOUND")
                all_present = False
        
        if all_present:
            print("\n✅ All required methods and properties are present!")
            return True
        else:
            raise AssertionError("Some required methods are missing")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports_work():
    """Test that all imports work correctly."""
    
    print("\n" + "="*70)
    print("TEST: Import Verification")
    print("="*70)
    
    try:
        print("\n➤ Testing core imports...")
        
        from QES.NQS.nqs import NQS
        print("  ✓ NQS imported successfully")
        
        from QES.NQS.src.compute_local_energy import ComputeLocalEnergy
        print("  ✓ ComputeLocalEnergy imported successfully")
        
        from QES.NQS.src.unified_evaluation_engine import UnifiedEvaluationEngine
        print("  ✓ UnifiedEvaluationEngine imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_signatures():
    """Test that method signatures are correct."""
    
    print("\n" + "="*70)
    print("TEST: Method Signatures")
    print("="*70)
    
    try:
        from QES.NQS.nqs import NQS
        import inspect
        
        print("\n➤ Checking method signatures...")
        
        # Check evaluate
        sig = inspect.signature(NQS.evaluate)
        params = list(sig.parameters.keys())
        expected = ['self', 'states', 'batch_size', 'params']
        assert params == expected, f"evaluate() signature mismatch. Expected {expected}, got {params}"
        print(f"  ✓ evaluate{sig}")
        
        # Check ansatz
        sig = inspect.signature(NQS.ansatz)
        params = list(sig.parameters.keys())
        expected = ['self', 'states', 'batch_size', 'params']
        assert params == expected, f"ansatz() signature mismatch. Expected {expected}, got {params}"
        print(f"  ✓ ansatz{sig}")
        
        # Check compute_energy
        sig = inspect.signature(NQS.compute_energy)
        params = list(sig.parameters.keys())
        expected = ['self', 'states', 'ham_action_func', 'params', 'probabilities', 'batch_size']
        assert params == expected, f"compute_energy() signature mismatch. Expected {expected}, got {params}"
        print(f"  ✓ compute_energy{sig}")
        
        # Check compute_observable
        sig = inspect.signature(NQS.compute_observable)
        params = list(sig.parameters.keys())
        assert 'observable_func' in params and 'states' in params, "compute_observable() missing required params"
        print(f"  ✓ compute_observable{sig}")
        
        # Check evaluate_function
        sig = inspect.signature(NQS.evaluate_function)
        params = list(sig.parameters.keys())
        assert 'func' in params and 'states' in params, "evaluate_function() missing required params"
        print(f"  ✓ evaluate_function{sig}")
        
        print("\n✅ All method signatures are correct!")
        return True
        
    except Exception as e:
        print(f"\n❌ Signature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_old_methods_removed():
    """Test that old internal methods have been removed."""
    
    print("\n" + "="*70)
    print("TEST: Cleanup Verification (Old Methods Removed)")
    print("="*70)
    
    try:
        from QES.NQS.nqs import NQS
        
        print("\n➤ Checking that old methods are removed...")
        
        # These old methods should NOT exist anymore (they were private/internal)
        # But we keep ansatz() and apply() as public wrappers for backwards compatibility
        
        # Private methods that should be gone
        removed_methods = {
            '_eval_jax': 'Old JAX evaluation method',
            '_eval_np': 'Old NumPy evaluation method',
        }
        
        all_removed = True
        for method_name, description in removed_methods.items():
            if not hasattr(NQS, method_name):
                print(f"  ✓ {method_name:.<30} removed ({description})")
            else:
                print(f"  ❌ {method_name:.<30} still exists ({description})")
                all_removed = False
        
        # Public methods that should still exist for compatibility
        print("\n➤ Checking that public methods still work...")
        public_methods = {
            'ansatz': 'Public backwards-compatible wrapper',
            'apply': 'Public apply method',
        }
        
        for method_name, description in public_methods.items():
            if hasattr(NQS, method_name):
                print(f"  ✓ {method_name:.<30} exists ({description})")
            else:
                print(f"  ❌ {method_name:.<30} missing ({description})")
                all_removed = False
        
        if all_removed:
            print("\n✅ Cleanup successful - old methods removed, public API preserved!")
            return True
        else:
            raise AssertionError("Cleanup incomplete")
        
    except Exception as e:
        print(f"\n❌ Cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EVALUATION INTERFACE CLEANUP - SMOKE TEST SUITE")
    print("="*70)
    
    results = []
    results.append(("Import Verification", test_imports_work()))
    results.append(("Method Existence", test_method_existence()))
    results.append(("Method Signatures", test_method_signatures()))
    results.append(("Cleanup Verification", test_old_methods_removed()))
    
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
        print("\n🎉 All smoke tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
