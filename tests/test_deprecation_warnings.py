"""
Test deprecation warnings for legacy parameters

This test ensures that deprecation warnings are properly raised when
using legacy parameters that will be removed in v1.0.
"""

import warnings
import pytest
from stockbench.core import PipelineContext
from stockbench.agents import decide_batch_dual_agent


def test_previous_decisions_deprecation_warning():
    """Test that using previous_decisions parameter raises DeprecationWarning"""
    
    features_list = [{"symbol": "AAPL", "features": {}}]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call with deprecated parameter
        result = decide_batch_dual_agent(
            features_list,
            enable_llm=False,  # Disable LLM to avoid API calls
            previous_decisions={"AAPL": {"action": "hold"}}
        )
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "previous_decisions" in str(w[0].message)
        assert "v1.0" in str(w[0].message)


def test_decision_history_deprecation_warning():
    """Test that using decision_history parameter raises DeprecationWarning"""
    
    features_list = [{"symbol": "AAPL", "features": {}}]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call with deprecated parameter
        result = decide_batch_dual_agent(
            features_list,
            enable_llm=False,
            decision_history={"AAPL": [{"date": "2024-01-01", "action": "hold"}]}
        )
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "decision_history" in str(w[0].message)
        assert "v1.0" in str(w[0].message)


def test_dict_ctx_deprecation_warning():
    """Test that using Dict ctx raises DeprecationWarning"""
    
    features_list = [{"symbol": "AAPL", "features": {}}]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call with deprecated Dict ctx
        result = decide_batch_dual_agent(
            features_list,
            enable_llm=False,
            ctx={"portfolio": {}, "config": {}}  # Old Dict style
        )
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        
        # Find the Dict ctx warning
        dict_ctx_warning = None
        for warning in deprecation_warnings:
            if "Dict as 'ctx'" in str(warning.message):
                dict_ctx_warning = warning
                break
        
        assert dict_ctx_warning is not None
        assert "PipelineContext" in str(dict_ctx_warning.message)


def test_no_warning_with_new_api():
    """Test that using new API does not raise deprecation warnings"""
    
    features_list = [{"symbol": "AAPL", "features": {}}]
    
    # Create proper PipelineContext
    ctx = PipelineContext(
        config={},
        run_id="test_run",
        date="2024-01-01",
        llm_client=None,  # Not needed for this test
        llm_config=None   # Not needed for this test
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call with new API (no deprecated parameters)
        result = decide_batch_dual_agent(
            features_list,
            enable_llm=False,
            ctx=ctx
        )
        
        # Check that no deprecation warnings were raised
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        legacy_param_warnings = [
            warning for warning in deprecation_warnings 
            if any(param in str(warning.message) 
                   for param in ["previous_decisions", "decision_history", "Dict as 'ctx'"])
        ]
        
        assert len(legacy_param_warnings) == 0, \
            f"Expected no legacy parameter warnings, but got: {[str(w.message) for w in legacy_param_warnings]}"


def test_memory_system_used_instead_of_parameters():
    """Test that memory system is used instead of legacy parameters"""
    
    features_list = [{"symbol": "AAPL", "features": {}}]
    
    # Create PipelineContext with memory enabled
    ctx = PipelineContext(
        config={"memory": {"enabled": True}},
        run_id="test_run",
        date="2024-01-01",
        llm_client=None,  # Not needed for this test
        llm_config=None   # Not needed for this test
    )
    
    # Call without deprecated parameters
    result = decide_batch_dual_agent(
        features_list,
        enable_llm=False,
        ctx=ctx
    )
    
    # Result should be valid (memory system handles history internally)
    assert result is not None
    assert isinstance(result, dict)


if __name__ == "__main__":
    # Run tests manually
    print("Testing deprecation warnings...")
    
    print("\n1. Testing previous_decisions warning...")
    test_previous_decisions_deprecation_warning()
    print("✅ PASSED")
    
    print("\n2. Testing decision_history warning...")
    test_decision_history_deprecation_warning()
    print("✅ PASSED")
    
    print("\n3. Testing Dict ctx warning...")
    test_dict_ctx_deprecation_warning()
    print("✅ PASSED")
    
    print("\n4. Testing new API (no warnings)...")
    test_no_warning_with_new_api()
    print("✅ PASSED")
    
    print("\n5. Testing memory system usage...")
    test_memory_system_used_instead_of_parameters()
    print("✅ PASSED")
    
    print("\n" + "="*60)
    print("All deprecation tests passed! ✅")
    print("="*60)
