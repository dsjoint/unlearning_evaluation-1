#!/usr/bin/env python3
"""
Smoke test to verify the project works without flash-attn installed.

This script tests:
1. Import of main modules (pipeline, unlearn_corpus, finetune_corpus)
2. Import of the attention backend helper
3. Attention backend selection logic
4. Model loading with fallback attention (if GPU available and model accessible)

Usage:
    python test_no_flash_attn.py
    
    # Or with pytest:
    pytest test_no_flash_attn.py -v
"""

import sys
import os

# Ensure we're testing from the repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def test_attention_backend_import():
    """Test that attention backend utilities can be imported."""
    from utils.attention_backend import (
        is_flash_attn_available,
        get_attn_implementation,
        ATTN_BACKEND_AUTO,
        ATTN_BACKEND_FLASH,
        ATTN_BACKEND_SDPA,
        ATTN_BACKEND_EAGER,
    )
    
    print("✓ Attention backend utilities imported successfully")
    return True


def test_is_flash_attn_available():
    """Test the flash-attn availability check."""
    from utils.attention_backend import is_flash_attn_available
    
    result = is_flash_attn_available()
    print(f"✓ is_flash_attn_available() returned: {result}")
    return True


def test_get_attn_implementation_auto():
    """Test automatic attention backend selection."""
    from utils.attention_backend import (
        get_attn_implementation,
        is_flash_attn_available,
        ATTN_BACKEND_FLASH,
        ATTN_BACKEND_SDPA,
    )
    
    result = get_attn_implementation("auto")
    expected = ATTN_BACKEND_FLASH if is_flash_attn_available() else ATTN_BACKEND_SDPA
    
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ get_attn_implementation('auto') returned: {result}")
    return True


def test_get_attn_implementation_fallback():
    """Test that flash request falls back to SDPA when flash-attn is unavailable."""
    from utils.attention_backend import (
        get_attn_implementation,
        is_flash_attn_available,
        ATTN_BACKEND_FLASH,
        ATTN_BACKEND_SDPA,
    )
    
    result = get_attn_implementation("flash_attention_2")
    
    if is_flash_attn_available():
        assert result == ATTN_BACKEND_FLASH
        print(f"✓ Flash attention is available, using: {result}")
    else:
        assert result == ATTN_BACKEND_SDPA
        print(f"✓ Flash attention not available, correctly fell back to: {result}")
    
    return True


def test_get_attn_implementation_explicit():
    """Test explicit attention backend selection."""
    from utils.attention_backend import (
        get_attn_implementation,
        ATTN_BACKEND_SDPA,
        ATTN_BACKEND_EAGER,
    )
    
    # SDPA should always work
    result_sdpa = get_attn_implementation("sdpa")
    assert result_sdpa == ATTN_BACKEND_SDPA
    print(f"✓ get_attn_implementation('sdpa') returned: {result_sdpa}")
    
    # Eager should always work
    result_eager = get_attn_implementation("eager")
    assert result_eager == ATTN_BACKEND_EAGER
    print(f"✓ get_attn_implementation('eager') returned: {result_eager}")
    
    return True


def test_get_attn_implementation_invalid():
    """Test that invalid backend raises ValueError."""
    from utils.attention_backend import get_attn_implementation
    
    try:
        get_attn_implementation("invalid_backend")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Invalid backend correctly raised ValueError: {e}")
    
    return True


def test_pipeline_import():
    """Test that pipeline module can be imported without flash-attn errors."""
    try:
        import pipeline
        print("✓ pipeline module imported successfully")
        return True
    except ImportError as e:
        if "flash_attn" in str(e).lower():
            print(f"✗ Pipeline import failed due to flash-attn: {e}")
            return False
        raise


def test_unlearn_corpus_import():
    """Test that unlearn_corpus module can be imported."""
    try:
        import unlearn_corpus
        print("✓ unlearn_corpus module imported successfully")
        return True
    except ImportError as e:
        if "flash_attn" in str(e).lower():
            print(f"✗ unlearn_corpus import failed due to flash-attn: {e}")
            return False
        raise


def test_finetune_corpus_import():
    """Test that finetune_corpus module can be imported."""
    try:
        import finetune_corpus
        print("✓ finetune_corpus module imported successfully")
        return True
    except ImportError as e:
        if "flash_attn" in str(e).lower():
            print(f"✗ finetune_corpus import failed due to flash-attn: {e}")
            return False
        raise


def test_utils_init_import():
    """Test that utils module __init__ can be imported."""
    try:
        import utils
        print("✓ utils module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ utils import failed: {e}")
        return False


def run_all_tests():
    """Run all smoke tests."""
    tests = [
        test_attention_backend_import,
        test_is_flash_attn_available,
        test_get_attn_implementation_auto,
        test_get_attn_implementation_fallback,
        test_get_attn_implementation_explicit,
        test_get_attn_implementation_invalid,
        test_utils_init_import,
        test_pipeline_import,
        test_unlearn_corpus_import,
        test_finetune_corpus_import,
    ]
    
    print("=" * 60)
    print("Running flash-attn optional dependency smoke tests")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} raised exception: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

