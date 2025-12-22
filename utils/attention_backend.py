"""
Attention backend utilities for optional flash-attn support.

This module provides a centralized way to select the attention implementation
for HuggingFace models. It supports automatic fallback when flash-attn is not
available.

Usage:
    from utils.attention_backend import get_attn_implementation
    
    attn_impl = get_attn_implementation(preference="auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        attn_implementation=attn_impl,
        ...
    )
"""

import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Backend constants
ATTN_BACKEND_AUTO = "auto"
ATTN_BACKEND_FLASH = "flash_attention_2"
ATTN_BACKEND_SDPA = "sdpa"
ATTN_BACKEND_EAGER = "eager"

# Valid backend options
VALID_BACKENDS = {ATTN_BACKEND_AUTO, ATTN_BACKEND_FLASH, ATTN_BACKEND_SDPA, ATTN_BACKEND_EAGER}

# Cache the flash-attn availability check
_flash_attn_available: Optional[bool] = None


def is_flash_attn_available() -> bool:
    """
    Check if flash-attn is available and can be used.
    
    Returns:
        True if flash-attn is installed and importable, False otherwise.
    """
    global _flash_attn_available
    
    if _flash_attn_available is not None:
        return _flash_attn_available
    
    try:
        import flash_attn  # noqa: F401
        _flash_attn_available = True
        logger.debug("flash-attn is available")
    except ImportError:
        _flash_attn_available = False
        logger.debug("flash-attn is not available, will use fallback attention")
    
    return _flash_attn_available


def get_attn_implementation(
    preference: Optional[str] = None,
) -> str:
    """
    Get the appropriate attention implementation based on preference and availability.
    
    This function implements the following logic:
    - If preference is "auto" or None: use flash_attention_2 if available, else sdpa
    - If preference is "flash_attention_2": use flash if available, else raise warning and fall back to sdpa
    - If preference is "sdpa": always use sdpa (PyTorch scaled dot-product attention)
    - If preference is "eager": always use eager (standard PyTorch attention)
    
    Args:
        preference: One of "auto", "flash_attention_2", "sdpa", "eager", or None.
                   None is treated as "auto".
    
    Returns:
        The attention implementation string to pass to HuggingFace model loading.
        One of: "flash_attention_2", "sdpa", "eager"
    
    Raises:
        ValueError: If preference is not a valid backend option.
    """
    if preference is None:
        preference = ATTN_BACKEND_AUTO
    
    if preference not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid attention backend preference: {preference}. "
            f"Must be one of: {VALID_BACKENDS}"
        )
    
    # Handle auto mode
    if preference == ATTN_BACKEND_AUTO:
        if is_flash_attn_available():
            logger.info("Using flash_attention_2 (auto-detected)")
            return ATTN_BACKEND_FLASH
        else:
            logger.info("Using sdpa (flash-attn not available)")
            return ATTN_BACKEND_SDPA
    
    # Handle explicit flash request
    if preference == ATTN_BACKEND_FLASH:
        if is_flash_attn_available():
            logger.info("Using flash_attention_2 (explicitly requested)")
            return ATTN_BACKEND_FLASH
        else:
            logger.warning(
                "flash_attention_2 was requested but flash-attn is not available. "
                "Falling back to sdpa."
            )
            return ATTN_BACKEND_SDPA
    
    # Handle sdpa or eager (no fallback needed)
    logger.info(f"Using {preference} (explicitly requested)")
    return preference


def log_attention_backend_info():
    """Log information about the current attention backend configuration."""
    flash_available = is_flash_attn_available()
    logger.info(f"Flash attention available: {flash_available}")
    if flash_available:
        try:
            import flash_attn
            version = getattr(flash_attn, "__version__", "unknown")
            logger.info(f"flash-attn version: {version}")
        except Exception:
            pass

