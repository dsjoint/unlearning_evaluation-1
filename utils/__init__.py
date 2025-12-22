# Utils module for unlearning evaluation
from utils.attention_backend import (
    is_flash_attn_available,
    get_attn_implementation,
    ATTN_BACKEND_AUTO,
    ATTN_BACKEND_FLASH,
    ATTN_BACKEND_SDPA,
    ATTN_BACKEND_EAGER,
)

__all__ = [
    "is_flash_attn_available",
    "get_attn_implementation",
    "ATTN_BACKEND_AUTO",
    "ATTN_BACKEND_FLASH",
    "ATTN_BACKEND_SDPA",
    "ATTN_BACKEND_EAGER",
]

