"""Metrics utilities for unlearning evaluation.

Provides helpers for converting epoch-keyed accuracy dicts to scalar values.
"""

from typing import Union
import math

# Default accuracy selection rule - can be overridden via Hydra config
ACC_SELECTION_RULE = "final_epoch"


def select_scalar_acc(
    acc_dict: dict,
    rule: str = "final_epoch"
) -> float:
    """Convert {epoch: acc} dict to a single scalar accuracy value.
    
    Args:
        acc_dict: Dict mapping epoch (int or str) to accuracy float.
                  Example: {0: 0.25, 1: 0.35, 2: 0.42}
        rule: Selection rule. One of:
              - "final_epoch": Return accuracy at the last (highest) epoch
              - "max_epoch": Return maximum accuracy across all epochs
    
    Returns:
        Single float accuracy value, or float('nan') if dict is empty/invalid.
    
    Raises:
        ValueError: If rule is not recognized.
    
    Examples:
        >>> select_scalar_acc({0: 0.25, 1: 0.35, 2: 0.42}, "final_epoch")
        0.42
        >>> select_scalar_acc({0: 0.50, 1: 0.35, 2: 0.42}, "max_epoch")
        0.50
    """
    if not acc_dict:
        return float('nan')
    
    try:
        # Sort epochs numerically (handles both int and str keys)
        epochs = sorted(acc_dict.keys(), key=lambda x: int(x))
        
        if rule == "final_epoch":
            return float(acc_dict[epochs[-1]])
        elif rule == "max_epoch":
            return float(max(acc_dict.values()))
        else:
            raise ValueError(f"Unknown accuracy selection rule: {rule}. "
                           f"Expected 'final_epoch' or 'max_epoch'.")
    except (TypeError, ValueError) as e:
        # Handle malformed acc_dict gracefully
        return float('nan')


def avg_scalar_acc(
    acc_dicts: list[dict],
    rule: str = "final_epoch"
) -> float:
    """Average scalar accuracy across multiple acc_dicts.
    
    Args:
        acc_dicts: List of {epoch: acc} dicts
        rule: Selection rule passed to select_scalar_acc
    
    Returns:
        Mean of scalar accuracies, ignoring NaN values.
        Returns NaN if all values are NaN or list is empty.
    """
    if not acc_dicts:
        return float('nan')
    
    scalars = [select_scalar_acc(d, rule) for d in acc_dicts]
    valid = [s for s in scalars if not math.isnan(s)]
    
    if not valid:
        return float('nan')
    
    return sum(valid) / len(valid)

