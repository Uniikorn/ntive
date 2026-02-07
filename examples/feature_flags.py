"""
feature_flags.py

Adapter for Feature Flag Evaluation using the Decision Trace Engine.
Converts flag definitions + user context into SemanticIR for traceable evaluation.

Use case: Audit why a feature was enabled/disabled for a specific user.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from ir import SemanticIR, IRStep


@dataclass
class FlagRule:
    """A single rule for flag evaluation."""
    attribute: str      # user attribute to check (e.g., "plan", "country")
    operator: str       # "eq", "in", "gte", "lte"
    value: Any          # expected value or list


@dataclass
class FeatureFlag:
    """A feature flag definition."""
    name: str
    default: bool
    rules: List[FlagRule]
    rollout_percent: Optional[int] = None  # 0-100, None = no rollout


def flags_to_ir(flags: List[FeatureFlag], user_context: Dict[str, Any]) -> SemanticIR:
    """
    Convert feature flags + user context into SemanticIR.
    
    Each flag becomes a sequence of steps:
    1. set: Load relevant user attributes
    2. validate: Check if attributes exist
    3. emit: Emit the flag decision
    
    Args:
        flags: List of feature flags to evaluate
        user_context: User attributes (e.g., {"user_id": "123", "plan": "pro"})
    
    Returns:
        SemanticIR ready for execution
    """
    # Build context: user attributes + flag definitions
    context: Dict[str, Any] = {}
    
    # Add user context with "user." prefix
    for key, value in user_context.items():
        context[f"user.{key}"] = value
    
    # Add flag defaults
    for flag in flags:
        context[f"flag.{flag.name}.default"] = flag.default
        context[f"flag.{flag.name}.rollout"] = flag.rollout_percent
    
    # Precompute flag decisions based on rules
    for flag in flags:
        decision = _evaluate_flag(flag, user_context)
        context[f"flag.{flag.name}.enabled"] = decision
    
    # Build steps
    steps: List[IRStep] = []
    
    for flag in flags:
        # Set: load the computed decision
        steps.append(IRStep(
            action="set",
            params={"key": f"flag.{flag.name}.enabled"}
        ))
        
        # Emit: output the flag result
        steps.append(IRStep(
            action="emit",
            params={"target": f"flag.{flag.name}"}
        ))
    
    return SemanticIR(
        goal=f"Evaluate {len(flags)} feature flags for user",
        context=context,
        steps=steps
    )


def _evaluate_flag(flag: FeatureFlag, user_context: Dict[str, Any]) -> bool:
    """Evaluate a single flag against user context."""
    # Check rules in order
    for rule in flag.rules:
        attr_value = user_context.get(rule.attribute)
        
        if attr_value is None:
            continue  # Skip if attribute missing
        
        match = _check_rule(attr_value, rule.operator, rule.value)
        if match:
            return True
    
    # No rules matched, use default
    return flag.default


def _check_rule(attr_value: Any, operator: str, expected: Any) -> bool:
    """Check if a rule matches."""
    if operator == "eq":
        return attr_value == expected
    elif operator == "in":
        return attr_value in expected
    elif operator == "gte":
        return attr_value >= expected
    elif operator == "lte":
        return attr_value <= expected
    return False
