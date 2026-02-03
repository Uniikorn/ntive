"""
executor.py

Pure interpreter for Semantic IR.
Emits TraceNodes ONLY for explicit IR steps.

Supported actions:
    - set: Get value from context by key
    - validate: Check if key exists in context
    - emit: Emit an event/result
    - compute: Perform basic computations
    - transform: Transform data
"""

from typing import Any, Dict, List
import uuid
import time

from ir import SemanticIR, IRStep
from trace import TraceNode, TraceLog


# Supported actions registry
SUPPORTED_ACTIONS: List[str] = ["set", "validate", "emit", "compute", "transform"]


class ExecutionError(Exception):
    """Raised when execution fails due to invalid IR or missing data."""
    pass


def execute(ir: SemanticIR, trace_log: TraceLog) -> Dict[str, Any]:
    """
    Execute the Semantic IR deterministically.
    Emits exactly one TraceNode per IR step.
    Returns the final output.
    """
    output: Dict[str, Any] = {"goal": ir.goal, "result": {}}
    parent_id: str | None = None
    depth: int = 0

    # Execute each step — no implicit nodes
    for step in ir.steps:
        step_output = _execute_step(step, ir.context)

        node = TraceNode(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            depth=depth,
            action=step.action,
            input={"params": step.params},
            output=step_output,
            reason=_build_reason(step, ir.context),
            timestamp=time.time()
        )
        trace_log.append(node)
        parent_id = node.id
        depth += 1

        # Merge step output into final result
        if step_output:
            output["result"].update(step_output)

    return output


def _execute_step(step: IRStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single step. Returns the output of that step.
    
    Args:
        step: The IRStep to execute
        context: The IR context containing data
        
    Returns:
        Dict with step output
        
    Raises:
        ExecutionError: If action is unknown or params are invalid
    """
    action = step.action
    params = step.params
    
    if action not in SUPPORTED_ACTIONS:
        raise ExecutionError(
            f"Unknown action '{action}'. Supported: {SUPPORTED_ACTIONS}"
        )
    
    if action == "set":
        key = params.get("key")
        if not key:
            raise ExecutionError("Action 'set' requires 'key' parameter")
        value = context.get(key)
        return {key: value}

    elif action == "validate":
        key = params.get("key")
        if not key:
            raise ExecutionError("Action 'validate' requires 'key' parameter")
        return {"validated": key, "exists": key in context}

    elif action == "emit":
        target = params.get("target")
        if not target:
            raise ExecutionError("Action 'emit' requires 'target' parameter")
        return {"emitted": target}
    
    elif action == "compute":
        # Basic computation: supports 'expression' on context values
        expr = params.get("expression")
        result_key = params.get("result", "computed")
        if not expr:
            raise ExecutionError("Action 'compute' requires 'expression' parameter")
        # Simple key lookup for now (can be extended)
        if expr in context:
            return {result_key: context[expr]}
        return {result_key: None}
    
    elif action == "transform":
        # Transform: apply simple transformations
        source = params.get("source")
        operation = params.get("operation", "identity")
        if not source:
            raise ExecutionError("Action 'transform' requires 'source' parameter")
        value = context.get(source)
        if operation == "upper" and isinstance(value, str):
            return {"transformed": value.upper()}
        elif operation == "lower" and isinstance(value, str):
            return {"transformed": value.lower()}
        elif operation == "identity":
            return {"transformed": value}
        return {"transformed": value}

    # This should never happen due to the check above
    raise ExecutionError(f"Unhandled action: {action}")


def _build_reason(step: IRStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured causal reason. Always references IR context."""
    key = step.params.get("key")
    
    if step.action == "set":
        return {
            "type": "constraint",
            "source": "ir",
            "ref": f"context.{key}",
            "value": context.get(key)
        }
    
    elif step.action == "validate":
        return {
            "type": "constraint",
            "source": "ir",
            "ref": f"context.{key}",
            "value": context.get(key)
        }
    
    elif step.action == "emit":
        target = step.params.get("target")
        return {
            "type": "constraint",
            "source": "ir",
            "ref": "params.target",
            "value": target
        }
    
    elif step.action == "compute":
        expr = step.params.get("expression")
        return {
            "type": "constraint",
            "source": "ir",
            "ref": f"context.{expr}",
            "value": context.get(expr) if expr else None
        }
    
    elif step.action == "transform":
        source = step.params.get("source")
        return {
            "type": "constraint",
            "source": "ir",
            "ref": f"context.{source}",
            "value": context.get(source) if source else None
        }
    
    else:
        # Fallback for any action (should not reach here if validation works)
        return {
            "type": "constraint",
            "source": "ir",
            "ref": "step.action",
            "value": step.action
        }
