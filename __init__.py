"""
Ntive — Causal Execution Engine

A deterministic execution engine with full causal traceability.
Every decision has an explicit origin in the Semantic IR.

Usage:
    from ntive import SemanticIR, IRStep, execute, TraceLog, CausalGraph
    
    ir = SemanticIR(
        goal="Process order",
        context={"user_id": "123"},
        steps=[IRStep(action="set", params={"key": "user_id"})]
    )
    
    trace = TraceLog("trace.jsonl")
    result = execute(ir, trace)
    
    graph = CausalGraph.from_jsonl("trace.jsonl")
    print(graph.trace_cause(list(graph.nodes.keys())[-1]))
"""

__version__ = "0.1.0"

from ir import SemanticIR, IRStep
from executor import execute
from trace import TraceNode, TraceLog
from graph import CausalGraph

# Optional: import if available
try:
    from executor import ExecutionError, SUPPORTED_ACTIONS
except ImportError:
    ExecutionError = Exception
    SUPPORTED_ACTIONS = ["set", "validate", "emit"]

__all__ = [
    # Core IR
    "SemanticIR",
    "IRStep",
    # Execution
    "execute",
    "ExecutionError",
    "SUPPORTED_ACTIONS",
    # Tracing
    "TraceNode",
    "TraceLog",
    # Graph analysis
    "CausalGraph",
    # Version
    "__version__",
]
