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
__all__ = ["__version__"]

# Optional imports - may fail when imported as a package from subdirectory
try:
    from ir import SemanticIR, IRStep
    from executor import execute
    from trace import TraceNode, TraceLog
    from graph import CausalGraph
    
    __all__.extend([
        "SemanticIR",
        "IRStep",
        "execute",
        "TraceNode",
        "TraceLog",
        "CausalGraph",
    ])
except ImportError:
    pass  # Submodule-only import

# Optional: import if available
try:
    from executor import ExecutionError, SUPPORTED_ACTIONS
    __all__.extend(["ExecutionError", "SUPPORTED_ACTIONS"])
except ImportError:
    pass
