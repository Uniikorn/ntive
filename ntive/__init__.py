"""
Ntive — Causal Execution Engine

A deterministic execution engine with full causal traceability.
Every decision has an explicit origin in the Semantic IR.
"""

__version__ = "0.1.0"
__all__ = []

# Optional imports from root-level modules (may not be available when
# importing submodules like ntive.parser or ntive.runtime directly)
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
