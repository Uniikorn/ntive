"""
Ntive — Causal Execution Engine

A deterministic execution engine with full causal traceability.
Every decision has an explicit origin in the Semantic IR.
"""

from ir import SemanticIR, IRStep
from executor import execute
from trace import TraceNode, TraceLog
from graph import CausalGraph

__version__ = "0.1.0"
__all__ = [
    "SemanticIR",
    "IRStep",
    "execute",
    "TraceNode",
    "TraceLog",
    "CausalGraph",
]
