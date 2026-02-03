# Ntive — Causal Execution Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Ntive** is a deterministic execution engine with full causal traceability. Every decision has an explicit origin in the Semantic IR — no implicit or "magic" behaviors.

## Why Ntive?

| Problem | Ntive Solution |
|---------|----------------|
| AI agents are opaque | Every action has a traceable `reason` |
| Non-reproducible results | Same IR → Same execution (deterministic) |
| Impossible to audit | Immutable JSONL trace log |
| Compliance (EU AI Act) | Full causal chain for any decision |
| Debugging is hard | Navigate causal graph to find root cause |

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ntive.git
cd ntive
pip install -e .
```

### Basic Usage

```python
from ntive import SemanticIR, IRStep, execute, TraceLog, CausalGraph

# 1. Define your Semantic IR
ir = SemanticIR(
    goal="Process user order",
    context={"user_id": "u123", "amount": 99.99},
    steps=[
        IRStep(action="set", params={"key": "user_id"}),
        IRStep(action="validate", params={"key": "amount"}),
        IRStep(action="emit", params={"target": "order_created"})
    ]
)

# 2. Execute with trace logging
trace = TraceLog("trace.jsonl")
result = execute(ir, trace)

print(result)
# {'goal': 'Process user order', 'result': {'user_id': 'u123', 'validated': 'amount', 'exists': True, 'emitted': 'order_created'}}

# 3. Analyze causal chain
graph = CausalGraph.from_jsonl("trace.jsonl")
print(graph.summary())
# {'total_nodes': 3, 'root': '...', 'max_depth': 2}

# Trace cause of any node
last_node = list(graph.nodes.keys())[-1]
causal_chain = graph.trace_cause(last_node)
for node in causal_chain:
    print(f"  {node['action']} <- {node['reason']}")
```

## Core Concepts

### Semantic IR

The **Semantic IR** is a structured representation of intent:

```python
@dataclass
class SemanticIR:
    goal: str                    # Human-readable intent
    context: Dict[str, Any]      # Domain data for execution
    steps: List[IRStep]          # Ordered actions to execute

@dataclass
class IRStep:
    action: str                  # Action name: "set", "validate", "emit"
    params: Dict[str, Any]       # Action parameters
```

### Trace Nodes

Every execution step produces a **TraceNode** with causal linkage:

```python
@dataclass
class TraceNode:
    id: str                      # Unique identifier
    parent_id: Optional[str]     # Link to previous node
    depth: int                   # Distance from root
    action: str                  # Action executed
    input: dict                  # Input parameters
    output: dict                 # Execution result
    reason: dict                 # Causal reason (type, source, ref, value)
    timestamp: float             # Execution time
```

### Causal Graph

The **CausalGraph** enables traversal and analysis:

```python
graph = CausalGraph.from_jsonl("trace.jsonl")

# Navigation
graph.get_root()                 # Root node ID
graph.get_ancestors(node_id)     # All ancestors to root
graph.get_descendants(node_id)   # All descendants

# Causal analysis
graph.trace_cause(node_id)       # Full causal chain as list of nodes
```

## Available Actions

| Action | Description | Params |
|--------|-------------|--------|
| `set` | Get value from context | `key`: context key |
| `validate` | Check if key exists | `key`: context key |
| `emit` | Emit an event/result | `target`: event name |

## Key Properties

### Determinism

Same IR always produces the same execution trace:

```python
ir1 = SemanticIR(goal="test", context={"x": 1}, steps=[...])
ir2 = SemanticIR(goal="test", context={"x": 1}, steps=[...])

# Execute both
result1 = execute(ir1, TraceLog("t1.jsonl"))
result2 = execute(ir2, TraceLog("t2.jsonl"))

assert result1 == result2  # Always true
```

### Prompt Independence

Different prompts that produce the same IR yield identical executions:

```python
# These prompts compress to the same IR:
prompt_a = "fetch user data for user 123"
prompt_b = "get the user info, id=123"

# Same IR → Same execution → Same trace structure
```

### Explicit Causality

Every `reason` field traces back to the IR:

```python
{
    "type": "constraint",      # Always "constraint" (no inference)
    "source": "ir",            # Always "ir" (no implicit sources)
    "ref": "context.user_id",  # Explicit reference
    "value": "u123"            # Actual value used
}
```

## Project Structure

```
ntive/
├── ir.py           # SemanticIR and IRStep definitions
├── executor.py     # Deterministic execution engine
├── trace.py        # TraceNode and TraceLog (JSONL)
├── graph.py        # CausalGraph for traversal/analysis
├── demo.py         # Example usage
└── test_trace.py   # Comprehensive test suite (800+ lines)
```

## Running Tests

```bash
pytest test_trace.py -v
```

Tests verify:
- ✅ Deterministic execution
- ✅ Valid causal chains
- ✅ All nodes have explicit reasons
- ✅ IR independence from prompts
- ✅ Edge cases (empty context, many steps)

## Use Cases

- **Regulated Industries**: Finance, healthcare, legal — where audit trails are mandatory
- **AI Agent Debugging**: Trace why an agent made a specific decision
- **Prompt Optimization**: Compare different prompts by their IR efficiency
- **Compliance**: EU AI Act, SOC2, HIPAA requirements for explainability

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Ntive**: *Every decision has a why.*
