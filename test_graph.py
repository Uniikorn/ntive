"""Quick test of the causal graph."""
from graph import CausalGraph

g = CausalGraph.from_jsonl("trace.jsonl")

# Get last node
last = list(g.nodes.keys())[-1]
print(f"Last node: {last}")

# Trace cause back to root
chain = g.trace_cause(last)
print(f"\nCausal chain ({len(chain)} nodes):")
for node in chain:
    action = node.get("action")
    ref = node.get("reason", {}).get("ref", "N/A")
    print(f"  {action} <- {ref}")

# Forward traversal from root
root = g.get_root()
print(f"\nRoot: {root}")
print(f"Children of root: {g.get_children(root)}")
print(f"All descendants: {len(g.get_descendants(root))}")
