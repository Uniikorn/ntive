"""
graph.py

Minimal causal graph layer for the Decision Trace Engine.
Loads TraceNodes from trace.jsonl and builds traversable causal relationships.

Graph representation:
  children: dict[node_id] -> list[child_node_ids]  (forward traversal)
  parents:  dict[node_id] -> parent_node_id        (backward traversal)
"""

import json
from typing import Dict, List, Optional, Any


class CausalGraph:
    """A minimal causal graph built from TraceNode parent_id relationships."""

    def __init__(self):
        self.nodes: Dict[str, dict] = {}           # node_id -> full node data
        self.children: Dict[str, List[str]] = {}   # node_id -> list of child ids
        self.parents: Dict[str, Optional[str]] = {} # node_id -> parent id

    @classmethod
    def from_jsonl(cls, path: str) -> "CausalGraph":
        """Load a causal graph from a trace.jsonl file."""
        graph = cls()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    node = json.loads(line)
                    graph._add_node(node)
        return graph

    def _add_node(self, node: dict) -> None:
        """Add a node and update parent/child relationships."""
        node_id = node["id"]
        parent_id = node.get("parent_id")

        self.nodes[node_id] = node
        self.parents[node_id] = parent_id
        self.children.setdefault(node_id, [])

        if parent_id is not None:
            self.children.setdefault(parent_id, [])
            self.children[parent_id].append(node_id)

    # --- Traversal methods ---

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get full node data by id."""
        return self.nodes.get(node_id)

    def get_parent(self, node_id: str) -> Optional[str]:
        """Get parent node id."""
        return self.parents.get(node_id)

    def get_children(self, node_id: str) -> List[str]:
        """Get list of child node ids."""
        return self.children.get(node_id, [])

    def get_root(self) -> Optional[str]:
        """Get the root node id (node with no parent)."""
        for node_id, parent_id in self.parents.items():
            if parent_id is None:
                return node_id
        return None

    def get_ancestors(self, node_id: str) -> List[str]:
        """Get all ancestors from node to root (excluding self)."""
        ancestors = []
        current = self.get_parent(node_id)
        while current is not None:
            ancestors.append(current)
            current = self.get_parent(current)
        return ancestors

    def get_descendants(self, node_id: str) -> List[str]:
        """Get all descendants (excluding self), breadth-first."""
        descendants = []
        queue = self.get_children(node_id)[:]
        while queue:
            current = queue.pop(0)
            descendants.append(current)
            queue.extend(self.get_children(current))
        return descendants

    def trace_cause(self, node_id: str) -> List[dict]:
        """Trace causal chain from node back to root. Returns list of nodes."""
        chain = []
        current = node_id
        while current is not None:
            node = self.get_node(current)
            if node:
                chain.append(node)
            current = self.get_parent(current)
        return chain

    # --- Summary ---

    def summary(self) -> dict:
        """Return a summary of the graph."""
        return {
            "total_nodes": len(self.nodes),
            "root": self.get_root(),
            "max_depth": self._max_depth()
        }

    def _max_depth(self) -> int:
        """Calculate max depth of the graph."""
        max_d = 0
        for node_id in self.nodes:
            depth = len(self.get_ancestors(node_id))
            if depth > max_d:
                max_d = depth
        return max_d


# --- CLI for quick inspection ---

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl"
    
    graph = CausalGraph.from_jsonl(path)
    print(f"Graph summary: {graph.summary()}")
    
    root = graph.get_root()
    if root:
        print(f"\nRoot node: {root}")
        print(f"Root action: {graph.get_node(root)['action']}")
        print(f"Descendants: {len(graph.get_descendants(root))}")
