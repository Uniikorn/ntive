# TraceNode dataclass and TraceLog placeholder
"""
trace.py

Defines the core trace data structures for the Decision Trace Engine (DTE).
 TraceNode: represents a single decision and its causal link.
 TraceLog: manages a collection of trace nodes.
"""

# Imports
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class TraceNode:
	"""A single decision node with explicit causal linkage.
	
	reason fields:
	  - type: constraint | inference | default | derived
	  - source: user | system | ir | executor
	  - ref: path or key referencing the causal origin
	  - value: the causal value
	"""
	id: str
	parent_id: Optional[str]
	depth: int                # hops from root (root=0)
	action: str
	input: dict
	output: dict
	reason: dict              # {type, source, ref, value}
	timestamp: float

class TraceLog:
    """Append-only trace log that writes to a JSONL file."""
    
    def __init__(self, path: str, truncate: bool = True):
        """Initialize trace log.
        
        Args:
            path: Path to the JSONL file
            truncate: If True, truncate file on init. If False, append to existing.
        """
        self.path = path
        self._nodes: list[TraceNode] = []
        
        if truncate:
            # Explicitly create/truncate the file on init.
            with open(self.path, 'w', encoding='utf-8') as f:
                pass  # Empty file, ready for append

    def append(self, node: TraceNode) -> None:
        """Append a trace node to the log."""
        self._nodes.append(node)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node.__dict__) + '\n')
    
    def read(self) -> list[TraceNode]:
        """Read all trace nodes from the file.
        
        Returns:
            List of TraceNode objects
        """
        nodes = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    nodes.append(TraceNode(**data))
        return nodes
    
    def count(self) -> int:
        """Return the number of nodes written in this session."""
        return len(self._nodes)
    
    @classmethod
    def from_file(cls, path: str) -> "TraceLog":
        """Load an existing trace log without truncating."""
        log = cls(path, truncate=False)
        log._nodes = log.read()
        return log
