"""
ir.py

Defines the minimal Semantic IR schema for the Decision Trace Engine.
- SemanticIR: the top-level structured intent
- IRStep: a single action within the IR

Design assumptions:
- 'goal' is a human-readable description of intent (used for tracing, not execution)
- 'context' holds domain-specific data needed by the executor
- 'steps' are executed in order; executor interprets each action
- All fields are JSON-serializable by design
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import json


@dataclass
class IRStep:
    """A single executable action in the IR."""
    action: str              # action name (e.g., "set", "validate", "emit")
    params: Dict[str, Any] = field(default_factory=dict)  # action parameters

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticIR:
    """
    Minimal Semantic IR.
    - goal: what the IR intends to achieve (human-readable)
    - context: domain-specific data for execution
    - steps: ordered list of actions to execute
    """
    goal: str
    context: Dict[str, Any]
    steps: List[IRStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "context": self.context,
            "steps": [step.to_dict() for step in self.steps]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticIR":
        steps = [IRStep(**s) for s in data.get("steps", [])]
        return cls(goal=data["goal"], context=data["context"], steps=steps)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SemanticIR":
        """Create SemanticIR from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
