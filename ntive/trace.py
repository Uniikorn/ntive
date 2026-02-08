"""
trace.py

Ntive Trace Primitive — Semantic Decision Specification v1.0.0

A Trace is an immutable, ordered record of Decisions forming a causal chain.
It exists ONLY to answer: "Why was this decision reached?"

Design Invariants:
- Pure data structure (no side effects)
- Immutable after creation
- Deterministic serialization (sorted keys, content-based hash)
- No execution logic
- No timestamps, logging, memory, or policies
- Contains ONLY Decision instances
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ntive.decision import Decision

# =============================================================================
# Trace Errors
# =============================================================================

class TraceValidationError(Exception):
    """
    Raised when a Trace cannot be constructed due to validation failure.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "T000",
    ):
        self.message = message
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        return f"[{self.error_code}] Trace validation failed: {self.message}"


class InvalidTraceNodeError(TraceValidationError):
    """Raised when a non-Decision is appended to a Trace."""

    def __init__(self, actual_type: str, index: Optional[int] = None):
        location = f" at index {index}" if index is not None else ""
        super().__init__(
            message=f"Trace nodes must be Decision instances{location}, "
                    f"got {actual_type}",
            error_code="T001",
        )
        self.actual_type = actual_type
        self.index = index


class InvalidCausalReasonError(TraceValidationError):
    """Raised when a causal reason has invalid structure."""

    def __init__(self, message: str, index: Optional[int] = None):
        location = f" at index {index}" if index is not None else ""
        super().__init__(
            message=f"Invalid causal reason{location}: {message}",
            error_code="T002",
        )
        self.index = index


class TraceImmutabilityError(Exception):
    """Raised when attempting to mutate an immutable Trace."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: Trace is immutable after creation"
        )


# =============================================================================
# CausalReason — Optional metadata per step
# =============================================================================

@dataclass(frozen=True)
class CausalReason:
    """
    Optional causal metadata attached to a Decision in a Trace.

    Explains WHY this Decision follows from the previous one.
    Pure data, no execution semantics.
    """
    reason: str
    category: Optional[str] = None  # e.g., "inference", "constraint", "default"

    def __post_init__(self):
        if not isinstance(self.reason, str):
            raise InvalidCausalReasonError(
                f"reason must be str, got {type(self.reason).__name__}"
            )
        if not self.reason.strip():
            raise InvalidCausalReasonError("reason cannot be empty")
        if self.category is not None and not isinstance(self.category, str):
            raise InvalidCausalReasonError(
                f"category must be str or None, got {type(self.category).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"reason": self.reason}
        if self.category is not None:
            result["category"] = self.category
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalReason":
        """Reconstruct from dictionary."""
        return cls(
            reason=data.get("reason", ""),
            category=data.get("category"),
        )


# =============================================================================
# TraceNode — Internal container for Decision + CausalReason
# =============================================================================

@dataclass(frozen=True)
class TraceNode:
    """
    Internal node in a Trace containing a Decision and optional causal reason.

    Not exposed directly to users; used internally by Trace.
    """
    decision: Decision
    causal_reason: Optional[CausalReason] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"decision": self.decision.to_dict()}
        if self.causal_reason is not None:
            result["causal_reason"] = self.causal_reason.to_dict()
        return dict(sorted(result.items()))


# =============================================================================
# Trace — Immutable causal chain of Decisions
# =============================================================================

class Trace:
    """
    Immutable, ordered record of Decisions forming a causal chain.

    A Trace exists to answer: "Why was this decision reached?"

    Design:
    - Contains only Decision instances
    - Each decision can have optional causal metadata (CausalReason)
    - Supports parent Trace for branching/retries
    - Deterministic serialization (content-based trace_id)
    - Immutable after creation

    Example:
        trace = Trace.build([
            (decision1, None),
            (decision2, CausalReason("Follows from user intent")),
            (decision3, CausalReason("Constraint satisfied", "constraint")),
        ])

        # Or incrementally:
        trace = Trace.empty()
        trace = trace.append(decision1)
        trace = trace.append(decision2, reason="User requested this")
    """

    __slots__ = ('_nodes', '_parent', '_trace_id', '_frozen')

    def __init__(
        self,
        nodes: Tuple[TraceNode, ...],
        parent: Optional["Trace"] = None,
    ):
        """
        Create a Trace from pre-validated nodes.

        Use Trace.build() or Trace.empty().append() for construction.

        Args:
            nodes: Tuple of TraceNode instances (already validated)
            parent: Optional parent Trace for branching
        """
        object.__setattr__(self, '_frozen', False)

        # Validate nodes tuple
        if not isinstance(nodes, tuple):
            raise TraceValidationError(
                f"nodes must be tuple, got {type(nodes).__name__}",
                error_code="T003",
            )

        for i, node in enumerate(nodes):
            if not isinstance(node, TraceNode):
                raise TraceValidationError(
                    f"nodes[{i}] must be TraceNode, got {type(node).__name__}",
                    error_code="T003",
                )

        # Validate parent
        if parent is not None and not isinstance(parent, Trace):
            raise TraceValidationError(
                f"parent must be Trace or None, got {type(parent).__name__}",
                error_code="T004",
            )

        self._nodes = nodes
        self._parent = parent
        self._trace_id = self._compute_trace_id()

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after construction."""
        if getattr(self, '_frozen', False):
            raise TraceImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of attributes."""
        raise TraceImmutabilityError(f"delete attribute '{name}'")

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def empty(cls, parent: Optional["Trace"] = None) -> "Trace":
        """
        Create an empty Trace.

        Args:
            parent: Optional parent Trace for branching

        Returns:
            Empty Trace instance
        """
        return cls(nodes=(), parent=parent)

    @classmethod
    def build(
        cls,
        items: List[Union[
            Decision,
            Tuple[Decision, None],
            Tuple[Decision, str],
            Tuple[Decision, CausalReason],
        ]],
        parent: Optional["Trace"] = None,
    ) -> "Trace":
        """
        Build a Trace from a list of decisions with optional reasons.

        Args:
            items: List of decisions or (decision, reason) tuples.
                   Reason can be None, a string, or a CausalReason.
            parent: Optional parent Trace for branching

        Returns:
            New Trace instance

        Raises:
            InvalidTraceNodeError: If any item is not a Decision
            InvalidCausalReasonError: If any reason is invalid

        Example:
            Trace.build([
                decision1,  # No reason
                (decision2, "Because of X"),  # String reason
                (decision3, CausalReason("Y", "inference")),  # Full reason
            ])
        """
        nodes = []

        for i, item in enumerate(items):
            # Handle bare Decision
            if isinstance(item, Decision):
                nodes.append(TraceNode(decision=item, causal_reason=None))
                continue

            # Handle tuple (decision, reason)
            if isinstance(item, tuple) and len(item) == 2:
                decision, reason = item

                # Validate decision
                if not isinstance(decision, Decision):
                    raise InvalidTraceNodeError(type(decision).__name__, i)

                # Convert reason
                causal_reason = None
                if reason is not None:
                    if isinstance(reason, str):
                        causal_reason = CausalReason(reason=reason)
                    elif isinstance(reason, CausalReason):
                        causal_reason = reason
                    else:
                        raise InvalidCausalReasonError(
                            f"must be str, CausalReason, or None, "
                            f"got {type(reason).__name__}",
                            index=i,
                        )

                nodes.append(TraceNode(decision=decision, causal_reason=causal_reason))
                continue

            # Invalid item type
            raise InvalidTraceNodeError(type(item).__name__, i)

        return cls(nodes=tuple(nodes), parent=parent)

    # =========================================================================
    # Append (returns new Trace)
    # =========================================================================

    def append(
        self,
        decision: Decision,
        reason: Optional[Union[str, CausalReason]] = None,
    ) -> "Trace":
        """
        Return a new Trace with the decision appended.

        Does NOT mutate the original Trace.

        Args:
            decision: Decision to append
            reason: Optional causal reason (string or CausalReason)

        Returns:
            New Trace with the decision appended

        Raises:
            InvalidTraceNodeError: If decision is not a Decision instance
            InvalidCausalReasonError: If reason is invalid
        """
        if not isinstance(decision, Decision):
            raise InvalidTraceNodeError(type(decision).__name__)

        # Convert reason
        causal_reason = None
        if reason is not None:
            if isinstance(reason, str):
                causal_reason = CausalReason(reason=reason)
            elif isinstance(reason, CausalReason):
                causal_reason = reason
            else:
                raise InvalidCausalReasonError(
                    f"must be str, CausalReason, or None, "
                    f"got {type(reason).__name__}"
                )

        new_node = TraceNode(decision=decision, causal_reason=causal_reason)
        return Trace(
            nodes=self._nodes + (new_node,),
            parent=self._parent,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def trace_id(self) -> str:
        """
        Deterministic content-based hash of this Trace.

        Same decisions + same order + same reasons = identical trace_id
        """
        return self._trace_id

    @property
    def parent(self) -> Optional["Trace"]:
        """Parent Trace for branching/retries, or None."""
        return self._parent

    @property
    def decisions(self) -> Tuple[Decision, ...]:
        """Ordered tuple of Decisions (immutable)."""
        return tuple(node.decision for node in self._nodes)

    @property
    def reasons(self) -> Tuple[Optional[CausalReason], ...]:
        """Ordered tuple of causal reasons (immutable)."""
        return tuple(node.causal_reason for node in self._nodes)

    @property
    def nodes(self) -> Tuple[TraceNode, ...]:
        """Internal nodes (Decision + CausalReason pairs)."""
        return self._nodes

    # =========================================================================
    # Length and Iteration
    # =========================================================================

    def __len__(self) -> int:
        """Number of decisions in this Trace."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Decision]:
        """Iterate over decisions in order."""
        return iter(self.decisions)

    def __getitem__(self, index: int) -> Decision:
        """Get decision at index."""
        return self._nodes[index].decision

    def __bool__(self) -> bool:
        """True if Trace contains any decisions."""
        return len(self._nodes) > 0

    # =========================================================================
    # Trace ID Computation
    # =========================================================================

    def _compute_trace_id(self) -> str:
        """
        Compute deterministic content-based hash.

        Uses SHA-256 of the JSON representation for stability.
        Includes parent trace_id if present.
        """
        content = {
            "nodes": [node.to_dict() for node in self._nodes],
        }
        if self._parent is not None:
            content["parent_trace_id"] = self._parent.trace_id

        # Deterministic JSON
        json_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        hash_bytes = hashlib.sha256(json_str.encode('utf-8')).digest()

        # Return hex string (64 chars for SHA-256)
        return hash_bytes.hex()

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Keys are sorted for deterministic output.
        """
        result = {
            "nodes": [node.to_dict() for node in self._nodes],
            "trace_id": self._trace_id,
        }

        if self._parent is not None:
            result["parent_trace_id"] = self._parent.trace_id

        return dict(sorted(result.items()))

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """
        Serialize to JSON string.

        Guarantees deterministic output:
        - Keys are sorted alphabetically at all levels
        - Identical traces produce identical JSON

        Args:
            indent: Optional indentation for pretty-printing

        Returns:
            JSON string representation
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        parent: Optional["Trace"] = None,
    ) -> "Trace":
        """
        Reconstruct a Trace from a dictionary.

        Note: Parent trace must be provided separately; parent_trace_id
        is stored for reference but cannot be used to reconstruct the parent.

        Args:
            data: Dictionary with trace fields
            parent: Parent Trace instance (optional)

        Returns:
            New Trace instance
        """

        nodes_data = data.get("nodes", [])
        nodes = []

        for node_data in nodes_data:
            # Reconstruct Decision
            decision_data = node_data.get("decision", {})
            decision = Decision.from_dict(decision_data)

            # Reconstruct CausalReason
            causal_reason = None
            if "causal_reason" in node_data and node_data["causal_reason"]:
                causal_reason = CausalReason.from_dict(node_data["causal_reason"])

            nodes.append(TraceNode(decision=decision, causal_reason=causal_reason))

        return cls(nodes=tuple(nodes), parent=parent)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        parent: Optional["Trace"] = None,
    ) -> "Trace":
        """
        Reconstruct a Trace from a JSON string.

        Args:
            json_str: JSON string representation
            parent: Parent Trace instance (optional)

        Returns:
            New Trace instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data, parent=parent)

    # =========================================================================
    # Equality and Hashing
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        """Two traces are equal if their trace_ids are identical."""
        if not isinstance(other, Trace):
            return NotImplemented
        return self._trace_id == other._trace_id

    def __hash__(self) -> int:
        """Hash based on trace_id."""
        return hash(self._trace_id)

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        parent_info = f", parent={self._parent.trace_id[:8]}..." if self._parent else ""
        return f"Trace(id={self._trace_id[:16]}..., len={len(self)}{parent_info})"

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [f"Trace ({len(self)} decisions):"]
        for i, node in enumerate(self._nodes):
            decision = node.decision
            reason = node.causal_reason
            reason_str = f" [{reason.reason}]" if reason else ""
            lines.append(f"  {i+1}. {decision.selected_option}{reason_str}")
        return "\n".join(lines)

    # =========================================================================
    # Causal Chain Navigation
    # =========================================================================

    def first(self) -> Optional[Decision]:
        """Return the first decision, or None if empty."""
        return self._nodes[0].decision if self._nodes else None

    def last(self) -> Optional[Decision]:
        """Return the last decision, or None if empty."""
        return self._nodes[-1].decision if self._nodes else None

    def chain(self) -> List["Trace"]:
        """
        Return the full chain of traces from root to this trace.

        Returns:
            List of traces, oldest (root) first
        """
        chain = [self]
        current = self._parent
        while current is not None:
            chain.append(current)
            current = current._parent
        return list(reversed(chain))

    def depth(self) -> int:
        """
        Return the depth of this trace in the parent chain.

        Root trace (no parent) has depth 0.
        """
        d = 0
        current = self._parent
        while current is not None:
            d += 1
            current = current._parent
        return d
