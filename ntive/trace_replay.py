"""
trace_replay.py

Ntive Trace Replay Tool — Semantic Decision Specification v1.0.0

A read-only tool for replaying and auditing past decision traces.
Supports loading serialized traces, verifying determinism, detecting
broken causal chains, and outputting human-readable explanations.

Design Invariants:
- Read-only (no execution, no mutation)
- Deterministic analysis
- Supports multiple serialization formats (JSON, JSONL)
- Pure audit/compliance/debugging tool

Use Cases:
- Audit: Verify decisions match expected outcomes
- Compliance: Document decision rationale for regulatory review
- Debugging: Trace why a specific decision was made
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ntive.decision import Decision
from ntive.trace import CausalReason, Trace, TraceNode

# =============================================================================
# Exceptions
# =============================================================================

class TraceReplayError(Exception):
    """Base exception for trace replay errors."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "TR000",
    ):
        self.message = message
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        return f"[{self.error_code}] Trace replay error: {self.message}"


class TraceLoadError(TraceReplayError):
    """Failed to load trace from file or data."""

    def __init__(self, message: str, source: Optional[str] = None):
        self.source = source
        source_info = f" from '{source}'" if source else ""
        super().__init__(
            message=f"Failed to load trace{source_info}: {message}",
            error_code="TR001",
        )


class BrokenCausalChainError(TraceReplayError):
    """Detected a broken causal chain in the trace."""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        expected_parent: Optional[str] = None,
    ):
        self.node_id = node_id
        self.expected_parent = expected_parent
        super().__init__(
            message=message,
            error_code="TR002",
        )


class DeterminismError(TraceReplayError):
    """Trace failed determinism verification."""

    def __init__(
        self,
        message: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
    ):
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            message=message,
            error_code="TR003",
        )


# =============================================================================
# Data Structures
# =============================================================================

class ChainStatus(Enum):
    """Status of a causal chain verification."""
    VALID = "valid"
    BROKEN = "broken"
    ORPHAN = "orphan"  # Node without valid parent
    CYCLE = "cycle"    # Circular reference detected


@dataclass(frozen=True)
class ReplayNode:
    """
    Immutable representation of a single node in a replay trace.

    Supports both Trace primitive format and JSONL format.
    """
    node_id: str
    parent_id: Optional[str]
    depth: int
    action: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reason: Optional[Dict[str, Any]]
    timestamp: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "action": self.action,
            "input": self.inputs,
            "output": self.outputs,
        }
        if self.reason:
            result["reason"] = self.reason
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayNode":
        """Construct from dictionary."""
        return cls(
            node_id=str(data.get("id", "")),
            parent_id=data.get("parent_id"),
            depth=int(data.get("depth", 0)),
            action=str(data.get("action", "")),
            inputs=dict(data.get("input", {})),
            outputs=dict(data.get("output", {})),
            reason=data.get("reason"),
            timestamp=data.get("timestamp"),
        )

    @classmethod
    def from_trace_node(cls, node: TraceNode, index: int, parent_id: Optional[str] = None) -> "ReplayNode":
        """Construct from a Trace primitive TraceNode."""
        decision = node.decision
        return cls(
            node_id=decision.decision_id,
            parent_id=parent_id,
            depth=index,
            action="decision",
            inputs=dict(decision.inputs),
            outputs={"selected_option": decision.selected_option},
            reason={
                "rationale": decision.rationale,
                "causal_reason": node.causal_reason.to_dict() if node.causal_reason else None,
            },
            timestamp=None,  # Trace primitive doesn't store timestamp per node
        )


@dataclass(frozen=True)
class ChainValidation:
    """Result of causal chain validation."""
    status: ChainStatus
    valid_nodes: Tuple[str, ...]
    broken_links: Tuple[Tuple[str, str], ...]  # (node_id, expected_parent_id)
    orphan_nodes: Tuple[str, ...]
    cycle_nodes: Tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """True if the chain is completely valid."""
        return self.status == ChainStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "valid_nodes": list(self.valid_nodes),
            "broken_links": [list(link) for link in self.broken_links],
            "orphan_nodes": list(self.orphan_nodes),
            "cycle_nodes": list(self.cycle_nodes),
        }


@dataclass(frozen=True)
class DeterminismCheck:
    """Result of determinism verification."""
    is_deterministic: bool
    trace_hash: str
    expected_hash: Optional[str]
    differences: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_deterministic": self.is_deterministic,
            "trace_hash": self.trace_hash,
            "expected_hash": self.expected_hash,
            "differences": list(self.differences),
        }


@dataclass(frozen=True)
class ReplayExplanation:
    """Human-readable explanation of a trace."""
    summary: str
    decision_count: int
    root_decision: Optional[str]
    final_decision: Optional[str]
    chain_depth: int
    steps: Tuple[str, ...]

    def to_text(self) -> str:
        """Format as plain text."""
        lines = [self.summary, ""]
        lines.append(f"Total decisions: {self.decision_count}")
        lines.append(f"Chain depth: {self.chain_depth}")
        if self.root_decision:
            lines.append(f"Root decision: {self.root_decision}")
        if self.final_decision:
            lines.append(f"Final decision: {self.final_decision}")
        lines.append("")
        lines.append("Decision sequence:")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "decision_count": self.decision_count,
            "root_decision": self.root_decision,
            "final_decision": self.final_decision,
            "chain_depth": self.chain_depth,
            "steps": list(self.steps),
        }


# =============================================================================
# TraceReplay — Main Tool
# =============================================================================

class TraceReplay:
    """
    Read-only trace replay and audit tool.

    Supports loading, reconstructing, verifying, and explaining past
    decision traces. Does NOT execute any logic - purely analyzes
    previously recorded traces.

    Example:
        # Load from JSONL file
        replay = TraceReplay.load_jsonl("trace.jsonl")

        # Verify causal chain
        validation = replay.validate_chain()
        if not validation.is_valid:
            print(f"Broken chain: {validation.broken_links}")

        # Get human-readable explanation
        explanation = replay.explain()
        print(explanation.to_text())

        # Verify determinism
        check = replay.verify_determinism()
        print(f"Deterministic: {check.is_deterministic}")
    """

    __slots__ = ('_nodes', '_nodes_by_id', '_root_nodes', '_trace_hash')

    def __init__(self, nodes: Tuple[ReplayNode, ...]):
        """
        Create a TraceReplay from pre-validated nodes.

        Use factory methods like load_jsonl() or from_trace() instead.

        Args:
            nodes: Tuple of ReplayNode instances
        """
        self._nodes = nodes

        # Build index for fast lookup
        self._nodes_by_id: Dict[str, ReplayNode] = {}
        self._root_nodes: List[str] = []

        for node in nodes:
            self._nodes_by_id[node.node_id] = node
            if node.parent_id is None:
                self._root_nodes.append(node.node_id)

        # Compute deterministic hash
        self._trace_hash = self._compute_hash()

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def load_jsonl(cls, path: Union[str, Path]) -> "TraceReplay":
        """
        Load a trace from a JSONL file.

        Each line should be a valid JSON object representing a node.

        Args:
            path: Path to the JSONL file

        Returns:
            TraceReplay instance

        Raises:
            TraceLoadError: If file cannot be loaded or parsed
        """
        path = Path(path)
        source = str(path)

        if not path.exists():
            raise TraceLoadError(f"File not found: {path}", source=source)

        nodes = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        nodes.append(ReplayNode.from_dict(data))
                    except json.JSONDecodeError as e:
                        raise TraceLoadError(
                            f"Invalid JSON at line {line_no}: {e}",
                            source=source,
                        )
        except IOError as e:
            raise TraceLoadError(f"IO error: {e}", source=source)

        return cls(nodes=tuple(nodes))

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "TraceReplay":
        """
        Load a trace from a JSON file containing a Trace dict.

        Args:
            path: Path to the JSON file

        Returns:
            TraceReplay instance

        Raises:
            TraceLoadError: If file cannot be loaded or parsed
        """
        path = Path(path)
        source = str(path)

        if not path.exists():
            raise TraceLoadError(f"File not found: {path}", source=source)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise TraceLoadError(str(e), source=source)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceReplay":
        """
        Create a TraceReplay from a Trace dictionary.

        Supports the Trace.to_dict() format.

        Args:
            data: Dictionary representation of a Trace

        Returns:
            TraceReplay instance
        """
        nodes_data = data.get("nodes", [])
        nodes = []

        prev_id = None
        for i, node_data in enumerate(nodes_data):
            decision_data = node_data.get("decision", {})
            causal_data = node_data.get("causal_reason")

            node = ReplayNode(
                node_id=decision_data.get("decision_id", f"node_{i}"),
                parent_id=prev_id,
                depth=i,
                action="decision",
                inputs=dict(decision_data.get("inputs", {})),
                outputs={"selected_option": decision_data.get("selected_option", "")},
                reason={
                    "rationale": decision_data.get("rationale", ""),
                    "causal_reason": causal_data,
                },
                timestamp=None,
            )
            nodes.append(node)
            prev_id = node.node_id

        return cls(nodes=tuple(nodes))

    @classmethod
    def from_json(cls, json_str: str) -> "TraceReplay":
        """
        Create a TraceReplay from a JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            TraceReplay instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_trace(cls, trace: Trace) -> "TraceReplay":
        """
        Create a TraceReplay from a Trace primitive.

        Args:
            trace: Trace instance

        Returns:
            TraceReplay instance
        """
        nodes = []
        prev_id = None

        for i, trace_node in enumerate(trace.nodes):
            node = ReplayNode.from_trace_node(trace_node, i, prev_id)
            nodes.append(node)
            prev_id = node.node_id

        return cls(nodes=tuple(nodes))

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def nodes(self) -> Tuple[ReplayNode, ...]:
        """All nodes in order."""
        return self._nodes

    @property
    def trace_hash(self) -> str:
        """Deterministic content-based hash."""
        return self._trace_hash

    @property
    def root_nodes(self) -> Tuple[str, ...]:
        """IDs of root nodes (no parent)."""
        return tuple(self._root_nodes)

    def __len__(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[ReplayNode]:
        """Iterate over nodes in order."""
        return iter(self._nodes)

    def __getitem__(self, index: int) -> ReplayNode:
        """Get node by index."""
        return self._nodes[index]

    # =========================================================================
    # Lookup
    # =========================================================================

    def get_node(self, node_id: str) -> Optional[ReplayNode]:
        """Get node by ID, or None if not found."""
        return self._nodes_by_id.get(node_id)

    def get_children(self, node_id: str) -> Tuple[ReplayNode, ...]:
        """Get all direct children of a node."""
        return tuple(
            node for node in self._nodes
            if node.parent_id == node_id
        )

    def get_ancestors(self, node_id: str) -> Tuple[ReplayNode, ...]:
        """Get all ancestors of a node from root to parent."""
        ancestors = []
        current = self.get_node(node_id)

        if current is None:
            return ()

        visited = {node_id}
        parent_id = current.parent_id

        while parent_id is not None:
            if parent_id in visited:
                break  # Cycle detected
            visited.add(parent_id)

            parent = self.get_node(parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            parent_id = parent.parent_id

        return tuple(reversed(ancestors))

    def get_descendants(self, node_id: str) -> Tuple[ReplayNode, ...]:
        """Get all descendants of a node (breadth-first)."""
        descendants = []
        queue = list(self.get_children(node_id))
        visited = {node_id}

        while queue:
            node = queue.pop(0)
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            descendants.append(node)
            queue.extend(self.get_children(node.node_id))

        return tuple(descendants)

    # =========================================================================
    # Reconstruction
    # =========================================================================

    def reconstruct_decisions(self) -> Tuple[Dict[str, Any], ...]:
        """
        Reconstruct the decision sequence from the trace.

        Returns a tuple of decision dictionaries in causal order.
        """
        decisions = []

        for node in self._nodes:
            decision = {
                "id": node.node_id,
                "action": node.action,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "depth": node.depth,
            }

            if node.reason:
                if "rationale" in node.reason:
                    decision["rationale"] = node.reason.get("rationale")
                if "causal_reason" in node.reason and node.reason["causal_reason"]:
                    decision["causal_reason"] = node.reason["causal_reason"]
                # JSONL format reason
                if "type" in node.reason:
                    decision["reason_type"] = node.reason.get("type")
                    decision["reason_source"] = node.reason.get("source")
                    decision["reason_ref"] = node.reason.get("ref")
                    decision["reason_value"] = node.reason.get("value")

            decisions.append(decision)

        return tuple(decisions)

    def reconstruct_trace(self) -> Optional[Trace]:
        """
        Attempt to reconstruct a Trace primitive from the replay data.

        Returns None if reconstruction is not possible (e.g., JSONL format
        without full Decision data).
        """
        nodes = []

        for replay_node in self._nodes:
            # Need full decision data
            if replay_node.action != "decision":
                return None  # Cannot reconstruct non-decision nodes

            try:
                rationale_value = (
                    replay_node.reason.get("rationale", "Reconstructed from trace")
                    if replay_node.reason else "Reconstructed from trace"
                )
                decision = Decision(
                    inputs=replay_node.inputs,
                    selected_option=replay_node.outputs.get("selected_option", "unknown"),
                    alternatives=[],  # Cannot reconstruct without original data
                    rationale=rationale_value,
                    decision_id=replay_node.node_id,
                )

                causal_reason = None
                if replay_node.reason and replay_node.reason.get("causal_reason"):
                    cr_data = replay_node.reason["causal_reason"]
                    causal_reason = CausalReason(
                        reason=cr_data.get("reason", ""),
                        category=cr_data.get("category"),
                    )

                nodes.append((decision, causal_reason))
            except Exception:
                return None  # Cannot reconstruct

        if not nodes:
            return Trace.empty()

        return Trace.build(nodes)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_chain(self) -> ChainValidation:
        """
        Validate the causal chain integrity.

        Checks for:
        - Broken links (parent doesn't exist)
        - Orphan nodes (non-root without valid parent)
        - Cycles (circular references)

        Returns:
            ChainValidation with status and details
        """
        valid_nodes = []
        broken_links = []
        orphan_nodes = []
        cycle_nodes = []

        for node in self._nodes:
            # Check for cycles by tracing ancestors
            visited = set()
            current_id = node.node_id
            has_cycle = False

            while current_id is not None:
                if current_id in visited:
                    has_cycle = True
                    break
                visited.add(current_id)

                current = self.get_node(current_id)
                if current is None:
                    break
                current_id = current.parent_id

            if has_cycle:
                cycle_nodes.append(node.node_id)
                continue

            # Root node
            if node.parent_id is None:
                valid_nodes.append(node.node_id)
                continue

            # Check parent exists
            parent = self.get_node(node.parent_id)
            if parent is None:
                broken_links.append((node.node_id, node.parent_id))
                orphan_nodes.append(node.node_id)
            else:
                valid_nodes.append(node.node_id)

        # Determine overall status
        if cycle_nodes:
            status = ChainStatus.CYCLE
        elif broken_links or orphan_nodes:
            status = ChainStatus.BROKEN
        elif not valid_nodes and len(self._nodes) > 0:
            status = ChainStatus.ORPHAN
        else:
            status = ChainStatus.VALID

        return ChainValidation(
            status=status,
            valid_nodes=tuple(valid_nodes),
            broken_links=tuple(broken_links),
            orphan_nodes=tuple(orphan_nodes),
            cycle_nodes=tuple(cycle_nodes),
        )

    def detect_broken_chains(self) -> List[BrokenCausalChainError]:
        """
        Detect all broken causal chains and return error objects.

        Returns:
            List of BrokenCausalChainError for each broken link
        """
        validation = self.validate_chain()
        errors = []

        for node_id, expected_parent in validation.broken_links:
            errors.append(BrokenCausalChainError(
                message=f"Node '{node_id}' references non-existent parent '{expected_parent}'",
                node_id=node_id,
                expected_parent=expected_parent,
            ))

        for node_id in validation.cycle_nodes:
            errors.append(BrokenCausalChainError(
                message=f"Node '{node_id}' is part of a circular reference",
                node_id=node_id,
            ))

        return errors

    # =========================================================================
    # Determinism Verification
    # =========================================================================

    def _compute_hash(self) -> str:
        """Compute deterministic content-based hash."""
        content = [node.to_dict() for node in self._nodes]
        json_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def verify_determinism(
        self,
        expected_hash: Optional[str] = None,
    ) -> DeterminismCheck:
        """
        Verify that the trace is deterministic.

        If expected_hash is provided, verifies the trace hash matches.
        Otherwise, just returns the computed hash.

        Args:
            expected_hash: Optional expected hash to compare against

        Returns:
            DeterminismCheck with results
        """
        current_hash = self._trace_hash

        if expected_hash is None:
            return DeterminismCheck(
                is_deterministic=True,
                trace_hash=current_hash,
                expected_hash=None,
                differences=(),
            )

        if current_hash == expected_hash:
            return DeterminismCheck(
                is_deterministic=True,
                trace_hash=current_hash,
                expected_hash=expected_hash,
                differences=(),
            )

        return DeterminismCheck(
            is_deterministic=False,
            trace_hash=current_hash,
            expected_hash=expected_hash,
            differences=(
                f"Hash mismatch: expected {expected_hash[:16]}..., got {current_hash[:16]}...",
            ),
        )

    def compare_with(self, other: "TraceReplay") -> Tuple[str, ...]:
        """
        Compare this trace with another and report differences.

        Args:
            other: Another TraceReplay to compare

        Returns:
            Tuple of difference descriptions
        """
        differences = []

        # Compare counts
        if len(self) != len(other):
            differences.append(f"Node count differs: {len(self)} vs {len(other)}")

        # Compare hashes
        if self._trace_hash != other._trace_hash:
            differences.append(
                f"Trace hash differs: {self._trace_hash[:16]}... vs {other._trace_hash[:16]}..."
            )

        # Compare nodes
        min_len = min(len(self), len(other))
        for i in range(min_len):
            n1, n2 = self._nodes[i], other._nodes[i]

            if n1.node_id != n2.node_id:
                differences.append(f"Node {i} ID differs: '{n1.node_id}' vs '{n2.node_id}'")
            if n1.action != n2.action:
                differences.append(f"Node {i} action differs: '{n1.action}' vs '{n2.action}'")
            if n1.inputs != n2.inputs:
                differences.append(f"Node {i} inputs differ")
            if n1.outputs != n2.outputs:
                differences.append(f"Node {i} outputs differ")

        return tuple(differences)

    # =========================================================================
    # Human-Readable Output
    # =========================================================================

    def explain(self) -> ReplayExplanation:
        """
        Generate a human-readable explanation of the trace.

        Returns:
            ReplayExplanation with summary and step-by-step details
        """
        if not self._nodes:
            return ReplayExplanation(
                summary="Empty trace (no decisions recorded)",
                decision_count=0,
                root_decision=None,
                final_decision=None,
                chain_depth=0,
                steps=(),
            )

        # Find max depth
        max_depth = max(node.depth for node in self._nodes)

        # Build step descriptions
        steps = []
        for node in self._nodes:
            step = self._format_step(node)
            steps.append(step)

        # Get root and final
        root = self._nodes[0] if self._nodes else None
        final = self._nodes[-1] if self._nodes else None

        root_desc = self._format_node_brief(root) if root else None
        final_desc = self._format_node_brief(final) if final else None

        summary = f"Trace with {len(self._nodes)} decision(s), max depth {max_depth}"

        return ReplayExplanation(
            summary=summary,
            decision_count=len(self._nodes),
            root_decision=root_desc,
            final_decision=final_desc,
            chain_depth=max_depth,
            steps=tuple(steps),
        )

    def _format_step(self, node: ReplayNode) -> str:
        """Format a single step for human reading."""
        indent = "  " * node.depth
        action = node.action.upper()

        # Format outputs
        output_parts = []
        for k, v in node.outputs.items():
            output_parts.append(f"{k}={v!r}")
        output_str = ", ".join(output_parts) if output_parts else "(no output)"

        # Format reason
        reason_str = ""
        if node.reason:
            if "rationale" in node.reason and node.reason["rationale"]:
                reason_str = f" | {node.reason['rationale']}"
            elif "ref" in node.reason:
                reason_str = f" | ref={node.reason['ref']}"

        return f"{indent}[{action}] {output_str}{reason_str}"

    def _format_node_brief(self, node: ReplayNode) -> str:
        """Format a brief description of a node."""
        if node.action == "decision":
            selected = node.outputs.get("selected_option", "unknown")
            return f"{node.action}: {selected}"
        return f"{node.action}: {node.outputs}"

    def to_text(self) -> str:
        """
        Generate full text representation of the trace.

        Returns:
            Multi-line string with all trace details
        """
        return self.explain().to_text()

    def to_markdown(self) -> str:
        """
        Generate Markdown representation of the trace.

        Returns:
            Markdown-formatted string
        """
        lines = ["# Trace Replay Report", ""]

        explanation = self.explain()
        lines.append(f"**Summary:** {explanation.summary}")
        lines.append("")
        lines.append("## Statistics")
        lines.append(f"- Decision count: {explanation.decision_count}")
        lines.append(f"- Chain depth: {explanation.chain_depth}")
        lines.append(f"- Trace hash: `{self._trace_hash[:16]}...`")
        lines.append("")

        # Validation
        validation = self.validate_chain()
        lines.append("## Chain Validation")
        lines.append(f"- Status: **{validation.status.value.upper()}**")
        if validation.broken_links:
            lines.append(f"- Broken links: {len(validation.broken_links)}")
        if validation.orphan_nodes:
            lines.append(f"- Orphan nodes: {len(validation.orphan_nodes)}")
        lines.append("")

        # Steps
        lines.append("## Decision Sequence")
        lines.append("```")
        for step in explanation.steps:
            lines.append(step)
        lines.append("```")

        return "\n".join(lines)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trace_hash": self._trace_hash,
            "node_count": len(self._nodes),
            "root_nodes": list(self._root_nodes),
            "nodes": [node.to_dict() for node in self._nodes],
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
        )

    def to_jsonl(self) -> str:
        """Serialize to JSONL format (one node per line)."""
        lines = []
        for node in self._nodes:
            lines.append(json.dumps(node.to_dict(), sort_keys=True, ensure_ascii=False))
        return "\n".join(lines)

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"TraceReplay(nodes={len(self)}, hash={self._trace_hash[:16]}...)"

    def __str__(self) -> str:
        """Human-readable representation."""
        return self.to_text()
