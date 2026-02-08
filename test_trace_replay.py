"""
test_trace_replay.py

Tests for the Ntive Trace Replay Tool.

Tests cover:
- Loading from JSONL and JSON files
- Loading from Trace primitive
- Causal chain validation
- Determinism verification
- Human-readable explanations
- Broken chain detection
- Serialization round-trips
"""

import json
import pytest
import tempfile
from pathlib import Path

from ntive.trace_replay import (
    TraceReplay,
    ReplayNode,
    ChainValidation,
    ChainStatus,
    DeterminismCheck,
    ReplayExplanation,
    TraceReplayError,
    TraceLoadError,
    BrokenCausalChainError,
    DeterminismError,
)
from ntive.trace import Trace, CausalReason
from ntive.decision import Decision, Alternative


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL trace content."""
    return '\n'.join([
        json.dumps({
            "id": "node-1",
            "parent_id": None,
            "depth": 0,
            "action": "set",
            "input": {"key": "ssl"},
            "output": {"ssl": True},
            "reason": {"type": "constraint", "source": "ir", "ref": "context.ssl", "value": True},
            "timestamp": 1000.0
        }),
        json.dumps({
            "id": "node-2",
            "parent_id": "node-1",
            "depth": 1,
            "action": "set",
            "input": {"key": "pool"},
            "output": {"pool": True},
            "reason": {"type": "constraint", "source": "ir", "ref": "context.pool", "value": True},
            "timestamp": 1001.0
        }),
        json.dumps({
            "id": "node-3",
            "parent_id": "node-2",
            "depth": 2,
            "action": "emit",
            "input": {"target": "config"},
            "output": {"emitted": "config"},
            "reason": {"type": "constraint", "source": "ir", "ref": "target", "value": "config"},
            "timestamp": 1002.0
        }),
    ])


@pytest.fixture
def sample_jsonl_file(sample_jsonl_content):
    """Create a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(sample_jsonl_content)
        return Path(f.name)


@pytest.fixture
def sample_trace():
    """Create a sample Trace primitive."""
    d1 = Decision(
        inputs={"query": "open file"},
        selected_option="open_dialog",
        alternatives=[Alternative("recent_files", "User said 'open'")],
        rationale="User explicitly requested file open",
    )
    d2 = Decision(
        inputs={"path": "/home/user"},
        selected_option="list_directory",
        alternatives=[],
        rationale="Default home directory",
    )
    d3 = Decision(
        inputs={"selection": "document.txt"},
        selected_option="open_file",
        alternatives=[],
        rationale="User selected specific file",
    )
    
    return Trace.build([
        (d1, "User initiated action"),
        (d2, CausalReason("Following open request", "navigation")),
        (d3, "Final selection"),
    ])


@pytest.fixture
def sample_replay_nodes():
    """Create sample ReplayNode instances."""
    return (
        ReplayNode(
            node_id="a",
            parent_id=None,
            depth=0,
            action="decision",
            inputs={"x": 1},
            outputs={"selected_option": "A"},
            reason={"rationale": "First choice"},
            timestamp=100.0,
        ),
        ReplayNode(
            node_id="b",
            parent_id="a",
            depth=1,
            action="decision",
            inputs={"x": 2},
            outputs={"selected_option": "B"},
            reason={"rationale": "Second choice"},
            timestamp=101.0,
        ),
        ReplayNode(
            node_id="c",
            parent_id="b",
            depth=2,
            action="decision",
            inputs={"x": 3},
            outputs={"selected_option": "C"},
            reason={"rationale": "Third choice"},
            timestamp=102.0,
        ),
    )


# =============================================================================
# ReplayNode Tests
# =============================================================================

class TestReplayNode:
    """Tests for ReplayNode data structure."""
    
    def test_create_minimal(self):
        """Create ReplayNode with minimal fields."""
        node = ReplayNode(
            node_id="n1",
            parent_id=None,
            depth=0,
            action="test",
            inputs={},
            outputs={},
            reason=None,
            timestamp=None,
        )
        assert node.node_id == "n1"
        assert node.parent_id is None
        assert node.depth == 0
    
    def test_create_with_all_fields(self):
        """Create ReplayNode with all fields."""
        node = ReplayNode(
            node_id="n1",
            parent_id="n0",
            depth=1,
            action="decision",
            inputs={"key": "value"},
            outputs={"result": 42},
            reason={"rationale": "Because"},
            timestamp=123.456,
        )
        assert node.parent_id == "n0"
        assert node.inputs == {"key": "value"}
        assert node.timestamp == 123.456
    
    def test_to_dict(self):
        """Convert to dictionary."""
        node = ReplayNode(
            node_id="n1",
            parent_id="n0",
            depth=1,
            action="set",
            inputs={"x": 1},
            outputs={"y": 2},
            reason={"type": "test"},
            timestamp=100.0,
        )
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["parent_id"] == "n0"
        assert d["depth"] == 1
        assert d["action"] == "set"
        assert d["input"] == {"x": 1}
        assert d["output"] == {"y": 2}
        assert d["reason"] == {"type": "test"}
        assert d["timestamp"] == 100.0
    
    def test_from_dict(self):
        """Reconstruct from dictionary."""
        data = {
            "id": "n1",
            "parent_id": "n0",
            "depth": 1,
            "action": "emit",
            "input": {"a": "b"},
            "output": {"c": "d"},
            "reason": {"type": "constraint"},
            "timestamp": 200.0,
        }
        node = ReplayNode.from_dict(data)
        assert node.node_id == "n1"
        assert node.parent_id == "n0"
        assert node.action == "emit"
    
    def test_from_dict_missing_fields(self):
        """Handle missing fields gracefully."""
        data = {"id": "n1"}
        node = ReplayNode.from_dict(data)
        assert node.node_id == "n1"
        assert node.parent_id is None
        assert node.depth == 0
        assert node.inputs == {}
        assert node.outputs == {}
    
    def test_immutability(self):
        """ReplayNode is immutable (frozen dataclass)."""
        node = ReplayNode(
            node_id="n1",
            parent_id=None,
            depth=0,
            action="test",
            inputs={},
            outputs={},
            reason=None,
            timestamp=None,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            node.node_id = "n2"


# =============================================================================
# TraceReplay Loading Tests
# =============================================================================

class TestTraceReplayLoading:
    """Tests for loading traces from various sources."""
    
    def test_load_jsonl(self, sample_jsonl_file):
        """Load trace from JSONL file."""
        replay = TraceReplay.load_jsonl(sample_jsonl_file)
        assert len(replay) == 3
        assert replay.nodes[0].node_id == "node-1"
        assert replay.nodes[1].parent_id == "node-1"
        assert replay.nodes[2].action == "emit"
    
    def test_load_jsonl_file_not_found(self):
        """Raise error for missing file."""
        with pytest.raises(TraceLoadError) as exc:
            TraceReplay.load_jsonl("/nonexistent/path/trace.jsonl")
        assert "not found" in str(exc.value).lower()
    
    def test_load_jsonl_invalid_json(self):
        """Raise error for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": true}\n')
            f.write('not valid json\n')
            path = Path(f.name)
        
        with pytest.raises(TraceLoadError) as exc:
            TraceReplay.load_jsonl(path)
        assert "line 2" in str(exc.value).lower() or "invalid" in str(exc.value).lower()
    
    def test_load_jsonl_empty_lines_ignored(self):
        """Empty lines in JSONL are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "n1", "action": "test"}\n')
            f.write('\n')
            f.write('{"id": "n2", "action": "test"}\n')
            path = Path(f.name)
        
        replay = TraceReplay.load_jsonl(path)
        assert len(replay) == 2
    
    def test_from_trace(self, sample_trace):
        """Create TraceReplay from Trace primitive."""
        replay = TraceReplay.from_trace(sample_trace)
        assert len(replay) == 3
        assert replay.nodes[0].action == "decision"
        assert "open_dialog" in str(replay.nodes[0].outputs)
    
    def test_from_trace_empty(self):
        """Create TraceReplay from empty Trace."""
        trace = Trace.empty()
        replay = TraceReplay.from_trace(trace)
        assert len(replay) == 0
    
    def test_from_dict(self):
        """Create TraceReplay from dictionary."""
        data = {
            "nodes": [
                {
                    "decision": {
                        "decision_id": "d1",
                        "inputs": {"x": 1},
                        "selected_option": "A",
                        "rationale": "Test",
                    }
                },
                {
                    "decision": {
                        "decision_id": "d2",
                        "inputs": {"x": 2},
                        "selected_option": "B",
                        "rationale": "Test 2",
                    },
                    "causal_reason": {"reason": "Follows A"},
                }
            ]
        }
        replay = TraceReplay.from_dict(data)
        assert len(replay) == 2
        assert replay.nodes[0].node_id == "d1"
        assert replay.nodes[1].parent_id == "d1"
    
    def test_from_json(self):
        """Create TraceReplay from JSON string."""
        data = {
            "trace_id": "abc",
            "nodes": [
                {"decision": {"decision_id": "d1", "inputs": {}, "selected_option": "X", "rationale": "R"}}
            ]
        }
        replay = TraceReplay.from_json(json.dumps(data))
        assert len(replay) == 1


# =============================================================================
# Chain Validation Tests
# =============================================================================

class TestChainValidation:
    """Tests for causal chain validation."""
    
    def test_valid_chain(self, sample_replay_nodes):
        """Valid chain passes validation."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        validation = replay.validate_chain()
        
        assert validation.status == ChainStatus.VALID
        assert validation.is_valid
        assert len(validation.valid_nodes) == 3
        assert len(validation.broken_links) == 0
        assert len(validation.orphan_nodes) == 0
    
    def test_broken_chain(self):
        """Detect broken parent link."""
        nodes = (
            ReplayNode("a", None, 0, "test", {}, {}, None, None),
            ReplayNode("b", "nonexistent", 1, "test", {}, {}, None, None),
        )
        replay = TraceReplay(nodes=nodes)
        validation = replay.validate_chain()
        
        assert validation.status == ChainStatus.BROKEN
        assert not validation.is_valid
        assert ("b", "nonexistent") in validation.broken_links
        assert "b" in validation.orphan_nodes
    
    def test_multiple_roots(self):
        """Multiple root nodes are valid."""
        nodes = (
            ReplayNode("a", None, 0, "test", {}, {}, None, None),
            ReplayNode("b", None, 0, "test", {}, {}, None, None),
            ReplayNode("c", "a", 1, "test", {}, {}, None, None),
        )
        replay = TraceReplay(nodes=nodes)
        validation = replay.validate_chain()
        
        assert validation.status == ChainStatus.VALID
        assert len(replay.root_nodes) == 2
    
    def test_cycle_detection(self):
        """Detect circular references."""
        # Create nodes that would form a cycle if parent links were followed
        # a -> b -> c -> a (cycle)
        nodes = (
            ReplayNode("a", "c", 0, "test", {}, {}, None, None),  # a's parent is c
            ReplayNode("b", "a", 1, "test", {}, {}, None, None),  # b's parent is a
            ReplayNode("c", "b", 2, "test", {}, {}, None, None),  # c's parent is b
        )
        replay = TraceReplay(nodes=nodes)
        validation = replay.validate_chain()
        
        assert validation.status == ChainStatus.CYCLE
        assert len(validation.cycle_nodes) > 0
    
    def test_detect_broken_chains_returns_errors(self):
        """detect_broken_chains returns error objects."""
        nodes = (
            ReplayNode("a", None, 0, "test", {}, {}, None, None),
            ReplayNode("b", "missing", 1, "test", {}, {}, None, None),
        )
        replay = TraceReplay(nodes=nodes)
        errors = replay.detect_broken_chains()
        
        assert len(errors) == 1
        assert isinstance(errors[0], BrokenCausalChainError)
        assert errors[0].node_id == "b"
        assert errors[0].expected_parent == "missing"
    
    def test_chain_validation_to_dict(self):
        """ChainValidation converts to dict."""
        validation = ChainValidation(
            status=ChainStatus.BROKEN,
            valid_nodes=("a",),
            broken_links=(("b", "x"),),
            orphan_nodes=("b",),
            cycle_nodes=(),
        )
        d = validation.to_dict()
        assert d["status"] == "broken"
        assert d["valid_nodes"] == ["a"]


# =============================================================================
# Determinism Verification Tests
# =============================================================================

class TestDeterminismVerification:
    """Tests for determinism verification."""
    
    def test_same_nodes_same_hash(self):
        """Identical nodes produce identical hash."""
        nodes = (
            ReplayNode("a", None, 0, "test", {"x": 1}, {"y": 2}, None, None),
        )
        replay1 = TraceReplay(nodes=nodes)
        replay2 = TraceReplay(nodes=nodes)
        
        assert replay1.trace_hash == replay2.trace_hash
    
    def test_different_nodes_different_hash(self):
        """Different nodes produce different hash."""
        replay1 = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {"x": 1}, {}, None, None),
        ))
        replay2 = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {"x": 2}, {}, None, None),
        ))
        
        assert replay1.trace_hash != replay2.trace_hash
    
    def test_verify_determinism_no_expected(self):
        """Verify determinism without expected hash."""
        replay = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {}, {}, None, None),
        ))
        check = replay.verify_determinism()
        
        assert check.is_deterministic
        assert check.trace_hash == replay.trace_hash
        assert check.expected_hash is None
    
    def test_verify_determinism_matching(self):
        """Verify determinism with matching expected hash."""
        replay = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {}, {}, None, None),
        ))
        check = replay.verify_determinism(expected_hash=replay.trace_hash)
        
        assert check.is_deterministic
        assert len(check.differences) == 0
    
    def test_verify_determinism_mismatch(self):
        """Verify determinism with mismatched expected hash."""
        replay = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {}, {}, None, None),
        ))
        check = replay.verify_determinism(expected_hash="wrong_hash")
        
        assert not check.is_deterministic
        assert check.expected_hash == "wrong_hash"
        assert len(check.differences) > 0
    
    def test_compare_with_identical(self, sample_replay_nodes):
        """Compare identical traces."""
        replay1 = TraceReplay(nodes=sample_replay_nodes)
        replay2 = TraceReplay(nodes=sample_replay_nodes)
        
        differences = replay1.compare_with(replay2)
        assert len(differences) == 0
    
    def test_compare_with_different(self):
        """Compare different traces."""
        replay1 = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {"x": 1}, {}, None, None),
        ))
        replay2 = TraceReplay(nodes=(
            ReplayNode("a", None, 0, "test", {"x": 2}, {}, None, None),
            ReplayNode("b", "a", 1, "test", {}, {}, None, None),
        ))
        
        differences = replay1.compare_with(replay2)
        assert len(differences) > 0
        assert any("count" in diff.lower() for diff in differences)
    
    def test_determinism_check_to_dict(self):
        """DeterminismCheck converts to dict."""
        check = DeterminismCheck(
            is_deterministic=True,
            trace_hash="abc123",
            expected_hash="abc123",
            differences=(),
        )
        d = check.to_dict()
        assert d["is_deterministic"] is True
        assert d["trace_hash"] == "abc123"


# =============================================================================
# Decision Reconstruction Tests
# =============================================================================

class TestDecisionReconstruction:
    """Tests for reconstructing decision sequences."""
    
    def test_reconstruct_decisions(self, sample_replay_nodes):
        """Reconstruct decision sequence."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        decisions = replay.reconstruct_decisions()
        
        assert len(decisions) == 3
        assert decisions[0]["id"] == "a"
        assert decisions[0]["inputs"] == {"x": 1}
        assert decisions[1]["id"] == "b"
        assert decisions[2]["depth"] == 2
    
    def test_reconstruct_decisions_with_reasons(self):
        """Reconstruct includes reason fields."""
        nodes = (
            ReplayNode(
                "n1", None, 0, "decision",
                {"query": "test"},
                {"selected_option": "A"},
                {"rationale": "Because A", "causal_reason": {"reason": "init"}},
                None,
            ),
        )
        replay = TraceReplay(nodes=nodes)
        decisions = replay.reconstruct_decisions()
        
        assert decisions[0]["rationale"] == "Because A"
        assert decisions[0]["causal_reason"] == {"reason": "init"}
    
    def test_reconstruct_decisions_jsonl_format(self):
        """Reconstruct from JSONL-format reasons."""
        nodes = (
            ReplayNode(
                "n1", None, 0, "set",
                {"key": "ssl"},
                {"ssl": True},
                {"type": "constraint", "source": "ir", "ref": "context.ssl", "value": True},
                1000.0,
            ),
        )
        replay = TraceReplay(nodes=nodes)
        decisions = replay.reconstruct_decisions()
        
        assert decisions[0]["reason_type"] == "constraint"
        assert decisions[0]["reason_source"] == "ir"
        assert decisions[0]["reason_ref"] == "context.ssl"
    
    def test_reconstruct_trace_from_decisions(self, sample_trace):
        """Reconstruct Trace primitive from replay."""
        replay = TraceReplay.from_trace(sample_trace)
        reconstructed = replay.reconstruct_trace()
        
        assert reconstructed is not None
        assert len(reconstructed) == len(sample_trace)
    
    def test_reconstruct_trace_non_decision_returns_none(self):
        """Cannot reconstruct Trace from non-decision actions."""
        nodes = (
            ReplayNode("n1", None, 0, "set", {}, {"x": 1}, None, None),
        )
        replay = TraceReplay(nodes=nodes)
        reconstructed = replay.reconstruct_trace()
        
        assert reconstructed is None


# =============================================================================
# Human-Readable Output Tests
# =============================================================================

class TestHumanReadableOutput:
    """Tests for human-readable explanations."""
    
    def test_explain_basic(self, sample_replay_nodes):
        """Generate basic explanation."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        explanation = replay.explain()
        
        assert explanation.decision_count == 3
        assert explanation.chain_depth == 2
        assert len(explanation.steps) == 3
    
    def test_explain_empty_trace(self):
        """Explain empty trace."""
        replay = TraceReplay(nodes=())
        explanation = replay.explain()
        
        assert explanation.decision_count == 0
        assert "empty" in explanation.summary.lower()
        assert len(explanation.steps) == 0
    
    def test_explain_to_text(self, sample_replay_nodes):
        """Explanation converts to text."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        text = replay.explain().to_text()
        
        assert "Total decisions: 3" in text
        assert "Chain depth: 2" in text
        assert "Decision sequence:" in text
    
    def test_explain_to_dict(self, sample_replay_nodes):
        """Explanation converts to dict."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        d = replay.explain().to_dict()
        
        assert d["decision_count"] == 3
        assert len(d["steps"]) == 3
    
    def test_to_text(self, sample_replay_nodes):
        """TraceReplay.to_text() works."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        text = replay.to_text()
        
        assert "Trace" in text
        assert "decisions" in text.lower()
    
    def test_to_markdown(self, sample_replay_nodes):
        """TraceReplay.to_markdown() works."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        md = replay.to_markdown()
        
        assert "# Trace Replay Report" in md
        assert "## Statistics" in md
        assert "## Chain Validation" in md
        assert "## Decision Sequence" in md
        assert "**VALID**" in md
    
    def test_str_representation(self, sample_replay_nodes):
        """String representation is human-readable."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        s = str(replay)
        assert "decisions" in s.lower() or "trace" in s.lower()
    
    def test_repr_representation(self, sample_replay_nodes):
        """Repr is developer-friendly."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        r = repr(replay)
        assert "TraceReplay" in r
        assert "nodes=3" in r


# =============================================================================
# Lookup Tests
# =============================================================================

class TestLookup:
    """Tests for node lookup operations."""
    
    def test_get_node(self, sample_replay_nodes):
        """Get node by ID."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        node = replay.get_node("b")
        assert node is not None
        assert node.node_id == "b"
    
    def test_get_node_not_found(self, sample_replay_nodes):
        """Get non-existent node returns None."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        assert replay.get_node("nonexistent") is None
    
    def test_get_children(self, sample_replay_nodes):
        """Get direct children of a node."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        children = replay.get_children("a")
        assert len(children) == 1
        assert children[0].node_id == "b"
    
    def test_get_children_none(self, sample_replay_nodes):
        """Leaf node has no children."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        children = replay.get_children("c")
        assert len(children) == 0
    
    def test_get_ancestors(self, sample_replay_nodes):
        """Get ancestors from root to parent."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        ancestors = replay.get_ancestors("c")
        assert len(ancestors) == 2
        assert ancestors[0].node_id == "a"  # root first
        assert ancestors[1].node_id == "b"  # then parent
    
    def test_get_ancestors_root(self, sample_replay_nodes):
        """Root node has no ancestors."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        ancestors = replay.get_ancestors("a")
        assert len(ancestors) == 0
    
    def test_get_descendants(self, sample_replay_nodes):
        """Get all descendants of a node."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        descendants = replay.get_descendants("a")
        assert len(descendants) == 2
        ids = {d.node_id for d in descendants}
        assert "b" in ids
        assert "c" in ids
    
    def test_iteration(self, sample_replay_nodes):
        """Iterate over nodes."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        node_ids = [n.node_id for n in replay]
        assert node_ids == ["a", "b", "c"]
    
    def test_indexing(self, sample_replay_nodes):
        """Access nodes by index."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        assert replay[0].node_id == "a"
        assert replay[1].node_id == "b"
        assert replay[-1].node_id == "c"
    
    def test_len(self, sample_replay_nodes):
        """Length returns node count."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        assert len(replay) == 3


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization methods."""
    
    def test_to_dict(self, sample_replay_nodes):
        """Convert to dictionary."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        d = replay.to_dict()
        
        assert "trace_hash" in d
        assert d["node_count"] == 3
        assert len(d["nodes"]) == 3
        assert "root_nodes" in d
    
    def test_to_json(self, sample_replay_nodes):
        """Serialize to JSON."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        j = replay.to_json()
        
        data = json.loads(j)
        assert data["node_count"] == 3
    
    def test_to_json_with_indent(self, sample_replay_nodes):
        """Serialize to indented JSON."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        j = replay.to_json(indent=2)
        
        assert "\n" in j
        assert "  " in j
    
    def test_to_jsonl(self, sample_replay_nodes):
        """Serialize to JSONL format."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        jsonl = replay.to_jsonl()
        
        lines = jsonl.strip().split('\n')
        assert len(lines) == 3
        
        for line in lines:
            data = json.loads(line)
            assert "id" in data
    
    def test_json_roundtrip(self, sample_replay_nodes):
        """JSON roundtrip preserves data."""
        replay1 = TraceReplay(nodes=sample_replay_nodes)
        j = replay1.to_json()
        
        # Note: from_json expects Trace format, not TraceReplay format
        # So we test via from_dict with nodes list
        data = json.loads(j)
        replay2 = TraceReplay(nodes=tuple(
            ReplayNode.from_dict(n) for n in data["nodes"]
        ))
        
        assert len(replay1) == len(replay2)
        assert replay1.trace_hash == replay2.trace_hash


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with real trace file."""
    
    def test_load_existing_trace_file(self):
        """Load the existing trace.jsonl file."""
        trace_path = Path(__file__).parent / "trace.jsonl"
        if not trace_path.exists():
            pytest.skip("trace.jsonl not found")
        
        replay = TraceReplay.load_jsonl(trace_path)
        
        assert len(replay) > 0
        validation = replay.validate_chain()
        assert validation.is_valid
    
    def test_full_audit_workflow(self, sample_trace):
        """Complete audit workflow."""
        # 1. Create trace from decisions
        replay = TraceReplay.from_trace(sample_trace)
        
        # 2. Validate chain
        validation = replay.validate_chain()
        assert validation.is_valid
        
        # 3. Get explanation
        explanation = replay.explain()
        assert explanation.decision_count == 3
        
        # 4. Verify determinism
        check = replay.verify_determinism()
        assert check.is_deterministic
        
        # 5. Check can reconstruct
        reconstructed = replay.reconstruct_trace()
        assert reconstructed is not None
        assert len(reconstructed) == 3
    
    def test_compliance_report_generation(self, sample_replay_nodes):
        """Generate compliance report."""
        replay = TraceReplay(nodes=sample_replay_nodes)
        
        # Generate various outputs
        text = replay.to_text()
        md = replay.to_markdown()
        decisions = replay.reconstruct_decisions()
        
        assert len(text) > 0
        assert "# Trace Replay Report" in md
        assert len(decisions) == 3
        
        # All outputs are deterministic
        assert replay.to_text() == text
        assert replay.to_markdown() == md


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error conditions."""
    
    def test_trace_load_error_format(self):
        """TraceLoadError has proper format."""
        error = TraceLoadError("test message", source="file.jsonl")
        assert "TR001" in str(error)
        assert "file.jsonl" in str(error)
    
    def test_broken_chain_error_format(self):
        """BrokenCausalChainError has proper format."""
        error = BrokenCausalChainError(
            "test",
            node_id="n1",
            expected_parent="n0",
        )
        assert "TR002" in str(error)
        assert error.node_id == "n1"
    
    def test_determinism_error_format(self):
        """DeterminismError has proper format."""
        error = DeterminismError(
            "hash mismatch",
            expected_hash="abc",
            actual_hash="def",
        )
        assert "TR003" in str(error)
        assert error.expected_hash == "abc"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_node_trace(self):
        """Trace with single node."""
        nodes = (ReplayNode("a", None, 0, "test", {}, {}, None, None),)
        replay = TraceReplay(nodes=nodes)
        
        assert len(replay) == 1
        assert replay.validate_chain().is_valid
        assert replay.explain().decision_count == 1
    
    def test_empty_trace(self):
        """Empty trace handling."""
        replay = TraceReplay(nodes=())
        
        assert len(replay) == 0
        assert replay.validate_chain().is_valid
        assert replay.explain().decision_count == 0
        assert replay.to_json() is not None
    
    def test_unicode_content(self):
        """Unicode in node content."""
        nodes = (
            ReplayNode(
                "n1", None, 0, "decision",
                {"query": "日本語テスト"},
                {"selected": "選択肢A"},
                {"rationale": "理由"},
                None,
            ),
        )
        replay = TraceReplay(nodes=nodes)
        
        j = replay.to_json()
        assert "日本語" in j
        
        explanation = replay.explain()
        assert len(explanation.steps) == 1
    
    def test_deep_chain(self):
        """Deep causal chain."""
        nodes = []
        prev_id = None
        for i in range(100):
            node = ReplayNode(
                f"n{i}", prev_id, i, "test", {}, {}, None, None
            )
            nodes.append(node)
            prev_id = f"n{i}"
        
        replay = TraceReplay(nodes=tuple(nodes))
        
        assert len(replay) == 100
        validation = replay.validate_chain()
        assert validation.is_valid
        
        ancestors = replay.get_ancestors("n99")
        assert len(ancestors) == 99
    
    def test_wide_tree(self):
        """Wide tree with many children per node."""
        nodes = [ReplayNode("root", None, 0, "test", {}, {}, None, None)]
        for i in range(50):
            nodes.append(
                ReplayNode(f"child{i}", "root", 1, "test", {}, {}, None, None)
            )
        
        replay = TraceReplay(nodes=tuple(nodes))
        
        assert len(replay) == 51
        children = replay.get_children("root")
        assert len(children) == 50
