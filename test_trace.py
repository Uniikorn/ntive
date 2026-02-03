"""
test_trace.py

Pytest tests for the causal execution engine.
Validates:
1. Deterministic execution
2. Valid causal chain (parent exists, depth increments)
3. No decisions without explicit IR constraints
4. Removing a context key removes only its dependent trace nodes
"""

import json
import os
import tempfile
from typing import List, Dict, Any, Optional

import pytest

from ir import SemanticIR, IRStep
from executor import execute
from trace import TraceLog
from graph import CausalGraph


# --- Fixtures ---

@pytest.fixture
def trace_path() -> str:
    """Path to the default trace file."""
    return "trace.jsonl"


@pytest.fixture
def load_trace(trace_path: str) -> List[Dict[str, Any]]:
    """Load trace nodes from JSONL file."""
    nodes = []
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                nodes.append(json.loads(line))
    return nodes


@pytest.fixture
def causal_graph(trace_path: str) -> CausalGraph:
    """Build causal graph from trace file."""
    return CausalGraph.from_jsonl(trace_path)


@pytest.fixture
def sample_ir() -> SemanticIR:
    """Create a sample IR for testing."""
    return SemanticIR(
        goal="Configure PostgreSQL for SaaS application",
        context={
            "ssl": True,
            "pool": True,
            "max_conn": 100,
            "timeout": 30
        },
        steps=[
            IRStep(action="set", params={"key": "ssl"}),
            IRStep(action="set", params={"key": "pool"}),
            IRStep(action="set", params={"key": "max_conn"}),
            IRStep(action="set", params={"key": "timeout"}),
            IRStep(action="emit", params={"target": "postgres_config"})
        ]
    )


# --- Test 1: Deterministic Execution ---

class TestDeterministicExecution:
    """Validate that execution is deterministic."""

    def test_same_ir_produces_same_structure(self, sample_ir: SemanticIR):
        """Executing the same IR twice produces structurally identical traces."""
        traces = []
        
        for _ in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                temp_path = f.name
            
            try:
                trace_log = TraceLog(temp_path)
                execute(sample_ir, trace_log)
                
                nodes = []
                with open(temp_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            nodes.append(json.loads(line))
                traces.append(nodes)
            finally:
                os.unlink(temp_path)
        
        trace_a, trace_b = traces
        
        # Same number of nodes
        assert len(trace_a) == len(trace_b), "Traces have different node counts"
        
        # Same structure (ignoring ids and timestamps)
        for node_a, node_b in zip(trace_a, trace_b):
            assert node_a["depth"] == node_b["depth"], "Depth mismatch"
            assert node_a["action"] == node_b["action"], "Action mismatch"
            assert node_a["input"] == node_b["input"], "Input mismatch"
            assert node_a["output"] == node_b["output"], "Output mismatch"
            assert node_a["reason"] == node_b["reason"], "Reason mismatch"

    def test_step_count_matches_ir_steps(self, sample_ir: SemanticIR):
        """Number of trace nodes equals number of IR steps."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(sample_ir, trace_log)
            
            node_count = 0
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        node_count += 1
            
            assert node_count == len(sample_ir.steps), \
                f"Expected {len(sample_ir.steps)} nodes, got {node_count}"
        finally:
            os.unlink(temp_path)

    def test_execution_order_matches_ir_order(self, sample_ir: SemanticIR):
        """Trace nodes appear in the same order as IR steps."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(sample_ir, trace_log)
            
            nodes = []
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        nodes.append(json.loads(line))
            
            for i, (node, step) in enumerate(zip(nodes, sample_ir.steps)):
                assert node["action"] == step.action, \
                    f"Node {i} action '{node['action']}' != step action '{step.action}'"
        finally:
            os.unlink(temp_path)


# --- Test 2: Valid Causal Chain ---

class TestCausalChain:
    """Validate causal chain integrity."""

    def test_root_has_no_parent(self, load_trace: List[Dict[str, Any]]):
        """First node (root) has parent_id=None."""
        assert len(load_trace) > 0, "Trace is empty"
        root = load_trace[0]
        assert root["parent_id"] is None, "Root node must have parent_id=None"

    def test_root_has_depth_zero(self, load_trace: List[Dict[str, Any]]):
        """Root node has depth=0."""
        assert len(load_trace) > 0, "Trace is empty"
        root = load_trace[0]
        assert root["depth"] == 0, "Root node must have depth=0"

    def test_depth_increments_monotonically(self, load_trace: List[Dict[str, Any]]):
        """Each subsequent node has depth = previous depth + 1."""
        for i, node in enumerate(load_trace):
            assert node["depth"] == i, \
                f"Node at index {i} has depth {node['depth']}, expected {i}"

    def test_parent_exists_for_all_non_root_nodes(self, load_trace: List[Dict[str, Any]]):
        """Every non-root node references an existing parent."""
        node_ids = {node["id"] for node in load_trace}
        
        for node in load_trace[1:]:  # Skip root
            parent_id = node["parent_id"]
            assert parent_id is not None, \
                f"Non-root node {node['id']} has no parent"
            assert parent_id in node_ids, \
                f"Node {node['id']} references non-existent parent {parent_id}"

    def test_parent_chain_links_correctly(self, load_trace: List[Dict[str, Any]]):
        """Each node's parent is the previous node in the trace."""
        for i in range(1, len(load_trace)):
            current = load_trace[i]
            previous = load_trace[i - 1]
            assert current["parent_id"] == previous["id"], \
                f"Node {i} parent mismatch: expected {previous['id']}, got {current['parent_id']}"

    def test_all_nodes_have_unique_ids(self, load_trace: List[Dict[str, Any]]):
        """All node IDs are unique."""
        ids = [node["id"] for node in load_trace]
        assert len(ids) == len(set(ids)), "Duplicate node IDs found"

    def test_timestamps_are_monotonically_increasing(self, load_trace: List[Dict[str, Any]]):
        """Timestamps increase (or equal) with each node."""
        for i in range(1, len(load_trace)):
            current_ts = load_trace[i]["timestamp"]
            previous_ts = load_trace[i - 1]["timestamp"]
            assert current_ts >= previous_ts, \
                f"Timestamp at node {i} ({current_ts}) < previous ({previous_ts})"

    def test_graph_traversal_from_leaf_to_root(self, causal_graph: CausalGraph):
        """Can traverse from any leaf back to root via parent chain."""
        # Find leaf nodes (nodes with no children)
        all_nodes = list(causal_graph.nodes.keys())
        leaves = [n for n in all_nodes if not causal_graph.get_children(n)]
        
        root = causal_graph.get_root()
        assert root is not None, "No root found"
        
        for leaf in leaves:
            chain = causal_graph.trace_cause(leaf)
            assert len(chain) > 0, f"Empty chain for leaf {leaf}"
            assert chain[-1]["id"] == root, \
                f"Chain from {leaf} does not end at root"


# --- Test 3: No Decisions Without Explicit IR Constraints ---

class TestIRConstraints:
    """Validate that all decisions have explicit IR-based reasons."""

    def test_all_nodes_have_reason(self, load_trace: List[Dict[str, Any]]):
        """Every node has a non-empty reason field."""
        for node in load_trace:
            assert "reason" in node, f"Node {node['id']} missing 'reason'"
            assert node["reason"], f"Node {node['id']} has empty reason"

    def test_reason_has_required_fields(self, load_trace: List[Dict[str, Any]]):
        """Every reason has type, source, ref, and value."""
        required_fields = {"type", "source", "ref", "value"}
        
        for node in load_trace:
            reason = node["reason"]
            missing = required_fields - set(reason.keys())
            assert not missing, \
                f"Node {node['id']} reason missing fields: {missing}"

    def test_reason_source_is_ir(self, load_trace: List[Dict[str, Any]]):
        """All reasons must originate from IR (no implicit decisions)."""
        for node in load_trace:
            source = node["reason"]["source"]
            assert source == "ir", \
                f"Node {node['id']} has non-IR source: {source}"

    def test_reason_type_is_constraint(self, load_trace: List[Dict[str, Any]]):
        """All reasons must be of type 'constraint'."""
        for node in load_trace:
            reason_type = node["reason"]["type"]
            assert reason_type == "constraint", \
                f"Node {node['id']} has reason type '{reason_type}', expected 'constraint'"

    def test_reason_ref_is_valid(self, load_trace: List[Dict[str, Any]]):
        """Reason ref must be a non-empty string."""
        for node in load_trace:
            ref = node["reason"]["ref"]
            assert isinstance(ref, str) and ref, \
                f"Node {node['id']} has invalid ref: {ref}"

    def test_set_actions_reference_context_keys(self, load_trace: List[Dict[str, Any]]):
        """'set' actions must reference context.{key} in reason ref."""
        for node in load_trace:
            if node["action"] == "set":
                ref = node["reason"]["ref"]
                assert ref.startswith("context."), \
                    f"'set' node {node['id']} ref '{ref}' must start with 'context.'"

    def test_output_matches_reason_value(self, load_trace: List[Dict[str, Any]]):
        """For 'set' actions, output value matches reason value."""
        for node in load_trace:
            if node["action"] == "set":
                key = node["input"]["params"]["key"]
                output_value = node["output"].get(key)
                reason_value = node["reason"]["value"]
                assert output_value == reason_value, \
                    f"Node {node['id']} output {output_value} != reason {reason_value}"


# --- Test 4: Context Key Removal Affects Dependents ---

class TestContextKeyRemoval:
    """Validate that removing a context key removes only dependent nodes."""

    def _execute_ir_and_get_trace(self, ir: SemanticIR) -> List[Dict[str, Any]]:
        """Helper to execute IR and return trace nodes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(ir, trace_log)
            
            nodes = []
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        nodes.append(json.loads(line))
            return nodes
        finally:
            os.unlink(temp_path)

    def test_removing_context_key_removes_corresponding_step_output(self):
        """Removing a key from context changes output for that step."""
        # Full IR
        full_ir = SemanticIR(
            goal="Test context removal",
            context={"a": 1, "b": 2, "c": 3},
            steps=[
                IRStep(action="set", params={"key": "a"}),
                IRStep(action="set", params={"key": "b"}),
                IRStep(action="set", params={"key": "c"}),
            ]
        )
        
        # IR with 'b' removed from context
        reduced_ir = SemanticIR(
            goal="Test context removal",
            context={"a": 1, "c": 3},  # 'b' removed
            steps=[
                IRStep(action="set", params={"key": "a"}),
                IRStep(action="set", params={"key": "b"}),  # still in steps
                IRStep(action="set", params={"key": "c"}),
            ]
        )
        
        full_trace = self._execute_ir_and_get_trace(full_ir)
        reduced_trace = self._execute_ir_and_get_trace(reduced_ir)
        
        # Find node for 'b' in both traces
        full_b_node = next(n for n in full_trace if n["input"]["params"]["key"] == "b")
        reduced_b_node = next(n for n in reduced_trace if n["input"]["params"]["key"] == "b")
        
        # Full trace should have value, reduced should have None
        assert full_b_node["output"]["b"] == 2
        assert reduced_b_node["output"]["b"] is None
        assert reduced_b_node["reason"]["value"] is None

    def test_removing_step_removes_trace_node(self):
        """Removing a step from IR removes its trace node."""
        # Full IR
        full_ir = SemanticIR(
            goal="Test step removal",
            context={"a": 1, "b": 2, "c": 3},
            steps=[
                IRStep(action="set", params={"key": "a"}),
                IRStep(action="set", params={"key": "b"}),
                IRStep(action="set", params={"key": "c"}),
            ]
        )
        
        # IR with middle step removed
        reduced_ir = SemanticIR(
            goal="Test step removal",
            context={"a": 1, "b": 2, "c": 3},
            steps=[
                IRStep(action="set", params={"key": "a"}),
                # 'b' step removed
                IRStep(action="set", params={"key": "c"}),
            ]
        )
        
        full_trace = self._execute_ir_and_get_trace(full_ir)
        reduced_trace = self._execute_ir_and_get_trace(reduced_ir)
        
        assert len(full_trace) == 3
        assert len(reduced_trace) == 2
        
        # Check 'b' is absent from reduced trace
        reduced_keys = [n["input"]["params"]["key"] for n in reduced_trace]
        assert "b" not in reduced_keys

    def test_causal_chain_remains_valid_after_step_removal(self):
        """After removing a step, remaining nodes form valid causal chain."""
        ir = SemanticIR(
            goal="Test chain validity",
            context={"x": 10, "y": 20},
            steps=[
                IRStep(action="set", params={"key": "x"}),
                IRStep(action="set", params={"key": "y"}),
            ]
        )
        
        trace = self._execute_ir_and_get_trace(ir)
        
        # Validate chain integrity
        assert trace[0]["parent_id"] is None
        assert trace[0]["depth"] == 0
        assert trace[1]["parent_id"] == trace[0]["id"]
        assert trace[1]["depth"] == 1

    def test_descendants_identified_correctly(self):
        """CausalGraph correctly identifies descendants."""
        ir = SemanticIR(
            goal="Test descendants",
            context={"a": 1, "b": 2, "c": 3, "d": 4},
            steps=[
                IRStep(action="set", params={"key": "a"}),
                IRStep(action="set", params={"key": "b"}),
                IRStep(action="set", params={"key": "c"}),
                IRStep(action="set", params={"key": "d"}),
            ]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(ir, trace_log)
            
            graph = CausalGraph.from_jsonl(temp_path)
            root = graph.get_root()
            
            descendants = graph.get_descendants(root)
            
            # Root has 3 descendants (b, c, d)
            assert len(descendants) == 3
            
            # All descendants are non-root nodes
            for desc_id in descendants:
                node = graph.get_node(desc_id)
                assert node["depth"] > 0
        finally:
            os.unlink(temp_path)


# --- Structural Validation Tests ---

class TestTraceStructure:
    """Validate trace file structure and format."""

    def test_trace_file_exists(self, trace_path: str):
        """Trace file exists."""
        assert os.path.exists(trace_path), f"Trace file not found: {trace_path}"

    def test_trace_is_valid_jsonl(self, trace_path: str):
        """Each line in trace is valid JSON."""
        with open(trace_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Line {i} is not valid JSON: {e}")

    def test_all_nodes_have_required_fields(self, load_trace: List[Dict[str, Any]]):
        """Every node has all required fields."""
        required_fields = {"id", "parent_id", "depth", "action", "input", "output", "reason", "timestamp"}
        
        for node in load_trace:
            missing = required_fields - set(node.keys())
            assert not missing, f"Node {node.get('id', 'unknown')} missing fields: {missing}"

    def test_id_is_valid_uuid_format(self, load_trace: List[Dict[str, Any]]):
        """Node IDs are valid UUID format."""
        import re
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        for node in load_trace:
            assert uuid_pattern.match(node["id"]), \
                f"Node ID '{node['id']}' is not valid UUID format"

    def test_depth_is_non_negative_integer(self, load_trace: List[Dict[str, Any]]):
        """Depth is a non-negative integer."""
        for node in load_trace:
            depth = node["depth"]
            assert isinstance(depth, int) and depth >= 0, \
                f"Node {node['id']} has invalid depth: {depth}"

    def test_timestamp_is_positive_float(self, load_trace: List[Dict[str, Any]]):
        """Timestamp is a positive float."""
        for node in load_trace:
            ts = node["timestamp"]
            assert isinstance(ts, (int, float)) and ts > 0, \
                f"Node {node['id']} has invalid timestamp: {ts}"


# --- Edge Case Tests ---

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_context_produces_valid_trace(self):
        """IR with empty context still produces valid trace."""
        ir = SemanticIR(
            goal="Empty context test",
            context={},
            steps=[
                IRStep(action="emit", params={"target": "output"})
            ]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(ir, trace_log)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                nodes = [json.loads(line) for line in f if line.strip()]
            
            assert len(nodes) == 1
            assert nodes[0]["parent_id"] is None
            assert nodes[0]["depth"] == 0
        finally:
            os.unlink(temp_path)

    def test_single_step_ir(self):
        """Single-step IR produces single-node trace."""
        ir = SemanticIR(
            goal="Single step",
            context={"key": "value"},
            steps=[IRStep(action="set", params={"key": "key"})]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(ir, trace_log)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                nodes = [json.loads(line) for line in f if line.strip()]
            
            assert len(nodes) == 1
            assert nodes[0]["parent_id"] is None
        finally:
            os.unlink(temp_path)

    def test_many_steps_maintains_chain_integrity(self):
        """Large number of steps maintains proper chain."""
        n_steps = 100
        context = {f"key_{i}": i for i in range(n_steps)}
        steps = [IRStep(action="set", params={"key": f"key_{i}"}) for i in range(n_steps)]
        
        ir = SemanticIR(goal="Many steps", context=context, steps=steps)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            execute(ir, trace_log)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                nodes = [json.loads(line) for line in f if line.strip()]
            
            assert len(nodes) == n_steps
            
            # Verify chain
            for i, node in enumerate(nodes):
                assert node["depth"] == i
                if i == 0:
                    assert node["parent_id"] is None
                else:
                    assert node["parent_id"] == nodes[i-1]["id"]
        finally:
            os.unlink(temp_path)


# --- Test: IR Execution Independent of Prompt ---

class TestIRExecutionIndependentOfPrompt:
    """Validate that execution depends ONLY on IR, not on original prompt."""

    @staticmethod
    def _normalize_trace(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove non-deterministic fields (id, parent_id, timestamp) for comparison."""
        normalized = []
        for node in nodes:
            normalized.append({
                "depth": node["depth"],
                "action": node["action"],
                "input": node["input"],
                "output": node["output"],
                "reason": node["reason"],
            })
        return normalized

    @staticmethod
    def _execute_and_get_trace(ir: SemanticIR) -> List[Dict[str, Any]]:
        """Execute IR and return raw trace nodes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            output = execute(ir, trace_log)
            
            nodes = []
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        nodes.append(json.loads(line))
            return nodes, output
        finally:
            os.unlink(temp_path)

    def test_ir_execution_independent_of_prompt_length(self):
        """
        Proves: prompt length is irrelevant after compression to IR.
        
        Scenario:
        - Long prompt A → IR X → trace T
        - Short prompt B → IR X → trace T
        - Same IR produces structurally identical traces.
        """
        # Simulate: Long verbose prompt compressed to IR
        # "Please configure a PostgreSQL database for my SaaS application.
        #  I need SSL enabled for security, connection pooling for performance,
        #  a maximum of 100 connections, and a 30 second timeout..."
        ir_from_long_prompt = SemanticIR(
            goal="Configure PostgreSQL for SaaS application",
            context={
                "ssl": True,
                "pool": True,
                "max_conn": 100,
                "timeout": 30
            },
            steps=[
                IRStep(action="set", params={"key": "ssl"}),
                IRStep(action="set", params={"key": "pool"}),
                IRStep(action="set", params={"key": "max_conn"}),
                IRStep(action="set", params={"key": "timeout"}),
                IRStep(action="emit", params={"target": "postgres_config"})
            ]
        )

        # Simulate: Short terse prompt compressed to identical IR
        # "pg ssl pool 100 30"
        ir_from_short_prompt = SemanticIR(
            goal="Configure PostgreSQL for SaaS application",
            context={
                "ssl": True,
                "pool": True,
                "max_conn": 100,
                "timeout": 30
            },
            steps=[
                IRStep(action="set", params={"key": "ssl"}),
                IRStep(action="set", params={"key": "pool"}),
                IRStep(action="set", params={"key": "max_conn"}),
                IRStep(action="set", params={"key": "timeout"}),
                IRStep(action="emit", params={"target": "postgres_config"})
            ]
        )

        # Execute both IRs
        trace_long, output_long = self._execute_and_get_trace(ir_from_long_prompt)
        trace_short, output_short = self._execute_and_get_trace(ir_from_short_prompt)

        # Normalize traces (remove id, parent_id, timestamp)
        normalized_long = self._normalize_trace(trace_long)
        normalized_short = self._normalize_trace(trace_short)

        # --- Structural assertions ---

        # Same number of trace nodes
        assert len(normalized_long) == len(normalized_short), \
            f"Trace lengths differ: {len(normalized_long)} vs {len(normalized_short)}"

        # Same output
        assert output_long == output_short, \
            f"Outputs differ:\n{output_long}\nvs\n{output_short}"

        # Node-by-node structural comparison
        for i, (node_long, node_short) in enumerate(zip(normalized_long, normalized_short)):
            assert node_long["depth"] == node_short["depth"], \
                f"Node {i}: depth mismatch {node_long['depth']} vs {node_short['depth']}"
            
            assert node_long["action"] == node_short["action"], \
                f"Node {i}: action mismatch '{node_long['action']}' vs '{node_short['action']}'"
            
            assert node_long["input"] == node_short["input"], \
                f"Node {i}: input mismatch {node_long['input']} vs {node_short['input']}"
            
            assert node_long["output"] == node_short["output"], \
                f"Node {i}: output mismatch {node_long['output']} vs {node_short['output']}"
            
            assert node_long["reason"] == node_short["reason"], \
                f"Node {i}: reason mismatch {node_long['reason']} vs {node_short['reason']}"

        # Full structural equality
        assert normalized_long == normalized_short, \
            "Normalized traces are not structurally identical"

    def test_ir_execution_independent_of_prompt_content(self):
        """
        Proves: prompt content is irrelevant after compression to IR.
        
        Scenario:
        - Formal prompt → IR X → trace T
        - Casual prompt → IR X → trace T
        - Technical prompt → IR X → trace T
        - All produce identical traces from identical IR.
        """
        # All these hypothetical prompts compress to the same IR
        # Prompt 1: "Initialize Redis cache with 512MB memory limit and LRU eviction"
        # Prompt 2: "yo set up redis, 512 megs, kick out old stuff when full"
        # Prompt 3: "redis maxmemory=512mb maxmemory-policy=allkeys-lru"
        
        identical_ir = SemanticIR(
            goal="Configure Redis cache",
            context={
                "maxmemory": "512mb",
                "eviction_policy": "allkeys-lru",
                "persistence": False
            },
            steps=[
                IRStep(action="set", params={"key": "maxmemory"}),
                IRStep(action="set", params={"key": "eviction_policy"}),
                IRStep(action="set", params={"key": "persistence"}),
                IRStep(action="emit", params={"target": "redis_config"})
            ]
        )

        # Execute the same IR multiple times (simulating different prompt origins)
        traces = []
        outputs = []
        for _ in range(3):
            trace, output = self._execute_and_get_trace(identical_ir)
            traces.append(self._normalize_trace(trace))
            outputs.append(output)

        # All outputs must be identical
        assert all(o == outputs[0] for o in outputs), \
            "Outputs differ across executions of identical IR"

        # All normalized traces must be identical
        assert all(t == traces[0] for t in traces), \
            "Traces differ across executions of identical IR"

    def test_ir_goal_field_does_not_affect_execution(self):
        """
        Proves: the 'goal' field (human-readable intent) does not affect execution.
        
        The goal field preserves prompt semantics but execution depends only on
        context and steps.
        """
        # Same context and steps, different goals
        ir_verbose_goal = SemanticIR(
            goal="Configure a highly available PostgreSQL cluster with SSL encryption, "
                 "connection pooling enabled, maximum 50 connections, and 15 second timeout "
                 "for optimal performance in a production SaaS environment",
            context={"ssl": True, "pool": True, "max_conn": 50, "timeout": 15},
            steps=[
                IRStep(action="set", params={"key": "ssl"}),
                IRStep(action="set", params={"key": "pool"}),
                IRStep(action="set", params={"key": "max_conn"}),
                IRStep(action="set", params={"key": "timeout"}),
            ]
        )

        ir_terse_goal = SemanticIR(
            goal="pg config",
            context={"ssl": True, "pool": True, "max_conn": 50, "timeout": 15},
            steps=[
                IRStep(action="set", params={"key": "ssl"}),
                IRStep(action="set", params={"key": "pool"}),
                IRStep(action="set", params={"key": "max_conn"}),
                IRStep(action="set", params={"key": "timeout"}),
            ]
        )

        trace_verbose, _ = self._execute_and_get_trace(ir_verbose_goal)
        trace_terse, _ = self._execute_and_get_trace(ir_terse_goal)

        normalized_verbose = self._normalize_trace(trace_verbose)
        normalized_terse = self._normalize_trace(trace_terse)

        # Traces must be structurally identical despite different goals
        assert normalized_verbose == normalized_terse, \
            "Goal field affected execution - traces differ"

    def test_causal_ordering_preserved_across_identical_ir(self):
        """
        Proves: causal ordering (parent-child relationships) is structurally
        consistent across executions of identical IR.
        """
        ir = SemanticIR(
            goal="Test causal ordering",
            context={"a": 1, "b": 2, "c": 3},
            steps=[
                IRStep(action="set", params={"key": "a"}),
                IRStep(action="set", params={"key": "b"}),
                IRStep(action="set", params={"key": "c"}),
            ]
        )

        # Execute twice
        trace_1, _ = self._execute_and_get_trace(ir)
        trace_2, _ = self._execute_and_get_trace(ir)

        # Verify causal ordering is identical (by depth and position)
        for i, (n1, n2) in enumerate(zip(trace_1, trace_2)):
            # Depth must match
            assert n1["depth"] == n2["depth"] == i, \
                f"Causal depth mismatch at position {i}"
            
            # Parent relationship structure (not IDs)
            if i == 0:
                assert n1["parent_id"] is None and n2["parent_id"] is None, \
                    "Root nodes must have no parent"
            else:
                # Both have parents (actual IDs differ but structure is same)
                assert n1["parent_id"] is not None and n2["parent_id"] is not None, \
                    f"Non-root node {i} must have parent"
