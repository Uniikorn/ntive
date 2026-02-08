"""
test_trace_primitive.py

Unit tests for the Ntive Trace primitive.

Tests prove:
- Validation (Decision instances only)
- Immutability (cannot modify after creation)
- Deterministic serialization (identical inputs → identical JSON/trace_id)
- Parent trace chaining
- Causal reason handling
"""

import pytest
import json
from ntive.decision import Decision, Alternative
from ntive.trace import (
    Trace,
    CausalReason,
    TraceValidationError,
    InvalidTraceNodeError,
    InvalidCausalReasonError,
    TraceImmutabilityError,
    TraceNode,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def decision1():
    """First test decision."""
    return Decision(
        inputs={"query": "open file"},
        selected_option="open_dialog",
        alternatives=[Alternative("recent", "Not mentioned")],
        rationale="User wants to open a file",
        decision_id="decision-001",
        timestamp="2026-02-08T12:00:00Z",
    )


@pytest.fixture
def decision2():
    """Second test decision."""
    return Decision(
        inputs={"file": "test.txt"},
        selected_option="confirm_open",
        alternatives=[],
        rationale="File selected by user",
        decision_id="decision-002",
        timestamp="2026-02-08T12:00:01Z",
    )


@pytest.fixture
def decision3():
    """Third test decision."""
    return Decision(
        inputs={"action": "save"},
        selected_option="save_file",
        alternatives=[Alternative("discard", "User chose save")],
        rationale="User requested save",
        decision_id="decision-003",
        timestamp="2026-02-08T12:00:02Z",
    )


@pytest.fixture
def sample_trace(decision1, decision2):
    """A sample trace with two decisions."""
    return Trace.build([
        (decision1, "Initial request"),
        (decision2, CausalReason("Follows from step 1", "inference")),
    ])


# =============================================================================
# SECTION 1: Validation Tests
# =============================================================================

class TestValidation:
    """Test that only Decision instances are accepted."""
    
    def test_non_decision_in_build_raises_error(self):
        """build() rejects non-Decision items."""
        with pytest.raises(InvalidTraceNodeError) as exc_info:
            Trace.build([{"not": "a decision"}])
        assert exc_info.value.actual_type == "dict"
        assert "T001" in str(exc_info.value)
    
    def test_non_decision_in_tuple_raises_error(self, decision1):
        """build() rejects non-Decision in (item, reason) tuple."""
        with pytest.raises(InvalidTraceNodeError) as exc_info:
            Trace.build([
                decision1,
                ("not a decision", "reason"),
            ])
        assert exc_info.value.index == 1
    
    def test_append_rejects_non_decision(self, sample_trace):
        """append() rejects non-Decision."""
        with pytest.raises(InvalidTraceNodeError) as exc_info:
            sample_trace.append("not a decision")
        assert exc_info.value.actual_type == "str"
    
    def test_append_rejects_dict(self, sample_trace):
        """append() rejects dict (common mistake)."""
        with pytest.raises(InvalidTraceNodeError):
            sample_trace.append({"selected_option": "fake"})
    
    def test_append_rejects_none(self, sample_trace):
        """append() rejects None."""
        with pytest.raises(InvalidTraceNodeError):
            sample_trace.append(None)
    
    def test_valid_decisions_accepted(self, decision1, decision2, decision3):
        """Valid Decision instances are accepted."""
        trace = Trace.build([decision1, decision2, decision3])
        assert len(trace) == 3
    
    def test_empty_trace_is_valid(self):
        """Empty trace is valid."""
        trace = Trace.empty()
        assert len(trace) == 0
        assert trace.trace_id is not None


class TestCausalReasonValidation:
    """Test CausalReason validation."""
    
    def test_empty_reason_rejected(self):
        """Empty reason string is rejected."""
        with pytest.raises(InvalidCausalReasonError):
            CausalReason("")
    
    def test_whitespace_only_reason_rejected(self):
        """Whitespace-only reason is rejected."""
        with pytest.raises(InvalidCausalReasonError):
            CausalReason("   ")
    
    def test_non_string_reason_rejected(self):
        """Non-string reason is rejected."""
        with pytest.raises(InvalidCausalReasonError):
            CausalReason(123)
    
    def test_non_string_category_rejected(self):
        """Non-string category is rejected."""
        with pytest.raises(InvalidCausalReasonError):
            CausalReason("valid", category=123)
    
    def test_valid_reason_accepted(self):
        """Valid reason is accepted."""
        reason = CausalReason("Because X", category="inference")
        assert reason.reason == "Because X"
        assert reason.category == "inference"
    
    def test_reason_without_category(self):
        """Reason without category is valid."""
        reason = CausalReason("Just because")
        assert reason.category is None
    
    def test_invalid_reason_type_in_build(self, decision1):
        """Invalid reason type in build() is rejected."""
        with pytest.raises(InvalidCausalReasonError) as exc_info:
            Trace.build([
                (decision1, 123),  # Invalid reason type
            ])
        assert exc_info.value.index == 0
    
    def test_invalid_reason_type_in_append(self, decision1):
        """Invalid reason type in append() is rejected."""
        trace = Trace.empty()
        with pytest.raises(InvalidCausalReasonError):
            trace.append(decision1, reason=["not", "valid"])


# =============================================================================
# SECTION 2: Immutability Tests
# =============================================================================

class TestImmutability:
    """Test that Trace is immutable after creation."""
    
    def test_cannot_set_attribute(self, sample_trace):
        """Setting any attribute raises TraceImmutabilityError."""
        with pytest.raises(TraceImmutabilityError):
            sample_trace._nodes = ()
    
    def test_cannot_delete_attribute(self, sample_trace):
        """Deleting any attribute raises TraceImmutabilityError."""
        with pytest.raises(TraceImmutabilityError):
            del sample_trace._trace_id
    
    def test_cannot_add_attribute(self, sample_trace):
        """Adding new attribute raises TraceImmutabilityError."""
        with pytest.raises(TraceImmutabilityError):
            sample_trace.new_field = "value"
    
    def test_append_returns_new_trace(self, decision1, decision2):
        """append() returns a new Trace, original unchanged."""
        original = Trace.build([decision1])
        original_id = original.trace_id
        original_len = len(original)
        
        new_trace = original.append(decision2)
        
        # Original unchanged
        assert len(original) == original_len
        assert original.trace_id == original_id
        
        # New trace has new content
        assert len(new_trace) == original_len + 1
        assert new_trace.trace_id != original_id
    
    def test_decisions_property_returns_tuple(self, sample_trace):
        """decisions property returns immutable tuple."""
        decisions = sample_trace.decisions
        assert isinstance(decisions, tuple)
    
    def test_reasons_property_returns_tuple(self, sample_trace):
        """reasons property returns immutable tuple."""
        reasons = sample_trace.reasons
        assert isinstance(reasons, tuple)
    
    def test_nodes_property_returns_tuple(self, sample_trace):
        """nodes property returns immutable tuple."""
        nodes = sample_trace.nodes
        assert isinstance(nodes, tuple)
    
    def test_causal_reason_is_frozen(self):
        """CausalReason is frozen dataclass."""
        reason = CausalReason("test")
        with pytest.raises(AttributeError):
            reason.reason = "changed"


# =============================================================================
# SECTION 3: Deterministic Serialization Tests
# =============================================================================

class TestDeterministicSerialization:
    """Test that serialization is deterministic."""
    
    def test_identical_traces_have_identical_trace_id(self, decision1, decision2):
        """Same decisions + same order = identical trace_id."""
        trace1 = Trace.build([decision1, decision2])
        trace2 = Trace.build([decision1, decision2])
        
        assert trace1.trace_id == trace2.trace_id
    
    def test_identical_traces_have_identical_json(self, decision1, decision2):
        """Same decisions + same order = identical JSON."""
        trace1 = Trace.build([decision1, decision2])
        trace2 = Trace.build([decision1, decision2])
        
        assert trace1.to_json() == trace2.to_json()
    
    def test_different_order_produces_different_trace_id(self, decision1, decision2):
        """Different order = different trace_id."""
        trace1 = Trace.build([decision1, decision2])
        trace2 = Trace.build([decision2, decision1])
        
        assert trace1.trace_id != trace2.trace_id
    
    def test_different_reasons_produce_different_trace_id(self, decision1):
        """Different reasons = different trace_id."""
        trace1 = Trace.build([(decision1, "reason A")])
        trace2 = Trace.build([(decision1, "reason B")])
        
        assert trace1.trace_id != trace2.trace_id
    
    def test_reason_vs_no_reason_produce_different_trace_id(self, decision1):
        """Having a reason vs not = different trace_id."""
        trace1 = Trace.build([decision1])
        trace2 = Trace.build([(decision1, "with reason")])
        
        assert trace1.trace_id != trace2.trace_id
    
    def test_json_keys_are_sorted(self, sample_trace):
        """JSON output has sorted keys."""
        json_str = sample_trace.to_json()
        data = json.loads(json_str)
        keys = list(data.keys())
        assert keys == sorted(keys)
    
    def test_multiple_serializations_identical(self, sample_trace):
        """Multiple calls to to_json() return identical strings."""
        json1 = sample_trace.to_json()
        json2 = sample_trace.to_json()
        json3 = sample_trace.to_json()
        
        assert json1 == json2 == json3
    
    def test_trace_id_is_content_based_hash(self, decision1):
        """trace_id is a hex string (SHA-256)."""
        trace = Trace.build([decision1])
        
        # SHA-256 produces 64 hex characters
        assert len(trace.trace_id) == 64
        assert all(c in '0123456789abcdef' for c in trace.trace_id)
    
    def test_roundtrip_preserves_data(self, sample_trace):
        """from_json(to_json()) preserves trace content."""
        json_str = sample_trace.to_json()
        restored = Trace.from_json(json_str)
        
        assert restored.trace_id == sample_trace.trace_id
        assert len(restored) == len(sample_trace)
        assert restored.to_json() == json_str
    
    def test_empty_trace_has_deterministic_id(self):
        """Empty traces have the same trace_id."""
        trace1 = Trace.empty()
        trace2 = Trace.empty()
        
        assert trace1.trace_id == trace2.trace_id


# =============================================================================
# SECTION 4: Parent Trace Chaining Tests
# =============================================================================

class TestParentTraceChaining:
    """Test parent trace functionality for branching/retries."""
    
    def test_trace_with_parent(self, decision1, decision2, decision3):
        """Trace can have a parent trace."""
        parent = Trace.build([decision1])
        child = Trace.build([decision2, decision3], parent=parent)
        
        assert child.parent is parent
        assert child.parent.trace_id == parent.trace_id
    
    def test_trace_without_parent(self, sample_trace):
        """Trace without parent has None parent."""
        assert sample_trace.parent is None
    
    def test_empty_with_parent(self, sample_trace):
        """Trace.empty() can have a parent."""
        child = Trace.empty(parent=sample_trace)
        assert child.parent is sample_trace
    
    def test_parent_affects_trace_id(self, decision1, decision2):
        """Different parents = different trace_ids (same content)."""
        parent1 = Trace.build([decision1])
        parent2 = Trace.build([decision2])
        
        child1 = Trace.build([decision1], parent=parent1)
        child2 = Trace.build([decision1], parent=parent2)
        
        assert child1.trace_id != child2.trace_id
    
    def test_parent_vs_no_parent_different_trace_id(self, decision1, decision2):
        """Having parent vs no parent = different trace_id."""
        parent = Trace.build([decision1])
        
        with_parent = Trace.build([decision2], parent=parent)
        without_parent = Trace.build([decision2])
        
        assert with_parent.trace_id != without_parent.trace_id
    
    def test_chain_method(self, decision1, decision2, decision3):
        """chain() returns full ancestry from root."""
        root = Trace.build([decision1])
        child = Trace.build([decision2], parent=root)
        grandchild = Trace.build([decision3], parent=child)
        
        chain = grandchild.chain()
        
        assert len(chain) == 3
        assert chain[0] is root
        assert chain[1] is child
        assert chain[2] is grandchild
    
    def test_chain_single_trace(self, sample_trace):
        """chain() on trace without parent returns list with just itself."""
        chain = sample_trace.chain()
        assert chain == [sample_trace]
    
    def test_depth_method(self, decision1, decision2, decision3):
        """depth() returns correct depth in parent chain."""
        root = Trace.build([decision1])
        child = Trace.build([decision2], parent=root)
        grandchild = Trace.build([decision3], parent=child)
        
        assert root.depth() == 0
        assert child.depth() == 1
        assert grandchild.depth() == 2
    
    def test_append_preserves_parent(self, decision1, decision2, decision3):
        """append() preserves parent reference."""
        parent = Trace.build([decision1])
        child = Trace.build([decision2], parent=parent)
        
        extended = child.append(decision3)
        
        assert extended.parent is parent


# =============================================================================
# SECTION 5: Length and Iteration Tests
# =============================================================================

class TestLengthAndIteration:
    """Test length and iteration behavior."""
    
    def test_len_returns_decision_count(self, decision1, decision2, decision3):
        """len() returns number of decisions."""
        trace = Trace.build([decision1, decision2, decision3])
        assert len(trace) == 3
    
    def test_len_empty_trace(self):
        """len() of empty trace is 0."""
        assert len(Trace.empty()) == 0
    
    def test_iteration_returns_decisions(self, decision1, decision2):
        """Iterating yields Decision instances."""
        trace = Trace.build([decision1, decision2])
        
        decisions = list(trace)
        
        assert len(decisions) == 2
        assert decisions[0] is decision1
        assert decisions[1] is decision2
    
    def test_indexing_returns_decision(self, decision1, decision2, decision3):
        """Indexing returns Decision at position."""
        trace = Trace.build([decision1, decision2, decision3])
        
        assert trace[0] is decision1
        assert trace[1] is decision2
        assert trace[2] is decision3
    
    def test_negative_indexing(self, decision1, decision2, decision3):
        """Negative indexing works."""
        trace = Trace.build([decision1, decision2, decision3])
        
        assert trace[-1] is decision3
        assert trace[-2] is decision2
    
    def test_bool_true_for_non_empty(self, sample_trace):
        """Non-empty trace is truthy."""
        assert bool(sample_trace) is True
    
    def test_bool_false_for_empty(self):
        """Empty trace is falsy."""
        assert bool(Trace.empty()) is False
    
    def test_first_returns_first_decision(self, decision1, decision2):
        """first() returns first decision."""
        trace = Trace.build([decision1, decision2])
        assert trace.first() is decision1
    
    def test_first_returns_none_for_empty(self):
        """first() returns None for empty trace."""
        assert Trace.empty().first() is None
    
    def test_last_returns_last_decision(self, decision1, decision2):
        """last() returns last decision."""
        trace = Trace.build([decision1, decision2])
        assert trace.last() is decision2
    
    def test_last_returns_none_for_empty(self):
        """last() returns None for empty trace."""
        assert Trace.empty().last() is None


# =============================================================================
# SECTION 6: Build Methods Tests
# =============================================================================

class TestBuildMethods:
    """Test various ways to build a Trace."""
    
    def test_build_with_bare_decisions(self, decision1, decision2):
        """build() accepts bare Decision instances."""
        trace = Trace.build([decision1, decision2])
        
        assert len(trace) == 2
        assert trace.reasons == (None, None)
    
    def test_build_with_tuples_and_string_reasons(self, decision1, decision2):
        """build() accepts (decision, string) tuples."""
        trace = Trace.build([
            (decision1, "First step"),
            (decision2, "Second step"),
        ])
        
        assert len(trace) == 2
        assert trace.reasons[0].reason == "First step"
        assert trace.reasons[1].reason == "Second step"
    
    def test_build_with_tuples_and_causal_reasons(self, decision1, decision2):
        """build() accepts (decision, CausalReason) tuples."""
        trace = Trace.build([
            (decision1, CausalReason("Step 1", "constraint")),
            (decision2, CausalReason("Step 2", "inference")),
        ])
        
        assert trace.reasons[0].category == "constraint"
        assert trace.reasons[1].category == "inference"
    
    def test_build_with_tuples_and_none_reasons(self, decision1, decision2):
        """build() accepts (decision, None) tuples."""
        trace = Trace.build([
            (decision1, None),
            (decision2, None),
        ])
        
        assert trace.reasons == (None, None)
    
    def test_build_mixed_formats(self, decision1, decision2, decision3):
        """build() accepts mixed formats."""
        trace = Trace.build([
            decision1,  # Bare
            (decision2, "string reason"),  # String
            (decision3, CausalReason("full reason", "default")),  # CausalReason
        ])
        
        assert len(trace) == 3
        assert trace.reasons[0] is None
        assert trace.reasons[1].reason == "string reason"
        assert trace.reasons[2].category == "default"
    
    def test_empty_then_append(self, decision1, decision2):
        """Build via empty() + append()."""
        trace = (
            Trace.empty()
            .append(decision1)
            .append(decision2, reason="After first")
        )
        
        assert len(trace) == 2
        assert trace.reasons[0] is None
        assert trace.reasons[1].reason == "After first"


# =============================================================================
# SECTION 7: Serialization Details Tests
# =============================================================================

class TestSerializationDetails:
    """Test serialization edge cases and details."""
    
    def test_to_dict_includes_trace_id(self, sample_trace):
        """to_dict() includes trace_id."""
        data = sample_trace.to_dict()
        assert "trace_id" in data
        assert data["trace_id"] == sample_trace.trace_id
    
    def test_to_dict_includes_nodes(self, sample_trace):
        """to_dict() includes nodes array."""
        data = sample_trace.to_dict()
        assert "nodes" in data
        assert len(data["nodes"]) == len(sample_trace)
    
    def test_to_dict_includes_parent_trace_id_when_present(
        self, decision1, decision2
    ):
        """to_dict() includes parent_trace_id when parent exists."""
        parent = Trace.build([decision1])
        child = Trace.build([decision2], parent=parent)
        
        data = child.to_dict()
        assert "parent_trace_id" in data
        assert data["parent_trace_id"] == parent.trace_id
    
    def test_to_dict_no_parent_trace_id_when_absent(self, sample_trace):
        """to_dict() omits parent_trace_id when no parent."""
        data = sample_trace.to_dict()
        assert "parent_trace_id" not in data
    
    def test_to_json_with_indent(self, sample_trace):
        """to_json() accepts indent parameter."""
        json_str = sample_trace.to_json(indent=2)
        assert "\n" in json_str
        assert "  " in json_str
    
    def test_from_dict_reconstructs_decisions(self, sample_trace):
        """from_dict() reconstructs Decision objects."""
        data = sample_trace.to_dict()
        restored = Trace.from_dict(data)
        
        for orig, rest in zip(sample_trace.decisions, restored.decisions):
            assert orig.selected_option == rest.selected_option
            assert orig.rationale == rest.rationale
    
    def test_from_dict_reconstructs_reasons(self, sample_trace):
        """from_dict() reconstructs CausalReason objects."""
        data = sample_trace.to_dict()
        restored = Trace.from_dict(data)
        
        for orig, rest in zip(sample_trace.reasons, restored.reasons):
            if orig is not None:
                assert rest is not None
                assert orig.reason == rest.reason
                assert orig.category == rest.category
    
    def test_empty_trace_serialization(self):
        """Empty trace serializes correctly."""
        trace = Trace.empty()
        json_str = trace.to_json()
        
        restored = Trace.from_json(json_str)
        assert len(restored) == 0
        assert restored.trace_id == trace.trace_id


# =============================================================================
# SECTION 8: Equality and Hashing Tests
# =============================================================================

class TestEqualityAndHashing:
    """Test equality and hash behavior."""
    
    def test_equal_traces_are_equal(self, decision1, decision2):
        """Traces with same content are equal."""
        trace1 = Trace.build([decision1, decision2])
        trace2 = Trace.build([decision1, decision2])
        
        assert trace1 == trace2
    
    def test_different_traces_not_equal(self, decision1, decision2):
        """Traces with different content are not equal."""
        trace1 = Trace.build([decision1])
        trace2 = Trace.build([decision2])
        
        assert trace1 != trace2
    
    def test_trace_is_hashable(self, sample_trace):
        """Trace can be used in sets."""
        s = {sample_trace}
        assert sample_trace in s
    
    def test_equal_traces_have_same_hash(self, decision1, decision2):
        """Equal traces have the same hash."""
        trace1 = Trace.build([decision1, decision2])
        trace2 = Trace.build([decision1, decision2])
        
        assert hash(trace1) == hash(trace2)
    
    def test_equal_traces_can_be_set_members(self, decision1):
        """Equal traces deduplicate in sets."""
        trace1 = Trace.build([decision1])
        trace2 = Trace.build([decision1])
        
        s = {trace1, trace2}
        assert len(s) == 1


# =============================================================================
# SECTION 9: String Representation Tests
# =============================================================================

class TestStringRepresentations:
    """Test __repr__ and __str__."""
    
    def test_repr_includes_trace_id(self, sample_trace):
        """__repr__ includes trace_id prefix."""
        r = repr(sample_trace)
        assert "Trace(" in r
        assert sample_trace.trace_id[:16] in r
    
    def test_repr_includes_length(self, sample_trace):
        """__repr__ includes length."""
        r = repr(sample_trace)
        assert f"len={len(sample_trace)}" in r
    
    def test_repr_includes_parent_when_present(self, decision1, decision2):
        """__repr__ includes parent info when present."""
        parent = Trace.build([decision1])
        child = Trace.build([decision2], parent=parent)
        
        r = repr(child)
        assert "parent=" in r
    
    def test_str_is_human_readable(self, sample_trace):
        """__str__ is human-readable."""
        s = str(sample_trace)
        assert "Trace" in s
        assert "decisions" in s


# =============================================================================
# SECTION 10: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_decision_trace(self, decision1):
        """Trace with single decision works."""
        trace = Trace.build([decision1])
        
        assert len(trace) == 1
        assert trace.first() is decision1
        assert trace.last() is decision1
    
    def test_many_decisions(self):
        """Trace with many decisions works."""
        decisions = [
            Decision(
                inputs={"i": i},
                selected_option=f"option_{i}",
                alternatives=[],
                rationale=f"Reason {i}",
                decision_id=f"d-{i:04d}",
                timestamp="2026-02-08T12:00:00Z",
            )
            for i in range(100)
        ]
        
        trace = Trace.build(decisions)
        assert len(trace) == 100
        
        # Verify determinism
        trace2 = Trace.build(decisions)
        assert trace.trace_id == trace2.trace_id
    
    def test_deep_parent_chain(self, decision1):
        """Deep parent chain works."""
        trace = Trace.build([decision1])
        
        for i in range(50):
            trace = Trace.build([decision1], parent=trace)
        
        assert trace.depth() == 50
        assert len(trace.chain()) == 51
    
    def test_unicode_in_reasons(self, decision1):
        """Unicode in causal reasons works."""
        trace = Trace.build([
            (decision1, CausalReason("日本語の理由", "推論")),
        ])
        
        json_str = trace.to_json()
        restored = Trace.from_json(json_str)
        
        assert restored.reasons[0].reason == "日本語の理由"
        assert restored.reasons[0].category == "推論"
    
    def test_very_long_reason(self, decision1):
        """Very long reason string works."""
        long_reason = "x" * 10000
        trace = Trace.build([
            (decision1, CausalReason(long_reason)),
        ])
        
        assert trace.reasons[0].reason == long_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
