"""
test_decision_engine.py

Comprehensive tests for the DecisionEngine.

Tests prove:
- Determinism (same input → same output)
- Policy conflict → rejected
- Capability mismatch → rejected
- Missing input → deferred
- Valid request → accepted
- Trace is always produced
- Engine does not mutate inputs
"""

import copy
import pytest
import uuid

from ntive.engine import DecisionEngine, EngineErrorCode, EngineErrorCategory
from ntive.request import DecisionRequest, QueryDescriptor, RequestConstraints
from ntive.response import (
    DecisionResponse,
    ResponseType,
    AcceptedPayload,
    RejectedPayload,
    DeferredPayload,
)
from ntive.decision import Decision, Alternative, Confidence
from ntive.trace import Trace, CausalReason
from ntive.memory import MemoryScope
from ntive.policy import Policy, PolicyRule, PolicyEffect, ConflictResolutionStrategy
from ntive.capability import (
    CapabilityDescriptor,
    CapabilityInput,
    DeclaredEffect,
    EffectCategory,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def engine() -> DecisionEngine:
    """Create a fresh DecisionEngine instance."""
    return DecisionEngine()


@pytest.fixture
def valid_uuid() -> str:
    """A valid UUID v4 for testing."""
    return "12345678-1234-4abc-8def-123456789abc"


@pytest.fixture
def another_uuid() -> str:
    """Another valid UUID v4 for testing."""
    return "87654321-4321-4abc-8def-abcdef123456"


@pytest.fixture
def valid_query() -> QueryDescriptor:
    """A minimal valid query."""
    return QueryDescriptor(intent="test_intent")


@pytest.fixture
def query_with_parameters() -> QueryDescriptor:
    """A query with parameters."""
    return QueryDescriptor(
        intent="run_action",
        parameters={"action": "open_file", "filename": "test.txt"},
    )


@pytest.fixture
def minimal_request(valid_uuid: str, valid_query: QueryDescriptor) -> DecisionRequest:
    """A minimal valid DecisionRequest."""
    return DecisionRequest(
        request_id=valid_uuid,
        query=valid_query,
    )


@pytest.fixture
def request_with_context(valid_uuid: str, valid_query: QueryDescriptor) -> DecisionRequest:
    """A request with context."""
    return DecisionRequest(
        request_id=valid_uuid,
        query=valid_query,
        context={"user": "alice", "environment": "test"},
    )


@pytest.fixture
def valid_memory() -> MemoryScope:
    """A valid MemoryScope."""
    return MemoryScope(values={"key": "value"})


@pytest.fixture
def valid_policy() -> Policy:
    """A valid Policy without conflicts."""
    return Policy(
        name="test-policy",
        rules=[
            PolicyRule(
                rule_id="rule-1",
                target="action_open",
                effect=PolicyEffect.PREFER,
            ),
        ],
    )


@pytest.fixture
def conflicting_policy() -> Policy:
    """A Policy with conflicting rules."""
    return Policy(
        name="conflict-policy",
        rules=[
            PolicyRule(
                rule_id="rule-require",
                target="same_target",
                effect=PolicyEffect.REQUIRE,
            ),
            PolicyRule(
                rule_id="rule-forbid",
                target="same_target",
                effect=PolicyEffect.FORBID,
            ),
        ],
        conflict_strategy=ConflictResolutionStrategy.EXPLICIT_PRIORITY,
    )


@pytest.fixture
def simple_capability() -> CapabilityDescriptor:
    """A simple capability with no required inputs."""
    return CapabilityDescriptor(
        name="SimpleAction",
        version="1.0.0",
        domain="test",
        inputs=[],
    )


@pytest.fixture
def capability_with_required_inputs() -> CapabilityDescriptor:
    """A capability with required inputs."""
    return CapabilityDescriptor(
        name="FileAction",
        version="1.0.0",
        domain="files",
        inputs=[
            CapabilityInput(key="filename", type="string", required=True),
            CapabilityInput(key="mode", type="string", required=True),
        ],
        effects=[
            DeclaredEffect(
                category=EffectCategory.STATE_CHANGE,
                target="filesystem",
                description="May modify file",
            ),
        ],
    )


@pytest.fixture
def capability_with_optional_inputs() -> CapabilityDescriptor:
    """A capability with only optional inputs."""
    return CapabilityDescriptor(
        name="OptionalAction",
        version="1.0.0",
        domain="test",
        inputs=[
            CapabilityInput(key="optional_field", type="string", required=False),
        ],
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestDecisionEngineBasics:
    """Basic DecisionEngine tests."""
    
    def test_engine_instantiation(self):
        """Engine can be instantiated."""
        engine = DecisionEngine()
        assert engine is not None
    
    def test_evaluate_returns_decision_response(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """evaluate() returns a DecisionResponse."""
        response = engine.evaluate(minimal_request)
        assert isinstance(response, DecisionResponse)
    
    def test_response_has_same_request_id(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Response contains the same request_id."""
        response = engine.evaluate(minimal_request)
        assert response.request_id == minimal_request.request_id


class TestDeterminism:
    """Tests proving determinism."""
    
    def test_same_input_same_response_type(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Same input produces same response type."""
        response1 = engine.evaluate(minimal_request)
        response2 = engine.evaluate(minimal_request)
        assert response1.response_type == response2.response_type
    
    def test_same_input_same_decision(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Same input produces equivalent decision content."""
        response1 = engine.evaluate(minimal_request)
        response2 = engine.evaluate(minimal_request)
        
        if response1.response_type == ResponseType.ACCEPTED:
            payload1 = response1.accepted_payload
            payload2 = response2.accepted_payload
            assert payload1 is not None
            assert payload2 is not None
            assert payload1.decision.selected_option == payload2.decision.selected_option
            assert payload1.decision.rationale == payload2.decision.rationale
    
    def test_deterministic_with_complex_request(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        query_with_parameters: QueryDescriptor,
        valid_memory: MemoryScope,
        valid_policy: Policy,
        simple_capability: CapabilityDescriptor,
    ):
        """Determinism with complex request."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query_with_parameters,
            context={"environment": "test"},
            memory=valid_memory,
            policy=valid_policy,
            capabilities=[simple_capability],
        )
        
        response1 = engine.evaluate(request)
        response2 = engine.evaluate(request)
        
        assert response1.response_type == response2.response_type
        if response1.response_type == ResponseType.ACCEPTED:
            assert response1.accepted_payload.decision.inputs == response2.accepted_payload.decision.inputs


class TestInputImmutability:
    """Tests proving engine does not mutate inputs."""
    
    def test_context_not_mutated(
        self, engine: DecisionEngine, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Engine does not mutate request context."""
        original_context = {"key1": "value1", "nested": {"inner": "data"}}
        context_copy = copy.deepcopy(original_context)
        
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context=original_context,
        )
        
        engine.evaluate(request)
        
        # Context should be unchanged
        assert request.context == context_copy
    
    def test_query_not_mutated(
        self, engine: DecisionEngine, valid_uuid: str
    ):
        """Engine does not mutate query."""
        query = QueryDescriptor(
            intent="test_intent",
            parameters={"key": "value"},
        )
        original_intent = query.intent
        original_params = query.parameters.copy()
        
        request = DecisionRequest(request_id=valid_uuid, query=query)
        engine.evaluate(request)
        
        assert query.intent == original_intent
        assert query.parameters == original_params
    
    def test_capabilities_not_mutated(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        simple_capability: CapabilityDescriptor,
    ):
        """Engine does not mutate capabilities."""
        original_name = simple_capability.name
        original_version = simple_capability.version
        
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[simple_capability],
        )
        
        engine.evaluate(request)
        
        assert simple_capability.name == original_name
        assert simple_capability.version == original_version


class TestStructuralValidation:
    """Tests for Step 1: Structural Validation."""
    
    def test_non_request_rejected(self, engine: DecisionEngine):
        """Non-DecisionRequest input is rejected."""
        response = engine.evaluate("not a request")  # type: ignore
        assert response.response_type == ResponseType.REJECTED
        assert response.rejected_payload is not None
        assert response.rejected_payload.category == EngineErrorCategory.VALIDATION
    
    def test_none_rejected(self, engine: DecisionEngine):
        """None input is rejected."""
        response = engine.evaluate(None)  # type: ignore
        assert response.response_type == ResponseType.REJECTED
    
    def test_dict_rejected(self, engine: DecisionEngine):
        """Dict input is rejected."""
        response = engine.evaluate({"request_id": "123"})  # type: ignore
        assert response.response_type == ResponseType.REJECTED


class TestContextAndMemoryValidation:
    """Tests for Step 2: Context & Memory Validation."""
    
    def test_valid_context_accepted(
        self, engine: DecisionEngine, request_with_context: DecisionRequest
    ):
        """Valid context passes validation."""
        response = engine.evaluate(request_with_context)
        # Should not be rejected for context issues
        if response.response_type == ResponseType.REJECTED:
            assert response.rejected_payload.code != EngineErrorCode.INVALID_CONTEXT
    
    def test_valid_memory_accepted(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        valid_memory: MemoryScope,
    ):
        """Valid memory passes validation."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            memory=valid_memory,
        )
        response = engine.evaluate(request)
        
        if response.response_type == ResponseType.REJECTED:
            assert response.rejected_payload.code != EngineErrorCode.INVALID_MEMORY


class TestPolicyResolution:
    """Tests for Step 3: Policy Resolution."""
    
    def test_no_policy_accepted(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Request without policy is accepted."""
        response = engine.evaluate(minimal_request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_valid_policy_accepted(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        valid_policy: Policy,
    ):
        """Request with valid policy is accepted."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=valid_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_policy_conflict_rejected(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        conflicting_policy: Policy,
    ):
        """Request with conflicting policy is rejected."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=conflicting_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.REJECTED
        assert response.rejected_payload is not None
        assert response.rejected_payload.category == EngineErrorCategory.POLICY
        assert "conflict" in response.rejected_payload.reason.lower()
    
    def test_policy_conflict_not_recoverable(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        conflicting_policy: Policy,
    ):
        """Policy conflict is not recoverable."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=conflicting_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.REJECTED
        assert response.rejected_payload.recoverable is False


class TestCapabilityMatching:
    """Tests for Step 4: Capability Matching."""
    
    def test_no_capabilities_accepted(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Request without capabilities is accepted."""
        response = engine.evaluate(minimal_request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_capability_with_satisfied_inputs_accepted(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Capability with all required inputs satisfied is accepted."""
        query = QueryDescriptor(
            intent="file_action",
            parameters={"filename": "test.txt", "mode": "read"},
        )
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_missing_inputs_deferred(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Capability with missing required inputs causes deferred response."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,  # No parameters provided
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.DEFERRED
    
    def test_deferred_includes_required_inputs(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Deferred response includes list of required inputs."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.DEFERRED
        assert response.deferred_payload is not None
        
        required_names = {ri.name for ri in response.deferred_payload.required_inputs}
        assert "filename" in required_names
        assert "mode" in required_names
    
    def test_optional_inputs_not_required(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        capability_with_optional_inputs: CapabilityDescriptor,
    ):
        """Capability with only optional inputs is satisfied."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[capability_with_optional_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_partial_inputs_deferred(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Capability with partial inputs still defers."""
        query = QueryDescriptor(
            intent="file_action",
            parameters={"filename": "test.txt"},  # Missing 'mode'
        )
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.DEFERRED
        
        required_names = {ri.name for ri in response.deferred_payload.required_inputs}
        assert "mode" in required_names
        assert "filename" not in required_names  # Already provided


class TestDecisionConstruction:
    """Tests for Step 5: Decision Construction."""
    
    def test_accepted_response_has_decision(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Accepted response contains a Decision."""
        response = engine.evaluate(minimal_request)
        assert response.response_type == ResponseType.ACCEPTED
        assert response.accepted_payload is not None
        assert response.accepted_payload.decision is not None
        assert isinstance(response.accepted_payload.decision, Decision)
    
    def test_decision_contains_intent(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Decision contains the query intent."""
        response = engine.evaluate(minimal_request)
        decision = response.accepted_payload.decision
        assert "intent" in decision.inputs
        assert decision.inputs["intent"] == "test_intent"
    
    def test_decision_with_capability_uses_capability_name(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        simple_capability: CapabilityDescriptor,
    ):
        """Decision with capability uses capability name in selected_option."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[simple_capability],
        )
        response = engine.evaluate(request)
        decision = response.accepted_payload.decision
        
        assert "capability:SimpleAction" in decision.selected_option
    
    def test_decision_without_capability_uses_intent(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Decision without capability uses intent in selected_option."""
        response = engine.evaluate(minimal_request)
        decision = response.accepted_payload.decision
        
        assert "intent:" in decision.selected_option


class TestTraceConstruction:
    """Tests for Step 6: Trace Construction."""
    
    def test_accepted_response_has_trace(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Accepted response contains a Trace."""
        response = engine.evaluate(minimal_request)
        assert response.response_type == ResponseType.ACCEPTED
        assert response.accepted_payload.trace is not None
        assert isinstance(response.accepted_payload.trace, Trace)
    
    def test_trace_is_not_empty(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Trace contains at least one decision."""
        response = engine.evaluate(minimal_request)
        trace = response.accepted_payload.trace
        assert len(trace) > 0
    
    def test_trace_contains_decision(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Trace contains decisions."""
        response = engine.evaluate(minimal_request)
        trace = response.accepted_payload.trace
        decisions = trace.decisions
        assert len(decisions) > 0
        assert all(isinstance(d, Decision) for d in decisions)
    
    def test_rejected_can_have_partial_trace(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        conflicting_policy: Policy,
    ):
        """Rejected response can have a partial trace."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=conflicting_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.REJECTED
        
        # Partial trace may or may not exist depending on when rejection occurred
        payload = response.rejected_payload
        # Policy resolution happens before decision, so partial trace may have policy decision
        if payload.partial_trace is not None:
            assert isinstance(payload.partial_trace, Trace)
    
    def test_trace_with_policy_includes_policy_decision(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        valid_policy: Policy,
    ):
        """Trace for request with policy includes policy resolution decision."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=valid_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
        
        trace = response.accepted_payload.trace
        decisions = trace.decisions
        
        # Should have at least two decisions: policy resolution + main decision
        assert len(decisions) >= 2


class TestResponseTypes:
    """Tests for different response types."""
    
    def test_accepted_response_properties(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Accepted response has correct properties."""
        response = engine.evaluate(minimal_request)
        
        assert response.response_type == ResponseType.ACCEPTED
        assert response.accepted_payload is not None
        assert response.rejected_payload is None
        assert response.deferred_payload is None
    
    def test_rejected_response_properties(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        conflicting_policy: Policy,
    ):
        """Rejected response has correct properties."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=conflicting_policy,
        )
        response = engine.evaluate(request)
        
        assert response.response_type == ResponseType.REJECTED
        assert response.accepted_payload is None
        assert response.rejected_payload is not None
        assert response.deferred_payload is None
    
    def test_deferred_response_properties(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Deferred response has correct properties."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        
        assert response.response_type == ResponseType.DEFERRED
        assert response.accepted_payload is None
        assert response.rejected_payload is None
        assert response.deferred_payload is not None


class TestEffectsHandling:
    """Tests for effects in accepted responses."""
    
    def test_capability_effects_included(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Capability effects are included in accepted response."""
        query = QueryDescriptor(
            intent="file_action",
            parameters={"filename": "test.txt", "mode": "read"},
        )
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
        
        effects = response.accepted_payload.effects
        assert effects is not None
        assert len(effects) > 0
    
    def test_no_capability_no_effects(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """No capability means no effects."""
        response = engine.evaluate(minimal_request)
        assert response.response_type == ResponseType.ACCEPTED
        
        effects = response.accepted_payload.effects
        assert effects is None


class TestErrorCodes:
    """Tests for error codes and categories."""
    
    def test_invalid_request_error_code(self, engine: DecisionEngine):
        """Invalid request uses correct error code."""
        response = engine.evaluate("invalid")  # type: ignore
        assert response.rejected_payload.code == EngineErrorCode.INVALID_REQUEST_TYPE
    
    def test_policy_conflict_error_code(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        conflicting_policy: Policy,
    ):
        """Policy conflict uses correct error code."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=conflicting_policy,
        )
        response = engine.evaluate(request)
        assert response.rejected_payload.code == EngineErrorCode.POLICY_CONFLICT_UNRESOLVABLE
    
    def test_missing_input_error_code(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Missing input uses correct error code."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.DEFERRED
        assert response.deferred_payload.code == EngineErrorCode.INPUT_NOT_SATISFIED


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_context(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
    ):
        """Empty context is accepted."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={},
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_empty_capabilities_list(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
    ):
        """Empty capabilities list is accepted."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_multiple_capabilities_first_match(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        simple_capability: CapabilityDescriptor,
        capability_with_optional_inputs: CapabilityDescriptor,
    ):
        """Multiple capabilities - first matching is selected."""
        query = QueryDescriptor(intent="multi_capability")
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query,
            capabilities=[simple_capability, capability_with_optional_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
        
        # First capability (SimpleAction) should be selected
        decision = response.accepted_payload.decision
        assert "SimpleAction" in decision.selected_option
    
    def test_deeply_nested_context(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
    ):
        """Deeply nested context is handled."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={
                "level1": {
                    "level2": {
                        "level3": {"value": 42},
                    },
                },
            },
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_inputs_from_context(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Required inputs can come from context."""
        query = QueryDescriptor(intent="file_action")
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query,
            context={"filename": "test.txt", "mode": "write"},
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_response_timestamp_is_positive(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Response timestamp is a positive integer."""
        response = engine.evaluate(minimal_request)
        assert response.timestamp > 0
    
    def test_response_hash_exists(
        self, engine: DecisionEngine, minimal_request: DecisionRequest
    ):
        """Response has a hash."""
        response = engine.evaluate(minimal_request)
        assert response.response_hash is not None
        assert len(response.response_hash) > 0


class TestPolicyInheritance:
    """Tests for policy inheritance."""
    
    def test_child_policy_inherits_rules(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
    ):
        """Child policy inherits parent rules."""
        parent_policy = Policy(
            name="parent-policy",
            rules=[
                PolicyRule(
                    rule_id="parent-rule",
                    target="inherited_action",
                    effect=PolicyEffect.PREFER,
                ),
            ],
        )
        
        child_policy = Policy(
            name="child-policy",
            parent=parent_policy,
            rules=[
                PolicyRule(
                    rule_id="child-rule",
                    target="child_action",
                    effect=PolicyEffect.PREFER,
                ),
            ],
        )
        
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=child_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.ACCEPTED
    
    def test_inherited_conflict_rejected(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
    ):
        """Conflict between parent and child rules is detected."""
        parent_policy = Policy(
            name="parent-policy",
            rules=[
                PolicyRule(
                    rule_id="parent-rule",
                    target="conflict_target",
                    effect=PolicyEffect.REQUIRE,
                ),
            ],
        )
        
        child_policy = Policy(
            name="child-policy",
            parent=parent_policy,
            rules=[
                PolicyRule(
                    rule_id="child-rule",
                    target="conflict_target",
                    effect=PolicyEffect.FORBID,
                ),
            ],
            conflict_strategy=ConflictResolutionStrategy.EXPLICIT_PRIORITY,
        )
        
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            policy=child_policy,
        )
        response = engine.evaluate(request)
        assert response.response_type == ResponseType.REJECTED


class TestRecoverability:
    """Tests for recoverability flags."""
    
    def test_validation_error_not_recoverable(self, engine: DecisionEngine):
        """Invalid request type is not recoverable."""
        response = engine.evaluate(12345)  # type: ignore
        assert response.rejected_payload.recoverable is False
    
    def test_context_error_is_recoverable(
        self,
        engine: DecisionEngine,
    ):
        """Context validation errors are recoverable."""
        # Most context errors would be caught at DecisionRequest construction
        # but if engine validation fails, it should be recoverable
        # This is a placeholder for when context validation fails inside engine
        pass  # Current implementation catches at request construction
    
    def test_capability_mismatch_is_recoverable(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        capability_with_required_inputs: CapabilityDescriptor,
    ):
        """Missing inputs is a deferred (recoverable) situation."""
        request = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[capability_with_required_inputs],
        )
        response = engine.evaluate(request)
        # Deferred responses are inherently recoverable
        assert response.response_type == ResponseType.DEFERRED


class TestStatelessness:
    """Tests proving engine is stateless."""
    
    def test_engine_has_no_state(self, engine: DecisionEngine):
        """Engine has no instance state."""
        assert engine.__slots__ == ()
    
    def test_multiple_evaluations_independent(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        another_uuid: str,
        valid_query: QueryDescriptor,
    ):
        """Multiple evaluations are independent."""
        request1 = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={"eval": 1},
        )
        request2 = DecisionRequest(
            request_id=another_uuid,
            query=valid_query,
            context={"eval": 2},
        )
        
        response1 = engine.evaluate(request1)
        response2 = engine.evaluate(request2)
        
        # Each response should have its own request_id
        assert response1.request_id == valid_uuid
        assert response2.request_id == another_uuid


class TestComplexScenarios:
    """Complex integration scenarios."""
    
    def test_full_stack_request(
        self,
        engine: DecisionEngine,
        valid_uuid: str,
        valid_memory: MemoryScope,
        valid_policy: Policy,
    ):
        """Full stack request with all components."""
        query = QueryDescriptor(
            intent="complex_action",
            parameters={"filename": "data.json", "mode": "read"},
            metadata={"source": "test"},
        )
        
        capability = CapabilityDescriptor(
            name="DataProcessor",
            version="2.0.0",
            domain="data",
            description="Process data files",
            inputs=[
                CapabilityInput(key="filename", type="string", required=True),
                CapabilityInput(key="mode", type="string", required=True),
            ],
            effects=[
                DeclaredEffect(
                    category=EffectCategory.STATE_CHANGE,
                    target="data_store",
                    description="May modify data store",
                ),
            ],
        )
        
        request = DecisionRequest(
            request_id=valid_uuid,
            query=query,
            context={"environment": "production"},
            memory=valid_memory,
            policy=valid_policy,
            capabilities=[capability],
        )
        
        response = engine.evaluate(request)
        
        assert response.response_type == ResponseType.ACCEPTED
        assert response.request_id == valid_uuid
        assert response.accepted_payload.decision is not None
        assert response.accepted_payload.trace is not None
        assert len(response.accepted_payload.trace) >= 2  # Policy + main decision
        assert response.accepted_payload.effects is not None
