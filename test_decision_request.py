"""
test_decision_request.py

Comprehensive tests for the DecisionRequest ecosystem primitive.
Target: ≥ 45 tests (half of 90+ total for request + response)
"""

import json
import pytest
import hashlib

from ntive.request import (
    DecisionRequest,
    QueryDescriptor,
    ContinuationRef,
    RequestConstraints,
    RequestValidationError,
    MissingRequestFieldError,
    InvalidRequestFieldError,
    InvalidRequestIdError,
    NonSerializableContextError,
    InvalidCapabilityError,
    InvalidPolicyError,
    ExecutableInRequestError,
    RequestImmutabilityError,
)
from ntive.memory import MemoryScope
from ntive.policy import Policy, PolicyRule, PolicyEffect
from ntive.capability import CapabilityDescriptor, CapabilityInput


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_uuid() -> str:
    """A valid UUID v4 for testing."""
    return "12345678-1234-4abc-8def-123456789abc"


@pytest.fixture
def valid_query() -> QueryDescriptor:
    """A minimal valid QueryDescriptor."""
    return QueryDescriptor(intent="test_intent")


@pytest.fixture
def valid_memory() -> MemoryScope:
    """A valid MemoryScope."""
    return MemoryScope(values={"key": "value"})


@pytest.fixture
def valid_policy() -> Policy:
    """A valid Policy."""
    return Policy(
        name="test-policy",
        rules=[
            PolicyRule(
                rule_id="rule-1",
                target="*",
                effect=PolicyEffect.PREFER,
            )
        ],
    )


@pytest.fixture
def valid_capability() -> CapabilityDescriptor:
    """A valid CapabilityDescriptor."""
    return CapabilityDescriptor(
        name="TestCapability",
        version="1.0.0",
        domain="test",
        inputs=[
            CapabilityInput(key="input1", type="string"),
        ],
    )


@pytest.fixture
def minimal_request(valid_uuid: str, valid_query: QueryDescriptor) -> DecisionRequest:
    """A minimal valid DecisionRequest."""
    return DecisionRequest(
        request_id=valid_uuid,
        query=valid_query,
    )


# =============================================================================
# Test: QueryDescriptor
# =============================================================================

class TestQueryDescriptor:
    """Tests for QueryDescriptor."""
    
    def test_create_minimal(self):
        """Create minimal QueryDescriptor with just intent."""
        qd = QueryDescriptor(intent="test_intent")
        assert qd.intent == "test_intent"
        assert qd.parameters == {}
        assert qd.metadata == {}
    
    def test_create_with_parameters(self):
        """Create QueryDescriptor with parameters."""
        qd = QueryDescriptor(
            intent="test_intent",
            parameters={"key": "value", "num": 42},
        )
        assert qd.parameters == {"key": "value", "num": 42}
    
    def test_create_with_metadata(self):
        """Create QueryDescriptor with metadata."""
        qd = QueryDescriptor(
            intent="test_intent",
            metadata={"source": "test"},
        )
        assert qd.metadata == {"source": "test"}
    
    def test_missing_intent_raises(self):
        """Missing intent raises error."""
        with pytest.raises(MissingRequestFieldError) as exc_info:
            QueryDescriptor(intent=None)  # type: ignore
        assert exc_info.value.error_code == "R001"
        assert "intent" in str(exc_info.value)
    
    def test_empty_intent_raises(self):
        """Empty intent raises error."""
        with pytest.raises(RequestValidationError):
            QueryDescriptor(intent="   ")
    
    def test_intent_invalid_type_raises(self):
        """Non-string intent raises error."""
        with pytest.raises(InvalidRequestFieldError) as exc_info:
            QueryDescriptor(intent=123)  # type: ignore
        assert exc_info.value.error_code == "R002"
    
    def test_parameters_must_be_dict(self):
        """Parameters must be a dict."""
        with pytest.raises(InvalidRequestFieldError):
            QueryDescriptor(intent="test", parameters="not_dict")  # type: ignore
    
    def test_parameters_must_be_json_serializable(self):
        """Parameters with executables raise ExecutableInRequestError."""
        with pytest.raises(ExecutableInRequestError):
            QueryDescriptor(
                intent="test",
                parameters={"func": lambda x: x},
            )
    
    def test_immutability(self):
        """QueryDescriptor is immutable."""
        qd = QueryDescriptor(intent="test_intent")
        with pytest.raises(RequestImmutabilityError):
            qd.intent = "changed"  # type: ignore
    
    def test_immutability_delete(self):
        """Cannot delete attributes."""
        qd = QueryDescriptor(intent="test_intent")
        with pytest.raises(RequestImmutabilityError):
            del qd.intent  # type: ignore
    
    def test_parameters_copy_protection(self):
        """Parameters are deep-copied."""
        params = {"key": "value"}
        qd = QueryDescriptor(intent="test", parameters=params)
        params["key"] = "changed"
        assert qd.parameters == {"key": "value"}
    
    def test_parameters_getter_returns_copy(self):
        """Getting parameters returns a copy."""
        qd = QueryDescriptor(intent="test", parameters={"key": "value"})
        params = qd.parameters
        params["key"] = "changed"
        assert qd.parameters == {"key": "value"}
    
    def test_equality(self):
        """Equal QueryDescriptors are equal."""
        qd1 = QueryDescriptor(intent="test", parameters={"a": 1})
        qd2 = QueryDescriptor(intent="test", parameters={"a": 1})
        assert qd1 == qd2
    
    def test_inequality(self):
        """Different QueryDescriptors are not equal."""
        qd1 = QueryDescriptor(intent="test1")
        qd2 = QueryDescriptor(intent="test2")
        assert qd1 != qd2
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        qd = QueryDescriptor(
            intent="test_intent",
            parameters={"key": "value"},
            metadata={"source": "test"},
        )
        d = qd.to_dict()
        assert d["intent"] == "test_intent"
        assert d["parameters"] == {"key": "value"}
        assert d["metadata"] == {"source": "test"}
    
    def test_to_json_deterministic(self):
        """JSON serialization is deterministic."""
        qd = QueryDescriptor(
            intent="test",
            parameters={"b": 2, "a": 1},
        )
        json1 = qd.to_json()
        json2 = qd.to_json()
        assert json1 == json2
        # Keys should be sorted
        assert '"a": 1' in json1  # a before b
    
    def test_from_dict(self):
        """from_dict correctly constructs QueryDescriptor."""
        data = {
            "intent": "test_intent",
            "parameters": {"key": "value"},
        }
        qd = QueryDescriptor.from_dict(data)
        assert qd.intent == "test_intent"
        assert qd.parameters == {"key": "value"}
    
    def test_from_json(self):
        """from_json correctly constructs QueryDescriptor."""
        json_str = '{"intent": "test_intent"}'
        qd = QueryDescriptor.from_json(json_str)
        assert qd.intent == "test_intent"


# =============================================================================
# Test: ContinuationRef
# =============================================================================

class TestContinuationRef:
    """Tests for ContinuationRef."""
    
    def test_create_minimal(self):
        """Create minimal ContinuationRef."""
        ref = ContinuationRef(ref_id="ref-123")
        assert ref.ref_id == "ref-123"
        assert ref.context == {}
    
    def test_create_with_context(self):
        """Create ContinuationRef with context."""
        ref = ContinuationRef(
            ref_id="ref-123",
            context={"step": 1},
        )
        assert ref.context == {"step": 1}
    
    def test_missing_ref_id_raises(self):
        """Missing ref_id raises error."""
        with pytest.raises(MissingRequestFieldError):
            ContinuationRef(ref_id=None)  # type: ignore
    
    def test_empty_ref_id_raises(self):
        """Empty ref_id raises error."""
        with pytest.raises(RequestValidationError):
            ContinuationRef(ref_id="   ")
    
    def test_immutability(self):
        """ContinuationRef is immutable."""
        ref = ContinuationRef(ref_id="ref-123")
        with pytest.raises(RequestImmutabilityError):
            ref.ref_id = "changed"  # type: ignore
    
    def test_context_invalid_type_raises(self):
        """Context must be a dict."""
        with pytest.raises(InvalidRequestFieldError):
            ContinuationRef(ref_id="ref-123", context="not_dict")  # type: ignore
    
    def test_context_not_serializable_raises(self):
        """Context with executables raises ExecutableInRequestError."""
        with pytest.raises(ExecutableInRequestError):
            ContinuationRef(ref_id="ref-123", context={"bad": lambda: None})
    
    def test_equality(self):
        """Equal ContinuationRefs are equal."""
        ref1 = ContinuationRef(ref_id="ref-123", context={"a": 1})
        ref2 = ContinuationRef(ref_id="ref-123", context={"a": 1})
        assert ref1 == ref2
    
    def test_hash(self):
        """ContinuationRefs are hashable."""
        ref1 = ContinuationRef(ref_id="ref-123")
        ref2 = ContinuationRef(ref_id="ref-123")
        assert hash(ref1) == hash(ref2)
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        ref = ContinuationRef(ref_id="ref-123", context={"a": 1})
        d = ref.to_dict()
        assert d["ref_id"] == "ref-123"
        assert d["context"] == {"a": 1}


# =============================================================================
# Test: RequestConstraints
# =============================================================================

class TestRequestConstraints:
    """Tests for RequestConstraints."""
    
    def test_create_default(self):
        """Create RequestConstraints with defaults."""
        rc = RequestConstraints()
        assert rc.timeout_hint is None
        assert rc.max_alternatives is None
        assert rc.priority == "normal"
    
    def test_create_with_values(self):
        """Create RequestConstraints with values."""
        rc = RequestConstraints(
            timeout_hint=1000,
            max_alternatives=5,
            priority="high",
        )
        assert rc.timeout_hint == 1000
        assert rc.max_alternatives == 5
        assert rc.priority == "high"
    
    def test_timeout_must_be_positive(self):
        """timeout_hint must be positive."""
        with pytest.raises(RequestValidationError):
            RequestConstraints(timeout_hint=-1)
        with pytest.raises(RequestValidationError):
            RequestConstraints(timeout_hint=0)
    
    def test_max_alternatives_must_be_positive(self):
        """max_alternatives must be positive."""
        with pytest.raises(RequestValidationError):
            RequestConstraints(max_alternatives=0)
    
    def test_priority_must_be_valid(self):
        """priority must be a valid value."""
        with pytest.raises(RequestValidationError):
            RequestConstraints(priority="invalid")
    
    def test_valid_priorities(self):
        """All valid priorities work."""
        for priority in ["low", "normal", "high", "critical"]:
            rc = RequestConstraints(priority=priority)
            assert rc.priority == priority
    
    def test_immutability(self):
        """RequestConstraints is immutable."""
        rc = RequestConstraints()
        with pytest.raises(RequestImmutabilityError):
            rc.priority = "high"  # type: ignore
    
    def test_equality(self):
        """Equal RequestConstraints are equal."""
        rc1 = RequestConstraints(timeout_hint=100, priority="high")
        rc2 = RequestConstraints(timeout_hint=100, priority="high")
        assert rc1 == rc2
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        rc = RequestConstraints(timeout_hint=100, priority="high")
        d = rc.to_dict()
        assert d["timeout_hint"] == 100
        assert d["priority"] == "high"


# =============================================================================
# Test: DecisionRequest - Basic Construction
# =============================================================================

class TestDecisionRequestConstruction:
    """Tests for DecisionRequest construction."""
    
    def test_create_minimal(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Create minimal DecisionRequest."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
        )
        assert req.request_id == valid_uuid
        assert req.query.intent == "test_intent"
        assert req.context == {}
        assert req.memory is None
        assert req.policy is None
        assert req.capabilities == ()
        assert req.continuation is None
        assert req.constraints is None  # Optional, defaults to None
    
    def test_create_fully_specified(
        self,
        valid_uuid: str,
        valid_query: QueryDescriptor,
        valid_memory: MemoryScope,
        valid_policy: Policy,
        valid_capability: CapabilityDescriptor,
    ):
        """Create fully specified DecisionRequest."""
        ref = ContinuationRef(ref_id="ref-123")
        constraints = RequestConstraints(timeout_hint=1000)
        
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={"key": "value"},
            memory=valid_memory,
            policy=valid_policy,
            capabilities=[valid_capability],
            continuation=ref,
            constraints=constraints,
        )
        
        assert req.context == {"key": "value"}
        assert req.memory is not None
        assert req.policy is not None
        assert len(req.capabilities) == 1
        assert req.continuation is not None
        assert req.constraints.timeout_hint == 1000
    
    def test_create_with_query_dict(self, valid_uuid: str):
        """Create DecisionRequest with query as dict."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query={"intent": "test_intent", "parameters": {"a": 1}},  # type: ignore
        )
        assert req.query.intent == "test_intent"
        assert req.query.parameters == {"a": 1}


# =============================================================================
# Test: DecisionRequest - Validation
# =============================================================================

class TestDecisionRequestValidation:
    """Tests for DecisionRequest validation."""
    
    def test_missing_request_id_raises(self, valid_query: QueryDescriptor):
        """Missing request_id raises error."""
        with pytest.raises(MissingRequestFieldError) as exc_info:
            DecisionRequest(request_id=None, query=valid_query)  # type: ignore
        assert exc_info.value.error_code == "R001"
        assert "request_id" in str(exc_info.value)
    
    def test_missing_query_raises(self, valid_uuid: str):
        """Missing query raises error."""
        with pytest.raises(MissingRequestFieldError):
            DecisionRequest(request_id=valid_uuid, query=None)  # type: ignore
    
    def test_invalid_uuid_raises(self, valid_query: QueryDescriptor):
        """Invalid UUID raises error."""
        with pytest.raises(InvalidRequestIdError) as exc_info:
            DecisionRequest(
                request_id="not-a-uuid",
                query=valid_query,
            )
        assert exc_info.value.error_code == "R003"
    
    def test_uuid_v1_rejected(self, valid_query: QueryDescriptor):
        """UUID v1 is rejected (only v4 allowed)."""
        uuid_v1 = "550e8400-e29b-11d4-a716-446655440000"
        with pytest.raises(InvalidRequestIdError):
            DecisionRequest(request_id=uuid_v1, query=valid_query)
    
    def test_non_serializable_context_raises(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Context with executables raises ExecutableInRequestError."""
        with pytest.raises(ExecutableInRequestError) as exc_info:
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"func": lambda x: x},
            )
        assert exc_info.value.error_code == "R007"
    
    def test_invalid_capability_type_raises(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Invalid capability type raises error."""
        with pytest.raises(InvalidCapabilityError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                capabilities=["not-a-capability"],  # type: ignore
            )
    
    def test_invalid_policy_type_raises(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Invalid policy type raises error."""
        with pytest.raises(InvalidPolicyError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                policy="not-a-policy",  # type: ignore
            )
    
    def test_executable_in_context_raises(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Executable in context raises error."""
        with pytest.raises(ExecutableInRequestError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"bad": print},
            )
    
    def test_class_in_context_raises(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Class in context raises error."""
        with pytest.raises(ExecutableInRequestError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"cls": dict},
            )


# =============================================================================
# Test: DecisionRequest - Immutability
# =============================================================================

class TestDecisionRequestImmutability:
    """Tests for DecisionRequest immutability."""
    
    def test_cannot_set_attribute(self, minimal_request: DecisionRequest):
        """Cannot set attribute after creation."""
        with pytest.raises(RequestImmutabilityError) as exc_info:
            minimal_request.request_id = "new-id"  # type: ignore
        assert "set attribute" in str(exc_info.value)
    
    def test_cannot_delete_attribute(self, minimal_request: DecisionRequest):
        """Cannot delete attribute after creation."""
        with pytest.raises(RequestImmutabilityError) as exc_info:
            del minimal_request.query  # type: ignore
        assert "delete attribute" in str(exc_info.value)
    
    def test_context_copy_protection(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Context is deep-copied on construction."""
        context = {"key": {"nested": "value"}}
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context=context,
        )
        context["key"]["nested"] = "changed"
        assert req.context["key"]["nested"] == "value"
    
    def test_context_getter_returns_copy(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Context getter returns a copy."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={"key": "value"},
        )
        ctx = req.context
        ctx["key"] = "changed"
        assert req.context["key"] == "value"
    
    def test_capabilities_returns_tuple(
        self, valid_uuid: str, valid_query: QueryDescriptor, valid_capability: CapabilityDescriptor
    ):
        """Capabilities returns a tuple (immutable)."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[valid_capability],
        )
        assert isinstance(req.capabilities, tuple)


# =============================================================================
# Test: DecisionRequest - Hashing and Equality
# =============================================================================

class TestDecisionRequestHashEquality:
    """Tests for DecisionRequest hashing and equality."""
    
    def test_equal_requests(self, valid_uuid: str):
        """Identical requests are equal."""
        qd = QueryDescriptor(intent="test")
        req1 = DecisionRequest(request_id=valid_uuid, query=qd)
        req2 = DecisionRequest(request_id=valid_uuid, query=qd)
        assert req1 == req2
    
    def test_equal_hash(self, valid_uuid: str):
        """Identical requests have same hash."""
        qd = QueryDescriptor(intent="test")
        req1 = DecisionRequest(request_id=valid_uuid, query=qd)
        req2 = DecisionRequest(request_id=valid_uuid, query=qd)
        assert hash(req1) == hash(req2)
    
    def test_different_request_id(self):
        """Different request_id means different requests."""
        qd = QueryDescriptor(intent="test")
        req1 = DecisionRequest(
            request_id="12345678-1234-4abc-8def-123456789abc",
            query=qd,
        )
        req2 = DecisionRequest(
            request_id="12345678-1234-4abc-8def-123456789abd",
            query=qd,
        )
        assert req1 != req2
    
    def test_different_query(self, valid_uuid: str):
        """Different query means different requests."""
        req1 = DecisionRequest(
            request_id=valid_uuid,
            query=QueryDescriptor(intent="test1"),
        )
        req2 = DecisionRequest(
            request_id=valid_uuid,
            query=QueryDescriptor(intent="test2"),
        )
        assert req1 != req2
    
    def test_request_hash_is_deterministic(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """request_hash is deterministic."""
        req = DecisionRequest(request_id=valid_uuid, query=valid_query)
        assert req.request_hash == req.request_hash
        # Hash should be a valid SHA-256 hex string
        assert len(req.request_hash) == 64
        assert all(c in "0123456789abcdef" for c in req.request_hash)


# =============================================================================
# Test: DecisionRequest - Serialization
# =============================================================================

class TestDecisionRequestSerialization:
    """Tests for DecisionRequest serialization."""
    
    def test_to_dict(self, valid_uuid: str, valid_query: QueryDescriptor):
        """to_dict returns correct dictionary."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={"key": "value"},
        )
        d = req.to_dict()
        assert d["request_id"] == valid_uuid
        assert d["query"]["intent"] == "test_intent"
        assert d["context"] == {"key": "value"}
        assert "request_hash" in d
    
    def test_to_json_deterministic(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """JSON serialization is deterministic."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={"b": 2, "a": 1},
        )
        json1 = req.to_json()
        json2 = req.to_json()
        assert json1 == json2
    
    def test_to_json_sorted_keys(self, valid_uuid: str, valid_query: QueryDescriptor):
        """JSON has sorted keys."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={"zebra": 1, "alpha": 2},
        )
        json_str = req.to_json()
        # alpha should appear before zebra
        assert json_str.index('"alpha"') < json_str.index('"zebra"')
    
    def test_to_json_indent(self, valid_uuid: str, valid_query: QueryDescriptor):
        """JSON can be indented."""
        req = DecisionRequest(request_id=valid_uuid, query=valid_query)
        json_str = req.to_json(indent=2)
        assert "\n" in json_str


# =============================================================================
# Test: DecisionRequest - Edge Cases
# =============================================================================

class TestDecisionRequestEdgeCases:
    """Edge case tests for DecisionRequest."""
    
    def test_empty_context(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Empty context is allowed."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context={},
        )
        assert req.context == {}
    
    def test_deeply_nested_context(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Deeply nested context is allowed."""
        context = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            context=context,
        )
        assert req.context == context
    
    def test_empty_capabilities_list(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Empty capabilities list is allowed."""
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[],
        )
        assert req.capabilities == ()
    
    def test_multiple_capabilities(
        self, valid_uuid: str, valid_query: QueryDescriptor
    ):
        """Multiple capabilities are preserved."""
        cap1 = CapabilityDescriptor(
            name="Cap1",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="in1", type="string")],
        )
        cap2 = CapabilityDescriptor(
            name="Cap2",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="in2", type="int")],
        )
        req = DecisionRequest(
            request_id=valid_uuid,
            query=valid_query,
            capabilities=[cap1, cap2],
        )
        assert len(req.capabilities) == 2
    
    def test_comparison_with_non_request(
        self, minimal_request: DecisionRequest
    ):
        """Comparing with non-DecisionRequest returns NotImplemented."""
        result = minimal_request.__eq__("not-a-request")
        assert result is NotImplemented
    
    def test_repr(self, minimal_request: DecisionRequest):
        """repr returns a string."""
        r = repr(minimal_request)
        assert "DecisionRequest" in r
        assert minimal_request.request_id in r
    
    def test_str(self, minimal_request: DecisionRequest):
        """str returns a string."""
        s = str(minimal_request)
        assert "Request" in s


# =============================================================================
# Test: Executable Detection
# =============================================================================

class TestExecutableDetection:
    """Tests for executable content detection."""
    
    def test_lambda_in_context(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Lambda in context is rejected."""
        with pytest.raises(ExecutableInRequestError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"func": lambda: None},
            )
    
    def test_function_in_context(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Function in context is rejected."""
        def my_func():
            pass
        
        with pytest.raises(ExecutableInRequestError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"func": my_func},
            )
    
    def test_nested_executable(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Nested executable is rejected."""
        with pytest.raises(ExecutableInRequestError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"nested": {"deep": {"func": print}}},
            )
    
    def test_executable_in_list(self, valid_uuid: str, valid_query: QueryDescriptor):
        """Executable in list is rejected."""
        with pytest.raises(ExecutableInRequestError):
            DecisionRequest(
                request_id=valid_uuid,
                query=valid_query,
                context={"items": [1, 2, print]},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
