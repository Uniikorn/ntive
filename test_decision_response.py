"""
test_decision_response.py

Comprehensive tests for the DecisionResponse ecosystem primitive.
Target: ≥ 45 tests (complements test_decision_request.py for 90+ total)
"""

import json
import pytest

from ntive.response import (
    DecisionResponse,
    ResponseType,
    AcceptedPayload,
    RejectedPayload,
    DeferredPayload,
    ContinuationToken,
    RequiredInput,
    ResponseValidationError,
    InvalidResponseTypeError,
    PayloadMismatchError,
    MissingPayloadFieldError,
    InvalidPayloadTypeError,
    InvalidTraceError,
    MissingResponseFieldError,
    ResponseImmutabilityError,
    ExecutableInResponseError,
)
from ntive.decision import Decision, Alternative, Confidence
from ntive.trace import Trace, CausalReason
from ntive.capability import DeclaredEffect, EffectCategory


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_uuid() -> str:
    """A valid UUID v4 for testing."""
    return "12345678-1234-4abc-8def-123456789abc"


@pytest.fixture
def valid_decision() -> Decision:
    """A valid Decision."""
    return Decision(
        inputs={"key": "value"},
        selected_option="option1",
        alternatives=[
            Alternative(option="option2", reason_not_selected="Not applicable"),
        ],
        rationale="Test rationale for the decision",
        confidence=Confidence(value=0.9),
    )


@pytest.fixture
def valid_trace(valid_decision: Decision) -> Trace:
    """A valid Trace."""
    return Trace.build([
        (valid_decision, CausalReason(reason="Test reason")),
    ])


@pytest.fixture
def valid_accepted_payload(valid_decision: Decision, valid_trace: Trace) -> AcceptedPayload:
    """A valid AcceptedPayload."""
    return AcceptedPayload(
        decision=valid_decision,
        trace=valid_trace,
    )


@pytest.fixture
def valid_rejected_payload() -> RejectedPayload:
    """A valid RejectedPayload."""
    return RejectedPayload(
        code="ERR001",
        category="validation",
        reason="Request validation failed",
        recoverable=True,
    )


@pytest.fixture
def valid_deferred_payload() -> DeferredPayload:
    """A valid DeferredPayload."""
    return DeferredPayload(
        code="DEF001",
        reason="Awaiting user input",
        required_inputs=[
            RequiredInput(name="user_input", type="string"),
        ],
    )


# =============================================================================
# Test: ResponseType
# =============================================================================

class TestResponseType:
    """Tests for ResponseType enum."""
    
    def test_accepted_value(self):
        """ACCEPTED has correct value."""
        assert ResponseType.ACCEPTED.value == "accepted"
    
    def test_rejected_value(self):
        """REJECTED has correct value."""
        assert ResponseType.REJECTED.value == "rejected"
    
    def test_deferred_value(self):
        """DEFERRED has correct value."""
        assert ResponseType.DEFERRED.value == "deferred"
    
    def test_all_types_exist(self):
        """All expected types exist."""
        types = {ResponseType.ACCEPTED, ResponseType.REJECTED, ResponseType.DEFERRED}
        assert len(types) == 3


# =============================================================================
# Test: ContinuationToken
# =============================================================================

class TestContinuationToken:
    """Tests for ContinuationToken."""
    
    def test_create_minimal(self):
        """Create minimal ContinuationToken."""
        token = ContinuationToken(token_id="token-123")
        assert token.token_id == "token-123"
        assert token.context == {}
        assert token.expires_at is None
    
    def test_create_with_all_fields(self):
        """Create ContinuationToken with all fields."""
        token = ContinuationToken(
            token_id="token-123",
            context={"step": 1},
            expires_at=1000,
        )
        assert token.context == {"step": 1}
        assert token.expires_at == 1000
    
    def test_missing_token_id_raises(self):
        """Missing token_id raises error."""
        with pytest.raises(MissingPayloadFieldError):
            ContinuationToken(token_id=None)  # type: ignore
    
    def test_empty_token_id_raises(self):
        """Empty token_id raises error."""
        with pytest.raises(ResponseValidationError):
            ContinuationToken(token_id="   ")
    
    def test_negative_expires_at_raises(self):
        """Negative expires_at raises error."""
        with pytest.raises(ResponseValidationError):
            ContinuationToken(token_id="token", expires_at=-1)
    
    def test_immutability(self):
        """ContinuationToken is immutable."""
        token = ContinuationToken(token_id="token-123")
        with pytest.raises(ResponseImmutabilityError):
            token.token_id = "changed"  # type: ignore
    
    def test_context_copy_protection(self):
        """Context is deep-copied."""
        ctx = {"key": "value"}
        token = ContinuationToken(token_id="token", context=ctx)
        ctx["key"] = "changed"
        assert token.context == {"key": "value"}
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        token = ContinuationToken(
            token_id="token-123",
            context={"a": 1},
            expires_at=1000,
        )
        d = token.to_dict()
        assert d["token_id"] == "token-123"
        assert d["context"] == {"a": 1}
        assert d["expires_at"] == 1000
    
    def test_from_dict(self):
        """from_dict correctly constructs ContinuationToken."""
        data = {"token_id": "token-123", "context": {"a": 1}}
        token = ContinuationToken.from_dict(data)
        assert token.token_id == "token-123"
        assert token.context == {"a": 1}
    
    def test_equality(self):
        """Equal tokens are equal."""
        t1 = ContinuationToken(token_id="token", context={"a": 1})
        t2 = ContinuationToken(token_id="token", context={"a": 1})
        assert t1 == t2


# =============================================================================
# Test: RequiredInput
# =============================================================================

class TestRequiredInput:
    """Tests for RequiredInput."""
    
    def test_create_minimal(self):
        """Create minimal RequiredInput."""
        ri = RequiredInput(name="input", type="string")
        assert ri.name == "input"
        assert ri.type == "string"
        assert ri.reason is None
        assert ri.constraints == {}
    
    def test_create_with_all_fields(self):
        """Create RequiredInput with all fields."""
        ri = RequiredInput(
            name="input",
            type="string",
            reason="Needed for validation",
            constraints={"min_length": 1},
        )
        assert ri.reason == "Needed for validation"
        assert ri.constraints == {"min_length": 1}
    
    def test_missing_name_raises(self):
        """Missing name raises error."""
        with pytest.raises(MissingPayloadFieldError):
            RequiredInput(name=None, type="string")  # type: ignore
    
    def test_missing_type_raises(self):
        """Missing type raises error."""
        with pytest.raises(InvalidPayloadTypeError):
            RequiredInput(name="input", type=None)  # type: ignore
    
    def test_immutability(self):
        """RequiredInput is immutable."""
        ri = RequiredInput(name="input", type="string")
        with pytest.raises(ResponseImmutabilityError):
            ri.name = "changed"  # type: ignore
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        ri = RequiredInput(name="input", type="string", reason="test")
        d = ri.to_dict()
        assert d["name"] == "input"
        assert d["type"] == "string"
        assert d["reason"] == "test"
    
    def test_from_dict(self):
        """from_dict correctly constructs RequiredInput."""
        data = {"name": "input", "type": "string"}
        ri = RequiredInput.from_dict(data)
        assert ri.name == "input"
        assert ri.type == "string"
    
    def test_equality(self):
        """Equal RequiredInputs are equal."""
        ri1 = RequiredInput(name="input", type="string")
        ri2 = RequiredInput(name="input", type="string")
        assert ri1 == ri2


# =============================================================================
# Test: AcceptedPayload
# =============================================================================

class TestAcceptedPayload:
    """Tests for AcceptedPayload."""
    
    def test_create_minimal(self, valid_decision: Decision, valid_trace: Trace):
        """Create minimal AcceptedPayload."""
        payload = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        assert payload.decision == valid_decision
        assert payload.trace == valid_trace
        assert payload.effects is None
        assert payload.continuation is None
    
    def test_create_with_effects(self, valid_decision: Decision, valid_trace: Trace):
        """Create AcceptedPayload with effects."""
        effect = DeclaredEffect(
            category=EffectCategory.STATE_CHANGE,
            target="test-target",
            description="Test effect",
        )
        payload = AcceptedPayload(
            decision=valid_decision,
            trace=valid_trace,
            effects=[effect],
        )
        assert payload.effects is not None
        assert len(payload.effects) == 1
    
    def test_create_with_continuation(self, valid_decision: Decision, valid_trace: Trace):
        """Create AcceptedPayload with continuation."""
        token = ContinuationToken(token_id="token-123")
        payload = AcceptedPayload(
            decision=valid_decision,
            trace=valid_trace,
            continuation=token,
        )
        assert payload.continuation is not None
        assert payload.continuation.token_id == "token-123"
    
    def test_create_with_continuation_dict(
        self, valid_decision: Decision, valid_trace: Trace
    ):
        """Create AcceptedPayload with continuation as dict."""
        payload = AcceptedPayload(
            decision=valid_decision,
            trace=valid_trace,
            continuation={"token_id": "token-123"},
        )
        assert payload.continuation is not None
        assert payload.continuation.token_id == "token-123"
    
    def test_missing_decision_raises(self, valid_trace: Trace):
        """Missing decision raises error."""
        with pytest.raises(MissingPayloadFieldError):
            AcceptedPayload(decision=None, trace=valid_trace)  # type: ignore
    
    def test_missing_trace_raises(self, valid_decision: Decision):
        """Missing trace raises error."""
        with pytest.raises(MissingPayloadFieldError):
            AcceptedPayload(decision=valid_decision, trace=None)  # type: ignore
    
    def test_invalid_decision_type_raises(self, valid_trace: Trace):
        """Invalid decision type raises error."""
        with pytest.raises(InvalidPayloadTypeError):
            AcceptedPayload(decision="not-a-decision", trace=valid_trace)  # type: ignore
    
    def test_invalid_trace_type_raises(self, valid_decision: Decision):
        """Invalid trace type raises error."""
        with pytest.raises(InvalidTraceError):
            AcceptedPayload(decision=valid_decision, trace="not-a-trace")  # type: ignore
    
    def test_immutability(self, valid_decision: Decision, valid_trace: Trace):
        """AcceptedPayload is immutable."""
        payload = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        with pytest.raises(ResponseImmutabilityError):
            payload.decision = None  # type: ignore
    
    def test_to_dict(self, valid_decision: Decision, valid_trace: Trace):
        """to_dict returns correct dictionary."""
        payload = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        d = payload.to_dict()
        assert d["type"] == "accepted"
        assert "decision" in d
        assert "trace_id" in d


# =============================================================================
# Test: RejectedPayload
# =============================================================================

class TestRejectedPayload:
    """Tests for RejectedPayload."""
    
    def test_create_minimal(self):
        """Create minimal RejectedPayload."""
        payload = RejectedPayload(
            code="ERR001",
            category="validation",
            reason="Invalid request",
            recoverable=True,
        )
        assert payload.code == "ERR001"
        assert payload.category == "validation"
        assert payload.reason == "Invalid request"
        assert payload.recoverable is True
        assert payload.details is None
        assert payload.suggestions is None
        assert payload.partial_trace is None
    
    def test_create_with_all_fields(self, valid_trace: Trace):
        """Create RejectedPayload with all fields."""
        payload = RejectedPayload(
            code="ERR001",
            category="validation",
            reason="Invalid request",
            recoverable=False,
            details={"field": "value"},
            suggestions=["Try X", "Try Y"],
            partial_trace=valid_trace,
        )
        assert payload.details == {"field": "value"}
        assert payload.suggestions == ("Try X", "Try Y")
        assert payload.partial_trace is not None
    
    def test_missing_code_raises(self):
        """Missing code raises error."""
        with pytest.raises(MissingPayloadFieldError):
            RejectedPayload(
                code=None,  # type: ignore
                category="cat",
                reason="reason",
                recoverable=True,
            )
    
    def test_missing_recoverable_raises(self):
        """Missing recoverable raises error."""
        with pytest.raises(MissingPayloadFieldError):
            RejectedPayload(
                code="ERR",
                category="cat",
                reason="reason",
                recoverable=None,  # type: ignore
            )
    
    def test_invalid_recoverable_type_raises(self):
        """Invalid recoverable type raises error."""
        with pytest.raises(InvalidPayloadTypeError):
            RejectedPayload(
                code="ERR",
                category="cat",
                reason="reason",
                recoverable="yes",  # type: ignore
            )
    
    def test_immutability(self):
        """RejectedPayload is immutable."""
        payload = RejectedPayload(
            code="ERR",
            category="cat",
            reason="reason",
            recoverable=True,
        )
        with pytest.raises(ResponseImmutabilityError):
            payload.code = "changed"  # type: ignore
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        payload = RejectedPayload(
            code="ERR001",
            category="validation",
            reason="Test",
            recoverable=True,
        )
        d = payload.to_dict()
        assert d["type"] == "rejected"
        assert d["code"] == "ERR001"
        assert d["category"] == "validation"


# =============================================================================
# Test: DeferredPayload
# =============================================================================

class TestDeferredPayload:
    """Tests for DeferredPayload."""
    
    def test_create_minimal(self):
        """Create minimal DeferredPayload."""
        payload = DeferredPayload(
            code="DEF001",
            reason="Need more info",
            required_inputs=[RequiredInput(name="x", type="int")],
        )
        assert payload.code == "DEF001"
        assert payload.reason == "Need more info"
        assert len(payload.required_inputs) == 1
        assert payload.timeout is None
        assert payload.partial_state is None
    
    def test_create_with_all_fields(self):
        """Create DeferredPayload with all fields."""
        payload = DeferredPayload(
            code="DEF001",
            reason="Need more info",
            required_inputs=[RequiredInput(name="x", type="int")],
            timeout=5000,
            partial_state={"step": 1},
        )
        assert payload.timeout == 5000
        assert payload.partial_state == {"step": 1}
    
    def test_create_with_input_dicts(self):
        """Create DeferredPayload with required_inputs as dicts."""
        payload = DeferredPayload(
            code="DEF001",
            reason="Need more info",
            required_inputs=[{"name": "x", "type": "int"}],
        )
        assert len(payload.required_inputs) == 1
        assert payload.required_inputs[0].name == "x"
    
    def test_missing_required_inputs_raises(self):
        """Missing required_inputs raises error."""
        with pytest.raises(MissingPayloadFieldError):
            DeferredPayload(
                code="DEF",
                reason="reason",
                required_inputs=None,  # type: ignore
            )
    
    def test_negative_timeout_raises(self):
        """Negative timeout raises error."""
        with pytest.raises(ResponseValidationError):
            DeferredPayload(
                code="DEF",
                reason="reason",
                required_inputs=[RequiredInput(name="x", type="int")],
                timeout=-1,
            )
    
    def test_immutability(self):
        """DeferredPayload is immutable."""
        payload = DeferredPayload(
            code="DEF",
            reason="reason",
            required_inputs=[RequiredInput(name="x", type="int")],
        )
        with pytest.raises(ResponseImmutabilityError):
            payload.code = "changed"  # type: ignore
    
    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        payload = DeferredPayload(
            code="DEF001",
            reason="Test",
            required_inputs=[RequiredInput(name="x", type="int")],
        )
        d = payload.to_dict()
        assert d["type"] == "deferred"
        assert d["code"] == "DEF001"
        assert len(d["required_inputs"]) == 1


# =============================================================================
# Test: DecisionResponse - Basic Construction
# =============================================================================

class TestDecisionResponseConstruction:
    """Tests for DecisionResponse construction."""
    
    def test_create_accepted(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Create accepted response."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        assert response.request_id == valid_uuid
        assert response.response_type == ResponseType.ACCEPTED
        assert response.timestamp == 100
        assert response.is_accepted
        assert not response.is_rejected
        assert not response.is_deferred
    
    def test_create_rejected(
        self, valid_uuid: str, valid_rejected_payload: RejectedPayload
    ):
        """Create rejected response."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.REJECTED,
            timestamp=100,
            payload=valid_rejected_payload,
        )
        assert response.response_type == ResponseType.REJECTED
        assert response.is_rejected
    
    def test_create_deferred(
        self, valid_uuid: str, valid_deferred_payload: DeferredPayload
    ):
        """Create deferred response."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.DEFERRED,
            timestamp=100,
            payload=valid_deferred_payload,
        )
        assert response.response_type == ResponseType.DEFERRED
        assert response.is_deferred
    
    def test_create_with_string_response_type(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Create response with string response_type."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type="accepted",
            timestamp=100,
            payload=valid_accepted_payload,
        )
        assert response.response_type == ResponseType.ACCEPTED


# =============================================================================
# Test: DecisionResponse - Validation
# =============================================================================

class TestDecisionResponseValidation:
    """Tests for DecisionResponse validation."""
    
    def test_missing_request_id_raises(
        self, valid_accepted_payload: AcceptedPayload
    ):
        """Missing request_id raises error."""
        with pytest.raises(MissingResponseFieldError):
            DecisionResponse(
                request_id=None,  # type: ignore
                response_type=ResponseType.ACCEPTED,
                timestamp=100,
                payload=valid_accepted_payload,
            )
    
    def test_invalid_uuid_raises(
        self, valid_accepted_payload: AcceptedPayload
    ):
        """Invalid UUID raises error."""
        with pytest.raises(ResponseValidationError):
            DecisionResponse(
                request_id="not-a-uuid",
                response_type=ResponseType.ACCEPTED,
                timestamp=100,
                payload=valid_accepted_payload,
            )
    
    def test_missing_response_type_raises(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Missing response_type raises error."""
        with pytest.raises(MissingResponseFieldError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type=None,  # type: ignore
                timestamp=100,
                payload=valid_accepted_payload,
            )
    
    def test_invalid_response_type_raises(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Invalid response_type raises error."""
        with pytest.raises(InvalidResponseTypeError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type="invalid",  # type: ignore
                timestamp=100,
                payload=valid_accepted_payload,
            )
    
    def test_missing_timestamp_raises(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Missing timestamp raises error."""
        with pytest.raises(MissingResponseFieldError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type=ResponseType.ACCEPTED,
                timestamp=None,  # type: ignore
                payload=valid_accepted_payload,
            )
    
    def test_negative_timestamp_raises(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Negative timestamp raises error."""
        with pytest.raises(ResponseValidationError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type=ResponseType.ACCEPTED,
                timestamp=-1,
                payload=valid_accepted_payload,
            )
    
    def test_missing_payload_raises(self, valid_uuid: str):
        """Missing payload raises error."""
        with pytest.raises(MissingResponseFieldError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type=ResponseType.ACCEPTED,
                timestamp=100,
                payload=None,  # type: ignore
            )
    
    def test_payload_mismatch_accepted_rejected(
        self, valid_uuid: str, valid_rejected_payload: RejectedPayload
    ):
        """ACCEPTED with RejectedPayload raises error."""
        with pytest.raises(PayloadMismatchError) as exc_info:
            DecisionResponse(
                request_id=valid_uuid,
                response_type=ResponseType.ACCEPTED,
                timestamp=100,
                payload=valid_rejected_payload,
            )
        assert "accepted" in str(exc_info.value)
    
    def test_payload_mismatch_rejected_accepted(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """REJECTED with AcceptedPayload raises error."""
        with pytest.raises(PayloadMismatchError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type=ResponseType.REJECTED,
                timestamp=100,
                payload=valid_accepted_payload,
            )
    
    def test_payload_mismatch_deferred_accepted(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """DEFERRED with AcceptedPayload raises error."""
        with pytest.raises(PayloadMismatchError):
            DecisionResponse(
                request_id=valid_uuid,
                response_type=ResponseType.DEFERRED,
                timestamp=100,
                payload=valid_accepted_payload,
            )


# =============================================================================
# Test: DecisionResponse - Immutability
# =============================================================================

class TestDecisionResponseImmutability:
    """Tests for DecisionResponse immutability."""
    
    def test_cannot_set_attribute(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Cannot set attribute after creation."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        with pytest.raises(ResponseImmutabilityError):
            response.timestamp = 200  # type: ignore
    
    def test_cannot_delete_attribute(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Cannot delete attribute after creation."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        with pytest.raises(ResponseImmutabilityError):
            del response.payload  # type: ignore


# =============================================================================
# Test: DecisionResponse - Hashing and Equality
# =============================================================================

class TestDecisionResponseHashEquality:
    """Tests for DecisionResponse hashing and equality."""
    
    def test_equal_responses(
        self, valid_uuid: str, valid_decision: Decision, valid_trace: Trace
    ):
        """Identical responses are equal."""
        payload1 = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        payload2 = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        
        resp1 = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=payload1,
        )
        resp2 = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=payload2,
        )
        assert resp1 == resp2
    
    def test_equal_hash(
        self, valid_uuid: str, valid_decision: Decision, valid_trace: Trace
    ):
        """Identical responses have same hash."""
        payload1 = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        payload2 = AcceptedPayload(decision=valid_decision, trace=valid_trace)
        
        resp1 = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=payload1,
        )
        resp2 = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=payload2,
        )
        assert hash(resp1) == hash(resp2)
    
    def test_different_timestamp(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Different timestamps mean different responses."""
        resp1 = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        resp2 = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=200,
            payload=valid_accepted_payload,
        )
        assert resp1 != resp2
    
    def test_response_hash_is_deterministic(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """response_hash is deterministic."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        assert response.response_hash == response.response_hash
        # Hash should be a valid SHA-256 hex string
        assert len(response.response_hash) == 64
        assert all(c in "0123456789abcdef" for c in response.response_hash)


# =============================================================================
# Test: DecisionResponse - Payload Accessors
# =============================================================================

class TestDecisionResponsePayloadAccessors:
    """Tests for payload accessor properties."""
    
    def test_accepted_payload_accessor(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """accepted_payload returns payload for accepted response."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        assert response.accepted_payload is not None
        assert response.rejected_payload is None
        assert response.deferred_payload is None
    
    def test_rejected_payload_accessor(
        self, valid_uuid: str, valid_rejected_payload: RejectedPayload
    ):
        """rejected_payload returns payload for rejected response."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.REJECTED,
            timestamp=100,
            payload=valid_rejected_payload,
        )
        assert response.accepted_payload is None
        assert response.rejected_payload is not None
        assert response.deferred_payload is None
    
    def test_deferred_payload_accessor(
        self, valid_uuid: str, valid_deferred_payload: DeferredPayload
    ):
        """deferred_payload returns payload for deferred response."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.DEFERRED,
            timestamp=100,
            payload=valid_deferred_payload,
        )
        assert response.accepted_payload is None
        assert response.rejected_payload is None
        assert response.deferred_payload is not None


# =============================================================================
# Test: DecisionResponse - Serialization
# =============================================================================

class TestDecisionResponseSerialization:
    """Tests for DecisionResponse serialization."""
    
    def test_to_dict(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """to_dict returns correct dictionary."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        d = response.to_dict()
        assert d["request_id"] == valid_uuid
        assert d["response_type"] == "accepted"
        assert d["timestamp"] == 100
        assert "payload" in d
        assert "response_hash" in d
    
    def test_to_json_deterministic(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """JSON serialization is deterministic."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        json1 = response.to_json()
        json2 = response.to_json()
        assert json1 == json2
    
    def test_to_json_sorted_keys(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """JSON has sorted keys."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        json_str = response.to_json()
        # payload should appear before request_id
        assert json_str.index('"payload"') < json_str.index('"request_id"')
    
    def test_to_json_indent(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """JSON can be indented."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        json_str = response.to_json(indent=2)
        assert "\n" in json_str


# =============================================================================
# Test: DecisionResponse - Edge Cases
# =============================================================================

class TestDecisionResponseEdgeCases:
    """Edge case tests for DecisionResponse."""
    
    def test_zero_timestamp(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Zero timestamp is allowed."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=0,
            payload=valid_accepted_payload,
        )
        assert response.timestamp == 0
    
    def test_comparison_with_non_response(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """Comparing with non-DecisionResponse returns NotImplemented."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        result = response.__eq__("not-a-response")
        assert result is NotImplemented
    
    def test_repr(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """repr returns a string."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        r = repr(response)
        assert "DecisionResponse" in r
        assert valid_uuid in r
    
    def test_str(
        self, valid_uuid: str, valid_accepted_payload: AcceptedPayload
    ):
        """str returns a string."""
        response = DecisionResponse(
            request_id=valid_uuid,
            response_type=ResponseType.ACCEPTED,
            timestamp=100,
            payload=valid_accepted_payload,
        )
        s = str(response)
        assert "Response" in s
        assert "accepted" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
