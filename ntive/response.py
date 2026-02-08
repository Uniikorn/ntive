"""
response.py

Ntive DecisionResponse Primitive — Ecosystem Interaction Specification v1.0.0

A DecisionResponse is an immutable, validated representation of a response
from the Ntive system. It captures:
- The response type (ACCEPTED, REJECTED, DEFERRED)
- The corresponding payload
- Traceability information

Design Invariants:
- Immutable after creation
- Pure data structure (no side effects)
- Deterministic serialization (sorted keys, content-based hash)
- No execution logic
- Exactly one payload type per response
- Payload must match response type
"""

import hashlib
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ntive.capability import DeclaredEffect
from ntive.decision import Decision
from ntive.trace import Trace

# Save reference to built-in type before any shadowing
_builtin_type = type


# =============================================================================
# UUID v4 Validation
# =============================================================================

_UUID_V4_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


def _is_valid_uuid_v4(value: str) -> bool:
    """Check if a string is a valid UUID v4."""
    if not isinstance(value, str):
        return False
    return _UUID_V4_PATTERN.match(value) is not None


# =============================================================================
# Response Type Enum
# =============================================================================

class ResponseType(Enum):
    """
    The type of response from the Ntive system.
    """
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


# =============================================================================
# Response Errors
# =============================================================================

class ResponseValidationError(Exception):
    """
    Raised when a DecisionResponse cannot be constructed due to validation failure.
    """

    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        error_code: str = "RS000",
    ):
        self.message = message
        self.field_name = field_name
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        if self.field_name:
            return f"[{self.error_code}] Response validation failed for '{self.field_name}': {self.message}"
        return f"[{self.error_code}] Response validation failed: {self.message}"


class InvalidResponseTypeError(ResponseValidationError):
    """RS001 — Raised when response_type is invalid."""

    def __init__(self, value: Any):
        super().__init__(
            message=f"Invalid response type: {value!r}",
            field_name="response_type",
            error_code="RS001",
        )
        self.value = value


class PayloadMismatchError(ResponseValidationError):
    """RS002 — Raised when payload type doesn't match response_type."""

    def __init__(self, response_type: str, payload_type: str):
        super().__init__(
            message=f"Payload type '{payload_type}' does not match "
                    f"response type '{response_type}'",
            error_code="RS002",
        )
        self.response_type = response_type
        self.payload_type = payload_type


class MissingPayloadFieldError(ResponseValidationError):
    """RS003 — Raised when a required payload field is missing."""

    def __init__(self, field_name: str, payload_type: str):
        super().__init__(
            message=f"Required field '{field_name}' is missing in {payload_type}",
            field_name=field_name,
            error_code="RS003",
        )
        self.payload_type = payload_type


class InvalidPayloadTypeError(ResponseValidationError):
    """RS004 — Raised when a payload field has invalid type."""

    def __init__(self, field_name: str, expected: str, actual: str):
        super().__init__(
            message=f"Expected {expected}, got {actual}",
            field_name=field_name,
            error_code="RS004",
        )
        self.expected = expected
        self.actual = actual


class InvalidTraceError(ResponseValidationError):
    """RS005 — Raised when a trace reference is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid trace: {reason}",
            field_name="trace",
            error_code="RS005",
        )
        self.reason = reason


class MissingResponseFieldError(ResponseValidationError):
    """Raised when a required response field is missing."""

    def __init__(self, field_name: str):
        super().__init__(
            message=f"Required field '{field_name}' is missing or None",
            field_name=field_name,
            error_code="RS003",
        )


class ResponseImmutabilityError(Exception):
    """Raised when attempting to mutate an immutable response."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: DecisionResponse is immutable after creation"
        )


class ExecutableInResponseError(ResponseValidationError):
    """Raised when executable content is detected in a response."""

    def __init__(self, location: str):
        super().__init__(
            message=f"Executable content detected in {location}. "
                    "Response payloads must be pure data only.",
            field_name=location,
            error_code="RS006",
        )
        self.location = location


# =============================================================================
# Helper Functions
# =============================================================================

def _check_for_executable(value: Any, path: str = "root") -> None:
    """
    Recursively check for executable content in a value.
    """
    if callable(value):
        raise ExecutableInResponseError(path)

    if hasattr(value, '__code__'):
        raise ExecutableInResponseError(path)

    if isinstance(value, type):
        raise ExecutableInResponseError(path)

    if isinstance(value, dict):
        for k, v in value.items():
            _check_for_executable(k, f"{path}.key({k!r})")
            _check_for_executable(v, f"{path}[{k!r}]")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_for_executable(item, f"{path}[{i}]")


def _validate_json_serializable(value: Any, field_name: str) -> Any:
    """Validate that a value is JSON-serializable and contains no executables."""
    _check_for_executable(value, field_name)

    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise ResponseValidationError(
            f"{field_name} must be JSON-serializable: {e}",
            field_name=field_name,
            error_code="RS004",
        )

    return value


def _deep_copy_json(value: Any) -> Any:
    """Create a deep copy of a JSON-serializable value."""
    return json.loads(json.dumps(value, sort_keys=True))


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of JSON-serialized data."""
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def _validate_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    """Validate that a value is a string."""
    if value is None:
        raise MissingPayloadFieldError(field_name, "payload")
    if not isinstance(value, str):
        raise InvalidPayloadTypeError(
            field_name,
            "str",
            _builtin_type(value).__name__,
        )
    if not allow_empty and not value.strip():
        raise ResponseValidationError(
            f"{field_name} cannot be empty or whitespace-only",
            field_name=field_name,
            error_code="RS004",
        )
    return value


# =============================================================================
# ContinuationToken
# =============================================================================

class ContinuationToken:
    """
    A token for continuing a deferred or ongoing decision process.

    Pure data, no execution semantics.

    Attributes:
        token_id: Unique token identifier
        context: Opaque continuation context
        expires_at: Optional logical expiration time
    """

    __slots__ = ('_token_id', '_context', '_expires_at', '_frozen')

    def __init__(
        self,
        *,
        token_id: str,
        context: Optional[Dict[str, Any]] = None,
        expires_at: Optional[int] = None,
    ):
        # Validate token_id
        if token_id is None:
            raise MissingPayloadFieldError("token_id", "ContinuationToken")
        if not isinstance(token_id, str):
            raise InvalidPayloadTypeError(
                "token_id",
                "str",
                _builtin_type(token_id).__name__,
            )
        if not token_id.strip():
            raise ResponseValidationError(
                "token_id cannot be empty",
                field_name="token_id",
                error_code="RS004",
            )

        # Validate context
        if context is not None:
            if not isinstance(context, dict):
                raise InvalidPayloadTypeError(
                    "context",
                    "dict",
                    _builtin_type(context).__name__,
                )
            _validate_json_serializable(context, "continuation_token.context")
            context = _deep_copy_json(context)
        else:
            context = {}

        # Validate expires_at
        if expires_at is not None:
            if not isinstance(expires_at, int):
                raise InvalidPayloadTypeError(
                    "expires_at",
                    "int",
                    _builtin_type(expires_at).__name__,
                )
            if expires_at < 0:
                raise ResponseValidationError(
                    "expires_at must be non-negative",
                    field_name="expires_at",
                    error_code="RS004",
                )

        object.__setattr__(self, '_token_id', token_id)
        object.__setattr__(self, '_context', context)
        object.__setattr__(self, '_expires_at', expires_at)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def token_id(self) -> str:
        return self._token_id

    @property
    def context(self) -> Dict[str, Any]:
        return _deep_copy_json(self._context)

    @property
    def expires_at(self) -> Optional[int]:
        return self._expires_at

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinuationToken):
            return NotImplemented
        return (
            self._token_id == other._token_id
            and self._context == other._context
            and self._expires_at == other._expires_at
        )

    def __hash__(self) -> int:
        return hash((
            self._token_id,
            json.dumps(self._context, sort_keys=True),
            self._expires_at,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "context": self._context,
            "token_id": self._token_id,
        }
        if self._expires_at is not None:
            result["expires_at"] = self._expires_at
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContinuationToken":
        """Construct from a dictionary."""
        return cls(
            token_id=data["token_id"],
            context=data.get("context"),
            expires_at=data.get("expires_at"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ContinuationToken":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"ContinuationToken(token_id={self._token_id!r})"


# =============================================================================
# RequiredInput
# =============================================================================

class RequiredInput:
    """
    Describes an input that is required for a deferred decision.

    Attributes:
        name: Name of the required input
        type: Symbolic type of the input
        reason: Why this input is required
        constraints: Optional constraints on the input
    """

    __slots__ = ('_name', '_type', '_reason', '_constraints', '_frozen')

    def __init__(
        self,
        *,
        name: str,
        type: str,
        reason: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ):
        # Validate name
        name = _validate_string(name, "required_input.name")

        # Validate type
        if not isinstance(type, str):
            raise InvalidPayloadTypeError(
                "required_input.type",
                "str",
                _builtin_type(type).__name__,
            )
        type = _validate_string(type, "required_input.type")

        # Validate reason
        if reason is not None:
            reason = _validate_string(reason, "required_input.reason", allow_empty=True)

        # Validate constraints
        if constraints is not None:
            if not isinstance(constraints, dict):
                raise InvalidPayloadTypeError(
                    "required_input.constraints",
                    "dict",
                    _builtin_type(constraints).__name__,
                )
            _validate_json_serializable(constraints, "required_input.constraints")
            constraints = _deep_copy_json(constraints)
        else:
            constraints = {}

        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_type', type)
        object.__setattr__(self, '_reason', reason)
        object.__setattr__(self, '_constraints', constraints)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @property
    def constraints(self) -> Dict[str, Any]:
        return _deep_copy_json(self._constraints)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RequiredInput):
            return NotImplemented
        return (
            self._name == other._name
            and self._type == other._type
            and self._reason == other._reason
            and self._constraints == other._constraints
        )

    def __hash__(self) -> int:
        return hash((
            self._name,
            self._type,
            self._reason,
            json.dumps(self._constraints, sort_keys=True),
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "constraints": self._constraints,
            "name": self._name,
            "type": self._type,
        }
        if self._reason is not None:
            result["reason"] = self._reason
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequiredInput":
        """Construct from a dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            reason=data.get("reason"),
            constraints=data.get("constraints"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "RequiredInput":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"RequiredInput(name={self._name!r}, type={self._type!r})"


# =============================================================================
# AcceptedPayload
# =============================================================================

class AcceptedPayload:
    """
    Payload for an ACCEPTED response.

    Contains the decision, trace, and optionally effects and continuation.

    Attributes:
        decision: The decision that was made
        trace: The causal trace of the decision
        effects: Optional declared effects
        continuation: Optional continuation token
    """

    __slots__ = ('_decision', '_trace', '_effects', '_continuation', '_frozen')

    def __init__(
        self,
        *,
        decision: Decision,
        trace: Trace,
        effects: Optional[List[DeclaredEffect]] = None,
        continuation: Optional[ContinuationToken | Dict[str, Any]] = None,
    ):
        # Validate decision
        if decision is None:
            raise MissingPayloadFieldError("decision", "AcceptedPayload")
        if not isinstance(decision, Decision):
            raise InvalidPayloadTypeError(
                "decision",
                "Decision",
                _builtin_type(decision).__name__,
            )

        # Validate trace
        if trace is None:
            raise MissingPayloadFieldError("trace", "AcceptedPayload")
        if not isinstance(trace, Trace):
            raise InvalidTraceError(
                f"Expected Trace, got {_builtin_type(trace).__name__}"
            )

        # Validate effects
        parsed_effects: List[DeclaredEffect] = []
        if effects is not None:
            if not isinstance(effects, list):
                raise InvalidPayloadTypeError(
                    "effects",
                    "list",
                    _builtin_type(effects).__name__,
                )
            for i, eff in enumerate(effects):
                if not isinstance(eff, DeclaredEffect):
                    raise InvalidPayloadTypeError(
                        f"effects[{i}]",
                        "DeclaredEffect",
                        _builtin_type(eff).__name__,
                    )
                parsed_effects.append(eff)

        # Validate continuation
        if continuation is not None:
            if isinstance(continuation, dict):
                continuation = ContinuationToken.from_dict(continuation)
            elif not isinstance(continuation, ContinuationToken):
                raise InvalidPayloadTypeError(
                    "continuation",
                    "ContinuationToken or dict",
                    _builtin_type(continuation).__name__,
                )

        object.__setattr__(self, '_decision', decision)
        object.__setattr__(self, '_trace', trace)
        object.__setattr__(self, '_effects', tuple(parsed_effects) if parsed_effects else None)
        object.__setattr__(self, '_continuation', continuation)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def decision(self) -> Decision:
        return self._decision

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def effects(self) -> Optional[Tuple[DeclaredEffect, ...]]:
        return self._effects

    @property
    def continuation(self) -> Optional[ContinuationToken]:
        return self._continuation

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AcceptedPayload):
            return NotImplemented
        return (
            self._decision == other._decision
            and self._trace == other._trace
            and self._effects == other._effects
            and self._continuation == other._continuation
        )

    def __hash__(self) -> int:
        return hash((
            self._decision,
            self._trace,
            self._effects,
            self._continuation,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "decision": self._decision.to_dict(),
            "trace_id": self._trace.trace_id,
            "type": "accepted",
        }
        if self._effects is not None:
            result["effects"] = [e.to_dict() for e in self._effects]
        if self._continuation is not None:
            result["continuation"] = self._continuation.to_dict()
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def __repr__(self) -> str:
        return (
            f"AcceptedPayload(decision_id={self._decision.decision_id!r}, "
            f"trace_id={self._trace.trace_id[:8]}...)"
        )


# =============================================================================
# RejectedPayload
# =============================================================================

class RejectedPayload:
    """
    Payload for a REJECTED response.

    Describes why the request was rejected.

    Attributes:
        code: Error code
        category: Error category
        reason: Human-readable reason
        recoverable: Whether the error is recoverable
        details: Optional additional details
        suggestions: Optional suggested actions
        partial_trace: Optional partial trace up to failure
    """

    __slots__ = (
        '_code',
        '_category',
        '_reason',
        '_recoverable',
        '_details',
        '_suggestions',
        '_partial_trace',
        '_frozen',
    )

    def __init__(
        self,
        *,
        code: str,
        category: str,
        reason: str,
        recoverable: bool,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        partial_trace: Optional[Trace] = None,
    ):
        # Validate required fields
        code = _validate_string(code, "code")
        category = _validate_string(category, "category")
        reason = _validate_string(reason, "reason")

        if recoverable is None:
            raise MissingPayloadFieldError("recoverable", "RejectedPayload")
        if not isinstance(recoverable, bool):
            raise InvalidPayloadTypeError(
                "recoverable",
                "bool",
                _builtin_type(recoverable).__name__,
            )

        # Validate details
        if details is not None:
            if not isinstance(details, dict):
                raise InvalidPayloadTypeError(
                    "details",
                    "dict",
                    _builtin_type(details).__name__,
                )
            _validate_json_serializable(details, "details")
            details = _deep_copy_json(details)

        # Validate suggestions
        if suggestions is not None:
            if not isinstance(suggestions, list):
                raise InvalidPayloadTypeError(
                    "suggestions",
                    "list",
                    _builtin_type(suggestions).__name__,
                )
            for i, s in enumerate(suggestions):
                if not isinstance(s, str):
                    raise InvalidPayloadTypeError(
                        f"suggestions[{i}]",
                        "str",
                        _builtin_type(s).__name__,
                    )
            suggestions = list(suggestions)  # Copy

        # Validate partial_trace
        if partial_trace is not None:
            if not isinstance(partial_trace, Trace):
                raise InvalidTraceError(
                    f"Expected Trace, got {_builtin_type(partial_trace).__name__}"
                )

        object.__setattr__(self, '_code', code)
        object.__setattr__(self, '_category', category)
        object.__setattr__(self, '_reason', reason)
        object.__setattr__(self, '_recoverable', recoverable)
        object.__setattr__(self, '_details', details)
        object.__setattr__(self, '_suggestions', tuple(suggestions) if suggestions else None)
        object.__setattr__(self, '_partial_trace', partial_trace)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def code(self) -> str:
        return self._code

    @property
    def category(self) -> str:
        return self._category

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def recoverable(self) -> bool:
        return self._recoverable

    @property
    def details(self) -> Optional[Dict[str, Any]]:
        return _deep_copy_json(self._details) if self._details else None

    @property
    def suggestions(self) -> Optional[Tuple[str, ...]]:
        return self._suggestions

    @property
    def partial_trace(self) -> Optional[Trace]:
        return self._partial_trace

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RejectedPayload):
            return NotImplemented
        return (
            self._code == other._code
            and self._category == other._category
            and self._reason == other._reason
            and self._recoverable == other._recoverable
            and self._details == other._details
            and self._suggestions == other._suggestions
            and self._partial_trace == other._partial_trace
        )

    def __hash__(self) -> int:
        return hash((
            self._code,
            self._category,
            self._reason,
            self._recoverable,
            json.dumps(self._details, sort_keys=True) if self._details else None,
            self._suggestions,
            self._partial_trace.trace_id if self._partial_trace else None,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "category": self._category,
            "code": self._code,
            "reason": self._reason,
            "recoverable": self._recoverable,
            "type": "rejected",
        }
        if self._details is not None:
            result["details"] = self._details
        if self._suggestions is not None:
            result["suggestions"] = list(self._suggestions)
        if self._partial_trace is not None:
            result["partial_trace_id"] = self._partial_trace.trace_id
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def __repr__(self) -> str:
        return f"RejectedPayload(code={self._code!r}, category={self._category!r})"


# =============================================================================
# DeferredPayload
# =============================================================================

class DeferredPayload:
    """
    Payload for a DEFERRED response.

    Indicates the decision cannot be made yet and what is needed.

    Attributes:
        code: Deferral code
        reason: Human-readable reason for deferral
        required_inputs: List of inputs needed to proceed
        timeout: Optional timeout (logical units)
        partial_state: Optional partial state to preserve
    """

    __slots__ = (
        '_code',
        '_reason',
        '_required_inputs',
        '_timeout',
        '_partial_state',
        '_frozen',
    )

    def __init__(
        self,
        *,
        code: str,
        reason: str,
        required_inputs: List[RequiredInput | Dict[str, Any]],
        timeout: Optional[int] = None,
        partial_state: Optional[Dict[str, Any]] = None,
    ):
        # Validate required fields
        code = _validate_string(code, "code")
        reason = _validate_string(reason, "reason")

        # Validate required_inputs
        if required_inputs is None:
            raise MissingPayloadFieldError("required_inputs", "DeferredPayload")
        if not isinstance(required_inputs, list):
            raise InvalidPayloadTypeError(
                "required_inputs",
                "list",
                _builtin_type(required_inputs).__name__,
            )

        parsed_inputs: List[RequiredInput] = []
        for i, inp in enumerate(required_inputs):
            if isinstance(inp, dict):
                ri = RequiredInput.from_dict(inp)
            elif isinstance(inp, RequiredInput):
                ri = inp
            else:
                raise InvalidPayloadTypeError(
                    f"required_inputs[{i}]",
                    "RequiredInput or dict",
                    _builtin_type(inp).__name__,
                )
            parsed_inputs.append(ri)

        # Validate timeout
        if timeout is not None:
            if not isinstance(timeout, int):
                raise InvalidPayloadTypeError(
                    "timeout",
                    "int",
                    _builtin_type(timeout).__name__,
                )
            if timeout < 0:
                raise ResponseValidationError(
                    "timeout must be non-negative",
                    field_name="timeout",
                    error_code="RS004",
                )

        # Validate partial_state
        if partial_state is not None:
            if not isinstance(partial_state, dict):
                raise InvalidPayloadTypeError(
                    "partial_state",
                    "dict",
                    _builtin_type(partial_state).__name__,
                )
            _validate_json_serializable(partial_state, "partial_state")
            partial_state = _deep_copy_json(partial_state)

        object.__setattr__(self, '_code', code)
        object.__setattr__(self, '_reason', reason)
        object.__setattr__(self, '_required_inputs', tuple(parsed_inputs))
        object.__setattr__(self, '_timeout', timeout)
        object.__setattr__(self, '_partial_state', partial_state)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def code(self) -> str:
        return self._code

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def required_inputs(self) -> Tuple[RequiredInput, ...]:
        return self._required_inputs

    @property
    def timeout(self) -> Optional[int]:
        return self._timeout

    @property
    def partial_state(self) -> Optional[Dict[str, Any]]:
        return _deep_copy_json(self._partial_state) if self._partial_state else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeferredPayload):
            return NotImplemented
        return (
            self._code == other._code
            and self._reason == other._reason
            and self._required_inputs == other._required_inputs
            and self._timeout == other._timeout
            and self._partial_state == other._partial_state
        )

    def __hash__(self) -> int:
        return hash((
            self._code,
            self._reason,
            self._required_inputs,
            self._timeout,
            json.dumps(self._partial_state, sort_keys=True) if self._partial_state else None,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "code": self._code,
            "reason": self._reason,
            "required_inputs": [ri.to_dict() for ri in self._required_inputs],
            "type": "deferred",
        }
        if self._timeout is not None:
            result["timeout"] = self._timeout
        if self._partial_state is not None:
            result["partial_state"] = self._partial_state
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def __repr__(self) -> str:
        return (
            f"DeferredPayload(code={self._code!r}, "
            f"required_inputs={len(self._required_inputs)})"
        )


# Type alias for payload union
ResponsePayload = Union[AcceptedPayload, RejectedPayload, DeferredPayload]


# =============================================================================
# DecisionResponse
# =============================================================================

class DecisionResponse:
    """
    An immutable response from the Ntive system.

    This is the boundary contract for outputs from Ntive.
    It captures the decision outcome with full traceability.

    Ntive does NOT:
    - Execute anything
    - Mutate external state
    - Access external systems

    Attributes:
        request_id: UUID v4 of the originating request
        response_type: ACCEPTED, REJECTED, or DEFERRED
        timestamp: Logical timestamp (not wall clock)
        payload: The response payload (matches response_type)
        response_hash: Content-based hash (computed)
    """

    __slots__ = (
        '_request_id',
        '_response_type',
        '_timestamp',
        '_payload',
        '_response_hash',
        '_frozen',
    )

    def __init__(
        self,
        *,
        request_id: str,
        response_type: ResponseType | str,
        timestamp: int,
        payload: ResponsePayload,
    ):
        # =====================================================================
        # Validate required fields
        # =====================================================================

        # request_id
        if request_id is None:
            raise MissingResponseFieldError("request_id")
        if not isinstance(request_id, str):
            raise InvalidPayloadTypeError(
                "request_id",
                "str",
                _builtin_type(request_id).__name__,
            )
        if not _is_valid_uuid_v4(request_id):
            raise ResponseValidationError(
                f"request_id must be a valid UUID v4, got: {request_id!r}",
                field_name="request_id",
                error_code="RS004",
            )

        # response_type
        if response_type is None:
            raise MissingResponseFieldError("response_type")
        if isinstance(response_type, str):
            try:
                response_type = ResponseType(response_type.lower())
            except ValueError:
                raise InvalidResponseTypeError(response_type)
        elif not isinstance(response_type, ResponseType):
            raise InvalidResponseTypeError(response_type)

        # timestamp
        if timestamp is None:
            raise MissingResponseFieldError("timestamp")
        if not isinstance(timestamp, int):
            raise InvalidPayloadTypeError(
                "timestamp",
                "int",
                _builtin_type(timestamp).__name__,
            )
        if timestamp < 0:
            raise ResponseValidationError(
                "timestamp must be non-negative",
                field_name="timestamp",
                error_code="RS004",
            )

        # payload
        if payload is None:
            raise MissingResponseFieldError("payload")

        # Validate payload type matches response_type
        _validate_payload_match(response_type, payload)

        object.__setattr__(self, '_request_id', request_id)
        object.__setattr__(self, '_response_type', response_type)
        object.__setattr__(self, '_timestamp', timestamp)
        object.__setattr__(self, '_payload', payload)

        # Compute content-based hash
        response_hash = self._compute_response_hash()
        object.__setattr__(self, '_response_hash', response_hash)

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise ResponseImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Internal Helper Methods
    # -------------------------------------------------------------------------

    def _compute_response_hash(self) -> str:
        """Compute content-based SHA-256 hash for response."""
        data = {
            "payload": self._payload.to_dict(),
            "request_id": self._request_id,
            "response_type": self._response_type.value,
            "timestamp": self._timestamp,
        }
        return _compute_hash(data)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def request_id(self) -> str:
        """UUID v4 of the originating request."""
        return self._request_id

    @property
    def response_type(self) -> ResponseType:
        """The response type."""
        return self._response_type

    @property
    def timestamp(self) -> int:
        """Logical timestamp."""
        return self._timestamp

    @property
    def payload(self) -> ResponsePayload:
        """The response payload."""
        return self._payload

    @property
    def response_hash(self) -> str:
        """Content-based hash of this response."""
        return self._response_hash

    # -------------------------------------------------------------------------
    # Convenience accessors
    # -------------------------------------------------------------------------

    @property
    def is_accepted(self) -> bool:
        """Whether this is an accepted response."""
        return self._response_type == ResponseType.ACCEPTED

    @property
    def is_rejected(self) -> bool:
        """Whether this is a rejected response."""
        return self._response_type == ResponseType.REJECTED

    @property
    def is_deferred(self) -> bool:
        """Whether this is a deferred response."""
        return self._response_type == ResponseType.DEFERRED

    @property
    def accepted_payload(self) -> Optional[AcceptedPayload]:
        """Get accepted payload if this is an accepted response."""
        if self._response_type == ResponseType.ACCEPTED:
            return self._payload  # type: ignore
        return None

    @property
    def rejected_payload(self) -> Optional[RejectedPayload]:
        """Get rejected payload if this is a rejected response."""
        if self._response_type == ResponseType.REJECTED:
            return self._payload  # type: ignore
        return None

    @property
    def deferred_payload(self) -> Optional[DeferredPayload]:
        """Get deferred payload if this is a deferred response."""
        if self._response_type == ResponseType.DEFERRED:
            return self._payload  # type: ignore
        return None

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DecisionResponse):
            return NotImplemented
        return self._response_hash == other._response_hash

    def __hash__(self) -> int:
        return hash(self._response_hash)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "payload": self._payload.to_dict(),
            "request_id": self._request_id,
            "response_hash": self._response_hash,
            "response_type": self._response_type.value,
            "timestamp": self._timestamp,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DecisionResponse(request_id={self._request_id!r}, "
            f"type={self._response_type.value}, "
            f"timestamp={self._timestamp})"
        )

    def __str__(self) -> str:
        return f"Response[{self._request_id[:8]}...]: {self._response_type.value}"


# =============================================================================
# Payload validation helper
# =============================================================================

def _validate_payload_match(response_type: ResponseType, payload: Any) -> None:
    """Validate that payload type matches response_type."""
    expected_types = {
        ResponseType.ACCEPTED: AcceptedPayload,
        ResponseType.REJECTED: RejectedPayload,
        ResponseType.DEFERRED: DeferredPayload,
    }

    expected = expected_types[response_type]

    if not isinstance(payload, expected):
        actual_name = _builtin_type(payload).__name__
        raise PayloadMismatchError(response_type.value, actual_name)
