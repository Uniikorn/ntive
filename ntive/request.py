"""
request.py

Ntive DecisionRequest Primitive — Ecosystem Interaction Specification v1.0.0

A DecisionRequest is an immutable, validated representation of a request
for a decision from the Ntive system. It captures:
- WHAT is being asked (query)
- WITH what context (context, memory)
- UNDER what constraints (policy, constraints)
- USING what capabilities

Design Invariants:
- Immutable after creation
- Pure data structure (no side effects)
- Deterministic serialization (sorted keys, content-based hash)
- No execution logic
- No evaluation of query or context
- All validation at construction time
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ntive.capability import CapabilityDescriptor
from ntive.memory import MemoryScope
from ntive.policy import Policy

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
# Request Errors
# =============================================================================

class RequestValidationError(Exception):
    """
    Raised when a DecisionRequest cannot be constructed due to validation failure.
    """

    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        error_code: str = "R000",
    ):
        self.message = message
        self.field_name = field_name
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        if self.field_name:
            return f"[{self.error_code}] Request validation failed for '{self.field_name}': {self.message}"
        return f"[{self.error_code}] Request validation failed: {self.message}"


class MissingRequestFieldError(RequestValidationError):
    """R001 — Raised when a required field is missing or None."""

    def __init__(self, field_name: str):
        super().__init__(
            message=f"Required field '{field_name}' is missing or None",
            field_name=field_name,
            error_code="R001",
        )


class InvalidRequestFieldError(RequestValidationError):
    """R002 — Raised when a field has an invalid type."""

    def __init__(self, field_name: str, expected: str, actual: str):
        super().__init__(
            message=f"Expected {expected}, got {actual}",
            field_name=field_name,
            error_code="R002",
        )
        self.expected = expected
        self.actual = actual


class InvalidRequestIdError(RequestValidationError):
    """R003 — Raised when request_id is not a valid UUID v4."""

    def __init__(self, value: str):
        super().__init__(
            message=f"request_id must be a valid UUID v4, got: {value!r}",
            field_name="request_id",
            error_code="R003",
        )
        self.value = value


class NonSerializableContextError(RequestValidationError):
    """R004 — Raised when context contains non-serializable content."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Context must be JSON-serializable: {reason}",
            field_name="context",
            error_code="R004",
        )
        self.reason = reason


class InvalidCapabilityError(RequestValidationError):
    """R005 — Raised when a capability reference is invalid."""

    def __init__(self, index: int, reason: str):
        super().__init__(
            message=f"Invalid capability at index {index}: {reason}",
            field_name="capabilities",
            error_code="R005",
        )
        self.index = index
        self.reason = reason


class InvalidPolicyError(RequestValidationError):
    """R006 — Raised when a policy reference is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid policy: {reason}",
            field_name="policy",
            error_code="R006",
        )
        self.reason = reason


class RequestImmutabilityError(Exception):
    """Raised when attempting to mutate an immutable request."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: DecisionRequest is immutable after creation"
        )


class ExecutableInRequestError(RequestValidationError):
    """Raised when executable content is detected in a request."""

    def __init__(self, location: str):
        super().__init__(
            message=f"Executable content detected in {location}. "
                    "DecisionRequest must be pure data only.",
            field_name=location,
            error_code="R007",
        )
        self.location = location


# =============================================================================
# Helper Functions
# =============================================================================

def _check_for_executable(value: Any, path: str = "root") -> None:
    """
    Recursively check for executable content in a value.

    Raises ExecutableInRequestError if found.
    """
    if callable(value):
        raise ExecutableInRequestError(path)

    if hasattr(value, '__code__'):
        raise ExecutableInRequestError(path)

    if isinstance(value, type):
        raise ExecutableInRequestError(path)

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
        raise NonSerializableContextError(str(e))

    return value


def _deep_copy_json(value: Any) -> Any:
    """Create a deep copy of a JSON-serializable value."""
    return json.loads(json.dumps(value, sort_keys=True))


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of JSON-serialized data."""
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# =============================================================================
# QueryDescriptor
# =============================================================================

class QueryDescriptor:
    """
    An opaque, immutable descriptor of what is being asked.

    Ntive does NOT interpret or execute the query.
    It is pure data that flows through the system.

    Attributes:
        intent: The symbolic intent identifier
        parameters: Opaque parameters for the query
        metadata: Additional metadata
    """

    __slots__ = ('_intent', '_parameters', '_metadata', '_frozen')

    def __init__(
        self,
        *,
        intent: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Validate intent
        if intent is None:
            raise MissingRequestFieldError("query.intent")
        if not isinstance(intent, str):
            raise InvalidRequestFieldError(
                "query.intent",
                "str",
                _builtin_type(intent).__name__,
            )
        if not intent.strip():
            raise RequestValidationError(
                "intent cannot be empty or whitespace-only",
                field_name="query.intent",
                error_code="R002",
            )

        # Validate parameters
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise InvalidRequestFieldError(
                    "query.parameters",
                    "dict",
                    _builtin_type(parameters).__name__,
                )
            _validate_json_serializable(parameters, "query.parameters")
            parameters = _deep_copy_json(parameters)
        else:
            parameters = {}

        # Validate metadata
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise InvalidRequestFieldError(
                    "query.metadata",
                    "dict",
                    _builtin_type(metadata).__name__,
                )
            _validate_json_serializable(metadata, "query.metadata")
            metadata = _deep_copy_json(metadata)
        else:
            metadata = {}

        object.__setattr__(self, '_intent', intent)
        object.__setattr__(self, '_parameters', parameters)
        object.__setattr__(self, '_metadata', metadata)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def intent(self) -> str:
        return self._intent

    @property
    def parameters(self) -> Dict[str, Any]:
        return _deep_copy_json(self._parameters)

    @property
    def metadata(self) -> Dict[str, Any]:
        return _deep_copy_json(self._metadata)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryDescriptor):
            return NotImplemented
        return (
            self._intent == other._intent
            and self._parameters == other._parameters
            and self._metadata == other._metadata
        )

    def __hash__(self) -> int:
        return hash((
            self._intent,
            json.dumps(self._parameters, sort_keys=True),
            json.dumps(self._metadata, sort_keys=True),
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "intent": self._intent,
            "metadata": self._metadata,
            "parameters": self._parameters,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryDescriptor":
        """Construct from a dictionary."""
        return cls(
            intent=data["intent"],
            parameters=data.get("parameters"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "QueryDescriptor":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"QueryDescriptor(intent={self._intent!r})"

    def __str__(self) -> str:
        return self._intent


# =============================================================================
# ContinuationRef
# =============================================================================

class ContinuationRef:
    """
    A reference to a continuation point.

    Used to resume or chain decision processes.
    Pure data, no execution semantics.

    Attributes:
        ref_id: Unique continuation identifier
        context: Opaque context for the continuation
    """

    __slots__ = ('_ref_id', '_context', '_frozen')

    def __init__(
        self,
        *,
        ref_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        # Validate ref_id
        if ref_id is None:
            raise MissingRequestFieldError("continuation.ref_id")
        if not isinstance(ref_id, str):
            raise InvalidRequestFieldError(
                "continuation.ref_id",
                "str",
                _builtin_type(ref_id).__name__,
            )
        if not ref_id.strip():
            raise RequestValidationError(
                "ref_id cannot be empty or whitespace-only",
                field_name="continuation.ref_id",
                error_code="R002",
            )

        # Validate context
        if context is not None:
            if not isinstance(context, dict):
                raise InvalidRequestFieldError(
                    "continuation.context",
                    "dict",
                    _builtin_type(context).__name__,
                )
            _validate_json_serializable(context, "continuation.context")
            context = _deep_copy_json(context)
        else:
            context = {}

        object.__setattr__(self, '_ref_id', ref_id)
        object.__setattr__(self, '_context', context)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def ref_id(self) -> str:
        return self._ref_id

    @property
    def context(self) -> Dict[str, Any]:
        return _deep_copy_json(self._context)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinuationRef):
            return NotImplemented
        return (
            self._ref_id == other._ref_id
            and self._context == other._context
        )

    def __hash__(self) -> int:
        return hash((
            self._ref_id,
            json.dumps(self._context, sort_keys=True),
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "context": self._context,
            "ref_id": self._ref_id,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContinuationRef":
        """Construct from a dictionary."""
        return cls(
            ref_id=data["ref_id"],
            context=data.get("context"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ContinuationRef":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"ContinuationRef(ref_id={self._ref_id!r})"

    def __str__(self) -> str:
        return self._ref_id


# =============================================================================
# RequestConstraints
# =============================================================================

_VALID_PRIORITIES = frozenset({"low", "normal", "high", "critical"})


class RequestConstraints:
    """
    Declarative constraints for a decision request.

    These are hints/limits, not execution rules.

    Attributes:
        timeout_hint: Suggested timeout (logical units only, must be > 0)
        max_alternatives: Maximum number of alternatives to consider (must be >= 1)
        priority: Request priority level ("low", "normal", "high", "critical")
        metadata: Additional constraint metadata
    """

    __slots__ = (
        '_timeout_hint',
        '_max_alternatives',
        '_priority',
        '_metadata',
        '_frozen',
    )

    def __init__(
        self,
        *,
        timeout_hint: Optional[int] = None,
        max_alternatives: Optional[int] = None,
        priority: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Validate timeout_hint
        if timeout_hint is not None:
            if not isinstance(timeout_hint, int):
                raise InvalidRequestFieldError(
                    "constraints.timeout_hint",
                    "int",
                    _builtin_type(timeout_hint).__name__,
                )
            if timeout_hint < 1:
                raise RequestValidationError(
                    "timeout_hint must be positive (> 0)",
                    field_name="constraints.timeout_hint",
                    error_code="R002",
                )

        # Validate max_alternatives
        if max_alternatives is not None:
            if not isinstance(max_alternatives, int):
                raise InvalidRequestFieldError(
                    "constraints.max_alternatives",
                    "int",
                    _builtin_type(max_alternatives).__name__,
                )
            if max_alternatives < 1:
                raise RequestValidationError(
                    "max_alternatives must be at least 1",
                    field_name="constraints.max_alternatives",
                    error_code="R002",
                )

        # Validate priority (default to "normal" if not specified)
        if priority is None:
            priority = "normal"
        if not isinstance(priority, str):
            raise InvalidRequestFieldError(
                "constraints.priority",
                "str",
                _builtin_type(priority).__name__,
            )
        priority = priority.lower()
        if priority not in _VALID_PRIORITIES:
            raise RequestValidationError(
                f"priority must be one of {sorted(_VALID_PRIORITIES)}, got {priority!r}",
                field_name="constraints.priority",
                error_code="R002",
            )

        # Validate metadata
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise InvalidRequestFieldError(
                    "constraints.metadata",
                    "dict",
                    _builtin_type(metadata).__name__,
                )
            _validate_json_serializable(metadata, "constraints.metadata")
            metadata = _deep_copy_json(metadata)
        else:
            metadata = {}

        object.__setattr__(self, '_timeout_hint', timeout_hint)
        object.__setattr__(self, '_max_alternatives', max_alternatives)
        object.__setattr__(self, '_priority', priority)
        object.__setattr__(self, '_metadata', metadata)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    @property
    def timeout_hint(self) -> Optional[int]:
        return self._timeout_hint

    @property
    def max_alternatives(self) -> Optional[int]:
        return self._max_alternatives

    @property
    def priority(self) -> str:
        return self._priority

    @property
    def metadata(self) -> Dict[str, Any]:
        return _deep_copy_json(self._metadata)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RequestConstraints):
            return NotImplemented
        return (
            self._timeout_hint == other._timeout_hint
            and self._max_alternatives == other._max_alternatives
            and self._priority == other._priority
            and self._metadata == other._metadata
        )

    def __hash__(self) -> int:
        return hash((
            self._timeout_hint,
            self._max_alternatives,
            self._priority,
            json.dumps(self._metadata, sort_keys=True),
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "metadata": self._metadata,
            "priority": self._priority,
        }
        if self._timeout_hint is not None:
            result["timeout_hint"] = self._timeout_hint
        if self._max_alternatives is not None:
            result["max_alternatives"] = self._max_alternatives
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestConstraints":
        """Construct from a dictionary."""
        return cls(
            timeout_hint=data.get("timeout_hint"),
            max_alternatives=data.get("max_alternatives"),
            priority=data.get("priority"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "RequestConstraints":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        parts = []
        if self._timeout_hint is not None:
            parts.append(f"timeout_hint={self._timeout_hint}")
        if self._max_alternatives is not None:
            parts.append(f"max_alternatives={self._max_alternatives}")
        if self._priority is not None:
            parts.append(f"priority={self._priority}")
        return f"RequestConstraints({', '.join(parts)})"


# =============================================================================
# DecisionRequest
# =============================================================================

class DecisionRequest:
    """
    An immutable request for a decision from the Ntive system.

    This is the boundary contract between external systems and Ntive.
    It captures what is being asked, with what context, and under what constraints.

    Ntive does NOT:
    - Execute anything in the request
    - Evaluate the query
    - Infer missing inputs
    - Access external systems

    Attributes:
        request_id: UUID v4 identifier (validated)
        query: What is being asked (opaque descriptor)
        context: Snapshot of relevant context (JSON-serializable)
        memory: Scoped memory state
        policy: Optional policy constraints
        capabilities: Available capabilities for this request
        continuation: Optional continuation reference
        constraints: Optional request constraints
        request_hash: Content-based hash (computed)
    """

    __slots__ = (
        '_request_id',
        '_query',
        '_context',
        '_memory',
        '_policy',
        '_capabilities',
        '_continuation',
        '_constraints',
        '_request_hash',
        '_frozen',
    )

    def __init__(
        self,
        *,
        request_id: str,
        query: QueryDescriptor | Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        memory: Optional[MemoryScope] = None,
        policy: Optional[Policy] = None,
        capabilities: Optional[List[CapabilityDescriptor]] = None,
        continuation: Optional[ContinuationRef | Dict[str, Any]] = None,
        constraints: Optional[RequestConstraints | Dict[str, Any]] = None,
    ):
        # =====================================================================
        # Validate required fields
        # =====================================================================

        # request_id
        if request_id is None:
            raise MissingRequestFieldError("request_id")
        if not isinstance(request_id, str):
            raise InvalidRequestFieldError(
                "request_id",
                "str",
                _builtin_type(request_id).__name__,
            )
        if not _is_valid_uuid_v4(request_id):
            raise InvalidRequestIdError(request_id)

        # query
        if query is None:
            raise MissingRequestFieldError("query")
        if isinstance(query, dict):
            query = QueryDescriptor.from_dict(query)
        elif not isinstance(query, QueryDescriptor):
            raise InvalidRequestFieldError(
                "query",
                "QueryDescriptor or dict",
                _builtin_type(query).__name__,
            )

        # =====================================================================
        # Validate optional fields
        # =====================================================================

        # context (optional, defaults to {})
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            raise InvalidRequestFieldError(
                "context",
                "dict",
                _builtin_type(context).__name__,
            )
        else:
            _validate_json_serializable(context, "context")
            context = _deep_copy_json(context)

        # memory (optional)
        if memory is not None:
            if not isinstance(memory, MemoryScope):
                raise InvalidRequestFieldError(
                    "memory",
                    "MemoryScope",
                    _builtin_type(memory).__name__,
                )

        # policy
        if policy is not None:
            if not isinstance(policy, Policy):
                raise InvalidPolicyError(
                    f"Expected Policy, got {_builtin_type(policy).__name__}"
                )

        # capabilities
        parsed_capabilities: List[CapabilityDescriptor] = []
        if capabilities is not None:
            if not isinstance(capabilities, list):
                raise InvalidRequestFieldError(
                    "capabilities",
                    "list",
                    _builtin_type(capabilities).__name__,
                )
            for i, cap in enumerate(capabilities):
                if not isinstance(cap, CapabilityDescriptor):
                    raise InvalidCapabilityError(
                        i,
                        f"Expected CapabilityDescriptor, got {_builtin_type(cap).__name__}",
                    )
                parsed_capabilities.append(cap)

        # continuation
        if continuation is not None:
            if isinstance(continuation, dict):
                continuation = ContinuationRef.from_dict(continuation)
            elif not isinstance(continuation, ContinuationRef):
                raise InvalidRequestFieldError(
                    "continuation",
                    "ContinuationRef or dict",
                    _builtin_type(continuation).__name__,
                )

        # constraints
        if constraints is not None:
            if isinstance(constraints, dict):
                constraints = RequestConstraints.from_dict(constraints)
            elif not isinstance(constraints, RequestConstraints):
                raise InvalidRequestFieldError(
                    "constraints",
                    "RequestConstraints or dict",
                    _builtin_type(constraints).__name__,
                )

        # =====================================================================
        # Set fields
        # =====================================================================

        object.__setattr__(self, '_request_id', request_id)
        object.__setattr__(self, '_query', query)
        object.__setattr__(self, '_context', context)
        object.__setattr__(self, '_memory', memory)
        object.__setattr__(self, '_policy', policy)
        object.__setattr__(self, '_capabilities', tuple(parsed_capabilities))
        object.__setattr__(self, '_continuation', continuation)
        object.__setattr__(self, '_constraints', constraints)

        # Compute content-based hash
        request_hash = self._compute_request_hash()
        object.__setattr__(self, '_request_hash', request_hash)

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise RequestImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Internal Helper Methods
    # -------------------------------------------------------------------------

    def _compute_request_hash(self) -> str:
        """Compute content-based SHA-256 hash for request."""
        data = {
            "capabilities": [cap.capability_id for cap in self._capabilities],
            "constraints": self._constraints.to_dict() if self._constraints is not None else None,
            "context": self._context,
            "continuation": self._continuation.to_dict() if self._continuation is not None else None,
            "memory_scope_id": self._memory.scope_id if self._memory is not None else None,
            "policy_id": self._policy.policy_id if self._policy is not None else None,
            "query": self._query.to_dict(),
            "request_id": self._request_id,
        }
        return _compute_hash(data)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def request_id(self) -> str:
        """UUID v4 request identifier."""
        return self._request_id

    @property
    def query(self) -> QueryDescriptor:
        """The query descriptor."""
        return self._query

    @property
    def context(self) -> Dict[str, Any]:
        """Return a copy of the context dict."""
        return _deep_copy_json(self._context)

    @property
    def memory(self) -> MemoryScope:
        """The memory scope."""
        return self._memory

    @property
    def policy(self) -> Optional[Policy]:
        """Optional policy."""
        return self._policy

    @property
    def capabilities(self) -> Tuple[CapabilityDescriptor, ...]:
        """Available capabilities."""
        return self._capabilities

    @property
    def continuation(self) -> Optional[ContinuationRef]:
        """Optional continuation reference."""
        return self._continuation

    @property
    def constraints(self) -> Optional[RequestConstraints]:
        """Optional request constraints."""
        return self._constraints

    @property
    def request_hash(self) -> str:
        """Content-based hash of this request."""
        return self._request_hash

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DecisionRequest):
            return NotImplemented
        return self._request_hash == other._request_hash

    def __hash__(self) -> int:
        return hash(self._request_hash)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result: Dict[str, Any] = {
            "capabilities": [cap.capability_id for cap in self._capabilities],
            "context": self._context,
            "memory_scope_id": self._memory.scope_id if self._memory is not None else None,
            "query": self._query.to_dict(),
            "request_hash": self._request_hash,
            "request_id": self._request_id,
        }

        if self._constraints is not None:
            result["constraints"] = self._constraints.to_dict()
        if self._continuation is not None:
            result["continuation"] = self._continuation.to_dict()
        if self._policy is not None:
            result["policy_id"] = self._policy.policy_id

        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    # NOTE: from_dict/from_json not provided because DecisionRequest requires
    # actual MemoryScope, Policy, CapabilityDescriptor objects which cannot
    # be reliably reconstructed from IDs alone without a registry.

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DecisionRequest(request_id={self._request_id!r}, "
            f"query={self._query.intent!r}, "
            f"capabilities={len(self._capabilities)})"
        )

    def __str__(self) -> str:
        return f"Request[{self._request_id[:8]}...]: {self._query.intent}"
