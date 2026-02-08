"""
Ntive Core — Deterministic Decision Engine
===========================================

Ntive Core provides a minimal, deterministic foundation for building
decision-making systems with full causal traceability.

Stability Guarantees (v1.x)
---------------------------
All symbols exported from this module are part of the **public API**
and follow semantic versioning:

- **v1.x.y**: No breaking changes to public API signatures or behavior.
  New features may be added in minor versions (1.x.0). Bug fixes in
  patch versions (1.x.y).

- **Backward Compatibility Promise**: Code written against ntive v1.0.0
  will continue to work with any v1.x.y release without modification.

- **Deprecation Policy**: Deprecated features will be marked with
  warnings for at least one minor release before removal in v2.0.0.

What's Public
-------------
Everything exported in ``__all__`` is public and stable:

- **Primitives**: Decision, Trace, MemoryScope, Policy, CapabilityDescriptor
- **Request/Response**: DecisionRequest, DecisionResponse and their types
- **Engine**: DecisionEngine (pure orchestration layer)
- **Exceptions**: All validation and runtime errors

What's Internal
---------------
The following are NOT part of the public API and may change without notice:

- ``ntive.parser`` — Experimental DSL parser (unstable)
- ``ntive.runtime`` — Experimental DSL runtime (unstable)
- ``ntive.ast_nodes`` — Internal AST representation
- ``ntive.errors`` — DSL-specific error classes
- Any symbol prefixed with underscore (``_``)

Example
-------
::

    from ntive import (
        DecisionRequest, DecisionResponse, DecisionEngine,
        CapabilityDescriptor, Policy, Decision, Trace
    )

    engine = DecisionEngine()
    response = engine.evaluate(request)

    if response.response_type == ResponseType.ACCEPTED:
        decision = response.payload.decision
        trace = response.payload.trace
"""

__version__ = "1.0.0"

# =============================================================================
# PUBLIC API — All symbols below are stable for v1.x
# =============================================================================

__all__ = [
    # --- Package Metadata ---
    "__version__",

    # --- Decision Primitive ---
    "Decision",
    "Alternative",
    "Confidence",
    "DecisionValidationError",
    "MissingRequiredFieldError",
    "InvalidFieldTypeError",
    "InvalidInputsError",
    "InvalidConfidenceError",
    "ImmutabilityViolationError",

    # --- Trace Primitive ---
    "Trace",
    "CausalReason",
    "TraceValidationError",
    "InvalidTraceNodeError",
    "InvalidCausalReasonError",
    "TraceImmutabilityError",

    # --- TraceReplay (Audit Tool) ---
    "TraceReplay",
    "ReplayNode",
    "ChainValidation",
    "ChainStatus",
    "DeterminismCheck",
    "ReplayExplanation",
    "TraceReplayError",
    "TraceLoadError",
    "BrokenCausalChainError",
    "DeterminismError",

    # --- MemoryScope Primitive ---
    "MemoryScope",
    "MemoryDiff",
    "MemoryValidationError",
    "InvalidMemoryKeyError",
    "InvalidMemoryValueError",
    "MemoryImmutabilityError",
    "KeyNotFoundError",

    # --- Policy Primitive ---
    "Policy",
    "PolicyRule",
    "PolicyEffectResult",
    "PolicyEffect",
    "ConflictResolutionStrategy",
    "PolicyValidationError",
    "InvalidPolicyRuleError",
    "DuplicateRuleIdError",
    "InvalidPolicyEffectError",
    "CyclicPolicyInheritanceError",
    "PolicyConflictError",
    "PolicyImmutabilityError",

    # --- CapabilityDescriptor Primitive ---
    "CapabilityDescriptor",
    "CapabilityInput",
    "DeclaredEffect",
    "Precondition",
    "Constraint",
    "EffectCategory",
    "CapabilityValidationError",
    "InvalidCapabilityInputError",
    "DuplicateInputKeyError",
    "InvalidPreconditionError",
    "InvalidEffectError",
    "InvalidConstraintError",
    "InvalidMetadataError",
    "ExecutableContentError",
    "CapabilityImmutabilityError",

    # --- DecisionRequest (Inbound) ---
    "DecisionRequest",
    "QueryDescriptor",
    "ContinuationRef",
    "RequestConstraints",
    "RequestValidationError",
    "MissingRequestFieldError",
    "InvalidRequestFieldError",
    "InvalidRequestIdError",
    "NonSerializableContextError",
    "InvalidCapabilityError",
    "InvalidPolicyError",
    "ExecutableInRequestError",
    "RequestImmutabilityError",

    # --- DecisionResponse (Outbound) ---
    "DecisionResponse",
    "ResponseType",
    "AcceptedPayload",
    "RejectedPayload",
    "DeferredPayload",
    "ContinuationToken",
    "RequiredInput",
    "ResponseValidationError",
    "InvalidResponseTypeError",
    "PayloadMismatchError",
    "MissingPayloadFieldError",
    "InvalidPayloadTypeError",
    "InvalidTraceError",
    "MissingResponseFieldError",
    "ResponseImmutabilityError",
    "ExecutableInResponseError",

    # --- DecisionEngine (Orchestration) ---
    "DecisionEngine",
    "EngineErrorCode",
    "EngineErrorCategory",
]

# =============================================================================
# IMPORTS — Core Primitives
# =============================================================================

from ntive.capability import (
    CapabilityDescriptor,
    CapabilityImmutabilityError,
    CapabilityInput,
    CapabilityValidationError,
    Constraint,
    DeclaredEffect,
    DuplicateInputKeyError,
    EffectCategory,
    ExecutableContentError,
    InvalidCapabilityInputError,
    InvalidConstraintError,
    InvalidEffectError,
    InvalidMetadataError,
    InvalidPreconditionError,
    Precondition,
)
from ntive.decision import (
    Alternative,
    Confidence,
    Decision,
    DecisionValidationError,
    ImmutabilityViolationError,
    InvalidConfidenceError,
    InvalidFieldTypeError,
    InvalidInputsError,
    MissingRequiredFieldError,
)
from ntive.engine import (
    DecisionEngine,
    EngineErrorCategory,
    EngineErrorCode,
)
from ntive.memory import (
    InvalidMemoryKeyError,
    InvalidMemoryValueError,
    KeyNotFoundError,
    MemoryDiff,
    MemoryImmutabilityError,
    MemoryScope,
    MemoryValidationError,
)
from ntive.policy import (
    ConflictResolutionStrategy,
    CyclicPolicyInheritanceError,
    DuplicateRuleIdError,
    InvalidPolicyEffectError,
    InvalidPolicyRuleError,
    Policy,
    PolicyConflictError,
    PolicyEffect,
    PolicyEffectResult,
    PolicyImmutabilityError,
    PolicyRule,
    PolicyValidationError,
)
from ntive.request import (
    ContinuationRef,
    DecisionRequest,
    ExecutableInRequestError,
    InvalidCapabilityError,
    InvalidPolicyError,
    InvalidRequestFieldError,
    InvalidRequestIdError,
    MissingRequestFieldError,
    NonSerializableContextError,
    QueryDescriptor,
    RequestConstraints,
    RequestImmutabilityError,
    RequestValidationError,
)
from ntive.response import (
    AcceptedPayload,
    ContinuationToken,
    DecisionResponse,
    DeferredPayload,
    ExecutableInResponseError,
    InvalidPayloadTypeError,
    InvalidResponseTypeError,
    InvalidTraceError,
    MissingPayloadFieldError,
    MissingResponseFieldError,
    PayloadMismatchError,
    RejectedPayload,
    RequiredInput,
    ResponseImmutabilityError,
    ResponseType,
    ResponseValidationError,
)
from ntive.trace import (
    CausalReason,
    InvalidCausalReasonError,
    InvalidTraceNodeError,
    Trace,
    TraceImmutabilityError,
    TraceValidationError,
)
from ntive.trace_replay import (
    BrokenCausalChainError,
    ChainStatus,
    ChainValidation,
    DeterminismCheck,
    DeterminismError,
    ReplayExplanation,
    ReplayNode,
    TraceLoadError,
    TraceReplay,
    TraceReplayError,
)

# =============================================================================
# INTERNAL MODULES — NOT part of public API (may change without notice)
# =============================================================================
# The following submodules are internal/experimental and should not be
# imported directly by users:
#
#   - ntive.parser      (experimental DSL parser)
#   - ntive.runtime     (experimental DSL runtime)
#   - ntive.ast_nodes   (internal AST representation)
#   - ntive.errors      (DSL-specific error classes)
#
# These modules are not exported and may be restructured or removed in any
# future release.
