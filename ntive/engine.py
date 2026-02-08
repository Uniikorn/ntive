"""
engine.py

Ntive DecisionEngine — Pure Orchestration Layer

The DecisionEngine is a coordinator that:
- Validates incoming DecisionRequests
- Resolves policies
- Matches capabilities
- Constructs Decisions
- Emits Traces
- Returns DecisionResponses

Design Invariants:
- Pure logic only
- No execution
- No mutation of inputs
- No undeclared state access
- No AI logic
- Deterministic
- Always emits Trace entries
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from ntive.capability import CapabilityDescriptor, CapabilityInput
from ntive.decision import Alternative, Decision
from ntive.memory import MemoryScope
from ntive.policy import Policy, PolicyEffect, PolicyRule
from ntive.request import (
    DecisionRequest,
)
from ntive.response import (
    AcceptedPayload,
    DecisionResponse,
    DeferredPayload,
    RejectedPayload,
    RequiredInput,
    ResponseType,
)
from ntive.trace import CausalReason, Trace

# =============================================================================
# Error Codes
# =============================================================================

class EngineErrorCode:
    """Error codes for engine failures."""
    # Structural validation
    INVALID_REQUEST_TYPE = "E001"
    INVALID_REQUEST_STRUCTURE = "E002"

    # Context/Memory validation
    INVALID_CONTEXT = "E010"
    INVALID_MEMORY = "E011"

    # Policy errors
    POLICY_VALIDATION_FAILED = "E020"
    POLICY_CONFLICT_UNRESOLVABLE = "E021"

    # Capability errors
    CAPABILITY_INVALID = "E030"
    CAPABILITY_MISMATCH = "E031"
    INPUT_NOT_SATISFIED = "E032"

    # General
    INTERNAL_ERROR = "E099"


class EngineErrorCategory:
    """Error categories for engine failures."""
    VALIDATION = "validation"
    POLICY = "policy"
    CAPABILITY = "capability"
    INTERNAL = "internal"


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _generate_logical_timestamp() -> int:
    """Generate a logical timestamp (monotonic)."""
    # Use nanoseconds for uniqueness
    return time.time_ns()


def _generate_continuation_token() -> str:
    """Generate a unique continuation token ID."""
    return f"cont_{uuid.uuid4().hex[:16]}"


# =============================================================================
# DecisionEngine
# =============================================================================

class DecisionEngine:
    """
    Pure orchestration layer for decision evaluation.

    The DecisionEngine:
    - Coordinates existing primitives
    - Validates requests
    - Resolves policies
    - Matches capabilities
    - Constructs decisions and traces
    - Returns responses

    The DecisionEngine does NOT:
    - Execute anything
    - Mutate inputs
    - Access external state
    - Contain AI logic
    - Make assumptions

    Determinism: Given the same input, the engine produces logically
    equivalent output. Timestamps are the only non-deterministic element
    but are used only for ordering, not for logic.
    """

    __slots__ = ()  # No instance state

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """
        Evaluate a DecisionRequest and produce a DecisionResponse.

        This method performs the following steps in order:
        1. Structural Validation
        2. Context & Memory Validation
        3. Policy Resolution
        4. Capability Matching
        5. Decision Construction
        6. Trace Construction
        7. Response Emission

        Args:
            request: The DecisionRequest to evaluate

        Returns:
            DecisionResponse: Either ACCEPTED, REJECTED, or DEFERRED
        """
        timestamp = _generate_logical_timestamp()

        # Track trace entries for partial trace on rejection
        trace_decisions: List[Tuple[Decision, Optional[CausalReason]]] = []

        # =====================================================================
        # Step 1: Structural Validation
        # =====================================================================

        validation_result = self._validate_request_structure(request)
        if validation_result is not None:
            return self._create_rejected_response(
                request_id=getattr(request, 'request_id', None) or str(uuid.uuid4()),
                timestamp=timestamp,
                code=EngineErrorCode.INVALID_REQUEST_TYPE,
                category=EngineErrorCategory.VALIDATION,
                reason=validation_result,
                recoverable=False,
            )

        request_id = request.request_id

        # =====================================================================
        # Step 2: Context & Memory Validation
        # =====================================================================

        context_result = self._validate_context_and_memory(request)
        if context_result is not None:
            return self._create_rejected_response(
                request_id=request_id,
                timestamp=timestamp,
                code=EngineErrorCode.INVALID_CONTEXT,
                category=EngineErrorCategory.VALIDATION,
                reason=context_result,
                recoverable=True,
            )

        # =====================================================================
        # Step 3: Policy Resolution
        # =====================================================================

        policy_result = self._resolve_policy(request, trace_decisions)
        if isinstance(policy_result, str):
            # Policy error - return rejected
            partial_trace = self._build_partial_trace(trace_decisions)
            return self._create_rejected_response(
                request_id=request_id,
                timestamp=timestamp,
                code=EngineErrorCode.POLICY_CONFLICT_UNRESOLVABLE,
                category=EngineErrorCategory.POLICY,
                reason=policy_result,
                recoverable=False,
                partial_trace=partial_trace,
            )

        resolved_rules, policy_decision = policy_result
        if policy_decision is not None:
            trace_decisions.append((
                policy_decision,
                CausalReason(reason="Policy resolution", category="policy"),
            ))

        # =====================================================================
        # Step 4: Capability Matching
        # =====================================================================

        capability_result = self._match_capabilities(request)

        if isinstance(capability_result, tuple) and len(capability_result) == 2:
            error_type, error_msg = capability_result
            if error_type == "missing_inputs":
                # Deferred - need more inputs
                required_inputs = self._extract_missing_inputs(request)
                partial_trace = self._build_partial_trace(trace_decisions)
                return self._create_deferred_response(
                    request_id=request_id,
                    timestamp=timestamp,
                    code=EngineErrorCode.INPUT_NOT_SATISFIED,
                    reason=error_msg,
                    required_inputs=required_inputs,
                    partial_trace=partial_trace,
                )
            else:
                # Capability mismatch - rejected
                partial_trace = self._build_partial_trace(trace_decisions)
                return self._create_rejected_response(
                    request_id=request_id,
                    timestamp=timestamp,
                    code=EngineErrorCode.CAPABILITY_MISMATCH,
                    category=EngineErrorCategory.CAPABILITY,
                    reason=error_msg,
                    recoverable=True,
                    partial_trace=partial_trace,
                )

        selected_capability = capability_result

        # =====================================================================
        # Step 5: Decision Construction
        # =====================================================================

        decision = self._construct_decision(
            request=request,
            selected_capability=selected_capability,
            resolved_rules=resolved_rules,
        )

        trace_decisions.append((
            decision,
            CausalReason(reason="Decision constructed from query and context"),
        ))

        # =====================================================================
        # Step 6: Trace Construction
        # =====================================================================

        trace = Trace.build([
            (d, r) for d, r in trace_decisions
        ])

        # =====================================================================
        # Step 7: Response Emission
        # =====================================================================

        # Get effects from selected capability
        effects = None
        if selected_capability is not None and selected_capability.effects:
            effects = list(selected_capability.effects)

        payload = AcceptedPayload(
            decision=decision,
            trace=trace,
            effects=effects,
        )

        return DecisionResponse(
            request_id=request_id,
            response_type=ResponseType.ACCEPTED,
            timestamp=timestamp,
            payload=payload,
        )

    # -------------------------------------------------------------------------
    # Step 1: Structural Validation
    # -------------------------------------------------------------------------

    def _validate_request_structure(
        self,
        request: Any,
    ) -> Optional[str]:
        """
        Validate that request is a valid DecisionRequest.

        Returns None if valid, error message string if invalid.
        """
        if not isinstance(request, DecisionRequest):
            return f"Expected DecisionRequest, got {type(request).__name__}"

        # DecisionRequest validates itself at construction, so if we have
        # a valid instance, structure is valid
        return None

    # -------------------------------------------------------------------------
    # Step 2: Context & Memory Validation
    # -------------------------------------------------------------------------

    def _validate_context_and_memory(
        self,
        request: DecisionRequest,
    ) -> Optional[str]:
        """
        Validate context and memory are usable.

        Returns None if valid, error message string if invalid.
        Does NOT modify them.
        """
        # Context is always a dict (guaranteed by DecisionRequest)
        context = request.context

        # Verify context is a plain dict (not a subclass that might have side effects)
        if type(context) is not dict:
            return f"Context must be a plain dict, got {type(context).__name__}"

        # Memory validation (if present)
        memory = request.memory
        if memory is not None:
            if not isinstance(memory, MemoryScope):
                return f"Memory must be MemoryScope, got {type(memory).__name__}"

        return None

    # -------------------------------------------------------------------------
    # Step 3: Policy Resolution
    # -------------------------------------------------------------------------

    def _resolve_policy(
        self,
        request: DecisionRequest,
        trace_decisions: List[Tuple[Decision, Optional[CausalReason]]],
    ) -> Union[Tuple[List[PolicyRule], Optional[Decision]], str]:
        """
        Resolve policy and detect conflicts.

        Returns:
            Tuple of (resolved_rules, policy_decision) if successful
            Error message string if unrecoverable conflict
        """
        policy = request.policy

        if policy is None:
            # No policy - no rules to apply
            return ([], None)

        if not isinstance(policy, Policy):
            return f"Policy must be Policy instance, got {type(policy).__name__}"

        # Collect all rules including inherited ones
        all_rules = self._collect_policy_rules(policy)

        # Check for conflicts
        conflict = self._detect_policy_conflicts(all_rules)
        if conflict is not None:
            return conflict

        # Create a decision recording policy resolution
        policy_decision = Decision(
            inputs={"policy_name": policy.name, "policy_version": policy.version},
            selected_option="policy_resolved",
            alternatives=[],
            rationale=f"Policy '{policy.name}' v{policy.version} resolved with {len(all_rules)} rules",
        )

        return (all_rules, policy_decision)

    def _collect_policy_rules(self, policy: Policy) -> List[PolicyRule]:
        """Collect all rules from a policy, including inherited rules."""
        # rules() method already handles inheritance
        return list(policy.rules())

    def _detect_policy_conflicts(
        self,
        rules: List[PolicyRule],
    ) -> Optional[str]:
        """
        Detect unresolvable policy conflicts.

        Returns error message if conflict found, None otherwise.
        """
        # Group rules by target
        rules_by_target: Dict[str, List[PolicyRule]] = {}
        for rule in rules:
            target = rule.target
            if target not in rules_by_target:
                rules_by_target[target] = []
            rules_by_target[target].append(rule)

        # Check for conflicting effects on same target
        for target, target_rules in rules_by_target.items():
            effects = set()
            for rule in target_rules:
                effects.add(rule.effect)

            # REQUIRE and FORBID on same target is a conflict
            if PolicyEffect.REQUIRE in effects and PolicyEffect.FORBID in effects:
                return f"Policy conflict: REQUIRE and FORBID both target '{target}'"

        return None

    # -------------------------------------------------------------------------
    # Step 4: Capability Matching
    # -------------------------------------------------------------------------

    def _match_capabilities(
        self,
        request: DecisionRequest,
    ) -> Union[Optional[CapabilityDescriptor], Tuple[str, str]]:
        """
        Match and validate capabilities.

        Returns:
            CapabilityDescriptor if a capability is matched
            None if no capabilities declared
            Tuple of (error_type, error_message) if error
        """
        capabilities = request.capabilities

        if not capabilities:
            # No capabilities declared - valid, just no capability selected
            return None

        # Validate all capabilities are valid instances
        for i, cap in enumerate(capabilities):
            if not isinstance(cap, CapabilityDescriptor):
                return ("invalid", f"capabilities[{i}] must be CapabilityDescriptor")

        # Get context for input matching
        context = request.context
        query = request.query

        # Build available inputs from context and query parameters
        available_inputs: Dict[str, Any] = {}
        available_inputs.update(context)
        if query.parameters:
            available_inputs.update(query.parameters)

        # Try to find a capability with satisfied inputs
        best_match: Optional[CapabilityDescriptor] = None
        missing_inputs: List[CapabilityInput] = []

        for cap in capabilities:
            cap_missing = self._get_missing_inputs(cap, available_inputs)
            if not cap_missing:
                # All required inputs satisfied
                best_match = cap
                break
            elif not missing_inputs or len(cap_missing) < len(missing_inputs):
                missing_inputs = cap_missing

        if best_match is not None:
            return best_match

        if missing_inputs:
            missing_names = ", ".join(inp.key for inp in missing_inputs)
            return ("missing_inputs", f"Required inputs not satisfied: {missing_names}")

        # No capabilities matched at all (shouldn't happen if list is non-empty)
        return ("invalid", "No capability could be matched")

    def _get_missing_inputs(
        self,
        capability: CapabilityDescriptor,
        available_inputs: Dict[str, Any],
    ) -> List[CapabilityInput]:
        """Get list of required inputs that are not satisfied."""
        missing = []
        for inp in capability.inputs:
            if inp.required and inp.key not in available_inputs:
                missing.append(inp)
        return missing

    def _extract_missing_inputs(
        self,
        request: DecisionRequest,
    ) -> List[RequiredInput]:
        """Extract RequiredInput list for deferred response."""
        capabilities = request.capabilities
        context = request.context
        query = request.query

        available_inputs: Dict[str, Any] = {}
        available_inputs.update(context)
        if query.parameters:
            available_inputs.update(query.parameters)

        required_inputs: List[RequiredInput] = []
        seen_keys: set = set()

        for cap in capabilities:
            for inp in cap.inputs:
                if inp.required and inp.key not in available_inputs and inp.key not in seen_keys:
                    required_inputs.append(RequiredInput(
                        name=inp.key,
                        type=inp.type,
                        reason=f"Required by capability '{cap.name}'",
                        constraints=inp.constraints if inp.constraints else None,
                    ))
                    seen_keys.add(inp.key)

        return required_inputs

    # -------------------------------------------------------------------------
    # Step 5: Decision Construction
    # -------------------------------------------------------------------------

    def _construct_decision(
        self,
        request: DecisionRequest,
        selected_capability: Optional[CapabilityDescriptor],
        resolved_rules: List[PolicyRule],
    ) -> Decision:
        """
        Construct the final Decision.

        Returns an immutable Decision instance.
        """
        query = request.query
        context = request.context

        # Build inputs from the request
        inputs: Dict[str, Any] = {
            "intent": query.intent,
        }
        if query.parameters:
            inputs["parameters"] = query.parameters
        if query.metadata:
            inputs["metadata"] = query.metadata

        # Add context keys (but not the full context to avoid bloat)
        if context:
            inputs["context_keys"] = list(context.keys())

        # Determine selected option
        if selected_capability is not None:
            selected_option = f"capability:{selected_capability.name}"
            rationale = (
                f"Selected capability '{selected_capability.name}' "
                f"v{selected_capability.version} to fulfill intent '{query.intent}'"
            )
        else:
            selected_option = f"intent:{query.intent}"
            rationale = f"No capability selected; proceeding with intent '{query.intent}'"

        # Build alternatives from other capabilities
        alternatives: List[Alternative] = []
        if selected_capability is not None:
            for cap in request.capabilities:
                if cap.capability_id != selected_capability.capability_id:
                    alternatives.append(Alternative(
                        option=f"capability:{cap.name}",
                        reason_not_selected="Alternative capability not selected",
                    ))

        # Add constraints from policy rules
        constraints: Optional[List[str]] = None
        if resolved_rules:
            constraints = [
                f"{rule.effect.value}:{rule.target}"
                for rule in resolved_rules
            ]

        return Decision(
            inputs=inputs,
            selected_option=selected_option,
            alternatives=alternatives,
            rationale=rationale,
            constraints=constraints,
        )

    # -------------------------------------------------------------------------
    # Step 6: Trace Construction
    # -------------------------------------------------------------------------

    def _build_partial_trace(
        self,
        trace_decisions: List[Tuple[Decision, Optional[CausalReason]]],
    ) -> Optional[Trace]:
        """Build a partial trace from accumulated decisions."""
        if not trace_decisions:
            return None
        return Trace.build([(d, r) for d, r in trace_decisions])

    # -------------------------------------------------------------------------
    # Step 7: Response Construction Helpers
    # -------------------------------------------------------------------------

    def _create_rejected_response(
        self,
        *,
        request_id: str,
        timestamp: int,
        code: str,
        category: str,
        reason: str,
        recoverable: bool,
        partial_trace: Optional[Trace] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> DecisionResponse:
        """Create a REJECTED DecisionResponse."""
        payload = RejectedPayload(
            code=code,
            category=category,
            reason=reason,
            recoverable=recoverable,
            details=details,
            suggestions=suggestions,
            partial_trace=partial_trace,
        )

        return DecisionResponse(
            request_id=request_id,
            response_type=ResponseType.REJECTED,
            timestamp=timestamp,
            payload=payload,
        )

    def _create_deferred_response(
        self,
        *,
        request_id: str,
        timestamp: int,
        code: str,
        reason: str,
        required_inputs: List[RequiredInput],
        partial_trace: Optional[Trace] = None,
    ) -> DecisionResponse:
        """Create a DEFERRED DecisionResponse."""
        payload = DeferredPayload(
            code=code,
            reason=reason,
            required_inputs=required_inputs,
        )

        return DecisionResponse(
            request_id=request_id,
            response_type=ResponseType.DEFERRED,
            timestamp=timestamp,
            payload=payload,
        )
