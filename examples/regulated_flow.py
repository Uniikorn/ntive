#!/usr/bin/env python3
"""
regulated_flow.py — Regulated System Decision Flow
===================================================

This example demonstrates how a regulated system (healthcare, finance,
or compliance-sensitive domain) would use Ntive Core to make decisions
under policy constraints.

Scenario:
    A healthcare system needs to decide whether to share patient data
    with a researcher. The decision must comply with privacy policies
    and be fully auditable.

Key Concepts:
    - Using Policy and PolicyRule for compliance constraints
    - Policy inheritance for layered regulations
    - Conflict resolution strategies (MOST_RESTRICTIVE, REJECT_CONFLICT)
    - Full audit trail via Trace

IMPORTANT:
    This example contains ZERO execution logic. Ntive only decides;
    the healthcare system executes data sharing operations.
"""

import uuid

from ntive import (
    # Request types
    DecisionRequest,
    QueryDescriptor,
    
    # Policy types
    Policy,
    PolicyRule,
    PolicyEffect,
    ConflictResolutionStrategy,
    
    # Capability types
    CapabilityDescriptor,
    CapabilityInput,
    DeclaredEffect,
    Precondition,
    EffectCategory,
    
    # Engine
    DecisionEngine,
    
    # Response types
    ResponseType,
)


# =============================================================================
# Policy Definitions
# =============================================================================

def create_hipaa_base_policy() -> Policy:
    """
    Create a base HIPAA compliance policy.
    
    This represents federal-level healthcare privacy regulations.
    """
    return Policy(
        name="hipaa_base",
        version="2.0.0",
        priority=100,  # High priority - federal law
        conflict_strategy=ConflictResolutionStrategy.MOST_RESTRICTIVE,
        rules=[
            PolicyRule(
                rule_id="hipaa_001",
                effect=PolicyEffect.FORBID,
                target="share_phi_unencrypted",
                condition={"data_type": "phi"},
            ),
            PolicyRule(
                rule_id="hipaa_002",
                effect=PolicyEffect.REQUIRE,
                target="audit_log_access",
                condition={"data_type": "phi"},
            ),
            PolicyRule(
                rule_id="hipaa_003",
                effect=PolicyEffect.REQUIRE,
                target="consent_verification",
                condition={"operation": "data_sharing"},
            ),
        ],
    )


def create_research_policy(parent: Policy) -> Policy:
    """
    Create a research-specific policy that inherits from HIPAA.
    
    This adds organization-level rules on top of federal requirements.
    """
    return Policy(
        name="research_data_policy",
        version="1.0.0",
        priority=50,  # Lower than HIPAA
        parent=parent,
        conflict_strategy=ConflictResolutionStrategy.MOST_RESTRICTIVE,
        rules=[
            # Additional requirement: IRB approval for research
            PolicyRule(
                rule_id="research_001",
                effect=PolicyEffect.REQUIRE,
                target="irb_approval",
                condition={"purpose": "research"},
            ),
            # Prefer de-identified data when possible
            PolicyRule(
                rule_id="research_002",
                effect=PolicyEffect.PREFER,
                target="use_deidentified_data",
                condition={"purpose": "research"},
                weight=0.8,
            ),
            # Allow limited data set with data use agreement
            PolicyRule(
                rule_id="research_003",
                effect=PolicyEffect.DEFAULT,
                target="limited_data_set",
                condition={
                    "purpose": "research",
                    "has_dua": True,
                },
            ),
        ],
    )


def create_emergency_override_policy(parent: Policy) -> Policy:
    """
    Create an emergency override policy for critical situations.
    
    This demonstrates how policies can have exceptions for emergencies,
    while still maintaining audit requirements.
    """
    return Policy(
        name="emergency_override",
        version="1.0.0",
        priority=200,  # Highest priority
        parent=parent,
        conflict_strategy=ConflictResolutionStrategy.EXPLICIT_PRIORITY,
        rules=[
            # In emergencies, allow immediate access
            PolicyRule(
                rule_id="emergency_001",
                effect=PolicyEffect.REQUIRE,
                target="immediate_access",
                condition={"emergency": True},
            ),
            # But STILL require audit logging (inherited requirement stands)
            PolicyRule(
                rule_id="emergency_002",
                effect=PolicyEffect.REQUIRE,
                target="emergency_audit_log",
                condition={"emergency": True},
            ),
        ],
    )


# =============================================================================
# Capability Definitions
# =============================================================================

def create_share_data_capability() -> CapabilityDescriptor:
    """Define the capability to share patient data."""
    return CapabilityDescriptor(
        name="share_patient_data",
        version="1.0.0",
        domain="healthcare_data",
        description="Share patient data with authorized parties",
        inputs=[
            CapabilityInput(
                key="patient_id",
                input_type="string",
                required=True,
                description="Unique patient identifier",
            ),
            CapabilityInput(
                key="recipient_id",
                input_type="string",
                required=True,
                description="ID of the authorized recipient",
            ),
            CapabilityInput(
                key="data_scope",
                input_type="string",
                required=True,
                description="Scope of data to share (full, limited, deidentified)",
            ),
            CapabilityInput(
                key="consent_token",
                input_type="string",
                required=False,
                description="Patient consent verification token",
            ),
        ],
        preconditions=[
            Precondition(
                predicate="recipient_authorized",
                description="Recipient must be authorized to receive data",
            ),
            Precondition(
                predicate="data_encrypted",
                description="Data must be encrypted for transmission",
            ),
        ],
        effects=[
            DeclaredEffect(
                category=EffectCategory.CREATE,
                target="audit_record",
                description="Creates an audit trail entry",
            ),
            DeclaredEffect(
                category=EffectCategory.MODIFY,
                target="access_log",
                description="Updates the data access log",
            ),
        ],
    )


# =============================================================================
# Request Builders
# =============================================================================

def build_research_request(
    patient_id: str,
    researcher_id: str,
    has_consent: bool,
    has_irb_approval: bool,
    data_scope: str = "limited",
) -> DecisionRequest:
    """Build a request for research data sharing."""
    return DecisionRequest(
        request_id=str(uuid.uuid4()),
        query=QueryDescriptor(
            intent="share_patient_data",
            parameters={
                "purpose": "research",
                "data_scope": data_scope,
            },
            metadata={
                "request_source": "research_portal",
                "timestamp": "2025-02-08T10:00:00Z",
            },
        ),
        context={
            "patient_id": patient_id,
            "recipient_id": researcher_id,
            "data_scope": data_scope,
            "data_type": "phi",
            "operation": "data_sharing",
            "purpose": "research",
            "has_consent": has_consent,
            "has_irb_approval": has_irb_approval,
            "has_dua": True,  # Data Use Agreement
            "emergency": False,
        },
        capabilities=[create_share_data_capability()],
    )


def build_emergency_request(
    patient_id: str,
    provider_id: str,
) -> DecisionRequest:
    """Build an emergency access request."""
    return DecisionRequest(
        request_id=str(uuid.uuid4()),
        query=QueryDescriptor(
            intent="share_patient_data",
            parameters={
                "purpose": "emergency_care",
                "data_scope": "full",
            },
            metadata={
                "request_source": "emergency_room",
                "priority": "critical",
            },
        ),
        context={
            "patient_id": patient_id,
            "recipient_id": provider_id,
            "data_scope": "full",
            "data_type": "phi",
            "operation": "data_sharing",
            "purpose": "emergency_care",
            "emergency": True,
            "has_consent": False,  # No time to get consent
        },
        capabilities=[create_share_data_capability()],
    )


# =============================================================================
# Response Handling
# =============================================================================

def display_policy_chain(policy: Policy) -> None:
    """Display the policy inheritance chain."""
    print("\n[POLICY CHAIN]")
    chain = policy.chain()
    for i, p in enumerate(chain):
        indent = "  " * i
        print(f"  {indent}├─ {p.name} v{p.version} (priority: {p.priority})")
        print(f"  {indent}│  Strategy: {p.conflict_strategy.value}")
        print(f"  {indent}│  Local Rules: {len(list(p))}")


def handle_regulated_response(
    response,
    scenario_name: str,
    policy: Policy | None = None,
) -> None:
    """Handle the response with compliance-focused output."""
    print(f"\n{'═' * 70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'═' * 70}")
    
    print(f"\n[RESPONSE SUMMARY]")
    print(f"  Type:       {response.response_type.value.upper()}")
    print(f"  Request ID: {response.request_id}")
    print(f"  Hash:       {response.response_hash[:16]}...")
    
    if policy:
        display_policy_chain(policy)
    
    if response.response_type == ResponseType.ACCEPTED:
        payload = response.payload
        decision = payload.decision
        trace = payload.trace
        
        print(f"\n[COMPLIANCE DECISION: APPROVED]")
        print(f"  Action:     {decision.action}")
        print(f"  Capability: {decision.capability}")
        print(f"  Confidence: {decision.confidence.value}")
        
        # Audit trail is critical in regulated systems
        print(f"\n[AUDIT TRAIL]")
        print(f"  Trace ID:      {trace.trace_id}")
        print(f"  Decisions:     {len(trace)}")
        print(f"  Deterministic: Yes (reproducible from inputs)")
        
        print(f"\n  Decision Chain (for compliance audit):")
        for i, (decision_node, reason) in enumerate(zip(trace.decisions, trace.reasons)):
            print(f"    [{i+1}] Action: {decision_node.action}")
            if reason:
                print(f"        Reason: {reason.text}")
        
        # Effects are important for understanding data flow
        if payload.effects:
            print(f"\n[DATA EFFECTS]")
            for effect in payload.effects:
                print(f"  - {effect.category.value}: {effect.target}")
    
    elif response.response_type == ResponseType.REJECTED:
        payload = response.payload
        
        print(f"\n[COMPLIANCE DECISION: DENIED]")
        print(f"  Error Code:  {payload.error_code}")
        print(f"  Category:    {payload.category}")
        print(f"  Recoverable: {payload.recoverable}")
        
        if payload.message:
            print(f"\n  Reason: {payload.message}")
        
        if payload.details:
            print(f"\n  Policy Violations:")
            for key, value in payload.details.items():
                print(f"    - {key}: {value}")
        
        # Even rejected requests have partial traces for audit
        if payload.partial_trace:
            print(f"\n[PARTIAL AUDIT TRAIL]")
            print(f"  Decisions before rejection: {len(payload.partial_trace)}")
    
    elif response.response_type == ResponseType.DEFERRED:
        payload = response.payload
        
        print(f"\n[COMPLIANCE DECISION: PENDING]")
        print(f"  Reason: {payload.reason}")
        
        print(f"\n  Required Compliance Steps:")
        for inp in payload.required_inputs:
            print(f"    □ {inp.name}: {inp.description or inp.input_type}")


def main():
    """
    Run the regulated decision flow.
    
    Demonstrates policy-based decisions in healthcare compliance scenarios.
    """
    engine = DecisionEngine()
    
    # Build policy hierarchy
    hipaa_policy = create_hipaa_base_policy()
    research_policy = create_research_policy(hipaa_policy)
    emergency_policy = create_emergency_override_policy(hipaa_policy)
    
    # =========================================================================
    # Scenario 1: Compliant Research Request
    # =========================================================================
    print("\n")
    request_compliant = build_research_request(
        patient_id="P-12345",
        researcher_id="R-67890",
        has_consent=True,
        has_irb_approval=True,
        data_scope="deidentified",
    )
    # Add policy to request
    request_compliant = DecisionRequest(
        request_id=request_compliant.request_id,
        query=request_compliant.query,
        context=request_compliant.context,
        capabilities=list(request_compliant.capabilities),
        policy=research_policy,
    )
    
    response_compliant = engine.evaluate(request_compliant)
    handle_regulated_response(
        response_compliant,
        "Research Request - All Approvals in Place",
        research_policy,
    )
    
    # =========================================================================
    # Scenario 2: Non-Compliant Request (missing consent)
    # =========================================================================
    request_no_consent = build_research_request(
        patient_id="P-12345",
        researcher_id="R-67890",
        has_consent=False,  # No patient consent
        has_irb_approval=True,
        data_scope="limited",
    )
    request_no_consent = DecisionRequest(
        request_id=request_no_consent.request_id,
        query=request_no_consent.query,
        context=request_no_consent.context,
        capabilities=list(request_no_consent.capabilities),
        policy=research_policy,
    )
    
    response_no_consent = engine.evaluate(request_no_consent)
    handle_regulated_response(
        response_no_consent,
        "Research Request - Missing Patient Consent",
        research_policy,
    )
    
    # =========================================================================
    # Scenario 3: Emergency Override
    # =========================================================================
    request_emergency = build_emergency_request(
        patient_id="P-99999",
        provider_id="DR-54321",
    )
    request_emergency = DecisionRequest(
        request_id=request_emergency.request_id,
        query=request_emergency.query,
        context=request_emergency.context,
        capabilities=list(request_emergency.capabilities),
        policy=emergency_policy,
    )
    
    response_emergency = engine.evaluate(request_emergency)
    handle_regulated_response(
        response_emergency,
        "Emergency Access - Life-Threatening Situation",
        emergency_policy,
    )
    
    # =========================================================================
    # Demonstration: Policy Conflict Detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("POLICY ANALYSIS: Rule Inheritance and Conflict Resolution")
    print("=" * 70)
    
    print("\n[HIPAA BASE POLICY]")
    print(f"  Rules: {len(list(hipaa_policy))}")
    for rule in hipaa_policy.rules():
        print(f"    - {rule.rule_id}: {rule.effect.value} → {rule.target}")
    
    print("\n[RESEARCH POLICY (inherits from HIPAA)]")
    print(f"  Local Rules: {len(list(research_policy))}")
    print(f"  Total Rules (with inheritance): {len(list(research_policy.rules()))}")
    
    inherited_rules = [r for r in research_policy.rules() if r.rule_id.startswith("hipaa")]
    local_rules = [r for r in research_policy.rules() if r.rule_id.startswith("research")]
    
    print(f"\n  Inherited from HIPAA ({len(inherited_rules)}):")
    for rule in inherited_rules:
        print(f"    - {rule.rule_id}: {rule.effect.value} → {rule.target}")
    
    print(f"\n  Local Rules ({len(local_rules)}):")
    for rule in local_rules:
        weight_str = f" (weight: {rule.weight})" if rule.weight else ""
        print(f"    - {rule.rule_id}: {rule.effect.value} → {rule.target}{weight_str}")


if __name__ == "__main__":
    main()
