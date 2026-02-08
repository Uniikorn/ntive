#!/usr/bin/env python3
"""
assistant_flow.py — AI Assistant Decision Flow
===============================================

This example demonstrates how an AI assistant system would use Ntive Core
to make structured decisions with full traceability.

Scenario:
    A user asks the assistant to "summarize the quarterly report". The
    assistant must decide whether it can fulfill this request based on
    available capabilities and context.

Key Concepts:
    - Constructing a DecisionRequest with query and context
    - Providing CapabilityDescriptors for available actions
    - Handling ACCEPTED, REJECTED, and DEFERRED responses
    - Inspecting the Trace for causal reasoning

IMPORTANT:
    This example contains ZERO execution logic. Ntive only decides;
    external systems execute.
"""

import uuid

from ntive import (
    # Request types
    DecisionRequest,
    QueryDescriptor,
    
    # Capability types
    CapabilityDescriptor,
    CapabilityInput,
    DeclaredEffect,
    EffectCategory,
    
    # Engine
    DecisionEngine,
    
    # Response types
    ResponseType,
)


def create_summarize_capability() -> CapabilityDescriptor:
    """
    Define a capability for summarizing documents.
    
    This describes WHAT the system CAN do, not HOW it does it.
    Ntive uses this to match requests to capabilities.
    """
    return CapabilityDescriptor(
        name="summarize_document",
        version="1.0.0",
        domain="text_processing",
        description="Summarize a document into key points",
        inputs=[
            CapabilityInput(
                key="document_content",
                input_type="string",
                required=True,
                description="The full text of the document to summarize",
            ),
            CapabilityInput(
                key="max_length",
                input_type="integer",
                required=False,
                default_value=500,
                description="Maximum length of the summary in words",
            ),
        ],
        effects=[
            DeclaredEffect(
                category=EffectCategory.CREATE,
                target="summary",
                description="Creates a text summary of the input document",
            ),
        ],
    )


def build_assistant_request(user_query: str, document_content: str | None) -> DecisionRequest:
    """
    Build a DecisionRequest from user input.
    
    This is where external systems serialize their request into
    Ntive's structured format.
    """
    return DecisionRequest(
        request_id=str(uuid.uuid4()),
        query=QueryDescriptor(
            intent="summarize_document",
            parameters={
                "target": "quarterly_report",
                "format": "bullet_points",
            },
            metadata={
                "user_query": user_query,
                "source": "voice_assistant",
            },
        ),
        context={
            # Document content may or may not be available
            "document_content": document_content,
            "user_preferences": {
                "language": "en",
                "verbosity": "concise",
            },
        },
        capabilities=[
            create_summarize_capability(),
        ],
    )


def handle_response(response) -> None:
    """
    Handle the DecisionResponse from the engine.
    
    This shows the three possible response types and how to
    process each one.
    """
    print(f"\n{'=' * 60}")
    print(f"Response Type: {response.response_type.value.upper()}")
    print(f"Request ID:    {response.request_id}")
    print(f"Timestamp:     {response.timestamp}")
    print(f"{'=' * 60}")
    
    if response.response_type == ResponseType.ACCEPTED:
        # =====================================================================
        # ACCEPTED: The request can be fulfilled
        # =====================================================================
        payload = response.payload
        decision = payload.decision
        trace = payload.trace
        
        print("\n[DECISION]")
        print(f"  Action:     {decision.action}")
        print(f"  Capability: {decision.capability}")
        print(f"  Confidence: {decision.confidence.value}")
        
        if decision.inputs:
            print(f"\n  Resolved Inputs:")
            for key, value in decision.inputs.items():
                # Truncate long values for display
                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else value
                print(f"    {key}: {display_value}")
        
        print(f"\n[TRACE] ({len(trace)} decisions in chain)")
        for i, node in enumerate(trace):
            print(f"  [{i}] {node.action}")
        
        if payload.effects:
            print(f"\n[EFFECTS]")
            for effect in payload.effects:
                print(f"  - {effect.category.value}: {effect.target}")
    
    elif response.response_type == ResponseType.REJECTED:
        # =====================================================================
        # REJECTED: The request cannot be fulfilled
        # =====================================================================
        payload = response.payload
        
        print("\n[REJECTION DETAILS]")
        print(f"  Error Code:   {payload.error_code}")
        print(f"  Category:     {payload.category}")
        print(f"  Recoverable:  {payload.recoverable}")
        
        if payload.message:
            print(f"  Message:      {payload.message}")
        
        if payload.details:
            print(f"\n  Additional Details:")
            for key, value in payload.details.items():
                print(f"    {key}: {value}")
        
        # Check if there's a partial trace
        if payload.partial_trace:
            print(f"\n[PARTIAL TRACE] ({len(payload.partial_trace)} decisions before failure)")
            for i, node in enumerate(payload.partial_trace):
                print(f"  [{i}] {node.action}")
    
    elif response.response_type == ResponseType.DEFERRED:
        # =====================================================================
        # DEFERRED: More information is needed
        # =====================================================================
        payload = response.payload
        
        print("\n[DEFERRED - ADDITIONAL INPUT REQUIRED]")
        print(f"  Reason: {payload.reason}")
        
        if payload.timeout_hint:
            print(f"  Timeout Hint: {payload.timeout_hint}ms")
        
        print(f"\n  Required Inputs:")
        for inp in payload.required_inputs:
            required_str = "(required)" if inp.required else "(optional)"
            print(f"    - {inp.name}: {inp.input_type} {required_str}")
            if inp.description:
                print(f"      {inp.description}")
        
        if payload.continuation_token:
            print(f"\n  Continuation Token: {payload.continuation_token.token_id}")
            print(f"  Expires At: {payload.continuation_token.expires_at}")


def main():
    """
    Run the assistant decision flow.
    
    This demonstrates two scenarios:
    1. Complete request (document available) -> ACCEPTED
    2. Incomplete request (document missing) -> DEFERRED
    """
    engine = DecisionEngine()
    
    # =========================================================================
    # Scenario 1: Complete Request
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCENARIO 1: Complete Request (document available)")
    print("=" * 60)
    
    request_complete = build_assistant_request(
        user_query="Can you summarize the quarterly report?",
        document_content="Q3 2025 Financial Report: Revenue increased 15%...",
    )
    
    response_complete = engine.evaluate(request_complete)
    handle_response(response_complete)
    
    # =========================================================================
    # Scenario 2: Incomplete Request (deferred)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCENARIO 2: Incomplete Request (document missing)")
    print("=" * 60)
    
    request_incomplete = build_assistant_request(
        user_query="Summarize the quarterly report",
        document_content=None,  # Document not yet loaded
    )
    
    response_incomplete = engine.evaluate(request_incomplete)
    handle_response(response_incomplete)


if __name__ == "__main__":
    main()
