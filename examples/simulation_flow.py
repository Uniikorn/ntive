#!/usr/bin/env python3
"""
simulation_flow.py — Simulation System Decision Flow
=====================================================

This example demonstrates how a simulation system (game, training,
or modeling environment) would use Ntive Core to make deterministic
decisions with full traceability.

Scenario:
    A strategy game AI needs to decide whether to "attack", "defend",
    or "wait" based on the current game state. The decision must be
    reproducible for replays and debugging.

Key Concepts:
    - Using MemoryScope for stateful context
    - Multiple capabilities representing possible actions
    - Deterministic decision-making for replay systems
    - Trace inspection for debugging AI behavior

IMPORTANT:
    This example contains ZERO execution logic. Ntive only decides;
    the game engine executes the chosen action.
"""

import uuid

from ntive import (
    # Request types
    DecisionRequest,
    QueryDescriptor,
    
    # Memory
    MemoryScope,
    
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


def create_attack_capability() -> CapabilityDescriptor:
    """Define the 'attack' action capability."""
    return CapabilityDescriptor(
        name="attack_enemy",
        version="1.0.0",
        domain="combat",
        description="Launch an attack on an enemy unit",
        inputs=[
            CapabilityInput(
                key="target_unit_id",
                input_type="string",
                required=True,
                description="ID of the enemy unit to attack",
            ),
            CapabilityInput(
                key="attack_strength",
                input_type="integer",
                required=True,
                description="Current attack strength of the unit",
            ),
        ],
        preconditions=[
            Precondition(
                predicate="has_ammunition",
                description="Unit must have ammunition to attack",
            ),
            Precondition(
                predicate="target_in_range",
                description="Target must be within attack range",
            ),
        ],
        effects=[
            DeclaredEffect(
                category=EffectCategory.MODIFY,
                target="enemy_health",
                description="Reduces enemy unit health",
            ),
            DeclaredEffect(
                category=EffectCategory.MODIFY,
                target="ammunition",
                description="Consumes ammunition",
            ),
        ],
    )


def create_defend_capability() -> CapabilityDescriptor:
    """Define the 'defend' action capability."""
    return CapabilityDescriptor(
        name="defensive_stance",
        version="1.0.0",
        domain="combat",
        description="Enter a defensive posture to reduce incoming damage",
        inputs=[
            CapabilityInput(
                key="defense_modifier",
                input_type="float",
                required=False,
                default_value=1.5,
                description="Multiplier for defense rating",
            ),
        ],
        effects=[
            DeclaredEffect(
                category=EffectCategory.MODIFY,
                target="defense_rating",
                description="Temporarily increases defense",
            ),
            DeclaredEffect(
                category=EffectCategory.MODIFY,
                target="movement_speed",
                description="Reduces movement while defending",
            ),
        ],
    )


def create_wait_capability() -> CapabilityDescriptor:
    """Define the 'wait' action capability."""
    return CapabilityDescriptor(
        name="wait_and_observe",
        version="1.0.0",
        domain="tactical",
        description="Wait and observe enemy movements",
        inputs=[],  # No inputs required
        effects=[
            DeclaredEffect(
                category=EffectCategory.MODIFY,
                target="action_points",
                description="Preserves action points for next turn",
            ),
        ],
    )


def build_game_state_memory(
    unit_health: int,
    ammunition: int,
    enemy_distance: int,
    turn_number: int,
) -> MemoryScope:
    """
    Build a MemoryScope representing the current game state.
    
    MemoryScope provides immutable, traceable state that Ntive
    can reference during decision-making.
    """
    return MemoryScope(
        values={
            "unit": {
                "health": unit_health,
                "max_health": 100,
                "ammunition": ammunition,
                "attack_strength": 25,
                "defense_rating": 10,
            },
            "enemy": {
                "distance": enemy_distance,
                "attack_range": 50,
                "health": 75,
            },
            "game": {
                "turn_number": turn_number,
                "max_turns": 100,
            },
        }
    )


def build_tactical_request(
    intent: str,
    memory: MemoryScope,
    target_unit_id: str | None = None,
) -> DecisionRequest:
    """
    Build a DecisionRequest for a tactical decision.
    """
    parameters = {"priority": "survival"}
    if target_unit_id:
        parameters["target_unit_id"] = target_unit_id
    
    return DecisionRequest(
        request_id=str(uuid.uuid4()),
        query=QueryDescriptor(
            intent=intent,
            parameters=parameters,
            metadata={
                "ai_personality": "aggressive",
                "difficulty": "hard",
            },
        ),
        context={
            # Context from the request (dynamic per-request data)
            "target_unit_id": target_unit_id,
            "attack_strength": memory.get("unit", {}).get("attack_strength", 0),
        },
        memory=memory,  # Scoped state for traceability
        capabilities=[
            create_attack_capability(),
            create_defend_capability(),
            create_wait_capability(),
        ],
    )


def inspect_trace_for_debugging(trace) -> None:
    """
    Inspect the trace for debugging AI decisions.
    
    In a replay system, this trace would be stored alongside
    the game state to enable step-by-step decision analysis.
    """
    print("\n[TRACE ANALYSIS FOR REPLAY/DEBUG]")
    print(f"  Trace ID: {trace.trace_id}")
    print(f"  Total Decisions: {len(trace)}")
    
    if trace.parent_trace_id:
        print(f"  Parent Trace: {trace.parent_trace_id}")
    
    print("\n  Decision Chain:")
    for i, decision in enumerate(trace):
        print(f"\n  --- Decision {i + 1} ---")
        print(f"    Action:     {decision.action}")
        print(f"    Capability: {decision.capability}")
        print(f"    Confidence: {decision.confidence.value}")
        
        # Get the reason for this decision
        reasons = trace.reasons
        if i < len(reasons) and reasons[i]:
            reason = reasons[i]
            print(f"    Reason:     {reason.text}")
            if reason.category:
                print(f"    Category:   {reason.category}")


def handle_simulation_response(response, scenario_name: str) -> None:
    """Handle the response from the engine."""
    print(f"\n{'─' * 60}")
    print(f"Scenario: {scenario_name}")
    print(f"Response: {response.response_type.value.upper()}")
    print(f"{'─' * 60}")
    
    if response.response_type == ResponseType.ACCEPTED:
        payload = response.payload
        decision = payload.decision
        
        print(f"\n[CHOSEN ACTION]")
        print(f"  Action:     {decision.action}")
        print(f"  Capability: {decision.capability}")
        print(f"  Confidence: {decision.confidence.value}")
        
        if decision.inputs:
            print(f"\n  Action Parameters:")
            for key, value in decision.inputs.items():
                print(f"    {key}: {value}")
        
        # For simulation systems, the trace is crucial for replays
        inspect_trace_for_debugging(payload.trace)
        
        # Effects tell the game engine what will change
        if payload.effects:
            print(f"\n[EXPECTED EFFECTS]")
            for effect in payload.effects:
                print(f"  - {effect.category.value}: {effect.target}")
                if effect.description:
                    print(f"    ({effect.description})")
    
    elif response.response_type == ResponseType.REJECTED:
        payload = response.payload
        print(f"\n[NO VALID ACTION]")
        print(f"  Reason: {payload.error_code}")
        if payload.message:
            print(f"  Details: {payload.message}")
        
        # In a game, this might mean "skip turn" or "default action"
        print("\n  Game Engine should: Fall back to default wait action")
    
    elif response.response_type == ResponseType.DEFERRED:
        payload = response.payload
        print(f"\n[ACTION INCOMPLETE - NEEDS INPUT]")
        print(f"  Reason: {payload.reason}")
        
        for inp in payload.required_inputs:
            print(f"    - Need: {inp.name} ({inp.input_type})")


def main():
    """
    Run the simulation decision flow.
    
    Demonstrates three tactical scenarios with different outcomes.
    """
    engine = DecisionEngine()
    
    # =========================================================================
    # Scenario 1: Good position - should attack
    # =========================================================================
    memory_aggressive = build_game_state_memory(
        unit_health=80,
        ammunition=10,
        enemy_distance=30,  # Within attack range (50)
        turn_number=5,
    )
    
    request_aggressive = build_tactical_request(
        intent="attack_enemy",
        memory=memory_aggressive,
        target_unit_id="enemy_unit_001",
    )
    
    response_aggressive = engine.evaluate(request_aggressive)
    handle_simulation_response(response_aggressive, "Aggressive - In Range with Ammo")
    
    # =========================================================================
    # Scenario 2: Low health - should defend
    # =========================================================================
    memory_defensive = build_game_state_memory(
        unit_health=15,  # Low health
        ammunition=5,
        enemy_distance=40,
        turn_number=10,
    )
    
    request_defensive = build_tactical_request(
        intent="defensive_stance",
        memory=memory_defensive,
    )
    
    response_defensive = engine.evaluate(request_defensive)
    handle_simulation_response(response_defensive, "Defensive - Low Health")
    
    # =========================================================================
    # Scenario 3: No target specified for attack - should defer
    # =========================================================================
    memory_incomplete = build_game_state_memory(
        unit_health=100,
        ammunition=20,
        enemy_distance=25,
        turn_number=1,
    )
    
    # Attack without specifying target
    request_incomplete = build_tactical_request(
        intent="attack_enemy",
        memory=memory_incomplete,
        target_unit_id=None,  # Missing required input
    )
    
    response_incomplete = engine.evaluate(request_incomplete)
    handle_simulation_response(response_incomplete, "Incomplete - No Target")
    
    # =========================================================================
    # Demonstration: Determinism
    # =========================================================================
    print("\n" + "=" * 60)
    print("DETERMINISM CHECK: Same input → Same decision")
    print("=" * 60)
    
    # Create identical requests
    request_a = build_tactical_request(
        intent="wait_and_observe",
        memory=memory_aggressive,
    )
    request_b = build_tactical_request(
        intent="wait_and_observe",
        memory=memory_aggressive,
    )
    
    # Note: request_id differs, but content is the same
    response_a = engine.evaluate(request_a)
    response_b = engine.evaluate(request_b)
    
    if response_a.response_type == response_b.response_type == ResponseType.ACCEPTED:
        decision_a = response_a.payload.decision
        decision_b = response_b.payload.decision
        
        match = (
            decision_a.action == decision_b.action
            and decision_a.capability == decision_b.capability
        )
        print(f"\n  Decision Match: {'✓ PASS' if match else '✗ FAIL'}")
        print(f"    Action A: {decision_a.action}")
        print(f"    Action B: {decision_b.action}")
    else:
        print(f"\n  Response types: {response_a.response_type}, {response_b.response_type}")


if __name__ == "__main__":
    main()
