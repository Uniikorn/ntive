# Ntive Core — Semantic Decision Infrastructure

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-856%20passing-brightgreen.svg)]()

**Ntive Core** is deterministic decision infrastructure that bridges human intent and AI systems. It provides the semantic primitives for building auditable, traceable, and reproducible decision-making pipelines.

## What Ntive Is

- **Semantic bridge** — Structured contracts between humans and AI systems
- **Decision infrastructure** — Primitives for capturing, tracing, and auditing decisions
- **Deterministic** — Same inputs always produce identical outputs
- **Immutable** — All data structures are frozen after construction
- **Non-executable** — Describes decisions, never performs them

## What Ntive Is NOT

| Not This | Why |
|----------|-----|
| An AI agent | Ntive describes decisions; it doesn't make them autonomously |
| An executor | Ntive traces what should happen; external systems act |
| A learning system | No adaptation, no hidden state, no inference |
| A chatbot | No NLU, no generation, no conversation management |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INBOUND                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Capability  │    │    Policy    │    │   Memory     │          │
│  │  Descriptor  │    │    Rules     │    │   Scope      │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             ▼                                       │
│                   ┌──────────────────┐                              │
│                   │ DecisionRequest  │                              │
│                   └────────┬─────────┘                              │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ DecisionEngine   │  ← Pure orchestration
                   │ (no side effects)│
                   └────────┬─────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                            ▼                         OUTBOUND       │
│                   ┌──────────────────┐                              │
│                   │ DecisionResponse │                              │
│                   └────────┬─────────┘                              │
│         ┌──────────────────┼──────────────────┐                     │
│         ▼                  ▼                  ▼                     │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                 │
│  │  ACCEPTED  │    │  REJECTED  │    │  DEFERRED  │                 │
│  │  + Decision│    │  + Reason  │    │  + Inputs  │                 │
│  │  + Trace   │    │  + Code    │    │  + Token   │                 │
│  └────────────┘    └────────────┘    └────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │   TraceReplay    │  ← Audit & verification
                   │ (read-only)      │
                   └──────────────────┘
```

---

## Core Primitives

### Decision

An immutable record of a decision:

```python
from ntive import Decision, Alternative, Confidence

decision = Decision(
    inputs={"user_id": "u123", "action": "transfer", "amount": 500},
    selected_option="approve",
    alternatives=[
        Alternative(option="reject", reason_not_selected="Amount within limits"),
        Alternative(option="escalate", reason_not_selected="User verified"),
    ],
    rationale="Transfer approved: amount $500 is within user's daily limit of $1000",
    confidence=Confidence(value=0.95, lower_bound=0.90, upper_bound=0.98),
)
```

### Trace

An ordered, immutable chain of decisions with causal links:

```python
from ntive import Trace, CausalReason

trace = Trace.build([
    (decision1, None),  # Root decision
    (decision2, CausalReason("Follows from approval", "policy")),
    (decision3, "User confirmed"),  # String shorthand
])

# Content-addressable
print(trace.trace_id)  # sha256 hash of content
```

### CapabilityDescriptor

Declares what an action can do:

```python
from ntive import CapabilityDescriptor, CapabilityInput, DeclaredEffect

capability = CapabilityDescriptor(
    name="transfer_funds",
    description="Transfer money between accounts",
    inputs=[
        CapabilityInput(key="from_account", type="string", required=True),
        CapabilityInput(key="to_account", type="string", required=True),
        CapabilityInput(key="amount", type="number", required=True),
    ],
    effects=[
        DeclaredEffect(category="financial", target="balance", description="Modifies account balance"),
    ],
)
```

### Policy

Declarative rules with conflict resolution:

```python
from ntive import Policy, PolicyRule, PolicyEffect

policy = Policy.build([
    PolicyRule(
        rule_id="max_transfer",
        condition="amount > 10000",
        effect=PolicyEffect.DENY,
        priority=100,
    ),
    PolicyRule(
        rule_id="verified_user",
        condition="user.verified == true",
        effect=PolicyEffect.ALLOW,
        priority=50,
    ),
])
```

---

## Request/Response Flow

```python
from ntive import DecisionRequest, DecisionEngine, ResponseType

# Build a request
request = DecisionRequest(
    query={"action": "transfer", "amount": 500},
    context={"user": {"id": "u123", "verified": True}},
    capabilities=[capability],
    policies=[policy],
)

# Evaluate (pure, no side effects)
engine = DecisionEngine()
response = engine.evaluate(request)

# Handle response
match response.response_type:
    case ResponseType.ACCEPTED:
        print(f"Decision: {response.payload.decision.selected_option}")
        print(f"Trace ID: {response.payload.trace.trace_id}")
    case ResponseType.REJECTED:
        print(f"Rejected: {response.payload.reason}")
    case ResponseType.DEFERRED:
        print(f"Need more info: {response.payload.required_inputs}")
```

---

## Trace Replay (Audit)

Replay and verify past decisions without re-execution:

```python
from ntive import TraceReplay

# Load from file
replay = TraceReplay.load_jsonl("trace.jsonl")

# Validate causal chain
validation = replay.validate_chain()
if not validation.is_valid:
    print(f"Broken links: {validation.broken_links}")

# Verify determinism
check = replay.verify_determinism(expected_hash="sha256:...")
print(f"Deterministic: {check.is_deterministic}")

# Human-readable explanation
explanation = replay.explain()
print(explanation.to_text())
```

---

## Guarantees

| Property | Guarantee |
|----------|-----------|
| **Deterministic** | Same request → same response, always |
| **Immutable** | All primitives frozen after construction |
| **Content-addressable** | Trace IDs = SHA-256 of content |
| **Serializable** | All types ↔ JSON, deterministically |
| **Non-executable** | Describes intent; never acts |
| **Auditable** | Every decision traceable to origin |

---

## Installation

```bash
git clone https://github.com/Uniikorn/ntive.git
cd ntive
pip install -e .
```

---

## Quality

| Tool | Status |
|------|--------|
| **pytest** | 856 tests passing |
| **ruff** | All checks passing |
| **flake8** | Zero errors |
| **pyright** | Type-checked (15 warnings tracked) |

---

## Project Structure

```
ntive/
├── __init__.py         # Public API (92 exports)
├── decision.py         # Decision, Alternative, Confidence
├── trace.py            # Trace, CausalReason
├── trace_replay.py     # TraceReplay (audit tool)
├── memory.py           # MemoryScope, MemoryDiff
├── policy.py           # Policy, PolicyRule
├── capability.py       # CapabilityDescriptor
├── request.py          # DecisionRequest
├── response.py         # DecisionResponse
├── engine.py           # DecisionEngine
├── parser.py           # (internal) DSL parser
├── runtime.py          # (internal) DSL runtime
└── errors.py           # (internal) Error classes
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

**Ntive Core**: *Decisions you can trace. Infrastructure you can trust.*

