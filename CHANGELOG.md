# Changelog

All notable changes to Ntive Core are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-08

### Summary

Ntive Core v1.0.0 establishes the foundational infrastructure for deterministic,
auditable decision-making systems. This release provides a complete semantic bridge
between human intent and AI systems, with full causal traceability.

### Added

#### Core Primitives
- **Decision** — Immutable record of a decision with inputs, selected option,
  alternatives considered, confidence score, and rationale
- **Trace** — Ordered, immutable chain of decisions with causal relationships;
  content-addressable via SHA-256 hash
- **MemoryScope** — Immutable key-value store with hierarchical scoping and
  diff tracking between states
- **Policy** — Declarative policy rules with conflict resolution strategies
  (FIRST_MATCH, MOST_SPECIFIC, PRIORITY, DENY_OVERRIDE)
- **CapabilityDescriptor** — Structured declaration of what an action can do,
  including inputs, preconditions, effects, and constraints

#### Request/Response Contract
- **DecisionRequest** — Inbound contract carrying context, query, capabilities,
  policies, memory, and constraints
- **DecisionResponse** — Outbound contract with three response types:
  - `ACCEPTED` — Decision made, includes trace
  - `REJECTED` — Decision refused with error code and reason
  - `DEFERRED` — Decision pending, specifies required inputs

#### Decision Engine
- **DecisionEngine** — Pure orchestration layer that evaluates requests and
  produces responses; no side effects, no external calls
- Validates request structure, context, and policy compliance
- Builds deterministic traces for all decisions
- Returns structured responses with full audit trail

#### Trace Replay (Audit Tool)
- **TraceReplay** — Read-only replay and verification of past decision traces
- Load from JSONL or JSON files
- **Chain validation** — Detect broken causal links, orphan nodes, cycles
- **Determinism verification** — Compare trace hashes for reproducibility
- **Human-readable explanations** — Generate text/markdown summaries

#### Examples
- `examples/assistant_flow.py` — Basic decision request/response flow
- `examples/simulation_flow.py` — Multi-step decision chains
- `examples/regulated_flow.py` — Policy-constrained decisions

### Design Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Deterministic** | Same inputs → same outputs, always |
| **Immutable** | All primitives are frozen after construction |
| **Non-executable** | Ntive describes decisions, never executes them |
| **Auditable** | Every decision has a causal chain back to origin |
| **Content-addressable** | Trace IDs derived from content (SHA-256) |
| **Serializable** | All primitives serialize to/from JSON deterministically |

### Test Coverage

- **856 tests passing** (pytest)
- Coverage includes: Decision, Trace, Memory, Policy, Capability, Request,
  Response, Engine, TraceReplay
- All primitives tested for immutability, validation, serialization

### Quality Assurance

- **Ruff** — All checks passing (import sorting, style)
- **Flake8** — Zero errors (line-length 110, E302 ignored for ruff compat)
- **Pyright** — 15 type warnings (tracked, downgraded from errors)
- Explicit ignore configuration in `.flake8` and `pyproject.toml`

### Non-Goals (Explicit)

Ntive Core is infrastructure, not an agent or assistant. It does NOT:

- Execute actions or call external systems
- Make autonomous decisions
- Maintain hidden state
- Learn or adapt behavior
- Provide natural language understanding

---

## [0.1.0] - 2026-01-15

### Added

- Initial parser and runtime for Ntive DSL
- AST nodes for press, type, wait, move, click commands
- Human-friendly error system
- Basic test infrastructure
