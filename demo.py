"""
demo.py

Minimal CLI demo for the Decision Trace Engine.
- Loads a hardcoded Semantic IR (simulating LLM compression output)
- Executes it deterministically
- Emits a causal trace to trace.jsonl
"""

import json

from ir import SemanticIR, IRStep
from executor import execute
from trace import TraceLog


def main():
    # --- Hardcoded IR (this is the artifact produced by the LLM) ---
    ir = SemanticIR(
        goal="Configure PostgreSQL for SaaS application",
        context={
            "ssl": True,
            "pool": True,
            "max_conn": 100,
            "timeout": 30
        },
        steps=[
            IRStep(action="set", params={"key": "ssl"}),
            IRStep(action="set", params={"key": "pool"}),
            IRStep(action="set", params={"key": "max_conn"}),
            IRStep(action="set", params={"key": "timeout"}),
            IRStep(action="emit", params={"target": "postgres_config"})
        ]
    )

    # --- Trace setup ---
    trace_path = "trace.jsonl"
    trace_log = TraceLog(trace_path)

    # --- Execute ---
    output = execute(ir, trace_log)

    # --- Result ---
    print("\nFinal Output:")
    print(json.dumps(output, indent=2))

    print(f"\nTrace written to: {trace_path}")
    print("Inspect with: cat trace.jsonl | jq .")


if __name__ == "__main__":
    main()
