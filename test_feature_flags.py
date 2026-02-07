"""
test_feature_flags.py

Integration tests for the Feature Flag Evaluation use case.
Proves the Decision Trace Engine works for real-world flag evaluation.
"""

import json
import os
import tempfile
from typing import List, Dict, Any

import pytest

from examples.feature_flags import FeatureFlag, FlagRule, flags_to_ir
from executor import execute
from trace import TraceLog
from graph import CausalGraph


class TestFeatureFlagAdapter:
    """Test the adapter converts flags to valid IR."""

    def test_single_flag_produces_valid_ir(self):
        """A single flag produces IR with correct structure."""
        flags = [
            FeatureFlag(name="new_dashboard", default=False, rules=[])
        ]
        user = {"user_id": "u123", "plan": "free"}
        
        ir = flags_to_ir(flags, user)
        
        assert ir.goal == "Evaluate 1 feature flags for user"
        assert "flag.new_dashboard.enabled" in ir.context
        assert len(ir.steps) == 2  # set + emit per flag

    def test_multiple_flags_produce_ordered_steps(self):
        """Multiple flags produce steps in order."""
        flags = [
            FeatureFlag(name="feature_a", default=True, rules=[]),
            FeatureFlag(name="feature_b", default=False, rules=[]),
        ]
        user = {"user_id": "u456"}
        
        ir = flags_to_ir(flags, user)
        
        assert len(ir.steps) == 4  # 2 steps per flag
        assert ir.steps[0].params["key"] == "flag.feature_a.enabled"
        assert ir.steps[2].params["key"] == "flag.feature_b.enabled"

    def test_user_context_prefixed_correctly(self):
        """User attributes get 'user.' prefix in context."""
        flags = [FeatureFlag(name="x", default=True, rules=[])]
        user = {"plan": "pro", "country": "US"}
        
        ir = flags_to_ir(flags, user)
        
        assert ir.context["user.plan"] == "pro"
        assert ir.context["user.country"] == "US"


class TestFeatureFlagExecution:
    """Test flag evaluation produces correct traces."""

    def _execute_flags(self, flags: List[FeatureFlag], user: Dict[str, Any]):
        """Helper: execute flags and return (trace_nodes, output)."""
        ir = flags_to_ir(flags, user)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            trace_log = TraceLog(temp_path)
            output = execute(ir, trace_log)
            
            nodes = []
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        nodes.append(json.loads(line))
            return nodes, output
        finally:
            os.unlink(temp_path)

    def test_flag_default_when_no_rules_match(self):
        """Flag returns default value when no rules match."""
        flags = [
            FeatureFlag(
                name="premium_feature",
                default=False,
                rules=[FlagRule(attribute="plan", operator="eq", value="enterprise")]
            )
        ]
        user = {"user_id": "u1", "plan": "free"}
        
        nodes, output = self._execute_flags(flags, user)
        
        # First node is the 'set' for flag decision
        set_node = nodes[0]
        assert set_node["output"]["flag.premium_feature.enabled"] is False

    def test_flag_enabled_when_rule_matches(self):
        """Flag enabled when user matches rule."""
        flags = [
            FeatureFlag(
                name="beta_feature",
                default=False,
                rules=[FlagRule(attribute="plan", operator="in", value=["pro", "enterprise"])]
            )
        ]
        user = {"user_id": "u2", "plan": "pro"}
        
        nodes, output = self._execute_flags(flags, user)
        
        set_node = nodes[0]
        assert set_node["output"]["flag.beta_feature.enabled"] is True

    def test_trace_has_causal_chain(self):
        """Flag evaluation produces valid causal chain."""
        flags = [
            FeatureFlag(name="f1", default=True, rules=[]),
            FeatureFlag(name="f2", default=False, rules=[]),
        ]
        user = {"user_id": "u3"}
        
        nodes, _ = self._execute_flags(flags, user)
        
        # Root has no parent
        assert nodes[0]["parent_id"] is None
        assert nodes[0]["depth"] == 0
        
        # Each subsequent node chains correctly
        for i in range(1, len(nodes)):
            assert nodes[i]["parent_id"] == nodes[i-1]["id"]
            assert nodes[i]["depth"] == i

    def test_reason_traces_to_ir_context(self):
        """Every decision traces back to IR context."""
        flags = [
            FeatureFlag(name="dark_mode", default=True, rules=[])
        ]
        user = {"user_id": "u4", "theme": "dark"}
        
        nodes, _ = self._execute_flags(flags, user)
        
        set_node = nodes[0]
        reason = set_node["reason"]
        
        assert reason["source"] == "ir"
        assert reason["type"] == "constraint"
        assert reason["ref"] == "context.flag.dark_mode.enabled"


class TestFeatureFlagDeterminism:
    """Test that flag evaluation is deterministic."""

    def test_same_input_produces_same_output(self):
        """Same flags + user = identical trace structure."""
        flags = [
            FeatureFlag(
                name="new_ui",
                default=False,
                rules=[
                    FlagRule(attribute="country", operator="in", value=["US", "CA"]),
                    FlagRule(attribute="beta_tester", operator="eq", value=True),
                ]
            )
        ]
        user = {"user_id": "u5", "country": "US", "beta_tester": False}
        
        traces = []
        for _ in range(2):
            ir = flags_to_ir(flags, user)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                temp_path = f.name
            
            try:
                trace_log = TraceLog(temp_path)
                execute(ir, trace_log)
                
                nodes = []
                with open(temp_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            nodes.append(json.loads(line))
                traces.append(nodes)
            finally:
                os.unlink(temp_path)
        
        # Compare structure (ignoring id/timestamp)
        for n1, n2 in zip(traces[0], traces[1]):
            assert n1["action"] == n2["action"]
            assert n1["output"] == n2["output"]
            assert n1["reason"] == n2["reason"]

    def test_different_user_produces_different_decision(self):
        """Different users can get different flag values."""
        flags = [
            FeatureFlag(
                name="vip_feature",
                default=False,
                rules=[FlagRule(attribute="tier", operator="gte", value=3)]
            )
        ]
        
        # Low tier user
        ir_low = flags_to_ir(flags, {"user_id": "u6", "tier": 1})
        # High tier user  
        ir_high = flags_to_ir(flags, {"user_id": "u7", "tier": 5})
        
        assert ir_low.context["flag.vip_feature.enabled"] is False
        assert ir_high.context["flag.vip_feature.enabled"] is True
