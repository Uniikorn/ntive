"""
test_policy_primitive.py

Unit tests for the Ntive Policy primitive.

Tests prove:
- Immutability (cannot modify after creation)
- Inheritance (child overrides parent)
- Conflict resolution (most_restrictive, explicit_priority, reject_conflict)
- Deterministic hashing (same content = same policy_id)
- Validation (invalid rules rejected, duplicate IDs rejected, cyclic inheritance rejected)
"""

import pytest
import json
from ntive.policy import (
    Policy,
    PolicyRule,
    PolicyEffectResult,
    PolicyEffect,
    ConflictResolutionStrategy,
    PolicyValidationError,
    InvalidPolicyRuleError,
    DuplicateRuleIdError,
    InvalidPolicyEffectError,
    CyclicPolicyInheritanceError,
    PolicyConflictError,
    PolicyImmutabilityError,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def simple_rule():
    """A simple policy rule."""
    return PolicyRule(
        rule_id="rule_1",
        effect="forbid",
        target="feature_x",
        condition={"env": "production"},
    )


@pytest.fixture
def root_policy():
    """Root policy with some rules."""
    return Policy(
        name="root_policy",
        version="1.0.0",
        rules=[
            PolicyRule(
                rule_id="root_rule_1",
                effect="forbid",
                target="dangerous_action",
            ),
            PolicyRule(
                rule_id="root_rule_2",
                effect="prefer",
                target="safe_mode",
                weight=0.8,
            ),
        ],
        priority=10,
    )


@pytest.fixture
def child_policy(root_policy):
    """Child policy that overrides some rules."""
    return Policy(
        name="child_policy",
        version="1.1.0",
        rules=[
            PolicyRule(
                rule_id="root_rule_1",  # Override parent rule
                effect="require",
                target="dangerous_action",
            ),
            PolicyRule(
                rule_id="child_rule_1",
                effect="default",
                target="new_feature",
            ),
        ],
        priority=20,
        parent=root_policy,
    )


# =============================================================================
# SECTION 1: PolicyRule Tests
# =============================================================================

class TestPolicyRule:
    """Test PolicyRule construction and behavior."""
    
    def test_basic_creation(self):
        """PolicyRule can be created with required fields."""
        rule = PolicyRule(
            rule_id="test_rule",
            effect="forbid",
            target="some_target",
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.effect == PolicyEffect.FORBID
        assert rule.target == "some_target"
        assert rule.condition == {}
        assert rule.weight is None
    
    def test_creation_with_all_fields(self):
        """PolicyRule can be created with all fields."""
        rule = PolicyRule(
            rule_id="full_rule",
            effect="prefer",
            target="feature_a",
            condition={"user_role": "admin"},
            weight=0.9,
        )
        
        assert rule.rule_id == "full_rule"
        assert rule.effect == PolicyEffect.PREFER
        assert rule.target == "feature_a"
        assert rule.condition == {"user_role": "admin"}
        assert rule.weight == 0.9
    
    def test_effect_enum_accepted(self):
        """PolicyRule accepts PolicyEffect enum."""
        rule = PolicyRule(
            rule_id="enum_rule",
            effect=PolicyEffect.REQUIRE,
            target="target",
        )
        
        assert rule.effect == PolicyEffect.REQUIRE
    
    def test_all_effect_types(self):
        """All effect types are valid."""
        for effect in ["require", "forbid", "prefer", "default"]:
            rule = PolicyRule(
                rule_id=f"rule_{effect}",
                effect=effect,
                target="target",
            )
            assert rule.effect.value == effect
    
    def test_condition_is_deep_copied(self):
        """Condition is deep copied on creation."""
        original = {"nested": {"value": 1}}
        rule = PolicyRule(
            rule_id="rule",
            effect="forbid",
            target="target",
            condition=original,
        )
        
        original["nested"]["value"] = 999
        assert rule.condition["nested"]["value"] == 1
    
    def test_condition_returned_is_copy(self):
        """Accessing condition returns a copy."""
        rule = PolicyRule(
            rule_id="rule",
            effect="forbid",
            target="target",
            condition={"list": [1, 2, 3]},
        )
        
        cond = rule.condition
        cond["list"].append(4)
        
        assert rule.condition["list"] == [1, 2, 3]
    
    def test_invalid_effect_rejected(self):
        """Invalid effect string is rejected."""
        with pytest.raises(InvalidPolicyEffectError) as exc_info:
            PolicyRule(
                rule_id="rule",
                effect="invalid_effect",
                target="target",
            )
        
        assert "invalid_effect" in str(exc_info.value)
        assert "P003" in str(exc_info.value)
    
    def test_empty_rule_id_rejected(self):
        """Empty rule_id is rejected."""
        with pytest.raises(PolicyValidationError):
            PolicyRule(
                rule_id="",
                effect="forbid",
                target="target",
            )
    
    def test_empty_target_rejected(self):
        """Empty target is rejected."""
        with pytest.raises(PolicyValidationError):
            PolicyRule(
                rule_id="rule",
                effect="forbid",
                target="",
            )
    
    def test_non_dict_condition_rejected(self):
        """Non-dict condition is rejected."""
        with pytest.raises(PolicyValidationError) as exc_info:
            PolicyRule(
                rule_id="rule",
                effect="forbid",
                target="target",
                condition="not a dict",
            )
        
        assert "P008" in str(exc_info.value)
    
    def test_non_json_serializable_condition_rejected(self):
        """Non-JSON-serializable condition is rejected."""
        with pytest.raises(PolicyValidationError):
            PolicyRule(
                rule_id="rule",
                effect="forbid",
                target="target",
                condition={"func": lambda x: x},
            )
    
    def test_non_numeric_weight_rejected(self):
        """Non-numeric weight is rejected."""
        with pytest.raises(InvalidPolicyRuleError):
            PolicyRule(
                rule_id="rule",
                effect="prefer",
                target="target",
                weight="high",
            )
    
    def test_nan_weight_rejected(self):
        """NaN weight is rejected."""
        with pytest.raises(InvalidPolicyRuleError):
            PolicyRule(
                rule_id="rule",
                effect="prefer",
                target="target",
                weight=float('nan'),
            )
    
    def test_infinity_weight_rejected(self):
        """Infinity weight is rejected."""
        with pytest.raises(InvalidPolicyRuleError):
            PolicyRule(
                rule_id="rule",
                effect="prefer",
                target="target",
                weight=float('inf'),
            )


class TestPolicyRuleSerialization:
    """Test PolicyRule serialization."""
    
    def test_to_dict(self, simple_rule):
        """to_dict() returns correct structure."""
        data = simple_rule.to_dict()
        
        assert data["rule_id"] == "rule_1"
        assert data["effect"] == "forbid"
        assert data["target"] == "feature_x"
        assert data["condition"] == {"env": "production"}
    
    def test_to_json(self, simple_rule):
        """to_json() returns valid JSON."""
        json_str = simple_rule.to_json()
        data = json.loads(json_str)
        
        assert data["rule_id"] == "rule_1"
    
    def test_from_dict(self):
        """from_dict() reconstructs rule."""
        data = {
            "rule_id": "test",
            "effect": "require",
            "target": "feature",
            "condition": {"key": "value"},
            "weight": 0.5,
        }
        
        rule = PolicyRule.from_dict(data)
        
        assert rule.rule_id == "test"
        assert rule.effect == PolicyEffect.REQUIRE
        assert rule.weight == 0.5
    
    def test_roundtrip(self, simple_rule):
        """from_json(to_json()) roundtrips correctly."""
        json_str = simple_rule.to_json()
        restored = PolicyRule.from_json(json_str)
        
        assert restored == simple_rule


# =============================================================================
# SECTION 2: Policy Immutability Tests
# =============================================================================

class TestImmutability:
    """Test that Policy and related objects are immutable."""
    
    def test_policy_cannot_set_attribute(self, root_policy):
        """Setting any attribute raises PolicyImmutabilityError."""
        with pytest.raises(PolicyImmutabilityError):
            root_policy._name = "changed"
    
    def test_policy_cannot_delete_attribute(self, root_policy):
        """Deleting any attribute raises PolicyImmutabilityError."""
        with pytest.raises(PolicyImmutabilityError):
            del root_policy._policy_id
    
    def test_policy_cannot_add_attribute(self, root_policy):
        """Adding new attribute raises PolicyImmutabilityError."""
        with pytest.raises(PolicyImmutabilityError):
            root_policy.new_field = "value"
    
    def test_rule_cannot_set_attribute(self, simple_rule):
        """PolicyRule cannot be mutated."""
        with pytest.raises(PolicyImmutabilityError):
            simple_rule._target = "changed"
    
    def test_effect_result_cannot_set_attribute(self):
        """PolicyEffectResult cannot be mutated."""
        result = PolicyEffectResult(
            policy_id="policy_1",
            rule_id="rule_1",
            effect="forbid",
            applied=True,
            reason="Test reason",
        )
        
        with pytest.raises(PolicyImmutabilityError):
            result._applied = False
    
    def test_add_rule_returns_new_policy(self, root_policy):
        """add_rule() returns a new Policy."""
        original_id = root_policy.policy_id
        
        new_policy = root_policy.add_rule(
            PolicyRule(
                rule_id="new_rule",
                effect="default",
                target="new_target",
            )
        )
        
        assert new_policy is not root_policy
        assert root_policy.policy_id == original_id
        assert not root_policy.has_rule("new_rule")
        assert new_policy.has_rule("new_rule")
    
    def test_with_priority_returns_new_policy(self, root_policy):
        """with_priority() returns a new Policy."""
        new_policy = root_policy.with_priority(100)
        
        assert new_policy is not root_policy
        assert root_policy.priority == 10
        assert new_policy.priority == 100
    
    def test_with_parent_returns_new_policy(self, root_policy):
        """with_parent() returns a new Policy."""
        parent = Policy.empty("new_parent")
        new_policy = root_policy.with_parent(parent)
        
        assert new_policy is not root_policy
        assert root_policy.parent is None
        assert new_policy.parent is parent


# =============================================================================
# SECTION 3: Policy Inheritance Tests
# =============================================================================

class TestInheritance:
    """Test policy inheritance."""
    
    def test_child_inherits_parent_rules(self, root_policy, child_policy):
        """Child policy can access parent rules."""
        # root_rule_2 is only in parent
        assert child_policy.has_rule("root_rule_2")
        rule = child_policy.get_rule("root_rule_2")
        assert rule.target == "safe_mode"
    
    def test_child_overrides_parent_rules(self, root_policy, child_policy):
        """Child rules override parent rules with same ID."""
        # root_rule_1 is overridden in child
        rule = child_policy.get_rule("root_rule_1")
        
        # Child's version is REQUIRE, parent's was FORBID
        assert rule.effect == PolicyEffect.REQUIRE
    
    def test_parent_rules_unchanged(self, root_policy, child_policy):
        """Parent rules are not affected by child overrides."""
        rule = root_policy.get_rule("root_rule_1")
        assert rule.effect == PolicyEffect.FORBID
    
    def test_deep_inheritance_chain(self):
        """Deep inheritance chains work correctly."""
        root = Policy(
            name="root",
            rules=[PolicyRule(rule_id="r1", effect="forbid", target="a")],
        )
        level1 = Policy(
            name="level1",
            rules=[PolicyRule(rule_id="r2", effect="prefer", target="b")],
            parent=root,
        )
        level2 = Policy(
            name="level2",
            rules=[PolicyRule(rule_id="r3", effect="default", target="c")],
            parent=level1,
        )
        level3 = Policy(
            name="level3",
            rules=[PolicyRule(rule_id="r4", effect="require", target="d")],
            parent=level2,
        )
        
        # All rules accessible
        assert level3.has_rule("r1")
        assert level3.has_rule("r2")
        assert level3.has_rule("r3")
        assert level3.has_rule("r4")
    
    def test_has_local_rule(self, root_policy, child_policy):
        """has_local_rule() only checks local rules."""
        assert child_policy.has_local_rule("child_rule_1")
        assert child_policy.has_local_rule("root_rule_1")  # Overridden locally
        assert not child_policy.has_local_rule("root_rule_2")  # Only in parent
    
    def test_chain_returns_all_ancestors(self, root_policy, child_policy):
        """chain() returns all policies from root to leaf."""
        chain = child_policy.chain()
        
        assert len(chain) == 2
        assert chain[0] is root_policy
        assert chain[1] is child_policy
    
    def test_root_returns_topmost_policy(self, root_policy, child_policy):
        """root() returns the root policy."""
        assert child_policy.root() is root_policy
        assert root_policy.root() is root_policy
    
    def test_depth_property(self, root_policy, child_policy):
        """depth property returns correct depth."""
        assert root_policy.depth == 0
        assert child_policy.depth == 1
    
    def test_rules_method_resolves_inheritance(self, child_policy):
        """rules() returns all visible rules with resolved overrides."""
        all_rules = child_policy.rules()
        rule_ids = [r.rule_id for r in all_rules]
        
        assert "root_rule_1" in rule_ids  # Overridden
        assert "root_rule_2" in rule_ids  # Inherited
        assert "child_rule_1" in rule_ids  # Local
    
    def test_rules_sorted_by_id(self, child_policy):
        """rules() returns rules sorted by rule_id."""
        all_rules = child_policy.rules()
        rule_ids = [r.rule_id for r in all_rules]
        
        assert rule_ids == sorted(rule_ids)


# =============================================================================
# SECTION 4: Cyclic Inheritance Tests
# =============================================================================

class TestCyclicInheritance:
    """Test cyclic inheritance detection."""
    
    def test_self_reference_rejected(self):
        """Policy cannot be its own parent."""
        policy = Policy.empty("self_ref")
        
        with pytest.raises(CyclicPolicyInheritanceError) as exc_info:
            Policy(
                name="self_ref",  # Same name as would-be parent
                rules=[],
                parent=policy,
            )
        
        assert "P004" in str(exc_info.value)
    
    def test_direct_cycle_rejected(self):
        """Direct A->B->A cycle is rejected."""
        a = Policy.empty("policy_a")
        b = Policy(name="policy_b", rules=[], parent=a)
        
        with pytest.raises(CyclicPolicyInheritanceError):
            Policy(
                name="policy_a",  # Same name, creates cycle
                rules=[],
                parent=b,
            )
    
    def test_indirect_cycle_rejected(self):
        """Indirect A->B->C->A cycle is rejected."""
        a = Policy.empty("policy_a")
        b = Policy(name="policy_b", rules=[], parent=a)
        c = Policy(name="policy_c", rules=[], parent=b)
        
        with pytest.raises(CyclicPolicyInheritanceError):
            Policy(
                name="policy_a",
                rules=[],
                parent=c,
            )
    
    def test_error_includes_cycle_path(self):
        """CyclicPolicyInheritanceError includes the cycle path."""
        a = Policy.empty("A")
        b = Policy(name="B", rules=[], parent=a)
        
        with pytest.raises(CyclicPolicyInheritanceError) as exc_info:
            Policy(name="A", rules=[], parent=b)
        
        assert exc_info.value.cycle_path == ["A", "B", "A"]


# =============================================================================
# SECTION 5: Conflict Resolution Tests
# =============================================================================

class TestConflictResolution:
    """Test conflict resolution strategies."""
    
    def test_most_restrictive_forbid_wins(self):
        """MOST_RESTRICTIVE: FORBID wins over all others."""
        policy = Policy(
            name="test",
            rules=[
                PolicyRule(rule_id="r1", effect="forbid", target="x"),
                PolicyRule(rule_id="r2", effect="require", target="x"),
                PolicyRule(rule_id="r3", effect="prefer", target="x"),
            ],
            conflict_strategy="most_restrictive",
        )
        
        assert policy.resolve_effect_for_target("x") == PolicyEffect.FORBID
    
    def test_most_restrictive_require_wins_over_prefer(self):
        """MOST_RESTRICTIVE: REQUIRE wins over PREFER."""
        policy = Policy(
            name="test",
            rules=[
                PolicyRule(rule_id="r1", effect="require", target="x"),
                PolicyRule(rule_id="r2", effect="prefer", target="x"),
            ],
            conflict_strategy="most_restrictive",
        )
        
        assert policy.resolve_effect_for_target("x") == PolicyEffect.REQUIRE
    
    def test_most_restrictive_prefer_wins_over_default(self):
        """MOST_RESTRICTIVE: PREFER wins over DEFAULT."""
        policy = Policy(
            name="test",
            rules=[
                PolicyRule(rule_id="r1", effect="prefer", target="x"),
                PolicyRule(rule_id="r2", effect="default", target="x"),
            ],
            conflict_strategy="most_restrictive",
        )
        
        assert policy.resolve_effect_for_target("x") == PolicyEffect.PREFER
    
    def test_explicit_priority_higher_wins(self):
        """EXPLICIT_PRIORITY: Higher priority policy wins."""
        low_priority = Policy(
            name="low",
            rules=[PolicyRule(rule_id="r1", effect="forbid", target="x")],
            priority=10,
        )
        high_priority = Policy(
            name="high",
            rules=[PolicyRule(rule_id="r2", effect="require", target="x")],
            priority=100,
            parent=low_priority,
            conflict_strategy="explicit_priority",
        )
        
        # High priority (REQUIRE) should win over low priority (FORBID)
        assert high_priority.resolve_effect_for_target("x") == PolicyEffect.REQUIRE
    
    def test_reject_conflict_raises_on_require_forbid(self):
        """REJECT_CONFLICT: REQUIRE + FORBID on same target raises error."""
        with pytest.raises(PolicyConflictError) as exc_info:
            Policy(
                name="conflicting",
                rules=[
                    PolicyRule(rule_id="r1", effect="require", target="x"),
                    PolicyRule(rule_id="r2", effect="forbid", target="x"),
                ],
                conflict_strategy="reject_conflict",
            )
        
        assert exc_info.value.target == "x"
        assert "P005" in str(exc_info.value)
    
    def test_reject_conflict_with_inheritance(self):
        """REJECT_CONFLICT: Conflicts with inherited rules are detected."""
        parent = Policy(
            name="parent",
            rules=[PolicyRule(rule_id="r1", effect="forbid", target="x")],
        )
        
        with pytest.raises(PolicyConflictError):
            Policy(
                name="child",
                rules=[PolicyRule(rule_id="r2", effect="require", target="x")],
                parent=parent,
                conflict_strategy="reject_conflict",
            )
    
    def test_prefer_prefer_allowed(self):
        """REJECT_CONFLICT: Multiple PREFER on same target is allowed."""
        policy = Policy(
            name="test",
            rules=[
                PolicyRule(rule_id="r1", effect="prefer", target="x", weight=0.5),
                PolicyRule(rule_id="r2", effect="prefer", target="x", weight=0.8),
            ],
            conflict_strategy="reject_conflict",
        )
        
        assert policy.resolve_effect_for_target("x") == PolicyEffect.PREFER
    
    def test_no_rules_for_target_returns_none(self, root_policy):
        """resolve_effect_for_target() returns None for unknown target."""
        result = root_policy.resolve_effect_for_target("nonexistent_target")
        assert result is None
    
    def test_get_all_targets(self, child_policy):
        """get_all_targets() returns all unique targets."""
        targets = child_policy.get_all_targets()
        
        assert "dangerous_action" in targets
        assert "safe_mode" in targets
        assert "new_feature" in targets


# =============================================================================
# SECTION 6: Deterministic Hashing Tests
# =============================================================================

class TestDeterministicHashing:
    """Test that policy_id is content-based and deterministic."""
    
    def test_identical_policies_have_same_id(self):
        """Same content = same policy_id."""
        policy1 = Policy(
            name="test",
            version="1.0.0",
            rules=[PolicyRule(rule_id="r1", effect="forbid", target="x")],
            priority=10,
        )
        policy2 = Policy(
            name="test",
            version="1.0.0",
            rules=[PolicyRule(rule_id="r1", effect="forbid", target="x")],
            priority=10,
        )
        
        assert policy1.policy_id == policy2.policy_id
    
    def test_different_name_different_id(self):
        """Different name = different policy_id."""
        policy1 = Policy(name="test1", rules=[])
        policy2 = Policy(name="test2", rules=[])
        
        assert policy1.policy_id != policy2.policy_id
    
    def test_different_version_different_id(self):
        """Different version = different policy_id."""
        policy1 = Policy(name="test", version="1.0.0", rules=[])
        policy2 = Policy(name="test", version="2.0.0", rules=[])
        
        assert policy1.policy_id != policy2.policy_id
    
    def test_different_priority_different_id(self):
        """Different priority = different policy_id."""
        policy1 = Policy(name="test", priority=10, rules=[])
        policy2 = Policy(name="test", priority=20, rules=[])
        
        assert policy1.policy_id != policy2.policy_id
    
    def test_different_rules_different_id(self):
        """Different rules = different policy_id."""
        policy1 = Policy(
            name="test",
            rules=[PolicyRule(rule_id="r1", effect="forbid", target="x")],
        )
        policy2 = Policy(
            name="test",
            rules=[PolicyRule(rule_id="r1", effect="require", target="x")],
        )
        
        assert policy1.policy_id != policy2.policy_id
    
    def test_different_parent_different_id(self):
        """Different parent = different policy_id."""
        parent1 = Policy(name="parent1", rules=[])
        parent2 = Policy(name="parent2", rules=[])
        
        child1 = Policy(name="child", rules=[], parent=parent1)
        child2 = Policy(name="child", rules=[], parent=parent2)
        
        assert child1.policy_id != child2.policy_id
    
    def test_policy_id_is_hex_string(self, root_policy):
        """policy_id is a hex string (SHA-256)."""
        assert len(root_policy.policy_id) == 64
        assert all(c in '0123456789abcdef' for c in root_policy.policy_id)
    
    def test_json_is_deterministic(self, root_policy):
        """Multiple to_json() calls return identical strings."""
        json1 = root_policy.to_json()
        json2 = root_policy.to_json()
        json3 = root_policy.to_json()
        
        assert json1 == json2 == json3
    
    def test_json_keys_are_sorted(self, root_policy):
        """JSON output has sorted keys."""
        json_str = root_policy.to_json()
        data = json.loads(json_str)
        keys = list(data.keys())
        
        assert keys == sorted(keys)
    
    def test_equal_policies_are_equal(self):
        """Policies with same policy_id are equal."""
        policy1 = Policy(name="test", rules=[])
        policy2 = Policy(name="test", rules=[])
        
        assert policy1 == policy2
    
    def test_equal_policies_have_same_hash(self):
        """Equal policies have the same hash."""
        policy1 = Policy(name="test", rules=[])
        policy2 = Policy(name="test", rules=[])
        
        assert hash(policy1) == hash(policy2)
    
    def test_policies_can_be_in_sets(self):
        """Policies can be used in sets."""
        policy1 = Policy(name="test", rules=[])
        policy2 = Policy(name="test", rules=[])
        
        s = {policy1, policy2}
        assert len(s) == 1  # Deduplicated


# =============================================================================
# SECTION 7: Validation Tests
# =============================================================================

class TestValidation:
    """Test validation of Policy construction."""
    
    def test_empty_name_rejected(self):
        """Empty name is rejected."""
        with pytest.raises(PolicyValidationError) as exc_info:
            Policy(name="", rules=[])
        
        assert "P007" in str(exc_info.value)
    
    def test_whitespace_name_rejected(self):
        """Whitespace-only name is rejected."""
        with pytest.raises(PolicyValidationError):
            Policy(name="   ", rules=[])
    
    def test_non_int_priority_rejected(self):
        """Non-int priority is rejected."""
        with pytest.raises(PolicyValidationError) as exc_info:
            Policy(name="test", priority=10.5, rules=[])
        
        assert "P010" in str(exc_info.value)
    
    def test_invalid_parent_type_rejected(self):
        """Non-Policy parent is rejected."""
        with pytest.raises(PolicyValidationError) as exc_info:
            Policy(name="test", parent="not a policy", rules=[])
        
        assert "P011" in str(exc_info.value)
    
    def test_duplicate_rule_ids_rejected(self):
        """Duplicate rule IDs are rejected."""
        with pytest.raises(DuplicateRuleIdError) as exc_info:
            Policy(
                name="test",
                rules=[
                    PolicyRule(rule_id="same_id", effect="forbid", target="x"),
                    PolicyRule(rule_id="same_id", effect="require", target="y"),
                ],
            )
        
        assert exc_info.value.rule_id == "same_id"
        assert "P002" in str(exc_info.value)
    
    def test_invalid_conflict_strategy_rejected(self):
        """Invalid conflict strategy is rejected."""
        with pytest.raises(PolicyValidationError) as exc_info:
            Policy(
                name="test",
                rules=[],
                conflict_strategy="invalid_strategy",
            )
        
        assert "P006" in str(exc_info.value)
    
    def test_rules_can_be_dicts(self):
        """Rules can be provided as dicts."""
        policy = Policy(
            name="test",
            rules=[
                {"rule_id": "r1", "effect": "forbid", "target": "x"},
                {"rule_id": "r2", "effect": "require", "target": "y"},
            ],
        )
        
        assert len(policy) == 2
        assert policy.has_rule("r1")
        assert policy.has_rule("r2")


# =============================================================================
# SECTION 8: PolicyEffectResult Tests
# =============================================================================

class TestPolicyEffectResult:
    """Test PolicyEffectResult construction and behavior."""
    
    def test_basic_creation(self):
        """PolicyEffectResult can be created."""
        result = PolicyEffectResult(
            policy_id="policy_1",
            rule_id="rule_1",
            effect="forbid",
            applied=True,
            reason="Target is forbidden in production",
        )
        
        assert result.policy_id == "policy_1"
        assert result.rule_id == "rule_1"
        assert result.effect == PolicyEffect.FORBID
        assert result.applied is True
        assert result.reason == "Target is forbidden in production"
    
    def test_effect_enum_accepted(self):
        """PolicyEffectResult accepts PolicyEffect enum."""
        result = PolicyEffectResult(
            policy_id="p1",
            rule_id="r1",
            effect=PolicyEffect.REQUIRE,
            applied=False,
            reason="Condition not met",
        )
        
        assert result.effect == PolicyEffect.REQUIRE
    
    def test_to_dict(self):
        """to_dict() returns correct structure."""
        result = PolicyEffectResult(
            policy_id="p1",
            rule_id="r1",
            effect="prefer",
            applied=True,
            reason="Preferred option",
        )
        
        data = result.to_dict()
        
        assert data["policy_id"] == "p1"
        assert data["rule_id"] == "r1"
        assert data["effect"] == "prefer"
        assert data["applied"] is True
        assert data["reason"] == "Preferred option"
    
    def test_roundtrip(self):
        """from_json(to_json()) roundtrips correctly."""
        result = PolicyEffectResult(
            policy_id="p1",
            rule_id="r1",
            effect="default",
            applied=False,
            reason="Fallback option",
        )
        
        json_str = result.to_json()
        restored = PolicyEffectResult.from_json(json_str)
        
        assert restored == result
    
    def test_invalid_applied_type_rejected(self):
        """Non-bool applied is rejected."""
        with pytest.raises(PolicyValidationError) as exc_info:
            PolicyEffectResult(
                policy_id="p1",
                rule_id="r1",
                effect="forbid",
                applied="yes",  # Should be bool
                reason="test",
            )
        
        assert "P009" in str(exc_info.value)


# =============================================================================
# SECTION 9: Serialization Tests
# =============================================================================

class TestPolicySerialization:
    """Test Policy serialization."""
    
    def test_to_dict_includes_policy_id(self, root_policy):
        """to_dict() includes policy_id."""
        data = root_policy.to_dict()
        assert "policy_id" in data
        assert data["policy_id"] == root_policy.policy_id
    
    def test_to_dict_includes_all_fields(self, root_policy):
        """to_dict() includes all fields."""
        data = root_policy.to_dict()
        
        assert "name" in data
        assert "version" in data
        assert "priority" in data
        assert "rules" in data
        assert "conflict_strategy" in data
    
    def test_to_dict_includes_parent_id_when_present(self, child_policy):
        """to_dict() includes parent_policy_id when parent exists."""
        data = child_policy.to_dict()
        assert "parent_policy_id" in data
    
    def test_to_dict_no_parent_id_when_absent(self, root_policy):
        """to_dict() omits parent_policy_id when no parent."""
        data = root_policy.to_dict()
        assert "parent_policy_id" not in data
    
    def test_to_json_with_indent(self, root_policy):
        """to_json() accepts indent parameter."""
        json_str = root_policy.to_json(indent=2)
        assert "\n" in json_str
        assert "  " in json_str
    
    def test_from_dict(self):
        """from_dict() reconstructs policy."""
        data = {
            "name": "test_policy",
            "version": "2.0.0",
            "priority": 50,
            "rules": [
                {"rule_id": "r1", "effect": "forbid", "target": "x"},
            ],
            "conflict_strategy": "explicit_priority",
        }
        
        policy = Policy.from_dict(data)
        
        assert policy.name == "test_policy"
        assert policy.version == "2.0.0"
        assert policy.priority == 50
        assert len(policy) == 1
        assert policy.conflict_strategy == ConflictResolutionStrategy.EXPLICIT_PRIORITY
    
    def test_from_json(self):
        """from_json() reconstructs policy."""
        json_str = '{"name": "json_policy", "rules": []}'
        policy = Policy.from_json(json_str)
        
        assert policy.name == "json_policy"


# =============================================================================
# SECTION 10: Iteration Tests
# =============================================================================

class TestIteration:
    """Test iteration and container protocols."""
    
    def test_len_returns_all_visible_rules(self, child_policy):
        """len() returns count of all visible rules."""
        # child has: root_rule_1 (override), root_rule_2 (inherited), child_rule_1 (local)
        assert len(child_policy) == 3
    
    def test_iter_yields_rules(self, root_policy):
        """Iterating yields PolicyRule objects."""
        rules = list(root_policy)
        assert len(rules) == 2
        assert all(isinstance(r, PolicyRule) for r in rules)
    
    def test_in_operator(self, root_policy):
        """'in' operator checks rule_id."""
        assert "root_rule_1" in root_policy
        assert "nonexistent" not in root_policy


# =============================================================================
# SECTION 11: String Representation Tests
# =============================================================================

class TestStringRepresentations:
    """Test __repr__ and __str__."""
    
    def test_repr_includes_key_info(self, root_policy):
        """__repr__ includes key information."""
        r = repr(root_policy)
        
        assert "Policy(" in r
        assert "root_policy" in r
        assert "1.0.0" in r
        assert "rules=" in r
    
    def test_repr_includes_parent_when_present(self, child_policy):
        """__repr__ includes parent when present."""
        r = repr(child_policy)
        assert "parent=" in r
        assert "root_policy" in r
    
    def test_str_is_human_readable(self, root_policy):
        """__str__ is human-readable."""
        s = str(root_policy)
        
        assert "Policy" in s
        assert "root_policy" in s
        assert "1.0.0" in s


# =============================================================================
# SECTION 12: Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_policy(self):
        """Empty policy works correctly."""
        policy = Policy.empty("empty")
        
        assert len(policy) == 0
        assert policy.rules() == []
        assert not policy.has_rule("anything")
    
    def test_unicode_names_and_targets(self):
        """Unicode names and targets work."""
        policy = Policy(
            name="日本語ポリシー",
            rules=[
                PolicyRule(rule_id="規則1", effect="forbid", target="機能X"),
            ],
        )
        
        assert policy.name == "日本語ポリシー"
        assert policy.has_rule("規則1")
    
    def test_very_deep_inheritance(self):
        """Deep inheritance chains work."""
        policy = Policy.empty("level_0")
        
        for i in range(1, 50):
            policy = Policy(
                name=f"level_{i}",
                rules=[PolicyRule(rule_id=f"r{i}", effect="prefer", target=f"t{i}")],
                parent=policy,
            )
        
        assert policy.depth == 49
        assert policy.has_rule("r1")  # From root
        assert policy.has_rule("r49")  # From leaf
    
    def test_many_rules(self):
        """Many rules work."""
        rules = [
            PolicyRule(rule_id=f"rule_{i:04d}", effect="prefer", target=f"target_{i}")
            for i in range(100)
        ]
        
        policy = Policy(name="many_rules", rules=rules)
        
        assert len(policy) == 100
        assert policy.has_rule("rule_0050")
    
    def test_rules_for_target_with_multiple_rules(self):
        """rules_for_target() returns all rules for a target."""
        policy = Policy(
            name="test",
            rules=[
                PolicyRule(rule_id="r1", effect="prefer", target="x", weight=0.5),
                PolicyRule(rule_id="r2", effect="prefer", target="x", weight=0.8),
                PolicyRule(rule_id="r3", effect="forbid", target="y"),
            ],
        )
        
        x_rules = policy.rules_for_target("x")
        
        assert len(x_rules) == 2
        assert all(r.target == "x" for r in x_rules)
    
    def test_with_strategy_changes_strategy(self, root_policy):
        """with_strategy() changes conflict resolution strategy."""
        new_policy = root_policy.with_strategy("explicit_priority")
        
        assert root_policy.conflict_strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE
        assert new_policy.conflict_strategy == ConflictResolutionStrategy.EXPLICIT_PRIORITY
    
    def test_conflict_strategy_enum_accepted(self):
        """Conflict strategy enum is accepted."""
        policy = Policy(
            name="test",
            rules=[],
            conflict_strategy=ConflictResolutionStrategy.REJECT_CONFLICT,
        )
        
        assert policy.conflict_strategy == ConflictResolutionStrategy.REJECT_CONFLICT


# =============================================================================
# SECTION 13: PolicyEffect Enum Tests
# =============================================================================

class TestPolicyEffectEnum:
    """Test PolicyEffect enum."""
    
    def test_from_string_valid(self):
        """from_string() works for valid effects."""
        assert PolicyEffect.from_string("require") == PolicyEffect.REQUIRE
        assert PolicyEffect.from_string("FORBID") == PolicyEffect.FORBID
        assert PolicyEffect.from_string("Prefer") == PolicyEffect.PREFER
        assert PolicyEffect.from_string("default") == PolicyEffect.DEFAULT
    
    def test_from_string_invalid(self):
        """from_string() raises for invalid effects."""
        with pytest.raises(InvalidPolicyEffectError):
            PolicyEffect.from_string("invalid")


# =============================================================================
# SECTION 14: ConflictResolutionStrategy Enum Tests
# =============================================================================

class TestConflictResolutionStrategyEnum:
    """Test ConflictResolutionStrategy enum."""
    
    def test_from_string_valid(self):
        """from_string() works for valid strategies."""
        assert ConflictResolutionStrategy.from_string("most_restrictive") == \
               ConflictResolutionStrategy.MOST_RESTRICTIVE
        assert ConflictResolutionStrategy.from_string("EXPLICIT_PRIORITY") == \
               ConflictResolutionStrategy.EXPLICIT_PRIORITY
        assert ConflictResolutionStrategy.from_string("Reject_Conflict") == \
               ConflictResolutionStrategy.REJECT_CONFLICT
    
    def test_from_string_invalid(self):
        """from_string() raises for invalid strategies."""
        with pytest.raises(PolicyValidationError):
            ConflictResolutionStrategy.from_string("invalid_strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
