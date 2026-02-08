"""
test_capability_descriptor.py

Comprehensive tests for the CapabilityDescriptor primitive.

Tests cover:
- Deterministic hashing
- Immutability
- Validation errors
- JSON serialization stability
- Input handling
- Effect handling
- Precondition handling
- Constraint handling
- Executable content rejection
"""

import json
import pytest
from ntive import (
    CapabilityDescriptor,
    CapabilityInput,
    DeclaredEffect,
    Precondition,
    Constraint,
    EffectCategory,
    CapabilityValidationError,
    InvalidCapabilityInputError,
    DuplicateInputKeyError,
    InvalidPreconditionError,
    InvalidEffectError,
    InvalidConstraintError,
    InvalidMetadataError,
    ExecutableContentError,
    CapabilityImmutabilityError,
)


# =============================================================================
# CapabilityInput Tests
# =============================================================================

class TestCapabilityInput:
    """Tests for CapabilityInput creation and validation."""
    
    def test_create_minimal_input(self):
        """Test creating an input with minimal fields."""
        inp = CapabilityInput(key="user_id", type="string")
        assert inp.key == "user_id"
        assert inp.type == "string"
        assert inp.required is True
        assert inp.constraints == {}
        assert inp.description is None
    
    def test_create_full_input(self):
        """Test creating an input with all fields."""
        inp = CapabilityInput(
            key="amount",
            type="decimal",
            required=False,
            constraints={"min": 0, "max": 1000},
            description="Transaction amount",
        )
        assert inp.key == "amount"
        assert inp.type == "decimal"
        assert inp.required is False
        assert inp.constraints == {"min": 0, "max": 1000}
        assert inp.description == "Transaction amount"
    
    def test_input_immutability(self):
        """Test that inputs are immutable."""
        inp = CapabilityInput(key="test", type="string")
        
        with pytest.raises(CapabilityImmutabilityError):
            inp.key = "modified"
        
        with pytest.raises(CapabilityImmutabilityError):
            del inp.key
    
    def test_input_constraints_copy(self):
        """Test that constraints are deep copied."""
        original = {"min": 0}
        inp = CapabilityInput(key="test", type="integer", constraints=original)
        
        # Modifying original doesn't affect input
        original["min"] = 100
        assert inp.constraints["min"] == 0
        
        # Modifying returned constraints doesn't affect input
        returned = inp.constraints
        returned["min"] = 200
        assert inp.constraints["min"] == 0
    
    def test_input_equality(self):
        """Test input equality comparison."""
        inp1 = CapabilityInput(key="test", type="string")
        inp2 = CapabilityInput(key="test", type="string")
        inp3 = CapabilityInput(key="test", type="integer")
        
        assert inp1 == inp2
        assert inp1 != inp3
        assert hash(inp1) == hash(inp2)
    
    def test_input_serialization(self):
        """Test JSON serialization/deserialization."""
        inp = CapabilityInput(
            key="email",
            type="email_address",
            required=True,
            constraints={"pattern": ".*@.*"},
            description="User email",
        )
        
        json_str = inp.to_json()
        restored = CapabilityInput.from_json(json_str)
        
        assert inp == restored
    
    def test_input_validation_empty_key(self):
        """Test that empty key is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityInput(key="", type="string")
        assert "C008" in str(exc_info.value)
    
    def test_input_validation_empty_type(self):
        """Test that empty type is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityInput(key="test", type="  ")
        assert "C008" in str(exc_info.value)
    
    def test_input_validation_non_string_type(self):
        """Test that non-string type is rejected."""
        with pytest.raises(InvalidCapabilityInputError) as exc_info:
            CapabilityInput(key="test", type=123)  # type: ignore
        assert "C001" in str(exc_info.value)
    
    def test_input_validation_non_bool_required(self):
        """Test that non-bool required is rejected."""
        with pytest.raises(InvalidCapabilityInputError) as exc_info:
            CapabilityInput(key="test", type="string", required=1)  # type: ignore
        assert "C001" in str(exc_info.value)
    
    def test_input_validation_non_dict_constraints(self):
        """Test that non-dict constraints is rejected."""
        with pytest.raises(InvalidCapabilityInputError) as exc_info:
            CapabilityInput(key="test", type="string", constraints=[1, 2, 3])  # type: ignore
        assert "C001" in str(exc_info.value)
    
    def test_input_str_representation(self):
        """Test string representation of inputs."""
        req_inp = CapabilityInput(key="required_field", type="string", required=True)
        opt_inp = CapabilityInput(key="optional_field", type="string", required=False)
        
        assert str(req_inp) == "required_field*: string"
        assert str(opt_inp) == "optional_field: string"
    
    def test_input_repr(self):
        """Test repr of inputs."""
        inp = CapabilityInput(key="test", type="string", required=True)
        assert "CapabilityInput" in repr(inp)
        assert "test" in repr(inp)
        assert "required" in repr(inp)


# =============================================================================
# DeclaredEffect Tests
# =============================================================================

class TestDeclaredEffect:
    """Tests for DeclaredEffect creation and validation."""
    
    def test_create_minimal_effect(self):
        """Test creating an effect with minimal fields."""
        eff = DeclaredEffect(category="state_change", target="user.balance")
        assert eff.category == "state_change"
        assert eff.target == "user.balance"
        assert eff.description is None
        assert eff.metadata == {}
    
    def test_create_full_effect(self):
        """Test creating an effect with all fields."""
        eff = DeclaredEffect(
            category="resource_consumption",
            target="api.rate_limit",
            description="Consumes API quota",
            metadata={"units": 1},
        )
        assert eff.category == "resource_consumption"
        assert eff.target == "api.rate_limit"
        assert eff.description == "Consumes API quota"
        assert eff.metadata == {"units": 1}
    
    def test_effect_with_enum_category(self):
        """Test creating effect with EffectCategory enum."""
        eff = DeclaredEffect(
            category=EffectCategory.IRREVERSIBLE_ACTION,
            target="data.deletion",
        )
        assert eff.category == "irreversible_action"
    
    def test_effect_custom_category(self):
        """Test that custom categories are accepted."""
        eff = DeclaredEffect(
            category="custom_effect_type",
            target="some.target",
        )
        assert eff.category == "custom_effect_type"
    
    def test_effect_immutability(self):
        """Test that effects are immutable."""
        eff = DeclaredEffect(category="state_change", target="db.record")
        
        with pytest.raises(CapabilityImmutabilityError):
            eff.category = "modified"
        
        with pytest.raises(CapabilityImmutabilityError):
            del eff.target
    
    def test_effect_metadata_copy(self):
        """Test that metadata is deep copied."""
        original = {"key": "value"}
        eff = DeclaredEffect(category="state_change", target="test", metadata=original)
        
        original["key"] = "modified"
        assert eff.metadata["key"] == "value"
    
    def test_effect_equality(self):
        """Test effect equality comparison."""
        eff1 = DeclaredEffect(category="state_change", target="test")
        eff2 = DeclaredEffect(category="state_change", target="test")
        eff3 = DeclaredEffect(category="state_change", target="other")
        
        assert eff1 == eff2
        assert eff1 != eff3
        assert hash(eff1) == hash(eff2)
    
    def test_effect_serialization(self):
        """Test JSON serialization/deserialization."""
        eff = DeclaredEffect(
            category="data_mutation",
            target="database.users",
            description="Modifies user record",
            metadata={"fields": ["name", "email"]},
        )
        
        json_str = eff.to_json()
        restored = DeclaredEffect.from_json(json_str)
        
        assert eff == restored
    
    def test_effect_validation_empty_target(self):
        """Test that empty target is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            DeclaredEffect(category="state_change", target="")
        assert "C008" in str(exc_info.value)
    
    def test_effect_str_representation(self):
        """Test string representation of effects."""
        eff = DeclaredEffect(category="state_change", target="user.status")
        assert str(eff) == "state_change: user.status"


# =============================================================================
# Precondition Tests
# =============================================================================

class TestPrecondition:
    """Tests for Precondition creation and validation."""
    
    def test_create_minimal_precondition(self):
        """Test creating a precondition with minimal fields."""
        pre = Precondition(condition="user.is_authenticated")
        assert pre.condition == "user.is_authenticated"
        assert pre.parameters == {}
        assert pre.description is None
    
    def test_create_full_precondition(self):
        """Test creating a precondition with all fields."""
        pre = Precondition(
            condition="user.has_permission",
            parameters={"permission": "admin"},
            description="User must be admin",
        )
        assert pre.condition == "user.has_permission"
        assert pre.parameters == {"permission": "admin"}
        assert pre.description == "User must be admin"
    
    def test_precondition_immutability(self):
        """Test that preconditions are immutable."""
        pre = Precondition(condition="test.condition")
        
        with pytest.raises(CapabilityImmutabilityError):
            pre.condition = "modified"
    
    def test_precondition_parameters_copy(self):
        """Test that parameters are deep copied."""
        original = {"key": "value"}
        pre = Precondition(condition="test", parameters=original)
        
        original["key"] = "modified"
        assert pre.parameters["key"] == "value"
    
    def test_precondition_equality(self):
        """Test precondition equality comparison."""
        pre1 = Precondition(condition="test")
        pre2 = Precondition(condition="test")
        pre3 = Precondition(condition="other")
        
        assert pre1 == pre2
        assert pre1 != pre3
        assert hash(pre1) == hash(pre2)
    
    def test_precondition_serialization(self):
        """Test JSON serialization/deserialization."""
        pre = Precondition(
            condition="balance.sufficient",
            parameters={"minimum": 100},
            description="Must have sufficient balance",
        )
        
        json_str = pre.to_json()
        restored = Precondition.from_json(json_str)
        
        assert pre == restored
    
    def test_precondition_str_representation(self):
        """Test string representation of preconditions."""
        pre = Precondition(condition="user.verified")
        assert str(pre) == "requires: user.verified"


# =============================================================================
# Constraint Tests
# =============================================================================

class TestConstraint:
    """Tests for Constraint creation and validation."""
    
    def test_create_minimal_constraint(self):
        """Test creating a constraint with minimal fields."""
        con = Constraint(name="rate_limit", rule="max_per_minute")
        assert con.name == "rate_limit"
        assert con.rule == "max_per_minute"
        assert con.parameters == {}
        assert con.description is None
    
    def test_create_full_constraint(self):
        """Test creating a constraint with all fields."""
        con = Constraint(
            name="transaction_limit",
            rule="max_amount",
            parameters={"limit": 10000},
            description="Maximum transaction amount",
        )
        assert con.name == "transaction_limit"
        assert con.rule == "max_amount"
        assert con.parameters == {"limit": 10000}
        assert con.description == "Maximum transaction amount"
    
    def test_constraint_immutability(self):
        """Test that constraints are immutable."""
        con = Constraint(name="test", rule="test_rule")
        
        with pytest.raises(CapabilityImmutabilityError):
            con.name = "modified"
    
    def test_constraint_equality(self):
        """Test constraint equality comparison."""
        con1 = Constraint(name="test", rule="rule1")
        con2 = Constraint(name="test", rule="rule1")
        con3 = Constraint(name="test", rule="rule2")
        
        assert con1 == con2
        assert con1 != con3
        assert hash(con1) == hash(con2)
    
    def test_constraint_serialization(self):
        """Test JSON serialization/deserialization."""
        con = Constraint(
            name="timeout",
            rule="max_execution_time",
            parameters={"seconds": 30},
            description="Request timeout",
        )
        
        json_str = con.to_json()
        restored = Constraint.from_json(json_str)
        
        assert con == restored


# =============================================================================
# CapabilityDescriptor Creation Tests
# =============================================================================

class TestCapabilityDescriptorCreation:
    """Tests for CapabilityDescriptor creation."""
    
    def test_create_minimal_capability(self):
        """Test creating a capability with minimal fields."""
        cap = CapabilityDescriptor(
            name="transfer_funds",
            version="1.0.0",
            domain="banking",
        )
        assert cap.name == "transfer_funds"
        assert cap.version == "1.0.0"
        assert cap.domain == "banking"
        assert cap.description is None
        assert len(cap.inputs) == 0
        assert len(cap.preconditions) == 0
        assert len(cap.effects) == 0
        assert len(cap.constraints) == 0
        assert cap.metadata == {}
        assert cap.capability_id  # Should have computed ID
    
    def test_create_full_capability(self):
        """Test creating a capability with all fields."""
        cap = CapabilityDescriptor(
            name="send_email",
            version="2.1.0",
            domain="communication",
            description="Send an email to a recipient",
            inputs=[
                CapabilityInput(key="to", type="email_address"),
                CapabilityInput(key="subject", type="string"),
                CapabilityInput(key="body", type="text"),
                CapabilityInput(key="cc", type="email_list", required=False),
            ],
            preconditions=[
                Precondition(condition="smtp.configured"),
                Precondition(condition="quota.available"),
            ],
            effects=[
                DeclaredEffect(category="communication", target="email.outbox"),
                DeclaredEffect(category="resource_consumption", target="email.quota"),
            ],
            constraints=[
                Constraint(name="max_attachments", rule="count_limit", parameters={"max": 10}),
            ],
            metadata={"category": "external"},
        )
        
        assert cap.name == "send_email"
        assert len(cap.inputs) == 4
        assert len(cap.preconditions) == 2
        assert len(cap.effects) == 2
        assert len(cap.constraints) == 1
        assert cap.metadata == {"category": "external"}
    
    def test_create_capability_with_dicts(self):
        """Test creating a capability with dict representations."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                {"key": "input1", "type": "string"},
                {"key": "input2", "type": "integer", "required": False},
            ],
            effects=[
                {"category": "state_change", "target": "data.record"},
            ],
            preconditions=[
                {"condition": "user.logged_in"},
            ],
            constraints=[
                {"name": "limit", "rule": "max_count"},
            ],
        )
        
        assert len(cap.inputs) == 2
        assert cap.inputs[0].key == "input1"
        assert cap.inputs[1].required is False
    
    def test_capability_immutability(self):
        """Test that capabilities are immutable."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
        )
        
        with pytest.raises(CapabilityImmutabilityError):
            cap.name = "modified"
        
        with pytest.raises(CapabilityImmutabilityError):
            del cap.version


# =============================================================================
# Deterministic Hashing Tests
# =============================================================================

class TestDeterministicHashing:
    """Tests for deterministic capability_id computation."""
    
    def test_identical_inputs_same_id(self):
        """Test that identical inputs produce identical IDs."""
        cap1 = CapabilityDescriptor(
            name="do_something",
            version="1.0.0",
            domain="actions",
            inputs=[CapabilityInput(key="x", type="integer")],
        )
        
        cap2 = CapabilityDescriptor(
            name="do_something",
            version="1.0.0",
            domain="actions",
            inputs=[CapabilityInput(key="x", type="integer")],
        )
        
        assert cap1.capability_id == cap2.capability_id
    
    def test_different_name_different_id(self):
        """Test that different names produce different IDs."""
        cap1 = CapabilityDescriptor(name="action_a", version="1.0.0", domain="test")
        cap2 = CapabilityDescriptor(name="action_b", version="1.0.0", domain="test")
        
        assert cap1.capability_id != cap2.capability_id
    
    def test_different_version_different_id(self):
        """Test that different versions produce different IDs."""
        cap1 = CapabilityDescriptor(name="action", version="1.0.0", domain="test")
        cap2 = CapabilityDescriptor(name="action", version="2.0.0", domain="test")
        
        assert cap1.capability_id != cap2.capability_id
    
    def test_different_domain_different_id(self):
        """Test that different domains produce different IDs."""
        cap1 = CapabilityDescriptor(name="action", version="1.0.0", domain="domain_a")
        cap2 = CapabilityDescriptor(name="action", version="1.0.0", domain="domain_b")
        
        assert cap1.capability_id != cap2.capability_id
    
    def test_different_inputs_different_id(self):
        """Test that different inputs produce different IDs."""
        cap1 = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="string")],
        )
        cap2 = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="integer")],
        )
        
        assert cap1.capability_id != cap2.capability_id
    
    def test_different_effects_different_id(self):
        """Test that different effects produce different IDs."""
        cap1 = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
            effects=[DeclaredEffect(category="state_change", target="a")],
        )
        cap2 = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
            effects=[DeclaredEffect(category="state_change", target="b")],
        )
        
        assert cap1.capability_id != cap2.capability_id
    
    def test_capability_id_is_sha256(self):
        """Test that capability_id is a valid SHA-256 hex digest."""
        cap = CapabilityDescriptor(name="test", version="1.0.0", domain="test")
        
        assert len(cap.capability_id) == 64  # SHA-256 = 64 hex chars
        assert all(c in "0123456789abcdef" for c in cap.capability_id)
    
    def test_serialization_determinism(self):
        """Test that serialization is deterministic."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="b", type="string"),
                CapabilityInput(key="a", type="integer"),
            ],
            metadata={"z": 1, "a": 2},
        )
        
        json1 = cap.to_json()
        json2 = cap.to_json()
        
        assert json1 == json2
    
    def test_from_json_produces_same_id(self):
        """Test that deserialization produces the same ID."""
        original = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="string")],
        )
        
        json_str = original.to_json()
        restored = CapabilityDescriptor.from_json(json_str)
        
        assert original.capability_id == restored.capability_id


# =============================================================================
# Input Access Tests
# =============================================================================

class TestInputAccess:
    """Tests for input access methods."""
    
    def test_get_input(self):
        """Test getting input by key."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="user_id", type="string"),
                CapabilityInput(key="amount", type="decimal"),
            ],
        )
        
        user_id = cap.get_input("user_id")
        assert user_id is not None
        assert user_id.key == "user_id"
        
        assert cap.get_input("nonexistent") is None
    
    def test_has_input(self):
        """Test checking if input exists."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="test_input", type="string")],
        )
        
        assert cap.has_input("test_input") is True
        assert cap.has_input("missing") is False
    
    def test_required_inputs(self):
        """Test getting required inputs."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="required1", type="string", required=True),
                CapabilityInput(key="optional1", type="string", required=False),
                CapabilityInput(key="required2", type="string", required=True),
            ],
        )
        
        required = cap.required_inputs()
        assert len(required) == 2
        assert all(i.required for i in required)
    
    def test_optional_inputs(self):
        """Test getting optional inputs."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="required1", type="string", required=True),
                CapabilityInput(key="optional1", type="string", required=False),
            ],
        )
        
        optional = cap.optional_inputs()
        assert len(optional) == 1
        assert optional[0].key == "optional1"
    
    def test_input_keys(self):
        """Test getting all input keys."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="a", type="string"),
                CapabilityInput(key="b", type="string"),
                CapabilityInput(key="c", type="string"),
            ],
        )
        
        keys = cap.input_keys()
        assert keys == ["a", "b", "c"]
    
    def test_contains_operator(self):
        """Test 'in' operator for input keys."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="exists", type="string")],
        )
        
        assert "exists" in cap
        assert "missing" not in cap
    
    def test_iteration(self):
        """Test iterating over inputs."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="a", type="string"),
                CapabilityInput(key="b", type="string"),
            ],
        )
        
        keys = [inp.key for inp in cap]
        assert keys == ["a", "b"]
    
    def test_len(self):
        """Test length (number of inputs)."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[
                CapabilityInput(key="a", type="string"),
                CapabilityInput(key="b", type="string"),
                CapabilityInput(key="c", type="string"),
            ],
        )
        
        assert len(cap) == 3


# =============================================================================
# Effect Query Tests
# =============================================================================

class TestEffectQueries:
    """Tests for effect query methods."""
    
    def test_effects_by_category(self):
        """Test filtering effects by category."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            effects=[
                DeclaredEffect(category="state_change", target="a"),
                DeclaredEffect(category="resource_consumption", target="b"),
                DeclaredEffect(category="state_change", target="c"),
            ],
        )
        
        state_changes = cap.effects_by_category("state_change")
        assert len(state_changes) == 2
    
    def test_has_effect_category(self):
        """Test checking if effect category exists."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            effects=[
                DeclaredEffect(category="state_change", target="a"),
            ],
        )
        
        assert cap.has_effect_category("state_change") is True
        assert cap.has_effect_category("irreversible_action") is False
    
    def test_has_irreversible_effects(self):
        """Test checking for irreversible effects."""
        cap_without = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            effects=[DeclaredEffect(category="state_change", target="a")],
        )
        
        cap_with = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            effects=[DeclaredEffect(category="irreversible_action", target="deletion")],
        )
        
        assert cap_without.has_irreversible_effects() is False
        assert cap_with.has_irreversible_effects() is True


# =============================================================================
# Qualified Name Tests
# =============================================================================

class TestQualifiedName:
    """Tests for qualified name."""
    
    def test_qualified_name_format(self):
        """Test qualified name format."""
        cap = CapabilityDescriptor(
            name="transfer_funds",
            version="1.2.3",
            domain="banking",
        )
        
        assert cap.qualified_name == "banking/transfer_funds@1.2.3"
    
    def test_str_is_qualified_name(self):
        """Test that str() returns qualified name."""
        cap = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
        )
        
        assert str(cap) == cap.qualified_name


# =============================================================================
# Validation Error Tests
# =============================================================================

class TestValidationErrors:
    """Tests for validation error handling."""
    
    def test_empty_name_rejected(self):
        """Test that empty name is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityDescriptor(name="", version="1.0.0", domain="test")
        assert "C008" in str(exc_info.value)
    
    def test_empty_version_rejected(self):
        """Test that empty version is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityDescriptor(name="test", version="  ", domain="test")
        assert "C008" in str(exc_info.value)
    
    def test_empty_domain_rejected(self):
        """Test that empty domain is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityDescriptor(name="test", version="1.0.0", domain="")
        assert "C008" in str(exc_info.value)
    
    def test_duplicate_input_key_rejected(self):
        """Test that duplicate input keys are rejected."""
        with pytest.raises(DuplicateInputKeyError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                inputs=[
                    CapabilityInput(key="same_key", type="string"),
                    CapabilityInput(key="same_key", type="integer"),
                ],
            )
        assert "C002" in str(exc_info.value)
        assert "same_key" in str(exc_info.value)
    
    def test_invalid_input_type_rejected(self):
        """Test that invalid input types are rejected."""
        with pytest.raises(InvalidCapabilityInputError):
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                inputs=[123],  # type: ignore
            )
    
    def test_non_serializable_metadata_rejected(self):
        """Test that non-serializable metadata is rejected."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata={"func": lambda x: x},  # type: ignore
            )
        # Could be C007 (executable) or C009 (not serializable)
        assert "C00" in str(exc_info.value)
    
    def test_non_dict_metadata_rejected(self):
        """Test that non-dict metadata is rejected."""
        with pytest.raises(InvalidMetadataError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata="not a dict",  # type: ignore
            )
        assert "C006" in str(exc_info.value)


# =============================================================================
# Executable Content Rejection Tests
# =============================================================================

class TestExecutableContentRejection:
    """Tests for rejection of executable content."""
    
    def test_lambda_in_metadata_rejected(self):
        """Test that lambda in metadata is rejected."""
        with pytest.raises(ExecutableContentError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata={"callback": lambda: None},  # type: ignore
            )
        assert "C007" in str(exc_info.value)
    
    def test_function_in_metadata_rejected(self):
        """Test that function reference in metadata is rejected."""
        def some_function():
            pass
        
        with pytest.raises(ExecutableContentError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata={"func": some_function},  # type: ignore
            )
        assert "C007" in str(exc_info.value)
    
    def test_class_in_metadata_rejected(self):
        """Test that class reference in metadata is rejected."""
        with pytest.raises(ExecutableContentError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata={"cls": int},  # type: ignore
            )
        assert "C007" in str(exc_info.value)
    
    def test_nested_lambda_rejected(self):
        """Test that nested lambda is rejected."""
        with pytest.raises((ExecutableContentError, CapabilityValidationError)):
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata={"nested": {"deep": {"callback": lambda: 1}}},  # type: ignore
            )
    
    def test_lambda_in_input_constraints_rejected(self):
        """Test that lambda in input constraints is rejected."""
        with pytest.raises((ExecutableContentError, CapabilityValidationError, InvalidCapabilityInputError)):
            CapabilityInput(
                key="test",
                type="string",
                constraints={"validator": lambda x: True},  # type: ignore
            )


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for JSON serialization/deserialization."""
    
    def test_full_roundtrip(self):
        """Test complete serialization roundtrip."""
        original = CapabilityDescriptor(
            name="complex_action",
            version="2.0.0",
            domain="operations",
            description="A complex action with all features",
            inputs=[
                CapabilityInput(
                    key="input1",
                    type="string",
                    required=True,
                    constraints={"max_length": 100},
                    description="First input",
                ),
                CapabilityInput(
                    key="input2",
                    type="integer",
                    required=False,
                ),
            ],
            preconditions=[
                Precondition(
                    condition="auth.valid",
                    parameters={"level": 2},
                    description="Must be authenticated",
                ),
            ],
            effects=[
                DeclaredEffect(
                    category="state_change",
                    target="data.record",
                    description="Modifies record",
                    metadata={"reversible": True},
                ),
            ],
            constraints=[
                Constraint(
                    name="rate_limit",
                    rule="requests_per_minute",
                    parameters={"limit": 60},
                    description="API rate limit",
                ),
            ],
            metadata={"category": "api", "tags": ["v2", "stable"]},
        )
        
        json_str = original.to_json()
        restored = CapabilityDescriptor.from_json(json_str)
        
        assert original == restored
        assert original.capability_id == restored.capability_id
    
    def test_to_dict(self):
        """Test conversion to dict."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="string")],
        )
        
        d = cap.to_dict()
        
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert d["domain"] == "test"
        assert len(d["inputs"]) == 1
        assert d["inputs"][0]["key"] == "x"
        assert d["capability_id"] == cap.capability_id
    
    def test_from_dict(self):
        """Test construction from dict."""
        d = {
            "name": "test",
            "version": "1.0.0",
            "domain": "test",
            "inputs": [{"key": "y", "type": "integer"}],
        }
        
        cap = CapabilityDescriptor.from_dict(d)
        
        assert cap.name == "test"
        assert cap.inputs[0].key == "y"
    
    def test_json_sorted_keys(self):
        """Test that JSON output has sorted keys."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            metadata={"z": 1, "a": 2},
        )
        
        json_str = cap.to_json()
        
        # Parse and check key order in outer object
        parsed = json.loads(json_str)
        keys = list(parsed.keys())
        assert keys == sorted(keys)


# =============================================================================
# Equality Tests
# =============================================================================

class TestEquality:
    """Tests for equality comparison."""
    
    def test_equal_capabilities(self):
        """Test that identical capabilities are equal."""
        cap1 = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="string")],
        )
        cap2 = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="string")],
        )
        
        assert cap1 == cap2
    
    def test_unequal_capabilities(self):
        """Test that different capabilities are not equal."""
        cap1 = CapabilityDescriptor(name="a", version="1.0.0", domain="test")
        cap2 = CapabilityDescriptor(name="b", version="1.0.0", domain="test")
        
        assert cap1 != cap2
    
    def test_hash_equality(self):
        """Test that equal capabilities have equal hashes."""
        cap1 = CapabilityDescriptor(name="test", version="1.0.0", domain="test")
        cap2 = CapabilityDescriptor(name="test", version="1.0.0", domain="test")
        
        assert hash(cap1) == hash(cap2)
    
    def test_can_use_in_set(self):
        """Test that capabilities can be used in sets."""
        cap1 = CapabilityDescriptor(name="a", version="1.0.0", domain="test")
        cap2 = CapabilityDescriptor(name="a", version="1.0.0", domain="test")
        cap3 = CapabilityDescriptor(name="b", version="1.0.0", domain="test")
        
        s = {cap1, cap2, cap3}
        assert len(s) == 2
    
    def test_can_use_as_dict_key(self):
        """Test that capabilities can be used as dict keys."""
        cap = CapabilityDescriptor(name="test", version="1.0.0", domain="test")
        
        d = {cap: "value"}
        assert d[cap] == "value"
    
    def test_not_equal_to_other_types(self):
        """Test that capabilities are not equal to other types."""
        cap = CapabilityDescriptor(name="test", version="1.0.0", domain="test")
        
        assert cap != "test/test@1.0.0"
        assert cap != 123
        assert cap != None


# =============================================================================
# String Representation Tests
# =============================================================================

class TestStringRepresentations:
    """Tests for __str__ and __repr__."""
    
    def test_str(self):
        """Test __str__ returns qualified name."""
        cap = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
        )
        
        assert str(cap) == "test/action@1.0.0"
    
    def test_repr(self):
        """Test __repr__ is informative."""
        cap = CapabilityDescriptor(
            name="action",
            version="1.0.0",
            domain="test",
            inputs=[CapabilityInput(key="x", type="string")],
            effects=[DeclaredEffect(category="state_change", target="y")],
        )
        
        r = repr(cap)
        assert "CapabilityDescriptor" in r
        assert "action" in r
        assert "test" in r
        assert "inputs=1" in r
        assert "effects=1" in r


# =============================================================================
# Metadata Copy Tests
# =============================================================================

class TestMetadataCopy:
    """Tests for metadata deep copy behavior."""
    
    def test_metadata_is_copied_on_construction(self):
        """Test that metadata is deep copied on construction."""
        original = {"key": "value", "nested": {"inner": 1}}
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            metadata=original,
        )
        
        original["key"] = "modified"
        original["nested"]["inner"] = 999
        
        assert cap.metadata["key"] == "value"
        assert cap.metadata["nested"]["inner"] == 1
    
    def test_metadata_is_copied_on_access(self):
        """Test that metadata is deep copied on access."""
        cap = CapabilityDescriptor(
            name="test",
            version="1.0.0",
            domain="test",
            metadata={"key": "value"},
        )
        
        returned = cap.metadata
        returned["key"] = "modified"
        
        assert cap.metadata["key"] == "value"


# =============================================================================
# EffectCategory Enum Tests
# =============================================================================

class TestEffectCategoryEnum:
    """Tests for EffectCategory enum."""
    
    def test_all_categories_have_values(self):
        """Test that all categories have string values."""
        assert EffectCategory.STATE_CHANGE.value == "state_change"
        assert EffectCategory.RESOURCE_CONSUMPTION.value == "resource_consumption"
        assert EffectCategory.EXTERNAL_DEPENDENCY.value == "external_dependency"
        assert EffectCategory.IRREVERSIBLE_ACTION.value == "irreversible_action"
        assert EffectCategory.DATA_MUTATION.value == "data_mutation"
        assert EffectCategory.COMMUNICATION.value == "communication"
        assert EffectCategory.OTHER.value == "other"
    
    def test_from_string_known_value(self):
        """Test from_string with known values."""
        assert EffectCategory.from_string("state_change") == EffectCategory.STATE_CHANGE
        assert EffectCategory.from_string("STATE_CHANGE") == EffectCategory.STATE_CHANGE
    
    def test_from_string_unknown_value(self):
        """Test from_string with unknown values returns OTHER."""
        assert EffectCategory.from_string("unknown_category") == EffectCategory.OTHER


# =============================================================================
# Capability Without Inputs (Edge Case)
# =============================================================================

class TestCapabilityWithoutInputs:
    """Tests for capabilities with no inputs."""
    
    def test_empty_inputs(self):
        """Test capability with no inputs."""
        cap = CapabilityDescriptor(
            name="no_input_action",
            version="1.0.0",
            domain="test",
        )
        
        assert len(cap) == 0
        assert list(cap) == []
        assert cap.input_keys() == []
        assert cap.required_inputs() == []
        assert cap.optional_inputs() == []
    
    def test_empty_capability_is_falsy_handled(self):
        """Test that empty capability doesn't cause issues with bool checks."""
        cap = CapabilityDescriptor(
            name="empty",
            version="1.0.0",
            domain="test",
        )
        
        # len() is 0 but capability should still be a valid object
        assert cap.name == "empty"
        assert cap.capability_id


# =============================================================================
# Error Code Tests
# =============================================================================

class TestErrorCodes:
    """Tests for error code presence in exceptions."""
    
    def test_c001_invalid_capability_input(self):
        """Test C001 error code for invalid input."""
        with pytest.raises(InvalidCapabilityInputError) as exc_info:
            CapabilityInput(key="test", type=123)  # type: ignore
        assert exc_info.value.error_code == "C001"
    
    def test_c002_duplicate_input_key(self):
        """Test C002 error code for duplicate key."""
        with pytest.raises(DuplicateInputKeyError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                inputs=[
                    CapabilityInput(key="dup", type="string"),
                    CapabilityInput(key="dup", type="integer"),
                ],
            )
        assert exc_info.value.error_code == "C002"
    
    def test_c006_invalid_metadata(self):
        """Test C006 error code for invalid metadata."""
        with pytest.raises(InvalidMetadataError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata="not a dict",  # type: ignore
            )
        assert exc_info.value.error_code == "C006"
    
    def test_c007_executable_content(self):
        """Test C007 error code for executable content."""
        with pytest.raises(ExecutableContentError) as exc_info:
            CapabilityDescriptor(
                name="test",
                version="1.0.0",
                domain="test",
                metadata={"fn": lambda: None},  # type: ignore
            )
        assert exc_info.value.error_code == "C007"
    
    def test_c008_string_validation(self):
        """Test C008 error code for string validation."""
        with pytest.raises(CapabilityValidationError) as exc_info:
            CapabilityDescriptor(name="", version="1.0.0", domain="test")
        assert exc_info.value.error_code == "C008"
