"""
test_decision.py

Unit tests for the Ntive Decision primitive.

Tests prove:
- Missing required fields are rejected
- Identical inputs produce identical serialized output (determinism)
- Decision cannot be mutated after creation (immutability)
- All validation rules are enforced
"""

import pytest
import json
from ntive.decision import (
    Decision,
    Alternative,
    Confidence,
    DecisionValidationError,
    MissingRequiredFieldError,
    InvalidFieldTypeError,
    InvalidInputsError,
    InvalidConfidenceError,
    ImmutabilityViolationError,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def valid_inputs():
    """Valid inputs dictionary."""
    return {"user_query": "open file", "context": "editor"}


@pytest.fixture
def valid_alternatives():
    """Valid list of alternatives."""
    return [
        Alternative("recent_files", "User didn't mention recent files"),
        Alternative("new_file", "User said 'open', not 'new'"),
    ]


@pytest.fixture
def valid_decision(valid_inputs, valid_alternatives):
    """A valid Decision for testing."""
    return Decision(
        inputs=valid_inputs,
        selected_option="open_file_dialog",
        alternatives=valid_alternatives,
        rationale="User explicitly requested to open a file",
        confidence=Confidence(0.9, lower_bound=0.85, upper_bound=0.95),
        constraints=["no_external_drives"],
    )


# =============================================================================
# SECTION 1: Missing Required Fields Are Rejected
# =============================================================================

class TestMissingRequiredFields:
    """Test that missing required fields raise MissingRequiredFieldError."""
    
    def test_missing_inputs_raises_error(self, valid_alternatives):
        """inputs is required."""
        with pytest.raises(MissingRequiredFieldError) as exc_info:
            Decision(
                inputs=None,
                selected_option="option",
                alternatives=valid_alternatives,
                rationale="Because reasons",
            )
        assert exc_info.value.field_name == "inputs"
        assert "D001" in str(exc_info.value)
    
    def test_missing_selected_option_raises_error(self, valid_inputs, valid_alternatives):
        """selected_option is required."""
        with pytest.raises(MissingRequiredFieldError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option=None,
                alternatives=valid_alternatives,
                rationale="Because reasons",
            )
        assert exc_info.value.field_name == "selected_option"
    
    def test_missing_alternatives_raises_error(self, valid_inputs):
        """alternatives is required (can be empty list, but not None)."""
        with pytest.raises(MissingRequiredFieldError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="option",
                alternatives=None,
                rationale="Because reasons",
            )
        assert exc_info.value.field_name == "alternatives"
    
    def test_missing_rationale_raises_error(self, valid_inputs, valid_alternatives):
        """rationale is required."""
        with pytest.raises(MissingRequiredFieldError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="option",
                alternatives=valid_alternatives,
                rationale=None,
            )
        assert exc_info.value.field_name == "rationale"
    
    def test_empty_alternatives_list_is_valid(self, valid_inputs):
        """Empty alternatives list is valid (no alternatives considered)."""
        decision = Decision(
            inputs=valid_inputs,
            selected_option="only_option",
            alternatives=[],
            rationale="No alternatives available",
        )
        assert decision.alternatives == ()
    
    def test_empty_selected_option_raises_error(self, valid_inputs, valid_alternatives):
        """Empty string is not a valid selected_option."""
        with pytest.raises(DecisionValidationError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="   ",  # Whitespace-only
                alternatives=valid_alternatives,
                rationale="Reason",
            )
        assert "D009" in str(exc_info.value)
    
    def test_empty_rationale_raises_error(self, valid_inputs, valid_alternatives):
        """Empty string is not a valid rationale."""
        with pytest.raises(DecisionValidationError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="option",
                alternatives=valid_alternatives,
                rationale="",
            )
        assert "D010" in str(exc_info.value)


# =============================================================================
# SECTION 2: Identical Inputs Produce Identical Serialized Output
# =============================================================================

class TestDeterministicSerialization:
    """Test that serialization is deterministic."""
    
    def test_identical_decisions_produce_identical_json(self):
        """Two decisions with identical inputs produce identical JSON."""
        inputs = {"a": 1, "b": 2, "c": 3}
        alternatives = [Alternative("alt1", "reason1")]
        
        # Create with explicit ID and timestamp to ensure reproducibility
        d1 = Decision(
            inputs=inputs,
            selected_option="option",
            alternatives=alternatives,
            rationale="reason",
            decision_id="test-id-123",
            timestamp="2026-02-08T12:00:00Z",
        )
        
        d2 = Decision(
            inputs=inputs,
            selected_option="option",
            alternatives=alternatives,
            rationale="reason",
            decision_id="test-id-123",
            timestamp="2026-02-08T12:00:00Z",
        )
        
        assert d1.to_json() == d2.to_json()
    
    def test_input_order_does_not_affect_serialization(self):
        """Input key order doesn't affect JSON output (sorted keys)."""
        alternatives = [Alternative("alt", "reason")]
        
        d1 = Decision(
            inputs={"z": 1, "a": 2, "m": 3},
            selected_option="opt",
            alternatives=alternatives,
            rationale="reason",
            decision_id="id",
            timestamp="2026-02-08T12:00:00Z",
        )
        
        d2 = Decision(
            inputs={"a": 2, "m": 3, "z": 1},
            selected_option="opt",
            alternatives=alternatives,
            rationale="reason",
            decision_id="id",
            timestamp="2026-02-08T12:00:00Z",
        )
        
        assert d1.to_json() == d2.to_json()
    
    def test_json_keys_are_sorted(self, valid_decision):
        """JSON output has sorted keys."""
        json_str = valid_decision.to_json()
        data = json.loads(json_str)
        keys = list(data.keys())
        assert keys == sorted(keys)
    
    def test_nested_dict_keys_are_sorted(self):
        """Nested dictionary keys are also sorted."""
        decision = Decision(
            inputs={"z": {"c": 1, "a": 2}, "m": {"b": 3}},
            selected_option="opt",
            alternatives=[],
            rationale="reason",
            decision_id="id",
            timestamp="2026-02-08T12:00:00Z",
        )
        
        json_str = decision.to_json()
        # Verify the JSON string has keys in sorted order
        assert '"a": 2' in json_str
        assert json_str.index('"a":') < json_str.index('"c":')
    
    def test_multiple_serializations_are_identical(self, valid_decision):
        """Calling to_json() multiple times returns identical results."""
        json1 = valid_decision.to_json()
        json2 = valid_decision.to_json()
        json3 = valid_decision.to_json()
        
        assert json1 == json2 == json3
    
    def test_roundtrip_preserves_data(self, valid_inputs, valid_alternatives):
        """from_json(to_json()) preserves all data."""
        original = Decision(
            inputs=valid_inputs,
            selected_option="option",
            alternatives=valid_alternatives,
            rationale="reason",
            confidence=Confidence(0.8),
            constraints=["c1", "c2"],
            decision_id="fixed-id",
            timestamp="2026-02-08T12:00:00Z",
        )
        
        json_str = original.to_json()
        restored = Decision.from_json(json_str)
        
        assert original == restored
        assert original.to_json() == restored.to_json()


# =============================================================================
# SECTION 3: Decision Cannot Be Mutated After Creation
# =============================================================================

class TestImmutability:
    """Test that Decision is truly immutable after creation."""
    
    def test_cannot_set_attribute(self, valid_decision):
        """Setting any attribute raises ImmutabilityViolationError."""
        with pytest.raises(ImmutabilityViolationError):
            valid_decision.selected_option = "other"
    
    def test_cannot_set_private_attribute(self, valid_decision):
        """Cannot set private attributes either."""
        with pytest.raises(ImmutabilityViolationError):
            valid_decision._selected_option = "other"
    
    def test_cannot_delete_attribute(self, valid_decision):
        """Cannot delete attributes."""
        with pytest.raises(ImmutabilityViolationError):
            del valid_decision.rationale
    
    def test_cannot_add_new_attribute(self, valid_decision):
        """Cannot add new attributes."""
        with pytest.raises(ImmutabilityViolationError):
            valid_decision.new_field = "value"
    
    def test_inputs_property_returns_copy(self, valid_decision):
        """Modifying the inputs dict doesn't affect the Decision."""
        inputs = valid_decision.inputs
        inputs["new_key"] = "new_value"
        
        # Original should be unchanged
        assert "new_key" not in valid_decision.inputs
    
    def test_alternatives_is_tuple(self, valid_decision):
        """Alternatives is returned as immutable tuple."""
        assert isinstance(valid_decision.alternatives, tuple)
    
    def test_constraints_is_tuple(self, valid_decision):
        """Constraints is returned as immutable tuple."""
        assert isinstance(valid_decision.constraints, tuple)
    
    def test_original_inputs_dict_mutation_doesnt_affect_decision(self, valid_alternatives):
        """Mutating the original inputs dict doesn't affect the Decision."""
        inputs = {"key": "value"}
        decision = Decision(
            inputs=inputs,
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        
        # Mutate original
        inputs["key"] = "changed"
        inputs["new"] = "added"
        
        # Decision should still have original value
        assert decision.inputs["key"] == "value"
        assert "new" not in decision.inputs
    
    def test_original_alternatives_list_mutation_doesnt_affect_decision(self, valid_inputs):
        """Mutating the original alternatives list doesn't affect the Decision."""
        alternatives = [Alternative("opt1", "reason1")]
        decision = Decision(
            inputs=valid_inputs,
            selected_option="opt",
            alternatives=alternatives,
            rationale="reason",
        )
        
        # Mutate original
        alternatives.append(Alternative("opt2", "reason2"))
        
        # Decision should still have original count
        assert len(decision.alternatives) == 1


# =============================================================================
# SECTION 4: Validation Rules
# =============================================================================

class TestTypeValidation:
    """Test that invalid types are rejected."""
    
    def test_inputs_must_be_dict(self, valid_alternatives):
        """inputs must be a dictionary."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=["not", "a", "dict"],
                selected_option="opt",
                alternatives=valid_alternatives,
                rationale="reason",
            )
        assert exc_info.value.field_name == "inputs"
        assert exc_info.value.expected == "dict"
        assert exc_info.value.actual == "list"
    
    def test_selected_option_must_be_string(self, valid_inputs, valid_alternatives):
        """selected_option must be a string."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option=123,
                alternatives=valid_alternatives,
                rationale="reason",
            )
        assert exc_info.value.field_name == "selected_option"
    
    def test_alternatives_must_be_list(self, valid_inputs):
        """alternatives must be a list."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="opt",
                alternatives="not a list",
                rationale="reason",
            )
        assert exc_info.value.field_name == "alternatives"
    
    def test_alternatives_items_must_be_alternative(self, valid_inputs):
        """alternatives items must be Alternative instances."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="opt",
                alternatives=[{"option": "x", "reason": "y"}],  # Dict, not Alternative
                rationale="reason",
            )
        assert "alternatives[0]" in exc_info.value.field_name
    
    def test_rationale_must_be_string(self, valid_inputs, valid_alternatives):
        """rationale must be a string."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="opt",
                alternatives=valid_alternatives,
                rationale=42,
            )
        assert exc_info.value.field_name == "rationale"
    
    def test_constraints_must_be_list(self, valid_inputs, valid_alternatives):
        """constraints must be a list if provided."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="opt",
                alternatives=valid_alternatives,
                rationale="reason",
                constraints="not a list",
            )
        assert exc_info.value.field_name == "constraints"
    
    def test_confidence_must_be_confidence_object(self, valid_inputs, valid_alternatives):
        """confidence must be a Confidence instance."""
        with pytest.raises(InvalidFieldTypeError) as exc_info:
            Decision(
                inputs=valid_inputs,
                selected_option="opt",
                alternatives=valid_alternatives,
                rationale="reason",
                confidence=0.9,  # Float, not Confidence
            )
        assert exc_info.value.field_name == "confidence"


class TestInputsValidation:
    """Test that inputs are validated for JSON serialization."""
    
    def test_inputs_rejects_non_string_keys(self, valid_alternatives):
        """Input dictionary keys must be strings."""
        with pytest.raises(InvalidInputsError):
            Decision(
                inputs={1: "value"},  # Integer key
                selected_option="opt",
                alternatives=valid_alternatives,
                rationale="reason",
            )
    
    def test_inputs_rejects_non_serializable_values(self, valid_alternatives):
        """Input values must be JSON-serializable."""
        with pytest.raises(InvalidInputsError):
            Decision(
                inputs={"func": lambda x: x},  # Function is not serializable
                selected_option="opt",
                alternatives=valid_alternatives,
                rationale="reason",
            )
    
    def test_inputs_accepts_nested_dicts(self, valid_alternatives):
        """Nested dictionaries are valid."""
        decision = Decision(
            inputs={"outer": {"inner": {"deep": "value"}}},
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        assert decision.inputs["outer"]["inner"]["deep"] == "value"
    
    def test_inputs_accepts_lists(self, valid_alternatives):
        """Lists are valid input values."""
        decision = Decision(
            inputs={"items": [1, 2, 3, {"nested": True}]},
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        assert decision.inputs["items"] == [1, 2, 3, {"nested": True}]
    
    def test_inputs_accepts_null(self, valid_alternatives):
        """None/null is a valid input value."""
        decision = Decision(
            inputs={"nullable": None},
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        assert decision.inputs["nullable"] is None


class TestConfidenceValidation:
    """Test Confidence validation rules."""
    
    def test_confidence_value_must_be_in_range(self):
        """Confidence value must be between 0.0 and 1.0."""
        with pytest.raises(InvalidConfidenceError):
            Confidence(1.5)
        
        with pytest.raises(InvalidConfidenceError):
            Confidence(-0.1)
    
    def test_confidence_bounds_must_be_in_range(self):
        """Confidence bounds must be between 0.0 and 1.0."""
        with pytest.raises(InvalidConfidenceError):
            Confidence(0.5, lower_bound=-0.1)
        
        with pytest.raises(InvalidConfidenceError):
            Confidence(0.5, upper_bound=1.5)
    
    def test_lower_bound_cannot_exceed_value(self):
        """lower_bound cannot be greater than value."""
        with pytest.raises(DecisionValidationError) as exc_info:
            Confidence(0.5, lower_bound=0.6)
        assert "D005" in str(exc_info.value)
    
    def test_upper_bound_cannot_be_less_than_value(self):
        """upper_bound cannot be less than value."""
        with pytest.raises(DecisionValidationError) as exc_info:
            Confidence(0.5, upper_bound=0.4)
        assert "D006" in str(exc_info.value)
    
    def test_valid_confidence_with_bounds(self):
        """Valid confidence with proper bounds."""
        conf = Confidence(0.5, lower_bound=0.4, upper_bound=0.6)
        assert conf.value == 0.5
        assert conf.lower_bound == 0.4
        assert conf.upper_bound == 0.6
    
    def test_confidence_edge_values(self):
        """Edge values 0.0 and 1.0 are valid."""
        Confidence(0.0)
        Confidence(1.0)
        Confidence(0.5, lower_bound=0.0, upper_bound=1.0)


class TestAlternativeValidation:
    """Test Alternative validation rules."""
    
    def test_alternative_option_cannot_be_empty(self):
        """Alternative option cannot be empty."""
        with pytest.raises(DecisionValidationError) as exc_info:
            Alternative("", "reason")
        assert "D007" in str(exc_info.value)
    
    def test_alternative_reason_cannot_be_empty(self):
        """Alternative reason_not_selected cannot be empty."""
        with pytest.raises(DecisionValidationError) as exc_info:
            Alternative("option", "")
        assert "D008" in str(exc_info.value)
    
    def test_alternative_option_must_be_string(self):
        """Alternative option must be a string."""
        with pytest.raises(InvalidFieldTypeError):
            Alternative(123, "reason")
    
    def test_alternative_reason_must_be_string(self):
        """Alternative reason_not_selected must be a string."""
        with pytest.raises(InvalidFieldTypeError):
            Alternative("option", 123)


# =============================================================================
# SECTION 5: Serialization/Deserialization
# =============================================================================

class TestSerialization:
    """Test serialization and deserialization."""
    
    def test_to_dict_returns_dict(self, valid_decision):
        """to_dict() returns a dictionary."""
        result = valid_decision.to_dict()
        assert isinstance(result, dict)
    
    def test_to_json_returns_string(self, valid_decision):
        """to_json() returns a JSON string."""
        result = valid_decision.to_json()
        assert isinstance(result, str)
        # Verify it's valid JSON
        json.loads(result)
    
    def test_to_json_with_indent(self, valid_decision):
        """to_json() accepts indent parameter."""
        result = valid_decision.to_json(indent=2)
        assert "\n" in result
        assert "  " in result
    
    def test_from_dict_reconstructs_decision(self, valid_decision):
        """from_dict() reconstructs a Decision."""
        data = valid_decision.to_dict()
        restored = Decision.from_dict(data)
        assert restored.selected_option == valid_decision.selected_option
        assert restored.rationale == valid_decision.rationale
    
    def test_from_json_reconstructs_decision(self, valid_decision):
        """from_json() reconstructs a Decision."""
        json_str = valid_decision.to_json()
        restored = Decision.from_json(json_str)
        assert restored == valid_decision
    
    def test_from_dict_with_invalid_data_raises_error(self):
        """from_dict() with invalid data raises appropriate error."""
        with pytest.raises(DecisionValidationError):
            Decision.from_dict({"selected_option": None})


# =============================================================================
# SECTION 6: Equality and Hashing
# =============================================================================

class TestEqualityAndHashing:
    """Test equality and hash behavior."""
    
    def test_equal_decisions_are_equal(self):
        """Two identical decisions are equal."""
        kwargs = {
            "inputs": {"a": 1},
            "selected_option": "opt",
            "alternatives": [],
            "rationale": "reason",
            "decision_id": "id",
            "timestamp": "2026-02-08T12:00:00Z",
        }
        d1 = Decision(**kwargs)
        d2 = Decision(**kwargs)
        assert d1 == d2
    
    def test_different_decisions_are_not_equal(self):
        """Different decisions are not equal."""
        base = {
            "inputs": {"a": 1},
            "alternatives": [],
            "rationale": "reason",
            "decision_id": "id",
            "timestamp": "2026-02-08T12:00:00Z",
        }
        d1 = Decision(selected_option="opt1", **base)
        d2 = Decision(selected_option="opt2", **base)
        assert d1 != d2
    
    def test_decision_is_hashable(self, valid_decision):
        """Decision can be used in sets."""
        s = {valid_decision}
        assert valid_decision in s
    
    def test_decisions_with_same_id_have_same_hash(self):
        """Decisions with the same ID have the same hash."""
        base = {
            "inputs": {"a": 1},
            "alternatives": [],
            "rationale": "reason",
            "timestamp": "2026-02-08T12:00:00Z",
        }
        d1 = Decision(selected_option="opt1", decision_id="same-id", **base)
        d2 = Decision(selected_option="opt2", decision_id="same-id", **base)
        assert hash(d1) == hash(d2)


# =============================================================================
# SECTION 7: String Representations
# =============================================================================

class TestStringRepresentations:
    """Test __repr__ and __str__."""
    
    def test_repr_includes_key_info(self, valid_decision):
        """__repr__ includes decision ID and selected option."""
        r = repr(valid_decision)
        assert "Decision(" in r
        assert valid_decision.decision_id in r
        assert valid_decision.selected_option in r
    
    def test_str_is_human_readable(self, valid_decision):
        """__str__ is human-readable."""
        s = str(valid_decision)
        assert "Decision:" in s
        assert valid_decision.selected_option in s


# =============================================================================
# SECTION 8: Auto-generated Fields
# =============================================================================

class TestAutoGeneratedFields:
    """Test auto-generated decision_id and timestamp."""
    
    def test_decision_id_is_auto_generated(self, valid_inputs, valid_alternatives):
        """decision_id is auto-generated if not provided."""
        decision = Decision(
            inputs=valid_inputs,
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        assert decision.decision_id is not None
        assert len(decision.decision_id) > 0
    
    def test_timestamp_is_auto_generated(self, valid_inputs, valid_alternatives):
        """timestamp is auto-generated if not provided."""
        decision = Decision(
            inputs=valid_inputs,
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        assert decision.timestamp is not None
        assert "T" in decision.timestamp  # ISO 8601 format
    
    def test_explicit_decision_id_is_used(self, valid_inputs, valid_alternatives):
        """Explicit decision_id is used when provided."""
        decision = Decision(
            inputs=valid_inputs,
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
            decision_id="my-custom-id",
        )
        assert decision.decision_id == "my-custom-id"
    
    def test_explicit_timestamp_is_used(self, valid_inputs, valid_alternatives):
        """Explicit timestamp is used when provided."""
        decision = Decision(
            inputs=valid_inputs,
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
            timestamp="2020-01-01T00:00:00Z",
        )
        assert decision.timestamp == "2020-01-01T00:00:00Z"


# =============================================================================
# SECTION 9: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_inputs_dict_is_valid(self, valid_alternatives):
        """Empty inputs dict is valid (decision with no explicit inputs)."""
        decision = Decision(
            inputs={},
            selected_option="opt",
            alternatives=valid_alternatives,
            rationale="reason",
        )
        assert decision.inputs == {}
    
    def test_unicode_in_all_fields(self):
        """Unicode characters are handled correctly."""
        decision = Decision(
            inputs={"query": "日本語テスト", "emoji": "🎉"},
            selected_option="選択肢",
            alternatives=[Alternative("別の選択", "理由")],
            rationale="決定の理由",
        )
        
        # Roundtrip should preserve unicode
        json_str = decision.to_json()
        restored = Decision.from_json(json_str)
        assert restored.inputs["query"] == "日本語テスト"
        assert restored.selected_option == "選択肢"
    
    def test_very_long_strings(self):
        """Very long strings are handled correctly."""
        long_string = "x" * 100000
        decision = Decision(
            inputs={"long": long_string},
            selected_option="opt",
            alternatives=[],
            rationale=long_string,
        )
        assert len(decision.rationale) == 100000
    
    def test_deeply_nested_inputs(self):
        """Deeply nested input structures are handled."""
        deep = {"level": 0}
        current = deep
        for i in range(1, 50):
            current["nested"] = {"level": i}
            current = current["nested"]
        
        decision = Decision(
            inputs=deep,
            selected_option="opt",
            alternatives=[],
            rationale="reason",
        )
        
        # Verify it serializes without error
        json_str = decision.to_json()
        restored = Decision.from_json(json_str)
        assert restored.inputs["level"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
