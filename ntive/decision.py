"""
decision.py

Ntive Decision Primitive — Semantic Decision Specification v1.0.0

A Decision is an immutable record of a semantic choice made by an
intelligent system. It captures WHAT was decided, WHY it was decided,
and WHAT alternatives were considered — without any execution logic.

Design Invariants:
- Immutable once created (frozen)
- All required fields validated on construction
- Missing inputs rejected explicitly
- No execution logic whatsoever
- Deterministic serialization (sorted keys, stable output)
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class DecisionValidationError(Exception):
    """
    Raised when a Decision cannot be constructed due to validation failure.

    This is NOT a runtime error — it's a construction-time error indicating
    that the caller attempted to create an invalid Decision.
    """

    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        error_code: str = "D000",
    ):
        self.message = message
        self.field_name = field_name
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        if self.field_name:
            return f"[{self.error_code}] Decision validation failed for '{self.field_name}': {self.message}"
        return f"[{self.error_code}] Decision validation failed: {self.message}"


class MissingRequiredFieldError(DecisionValidationError):
    """Raised when a required field is missing or None."""

    def __init__(self, field_name: str):
        super().__init__(
            message=f"Required field '{field_name}' is missing or None",
            field_name=field_name,
            error_code="D001",
        )


class InvalidFieldTypeError(DecisionValidationError):
    """Raised when a field has an invalid type."""

    def __init__(self, field_name: str, expected: str, actual: str):
        super().__init__(
            message=f"Expected {expected}, got {actual}",
            field_name=field_name,
            error_code="D002",
        )
        self.expected = expected
        self.actual = actual


class InvalidInputsError(DecisionValidationError):
    """Raised when inputs dict contains invalid entries."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            field_name="inputs",
            error_code="D003",
        )


class InvalidConfidenceError(DecisionValidationError):
    """Raised when confidence value is out of valid range [0.0, 1.0]."""

    def __init__(self, value: float):
        super().__init__(
            message=f"Confidence must be between 0.0 and 1.0, got {value}",
            field_name="confidence",
            error_code="D004",
        )
        self.value = value


class ImmutabilityViolationError(Exception):
    """Raised when attempting to mutate an immutable Decision."""

    def __init__(self, field_name: str):
        self.field_name = field_name
        super().__init__(f"Cannot modify Decision.{field_name}: Decision is immutable")


# === Confidence Type ===

@dataclass(frozen=True)
class Confidence:
    """
    Represents a confidence level with optional bounds.

    All values MUST be in the range [0.0, 1.0].
    """
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self):
        # Validate value
        if not isinstance(self.value, (int, float)):
            raise InvalidFieldTypeError("confidence.value", "float", type(self.value).__name__)
        if not 0.0 <= self.value <= 1.0:
            raise InvalidConfidenceError(self.value)

        # Validate bounds if provided
        if self.lower_bound is not None:
            if not isinstance(self.lower_bound, (int, float)):
                raise InvalidFieldTypeError("confidence.lower_bound", "float", type(self.lower_bound).__name__)  # noqa: E501
            if not 0.0 <= self.lower_bound <= 1.0:
                raise InvalidConfidenceError(self.lower_bound)
            if self.lower_bound > self.value:
                raise DecisionValidationError(
                    f"lower_bound ({self.lower_bound}) cannot exceed value ({self.value})",
                    field_name="confidence.lower_bound",
                    error_code="D005",
                )

        if self.upper_bound is not None:
            if not isinstance(self.upper_bound, (int, float)):
                raise InvalidFieldTypeError("confidence.upper_bound", "float", type(self.upper_bound).__name__)  # noqa: E501
            if not 0.0 <= self.upper_bound <= 1.0:
                raise InvalidConfidenceError(self.upper_bound)
            if self.upper_bound < self.value:
                raise DecisionValidationError(
                    f"upper_bound ({self.upper_bound}) cannot be less than value ({self.value})",
                    field_name="confidence.upper_bound",
                    error_code="D006",
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"value": self.value}
        if self.lower_bound is not None:
            result["lower_bound"] = self.lower_bound
        if self.upper_bound is not None:
            result["upper_bound"] = self.upper_bound
        return result


# === Alternative Type ===

@dataclass(frozen=True)
class Alternative:
    """
    Represents an alternative option that was considered but not selected.

    Captures what else could have been chosen and why it wasn't.
    """
    option: str
    reason_not_selected: str

    def __post_init__(self):
        if not isinstance(self.option, str):
            raise InvalidFieldTypeError("alternative.option", "str", type(self.option).__name__)
        if not self.option.strip():
            raise DecisionValidationError(
                "Alternative option cannot be empty",
                field_name="alternative.option",
                error_code="D007",
            )
        if not isinstance(self.reason_not_selected, str):
            raise InvalidFieldTypeError("alternative.reason_not_selected", "str", type(self.reason_not_selected).__name__)  # noqa: E501
        if not self.reason_not_selected.strip():
            raise DecisionValidationError(
                "Alternative must have a reason_not_selected",
                field_name="alternative.reason_not_selected",
                error_code="D008",
            )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization."""
        return {
            "option": self.option,
            "reason_not_selected": self.reason_not_selected,
        }


# === Helper Functions ===

def _freeze_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-freeze a dictionary of inputs for immutability.

    Converts mutable containers to immutable equivalents where possible.
    """
    def freeze_value(v: Any) -> Any:
        if isinstance(v, dict):
            return tuple(sorted((k, freeze_value(val)) for k, val in v.items()))
        elif isinstance(v, list):
            return tuple(freeze_value(item) for item in v)
        elif isinstance(v, set):
            return frozenset(freeze_value(item) for item in v)
        return v

    return {k: freeze_value(v) for k, v in inputs.items()}


def _validate_json_serializable(value: Any, path: str = "inputs") -> None:
    """
    Validate that a value can be serialized to JSON.

    Raises InvalidInputsError if the value contains non-serializable types.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    elif isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise InvalidInputsError(f"Dictionary keys must be strings at {path}, got {type(k).__name__}")
            _validate_json_serializable(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_json_serializable(item, f"{path}[{i}]")
    else:
        raise InvalidInputsError(f"Value at {path} is not JSON-serializable: {type(value).__name__}")


# === Decision Class ===

class Decision:
    """
    Immutable record of a semantic decision.

    A Decision captures:
    - WHAT was decided (selected_option)
    - WITH WHAT inputs (inputs)
    - WHY it was decided (rationale)
    - WHAT alternatives were considered (alternatives)
    - HOW confident the decision is (confidence)
    - WHAT constraints were active (constraints)

    Invariants:
    - Immutable after construction
    - All fields validated at construction time
    - Deterministic serialization (identical inputs → identical JSON)
    - No execution logic

    Example:
        decision = Decision(
            inputs={"user_query": "open file", "context": "editor"},
            selected_option="open_file_dialog",
            alternatives=[
                Alternative("recent_files", "User said 'open', not 'recent'"),
            ],
            rationale="User explicitly requested to open a file",
        )
    """

    __slots__ = (
        '_decision_id',
        '_inputs',
        '_selected_option',
        '_alternatives',
        '_rationale',
        '_confidence',
        '_constraints',
        '_timestamp',
        '_frozen',
    )

    def __init__(
        self,
        *,
        inputs: Dict[str, Any],
        selected_option: str,
        alternatives: List[Alternative],
        rationale: str,
        confidence: Optional[Confidence] = None,
        constraints: Optional[List[str]] = None,
        decision_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        """
        Create a new Decision.

        All parameters except confidence, constraints, decision_id, and timestamp are required.

        Args:
            inputs: Explicit input bindings that led to this decision.
                    Must be JSON-serializable.
            selected_option: The chosen option.
            alternatives: List of alternatives that were considered.
                          MUST be a list (can be empty).
            rationale: Explanation of why this option was selected.
            confidence: Optional confidence level with bounds.
            constraints: Optional list of active constraints.
            decision_id: Optional explicit ID (auto-generated if not provided).
            timestamp: Optional ISO 8601 timestamp (auto-generated if not provided).

        Raises:
            MissingRequiredFieldError: If a required field is missing.
            InvalidFieldTypeError: If a field has wrong type.
            InvalidInputsError: If inputs are not JSON-serializable.
            InvalidConfidenceError: If confidence is out of range.
        """
        # Mark as not frozen during construction
        object.__setattr__(self, '_frozen', False)

        # === Validate required fields ===

        # inputs
        if inputs is None:
            raise MissingRequiredFieldError("inputs")
        if not isinstance(inputs, dict):
            raise InvalidFieldTypeError("inputs", "dict", type(inputs).__name__)
        _validate_json_serializable(inputs)

        # selected_option
        if selected_option is None:
            raise MissingRequiredFieldError("selected_option")
        if not isinstance(selected_option, str):
            raise InvalidFieldTypeError("selected_option", "str", type(selected_option).__name__)
        if not selected_option.strip():
            raise DecisionValidationError(
                "selected_option cannot be empty",
                field_name="selected_option",
                error_code="D009",
            )

        # alternatives
        if alternatives is None:
            raise MissingRequiredFieldError("alternatives")
        if not isinstance(alternatives, list):
            raise InvalidFieldTypeError("alternatives", "list", type(alternatives).__name__)
        for i, alt in enumerate(alternatives):
            if not isinstance(alt, Alternative):
                raise InvalidFieldTypeError(f"alternatives[{i}]", "Alternative", type(alt).__name__)

        # rationale
        if rationale is None:
            raise MissingRequiredFieldError("rationale")
        if not isinstance(rationale, str):
            raise InvalidFieldTypeError("rationale", "str", type(rationale).__name__)
        if not rationale.strip():
            raise DecisionValidationError(
                "rationale cannot be empty",
                field_name="rationale",
                error_code="D010",
            )

        # === Validate optional fields ===

        if confidence is not None and not isinstance(confidence, Confidence):
            raise InvalidFieldTypeError("confidence", "Confidence", type(confidence).__name__)

        if constraints is not None:
            if not isinstance(constraints, list):
                raise InvalidFieldTypeError("constraints", "list", type(constraints).__name__)
            for i, c in enumerate(constraints):
                if not isinstance(c, str):
                    raise InvalidFieldTypeError(f"constraints[{i}]", "str", type(c).__name__)

        # === Generate auto fields ===

        if decision_id is None:
            decision_id = str(uuid.uuid4())
        elif not isinstance(decision_id, str):
            raise InvalidFieldTypeError("decision_id", "str", type(decision_id).__name__)

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        elif not isinstance(timestamp, str):
            raise InvalidFieldTypeError("timestamp", "str", type(timestamp).__name__)

        # === Store frozen values ===

        # Deep copy inputs to prevent external mutation
        self._inputs = json.loads(json.dumps(inputs))  # Ensures deep copy and JSON-validity
        self._selected_option = selected_option
        self._alternatives = tuple(alternatives)  # Freeze list
        self._rationale = rationale
        self._confidence = confidence
        self._constraints = tuple(constraints) if constraints else None
        self._decision_id = decision_id
        self._timestamp = timestamp

        # Freeze the object
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after construction."""
        if getattr(self, '_frozen', False):
            raise ImmutabilityViolationError(name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of attributes."""
        raise ImmutabilityViolationError(name)

    # === Properties (read-only access) ===

    @property
    def decision_id(self) -> str:
        """Unique identifier for this decision."""
        return self._decision_id

    @property
    def inputs(self) -> Dict[str, Any]:
        """Explicit input bindings (deep copy to prevent mutation)."""
        return json.loads(json.dumps(self._inputs))

    @property
    def selected_option(self) -> str:
        """The chosen option."""
        return self._selected_option

    @property
    def alternatives(self) -> Tuple[Alternative, ...]:
        """Considered alternatives (immutable tuple)."""
        return self._alternatives

    @property
    def rationale(self) -> str:
        """Explanation for this decision."""
        return self._rationale

    @property
    def confidence(self) -> Optional[Confidence]:
        """Confidence level with optional bounds."""
        return self._confidence

    @property
    def constraints(self) -> Optional[Tuple[str, ...]]:
        """Active constraints (immutable tuple or None)."""
        return self._constraints

    @property
    def timestamp(self) -> str:
        """ISO 8601 timestamp of decision creation."""
        return self._timestamp

    # === Serialization ===

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Keys are sorted for deterministic output.
        """
        result = {
            "alternatives": [alt.to_dict() for alt in self._alternatives],
            "decision_id": self._decision_id,
            "inputs": self._inputs,
            "rationale": self._rationale,
            "selected_option": self._selected_option,
            "timestamp": self._timestamp,
        }

        if self._confidence is not None:
            result["confidence"] = self._confidence.to_dict()

        if self._constraints is not None:
            result["constraints"] = list(self._constraints)

        return dict(sorted(result.items()))

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """
        Serialize to JSON string.

        Guarantees deterministic output:
        - Keys are sorted alphabetically at all levels
        - No trailing whitespace
        - Identical inputs always produce identical output

        Args:
            indent: Optional indentation for pretty-printing.

        Returns:
            JSON string representation.
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        """
        Reconstruct a Decision from a dictionary.

        Args:
            data: Dictionary with decision fields.

        Returns:
            New Decision instance.

        Raises:
            DecisionValidationError: If data is invalid.
        """
        if not isinstance(data, dict):
            raise InvalidFieldTypeError("data", "dict", type(data).__name__)

        # Parse alternatives
        alternatives_data = data.get("alternatives", [])
        alternatives = [
            Alternative(
                option=alt.get("option", ""),
                reason_not_selected=alt.get("reason_not_selected", ""),
            )
            for alt in alternatives_data
        ]

        # Parse confidence
        confidence = None
        if "confidence" in data and data["confidence"] is not None:
            conf_data = data["confidence"]
            confidence = Confidence(
                value=conf_data.get("value", 0.0),
                lower_bound=conf_data.get("lower_bound"),
                upper_bound=conf_data.get("upper_bound"),
            )

        return cls(
            decision_id=data.get("decision_id"),
            inputs=data.get("inputs"),
            selected_option=data.get("selected_option"),
            alternatives=alternatives,
            rationale=data.get("rationale"),
            confidence=confidence,
            constraints=data.get("constraints"),
            timestamp=data.get("timestamp"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Decision":
        """
        Reconstruct a Decision from a JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            New Decision instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # === Comparison ===

    def __eq__(self, other: object) -> bool:
        """Two decisions are equal if their serialized forms are identical."""
        if not isinstance(other, Decision):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        """Hash based on decision_id for use in sets/dicts."""
        return hash(self._decision_id)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Decision("
            f"id={self._decision_id!r}, "
            f"selected={self._selected_option!r}, "
            f"alternatives={len(self._alternatives)}, "
            f"confidence={self._confidence.value if self._confidence else None}"
            f")"
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"Decision: {self._selected_option} ({self._rationale})"
