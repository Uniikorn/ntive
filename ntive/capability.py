"""
capability.py

Ntive CapabilityDescriptor Primitive — Semantic Decision Specification v1.0.0

A CapabilityDescriptor is a declarative, immutable contract describing:
- What can be done
- Under what conditions
- With what inputs
- Producing what declared effects

Design Invariants:
- Immutable after creation
- Pure data structure (no side effects)
- Deterministic serialization (sorted keys, content-based hash)
- No execution logic
- No callable objects
- No runtime coupling
- Describes potential, not action
"""

import hashlib
import json
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Save reference to built-in type before any shadowing
_builtin_type = type


# =============================================================================
# Capability Errors
# =============================================================================

class CapabilityValidationError(Exception):
    """
    Raised when a CapabilityDescriptor cannot be constructed due to validation failure.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "C000",
    ):
        self.message = message
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        return f"[{self.error_code}] Capability validation failed: {self.message}"


class InvalidCapabilityInputError(CapabilityValidationError):
    """Raised when a CapabilityInput is invalid."""

    def __init__(self, key: str, reason: str):
        self.key = key
        self.reason = reason
        super().__init__(
            message=f"Invalid input '{key}': {reason}",
            error_code="C001",
        )


class DuplicateInputKeyError(CapabilityValidationError):
    """Raised when multiple inputs have the same key."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(
            message=f"Duplicate input key: '{key}'",
            error_code="C002",
        )


class InvalidPreconditionError(CapabilityValidationError):
    """Raised when a precondition is invalid."""

    def __init__(self, index: int, reason: str):
        self.index = index
        self.reason = reason
        super().__init__(
            message=f"Invalid precondition at index {index}: {reason}",
            error_code="C003",
        )


class InvalidEffectError(CapabilityValidationError):
    """Raised when an effect declaration is invalid."""

    def __init__(self, index: int, reason: str):
        self.index = index
        self.reason = reason
        super().__init__(
            message=f"Invalid effect at index {index}: {reason}",
            error_code="C004",
        )


class InvalidConstraintError(CapabilityValidationError):
    """Raised when a constraint is invalid."""

    def __init__(self, index: int, reason: str):
        self.index = index
        self.reason = reason
        super().__init__(
            message=f"Invalid constraint at index {index}: {reason}",
            error_code="C005",
        )


class InvalidMetadataError(CapabilityValidationError):
    """Raised when metadata contains non-serializable or executable content."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(
            message=f"Invalid metadata: {reason}",
            error_code="C006",
        )


class ExecutableContentError(CapabilityValidationError):
    """Raised when executable content (callable, lambda, etc.) is detected."""

    def __init__(self, location: str):
        self.location = location
        super().__init__(
            message=f"Executable content detected in {location}. "
                    "CapabilityDescriptor must be pure data only.",
            error_code="C007",
        )


class CapabilityImmutabilityError(Exception):
    """Raised when attempting to mutate an immutable capability object."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: Capability is immutable after creation"
        )


# =============================================================================
# Effect Category Enum
# =============================================================================

class EffectCategory(Enum):
    """
    Standard categories for declared effects.

    Effects are purely descriptive — they do NOT execute anything.
    """
    STATE_CHANGE = "state_change"
    RESOURCE_CONSUMPTION = "resource_consumption"
    EXTERNAL_DEPENDENCY = "external_dependency"
    IRREVERSIBLE_ACTION = "irreversible_action"
    DATA_MUTATION = "data_mutation"
    COMMUNICATION = "communication"
    OTHER = "other"

    @classmethod
    def from_string(cls, value: str) -> "EffectCategory":
        """Convert string to EffectCategory."""
        try:
            return cls(value.lower())
        except ValueError:
            # Allow custom effect categories
            return cls.OTHER


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    """Validate that a value is a string."""
    if not isinstance(value, str):
        raise CapabilityValidationError(
            f"{field_name} must be a string, got {_builtin_type(value).__name__}",
            error_code="C008",
        )
    if not allow_empty and not value.strip():
        raise CapabilityValidationError(
            f"{field_name} cannot be empty or whitespace-only",
            error_code="C008",
        )
    return value


def _check_for_executable(value: Any, path: str = "root") -> None:
    """
    Recursively check for executable content in a value.

    Raises ExecutableContentError if found.
    """
    # Check for callable
    if callable(value):
        raise ExecutableContentError(path)

    # Check for code objects
    if hasattr(value, '__code__'):
        raise ExecutableContentError(path)

    # Check for common dangerous types
    if isinstance(value, type):
        raise ExecutableContentError(path)

    # Recurse into containers
    if isinstance(value, dict):
        for k, v in value.items():
            _check_for_executable(k, f"{path}.key({k!r})")
            _check_for_executable(v, f"{path}[{k!r}]")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_for_executable(item, f"{path}[{i}]")


def _validate_json_serializable(value: Any, field_name: str) -> Any:
    """Validate that a value is JSON-serializable and contains no executables."""
    _check_for_executable(value, field_name)

    try:
        # Attempt serialization
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise CapabilityValidationError(
            f"{field_name} must be JSON-serializable: {e}",
            error_code="C009",
        )

    return value


def _deep_copy_json(value: Any) -> Any:
    """Create a deep copy of a JSON-serializable value."""
    return json.loads(json.dumps(value, sort_keys=True))


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of JSON-serialized data."""
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# =============================================================================
# CapabilityInput
# =============================================================================

class CapabilityInput:
    """
    A single input definition for a capability.

    Inputs describe what data is needed, not values themselves.
    Types are symbolic strings, not Python types.

    Attributes:
        key: Unique identifier for this input
        type: Symbolic type name (e.g., "string", "integer", "user_id")
        required: Whether this input must be provided
        constraints: Optional declarative constraints
        description: Optional human-readable description
    """

    __slots__ = ('_key', '_type', '_required', '_constraints', '_description', '_frozen')

    def __init__(
        self,
        *,
        key: str,
        type: str,
        required: bool = True,
        constraints: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        # Validate key
        key = _validate_string(key, "key")

        # Validate type (symbolic, not Python type)
        if not isinstance(type, str):
            raise InvalidCapabilityInputError(
                key,
                f"type must be a symbolic string, got {type!r}",
            )
        type = _validate_string(type, "type")

        # Validate required
        if not isinstance(required, bool):
            raise InvalidCapabilityInputError(
                key,
                f"required must be bool, got {_builtin_type(required).__name__}",
            )

        # Validate constraints
        if constraints is not None:
            if not isinstance(constraints, dict):
                raise InvalidCapabilityInputError(
                    key,
                    f"constraints must be dict, got {_builtin_type(constraints).__name__}",
                )
            _validate_json_serializable(constraints, f"input[{key}].constraints")
            constraints = _deep_copy_json(constraints)
        else:
            constraints = {}

        # Validate description
        if description is not None:
            description = _validate_string(description, "description", allow_empty=True)

        object.__setattr__(self, '_key', key)
        object.__setattr__(self, '_type', type)
        object.__setattr__(self, '_required', required)
        object.__setattr__(self, '_constraints', constraints)
        object.__setattr__(self, '_description', description)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def key(self) -> str:
        return self._key

    @property
    def type(self) -> str:
        return self._type

    @property
    def required(self) -> bool:
        return self._required

    @property
    def constraints(self) -> Dict[str, Any]:
        """Return a copy of the constraints dict."""
        return _deep_copy_json(self._constraints)

    @property
    def description(self) -> Optional[str]:
        return self._description

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CapabilityInput):
            return NotImplemented
        return (
            self._key == other._key
            and self._type == other._type
            and self._required == other._required
            and self._constraints == other._constraints
            and self._description == other._description
        )

    def __hash__(self) -> int:
        return hash((
            self._key,
            self._type,
            self._required,
            json.dumps(self._constraints, sort_keys=True),
            self._description,
        ))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "constraints": self._constraints,
            "key": self._key,
            "required": self._required,
            "type": self._type,
        }
        if self._description is not None:
            result["description"] = self._description
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityInput":
        """Construct from a dictionary."""
        return cls(
            key=data["key"],
            type=data["type"],
            required=data.get("required", True),
            constraints=data.get("constraints"),
            description=data.get("description"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CapabilityInput":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        req = "required" if self._required else "optional"
        return f"CapabilityInput(key={self._key!r}, type={self._type!r}, {req})"

    def __str__(self) -> str:
        marker = "*" if self._required else ""
        return f"{self._key}{marker}: {self._type}"


# =============================================================================
# DeclaredEffect
# =============================================================================

class DeclaredEffect:
    """
    A declared effect of a capability.

    Effects describe what WOULD change if the capability is enacted.
    They do NOT perform or encode execution logic.

    Attributes:
        category: Effect category (state_change, resource_consumption, etc.)
        target: What is affected (symbolic identifier)
        description: Human-readable description
        metadata: Additional structured data about the effect
    """

    __slots__ = ('_category', '_target', '_description', '_metadata', '_frozen')

    def __init__(
        self,
        *,
        category: str | EffectCategory,
        target: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Parse category
        if isinstance(category, str):
            # Accept any string as category, but map known ones to enum
            try:
                EffectCategory(category.lower())  # Validate against enum
            except ValueError:
                pass  # Accept unknown categories as OTHER
            category_str = category.lower()
        elif isinstance(category, EffectCategory):
            category_str = category.value
        else:
            raise InvalidEffectError(
                0,
                f"category must be string or EffectCategory, got {_builtin_type(category).__name__}",
            )

        # Validate target
        target = _validate_string(target, "effect.target")

        # Validate description
        if description is not None:
            description = _validate_string(description, "effect.description", allow_empty=True)

        # Validate metadata
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise InvalidEffectError(
                    0,
                    f"metadata must be dict, got {_builtin_type(metadata).__name__}",
                )
            _validate_json_serializable(metadata, "effect.metadata")
            metadata = _deep_copy_json(metadata)
        else:
            metadata = {}

        object.__setattr__(self, '_category', category_str)
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_description', description)
        object.__setattr__(self, '_metadata', metadata)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def category(self) -> str:
        return self._category

    @property
    def target(self) -> str:
        return self._target

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a copy of the metadata dict."""
        return _deep_copy_json(self._metadata)

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeclaredEffect):
            return NotImplemented
        return (
            self._category == other._category
            and self._target == other._target
            and self._description == other._description
            and self._metadata == other._metadata
        )

    def __hash__(self) -> int:
        return hash((
            self._category,
            self._target,
            self._description,
            json.dumps(self._metadata, sort_keys=True),
        ))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "category": self._category,
            "metadata": self._metadata,
            "target": self._target,
        }
        if self._description is not None:
            result["description"] = self._description
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeclaredEffect":
        """Construct from a dictionary."""
        return cls(
            category=data["category"],
            target=data["target"],
            description=data.get("description"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DeclaredEffect":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DeclaredEffect(category={self._category!r}, target={self._target!r})"

    def __str__(self) -> str:
        return f"{self._category}: {self._target}"


# =============================================================================
# Precondition
# =============================================================================

class Precondition:
    """
    A declarative precondition for a capability.

    Preconditions describe what must be true for the capability to be valid.
    They do NOT evaluate or check conditions at runtime.

    Attributes:
        condition: Symbolic condition identifier
        parameters: Condition parameters (opaque, declarative)
        description: Human-readable description
    """

    __slots__ = ('_condition', '_parameters', '_description', '_frozen')

    def __init__(
        self,
        *,
        condition: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        # Validate condition
        condition = _validate_string(condition, "precondition.condition")

        # Validate parameters
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise InvalidPreconditionError(
                    0,
                    f"parameters must be dict, got {_builtin_type(parameters).__name__}",
                )
            _validate_json_serializable(parameters, "precondition.parameters")
            parameters = _deep_copy_json(parameters)
        else:
            parameters = {}

        # Validate description
        if description is not None:
            description = _validate_string(description, "precondition.description", allow_empty=True)

        object.__setattr__(self, '_condition', condition)
        object.__setattr__(self, '_parameters', parameters)
        object.__setattr__(self, '_description', description)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def condition(self) -> str:
        return self._condition

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return a copy of the parameters dict."""
        return _deep_copy_json(self._parameters)

    @property
    def description(self) -> Optional[str]:
        return self._description

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Precondition):
            return NotImplemented
        return (
            self._condition == other._condition
            and self._parameters == other._parameters
            and self._description == other._description
        )

    def __hash__(self) -> int:
        return hash((
            self._condition,
            json.dumps(self._parameters, sort_keys=True),
            self._description,
        ))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "condition": self._condition,
            "parameters": self._parameters,
        }
        if self._description is not None:
            result["description"] = self._description
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Precondition":
        """Construct from a dictionary."""
        return cls(
            condition=data["condition"],
            parameters=data.get("parameters"),
            description=data.get("description"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Precondition":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Precondition(condition={self._condition!r})"

    def __str__(self) -> str:
        return f"requires: {self._condition}"


# =============================================================================
# Constraint
# =============================================================================

class Constraint:
    """
    A declarative constraint or invariant for a capability.

    Constraints describe limits or rules that apply to the capability.
    They do NOT enforce anything at runtime.

    Attributes:
        name: Constraint identifier
        rule: Symbolic rule definition
        parameters: Rule parameters
        description: Human-readable description
    """

    __slots__ = ('_name', '_rule', '_parameters', '_description', '_frozen')

    def __init__(
        self,
        *,
        name: str,
        rule: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        # Validate name
        name = _validate_string(name, "constraint.name")

        # Validate rule
        rule = _validate_string(rule, "constraint.rule")

        # Validate parameters
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise InvalidConstraintError(
                    0,
                    f"parameters must be dict, got {_builtin_type(parameters).__name__}",
                )
            _validate_json_serializable(parameters, "constraint.parameters")
            parameters = _deep_copy_json(parameters)
        else:
            parameters = {}

        # Validate description
        if description is not None:
            description = _validate_string(description, "constraint.description", allow_empty=True)

        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_rule', rule)
        object.__setattr__(self, '_parameters', parameters)
        object.__setattr__(self, '_description', description)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def rule(self) -> str:
        return self._rule

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return a copy of the parameters dict."""
        return _deep_copy_json(self._parameters)

    @property
    def description(self) -> Optional[str]:
        return self._description

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constraint):
            return NotImplemented
        return (
            self._name == other._name
            and self._rule == other._rule
            and self._parameters == other._parameters
            and self._description == other._description
        )

    def __hash__(self) -> int:
        return hash((
            self._name,
            self._rule,
            json.dumps(self._parameters, sort_keys=True),
            self._description,
        ))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "name": self._name,
            "parameters": self._parameters,
            "rule": self._rule,
        }
        if self._description is not None:
            result["description"] = self._description
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        """Construct from a dictionary."""
        return cls(
            name=data["name"],
            rule=data["rule"],
            parameters=data.get("parameters"),
            description=data.get("description"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Constraint":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Constraint(name={self._name!r}, rule={self._rule!r})"

    def __str__(self) -> str:
        return f"{self._name}: {self._rule}"


# =============================================================================
# CapabilityDescriptor
# =============================================================================

class CapabilityDescriptor:
    """
    A declarative contract describing what a capability can do.

    CapabilityDescriptor is pure data — it describes potential, not action.
    It does NOT execute, evaluate, or enforce anything.

    Attributes:
        capability_id: Content-based hash (computed, not user-provided)
        name: Unique capability identifier
        version: Semantic version string
        domain: Namespace for the capability
        description: Human-readable description
        inputs: List of required/optional inputs
        preconditions: List of declarative preconditions
        effects: List of declared effects
        constraints: List of constraints/invariants
        metadata: Additional JSON-serializable data
    """

    __slots__ = (
        '_capability_id',
        '_name',
        '_version',
        '_domain',
        '_description',
        '_inputs',
        '_inputs_by_key',
        '_preconditions',
        '_effects',
        '_constraints',
        '_metadata',
        '_frozen',
    )

    def __init__(
        self,
        *,
        name: str,
        version: str,
        domain: str,
        description: Optional[str] = None,
        inputs: Optional[List[CapabilityInput | Dict[str, Any]]] = None,
        preconditions: Optional[List[Precondition | Dict[str, Any]]] = None,
        effects: Optional[List[DeclaredEffect | Dict[str, Any]]] = None,
        constraints: Optional[List[Constraint | Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Validate required fields
        name = _validate_string(name, "name")
        version = _validate_string(version, "version")
        domain = _validate_string(domain, "domain")

        # Validate description
        if description is not None:
            description = _validate_string(description, "description", allow_empty=True)

        # Parse and validate inputs
        parsed_inputs: List[CapabilityInput] = []
        inputs_by_key: Dict[str, CapabilityInput] = {}

        if inputs:
            for i, inp in enumerate(inputs):
                if isinstance(inp, dict):
                    ci = CapabilityInput.from_dict(inp)
                elif isinstance(inp, CapabilityInput):
                    ci = inp
                else:
                    raise InvalidCapabilityInputError(
                        f"input[{i}]",
                        f"must be CapabilityInput or dict, got {_builtin_type(inp).__name__}",
                    )

                # Check for duplicate keys
                if ci.key in inputs_by_key:
                    raise DuplicateInputKeyError(ci.key)

                parsed_inputs.append(ci)
                inputs_by_key[ci.key] = ci

        # Parse and validate preconditions
        parsed_preconditions: List[Precondition] = []

        if preconditions:
            for i, pre in enumerate(preconditions):
                if isinstance(pre, dict):
                    try:
                        pc = Precondition.from_dict(pre)
                    except CapabilityValidationError as e:
                        raise InvalidPreconditionError(i, str(e))
                elif isinstance(pre, Precondition):
                    pc = pre
                else:
                    raise InvalidPreconditionError(
                        i,
                        f"must be Precondition or dict, got {_builtin_type(pre).__name__}",
                    )
                parsed_preconditions.append(pc)

        # Parse and validate effects
        parsed_effects: List[DeclaredEffect] = []

        if effects:
            for i, eff in enumerate(effects):
                if isinstance(eff, dict):
                    try:
                        de = DeclaredEffect.from_dict(eff)
                    except CapabilityValidationError as e:
                        raise InvalidEffectError(i, str(e))
                elif isinstance(eff, DeclaredEffect):
                    de = eff
                else:
                    raise InvalidEffectError(
                        i,
                        f"must be DeclaredEffect or dict, got {_builtin_type(eff).__name__}",
                    )
                parsed_effects.append(de)

        # Parse and validate constraints
        parsed_constraints: List[Constraint] = []

        if constraints:
            for i, con in enumerate(constraints):
                if isinstance(con, dict):
                    try:
                        c = Constraint.from_dict(con)
                    except CapabilityValidationError as e:
                        raise InvalidConstraintError(i, str(e))
                elif isinstance(con, Constraint):
                    c = con
                else:
                    raise InvalidConstraintError(
                        i,
                        f"must be Constraint or dict, got {_builtin_type(con).__name__},"
                    )
                parsed_constraints.append(c)

        # Validate metadata
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise InvalidMetadataError(
                    f"metadata must be dict, got {_builtin_type(metadata).__name__}"
                )
            _validate_json_serializable(metadata, "metadata")
            metadata = _deep_copy_json(metadata)
        else:
            metadata = {}

        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_version', version)
        object.__setattr__(self, '_domain', domain)
        object.__setattr__(self, '_description', description)
        object.__setattr__(self, '_inputs', tuple(parsed_inputs))
        object.__setattr__(self, '_inputs_by_key', inputs_by_key)
        object.__setattr__(self, '_preconditions', tuple(parsed_preconditions))
        object.__setattr__(self, '_effects', tuple(parsed_effects))
        object.__setattr__(self, '_constraints', tuple(parsed_constraints))
        object.__setattr__(self, '_metadata', metadata)

        # Compute content-based capability_id
        capability_id = self._compute_capability_id()
        object.__setattr__(self, '_capability_id', capability_id)

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise CapabilityImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Internal Helper Methods
    # -------------------------------------------------------------------------

    def _compute_capability_id(self) -> str:
        """Compute content-based hash for capability_id."""
        data = {
            "constraints": [c.to_dict() for c in self._constraints],
            "description": self._description,
            "domain": self._domain,
            "effects": [e.to_dict() for e in self._effects],
            "inputs": [i.to_dict() for i in self._inputs],
            "metadata": self._metadata,
            "name": self._name,
            "preconditions": [p.to_dict() for p in self._preconditions],
            "version": self._version,
        }
        return _compute_hash(data)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def capability_id(self) -> str:
        """Content-based hash of this capability descriptor."""
        return self._capability_id

    @property
    def name(self) -> str:
        """Unique capability identifier."""
        return self._name

    @property
    def version(self) -> str:
        """Semantic version string."""
        return self._version

    @property
    def domain(self) -> str:
        """Namespace for the capability."""
        return self._domain

    @property
    def description(self) -> Optional[str]:
        """Human-readable description."""
        return self._description

    @property
    def inputs(self) -> Tuple[CapabilityInput, ...]:
        """List of input definitions."""
        return self._inputs

    @property
    def preconditions(self) -> Tuple[Precondition, ...]:
        """List of preconditions."""
        return self._preconditions

    @property
    def effects(self) -> Tuple[DeclaredEffect, ...]:
        """List of declared effects."""
        return self._effects

    @property
    def constraints(self) -> Tuple[Constraint, ...]:
        """List of constraints."""
        return self._constraints

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a copy of the metadata dict."""
        return _deep_copy_json(self._metadata)

    # -------------------------------------------------------------------------
    # Input Access Methods
    # -------------------------------------------------------------------------

    def get_input(self, key: str) -> Optional[CapabilityInput]:
        """Get an input by key."""
        return self._inputs_by_key.get(key)

    def has_input(self, key: str) -> bool:
        """Check if an input exists."""
        return key in self._inputs_by_key

    def required_inputs(self) -> List[CapabilityInput]:
        """Get all required inputs."""
        return [i for i in self._inputs if i.required]

    def optional_inputs(self) -> List[CapabilityInput]:
        """Get all optional inputs."""
        return [i for i in self._inputs if not i.required]

    def input_keys(self) -> List[str]:
        """Get all input keys."""
        return [i.key for i in self._inputs]

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def effects_by_category(self, category: str) -> List[DeclaredEffect]:
        """Get all effects of a specific category."""
        return [e for e in self._effects if e.category == category.lower()]

    def has_effect_category(self, category: str) -> bool:
        """Check if any effect has the given category."""
        return any(e.category == category.lower() for e in self._effects)

    def has_irreversible_effects(self) -> bool:
        """Check if any effects are marked as irreversible."""
        return self.has_effect_category("irreversible_action")

    # -------------------------------------------------------------------------
    # Qualified Name
    # -------------------------------------------------------------------------

    @property
    def qualified_name(self) -> str:
        """Full qualified name: domain/name@version."""
        return f"{self._domain}/{self._name}@{self._version}"

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CapabilityDescriptor):
            return NotImplemented
        return self._capability_id == other._capability_id

    def __hash__(self) -> int:
        return hash(self._capability_id)

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return count of inputs."""
        return len(self._inputs)

    def __iter__(self) -> Iterator[CapabilityInput]:
        """Iterate over inputs."""
        return iter(self._inputs)

    def __contains__(self, key: str) -> bool:
        """Check if an input key exists."""
        return self.has_input(key)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "capability_id": self._capability_id,
            "constraints": [c.to_dict() for c in self._constraints],
            "domain": self._domain,
            "effects": [e.to_dict() for e in self._effects],
            "inputs": [i.to_dict() for i in self._inputs],
            "metadata": self._metadata,
            "name": self._name,
            "preconditions": [p.to_dict() for p in self._preconditions],
            "version": self._version,
        }
        if self._description is not None:
            result["description"] = self._description
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityDescriptor":
        """Construct from a dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            domain=data["domain"],
            description=data.get("description"),
            inputs=[CapabilityInput.from_dict(i) for i in data.get("inputs", [])],
            preconditions=[Precondition.from_dict(p) for p in data.get("preconditions", [])],
            effects=[DeclaredEffect.from_dict(e) for e in data.get("effects", [])],
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CapabilityDescriptor":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CapabilityDescriptor(name={self._name!r}, "
            f"domain={self._domain!r}, version={self._version!r}, "
            f"inputs={len(self._inputs)}, effects={len(self._effects)})"
        )

    def __str__(self) -> str:
        return self.qualified_name
