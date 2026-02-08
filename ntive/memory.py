"""
memory.py

Ntive MemoryScope Primitive — Semantic Decision Specification v1.0.0

A MemoryScope is an immutable, scoped container of key-value state.
It exists ONLY to answer: "What is known or assumed at this point?"

Design Invariants:
- Immutable after creation
- Pure data structure (no side effects)
- Deterministic serialization (sorted keys, content-based hash)
- No I/O, no time, no randomness
- No coupling with Decision or Trace logic
- Values are JSON-serializable primitives only
"""

import hashlib
import json
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# =============================================================================
# Memory Errors
# =============================================================================

class MemoryValidationError(Exception):
    """
    Raised when a MemoryScope cannot be constructed due to validation failure.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "M000",
    ):
        self.message = message
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        return f"[{self.error_code}] MemoryScope validation failed: {self.message}"


class InvalidMemoryKeyError(MemoryValidationError):
    """Raised when a key is not a valid string."""

    def __init__(self, key: Any):
        self.key = key
        self.key_type = type(key).__name__
        super().__init__(
            message=f"Keys must be non-empty strings, got {self.key_type}: {key!r}",
            error_code="M001",
        )


class InvalidMemoryValueError(MemoryValidationError):
    """Raised when a value is not JSON-serializable."""

    def __init__(self, key: str, value: Any, reason: str = ""):
        self.key = key
        self.value = value
        self.value_type = type(value).__name__
        detail = f": {reason}" if reason else ""
        super().__init__(
            message=f"Value for key '{key}' is not JSON-serializable "
                    f"({self.value_type}){detail}",
            error_code="M002",
        )


class MemoryImmutabilityError(Exception):
    """Raised when attempting to mutate an immutable MemoryScope."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: MemoryScope is immutable after creation"
        )


class KeyNotFoundError(Exception):
    """Raised when a key is not found and no default is provided."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Key not found: '{key}'")


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_key(key: Any) -> str:
    """
    Validate that a key is a non-empty string.

    Raises:
        InvalidMemoryKeyError: If key is invalid

    Returns:
        The validated key
    """
    if not isinstance(key, str):
        raise InvalidMemoryKeyError(key)
    if not key.strip():
        raise InvalidMemoryKeyError(key)
    return key


def _validate_value(key: str, value: Any) -> Any:
    """
    Validate that a value is JSON-serializable.

    Allowed types: None, bool, int, float, str, list, dict
    Lists and dicts are validated recursively.

    Raises:
        InvalidMemoryValueError: If value is not serializable

    Returns:
        The validated value (deep-copied for safety)
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        # Check for special float values
        if isinstance(value, float):
            if value != value:  # NaN check
                raise InvalidMemoryValueError(key, value, "NaN is not allowed")
            if value == float('inf') or value == float('-inf'):
                raise InvalidMemoryValueError(key, value, "Infinity is not allowed")
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return [_validate_value(f"{key}[{i}]", v) for i, v in enumerate(value)]

    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise InvalidMemoryValueError(
                    key, value, f"dict keys must be strings, got {type(k).__name__}"
                )
            result[k] = _validate_value(f"{key}.{k}", v)
        return result

    # Not a valid type
    raise InvalidMemoryValueError(key, value)


def _deep_copy_value(value: Any) -> Any:
    """
    Deep copy a value using JSON roundtrip for safety.

    This ensures no external references can mutate internal state.
    """
    if value is None:
        return None
    return json.loads(json.dumps(value, sort_keys=True))


# =============================================================================
# MemoryDiff — Result of comparing two scopes
# =============================================================================

class MemoryDiff:
    """
    Represents the difference between two MemoryScopes.

    Immutable record of what was added, removed, or changed.
    """

    __slots__ = ('_added', '_removed', '_changed', '_frozen')

    def __init__(
        self,
        added: Dict[str, Any],
        removed: Dict[str, Any],
        changed: Dict[str, Tuple[Any, Any]],
    ):
        """
        Create a MemoryDiff.

        Args:
            added: Keys present in new but not in old
            removed: Keys present in old but not in new
            changed: Keys present in both with different values (old, new)
        """
        object.__setattr__(self, '_frozen', False)
        self._added = dict(sorted(added.items()))
        self._removed = dict(sorted(removed.items()))
        self._changed = dict(sorted(changed.items()))
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise MemoryImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    @property
    def added(self) -> Dict[str, Any]:
        """Keys and values present in new but not in old."""
        return _deep_copy_value(self._added)

    @property
    def removed(self) -> Dict[str, Any]:
        """Keys and values present in old but not in new."""
        return _deep_copy_value(self._removed)

    @property
    def changed(self) -> Dict[str, Tuple[Any, Any]]:
        """Keys with different values: {key: (old_value, new_value)}."""
        return {k: (v[0], v[1]) for k, v in self._changed.items()}

    @property
    def is_empty(self) -> bool:
        """True if there are no differences."""
        return not self._added and not self._removed and not self._changed

    def __bool__(self) -> bool:
        """True if there are differences."""
        return not self.is_empty

    def __repr__(self) -> str:
        return (
            f"MemoryDiff(added={len(self._added)}, "
            f"removed={len(self._removed)}, "
            f"changed={len(self._changed)})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "added": self._added,
            "changed": {k: {"old": v[0], "new": v[1]} for k, v in self._changed.items()},
            "removed": self._removed,
        }


# =============================================================================
# MemoryScope — Immutable scoped state container
# =============================================================================

class MemoryScope:
    """
    Immutable, scoped container of key-value state.

    A MemoryScope represents semantic ground truth — what is known or
    assumed at a specific point in a decision chain.

    Design:
    - Immutable after creation
    - Supports parent scope inheritance (child overrides parent)
    - Content-based scope_id for determinism
    - JSON-serializable values only

    Example:
        # Create root scope
        root = MemoryScope(values={"user": "alice", "mode": "read"})

        # Create child scope (inherits from root)
        child = root.set("mode", "write")  # Overrides parent

        # Lookup resolves child → parent
        assert child.get("user") == "alice"  # From parent
        assert child.get("mode") == "write"  # From child
    """

    __slots__ = ('_values', '_parent', '_scope_id', '_frozen')

    def __init__(
        self,
        values: Optional[Dict[str, Any]] = None,
        parent: Optional["MemoryScope"] = None,
    ):
        """
        Create a new MemoryScope.

        Args:
            values: Initial key-value pairs (validated and deep-copied)
            parent: Optional parent scope for inheritance

        Raises:
            InvalidMemoryKeyError: If any key is invalid
            InvalidMemoryValueError: If any value is not JSON-serializable
            MemoryValidationError: If parent is not a MemoryScope
        """
        object.__setattr__(self, '_frozen', False)

        # Validate parent
        if parent is not None and not isinstance(parent, MemoryScope):
            raise MemoryValidationError(
                f"parent must be MemoryScope or None, got {type(parent).__name__}",
                error_code="M003",
            )

        # Validate and copy values
        validated_values: Dict[str, Any] = {}
        if values is not None:
            if not isinstance(values, dict):
                raise MemoryValidationError(
                    f"values must be dict or None, got {type(values).__name__}",
                    error_code="M004",
                )
            for k, v in values.items():
                validated_key = _validate_key(k)
                validated_value = _validate_value(validated_key, v)
                validated_values[validated_key] = _deep_copy_value(validated_value)

        self._values = validated_values
        self._parent = parent
        self._scope_id = self._compute_scope_id()

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after construction."""
        if getattr(self, '_frozen', False):
            raise MemoryImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of attributes."""
        raise MemoryImmutabilityError(f"delete attribute '{name}'")

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def empty(cls, parent: Optional["MemoryScope"] = None) -> "MemoryScope":
        """
        Create an empty MemoryScope.

        Args:
            parent: Optional parent scope

        Returns:
            Empty MemoryScope instance
        """
        return cls(values=None, parent=parent)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        parent: Optional["MemoryScope"] = None,
    ) -> "MemoryScope":
        """
        Create a MemoryScope from a dictionary.

        Args:
            data: Dictionary with scope fields
            parent: Optional parent scope

        Returns:
            New MemoryScope instance
        """
        values = data.get("values", {})
        return cls(values=values, parent=parent)

    # =========================================================================
    # Core Operations
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value by key, with parent resolution.

        Lookup order: this scope → parent → parent's parent → ...

        Args:
            key: The key to look up
            default: Value to return if key not found (default: None)

        Returns:
            The value, or default if not found
        """
        _validate_key(key)

        # Check this scope first
        if key in self._values:
            return _deep_copy_value(self._values[key])

        # Check parent chain
        if self._parent is not None:
            return self._parent.get(key, default)

        return default

    def get_strict(self, key: str) -> Any:
        """
        Get a value by key, raising KeyNotFoundError if not found.

        Args:
            key: The key to look up

        Returns:
            The value

        Raises:
            KeyNotFoundError: If key is not found in this scope or any parent
        """
        _validate_key(key)

        if key in self._values:
            return _deep_copy_value(self._values[key])

        if self._parent is not None:
            return self._parent.get_strict(key)

        raise KeyNotFoundError(key)

    def has(self, key: str) -> bool:
        """
        Check if a key exists in this scope or any parent.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        _validate_key(key)

        if key in self._values:
            return True

        if self._parent is not None:
            return self._parent.has(key)

        return False

    def has_local(self, key: str) -> bool:
        """
        Check if a key exists in THIS scope only (not parent).

        Args:
            key: The key to check

        Returns:
            True if key exists in this scope, False otherwise
        """
        _validate_key(key)
        return key in self._values

    def set(self, key: str, value: Any) -> "MemoryScope":
        """
        Return a NEW MemoryScope with the key set to value.

        Does NOT mutate this scope. Creates a child scope with
        this scope as parent.

        Args:
            key: The key to set
            value: The value (must be JSON-serializable)

        Returns:
            New MemoryScope with the key set

        Raises:
            InvalidMemoryKeyError: If key is invalid
            InvalidMemoryValueError: If value is not JSON-serializable
        """
        _validate_key(key)
        validated_value = _validate_value(key, value)

        return MemoryScope(
            values={key: validated_value},
            parent=self,
        )

    def set_many(self, values: Dict[str, Any]) -> "MemoryScope":
        """
        Return a NEW MemoryScope with multiple keys set.

        Does NOT mutate this scope. Creates a child scope with
        this scope as parent.

        Args:
            values: Dictionary of key-value pairs to set

        Returns:
            New MemoryScope with all keys set
        """
        if not values:
            return self

        return MemoryScope(values=values, parent=self)

    def remove(self, key: str) -> "MemoryScope":
        """
        Return a NEW MemoryScope with the key removed.

        This creates a child scope where the key maps to a sentinel
        indicating deletion. The key will not be visible in the new
        scope even if it exists in a parent.

        Note: This is implemented by creating a scope with all visible
        keys EXCEPT the removed one, breaking the parent chain for that key.

        Args:
            key: The key to remove

        Returns:
            New MemoryScope without the key
        """
        _validate_key(key)

        # Collect all visible keys except the one to remove
        all_values = {}
        for k in self.keys():
            if k != key:
                all_values[k] = self.get(k)

        # Create new scope without parent (flat copy)
        return MemoryScope(values=all_values, parent=None)

    # =========================================================================
    # Diff
    # =========================================================================

    def diff(self, other: "MemoryScope") -> MemoryDiff:
        """
        Compute the difference between this scope and another.

        Returns what changed from self (old) to other (new).

        Args:
            other: The other MemoryScope to compare with

        Returns:
            MemoryDiff describing the changes
        """
        if not isinstance(other, MemoryScope):
            raise MemoryValidationError(
                f"Cannot diff with {type(other).__name__}, expected MemoryScope",
                error_code="M005",
            )

        self_keys = set(self.keys())
        other_keys = set(other.keys())

        added = {}
        removed = {}
        changed = {}

        # Keys only in other (added)
        for key in other_keys - self_keys:
            added[key] = other.get(key)

        # Keys only in self (removed)
        for key in self_keys - other_keys:
            removed[key] = self.get(key)

        # Keys in both (check for changes)
        for key in self_keys & other_keys:
            self_val = self.get(key)
            other_val = other.get(key)
            # Compare via JSON for deep equality
            if json.dumps(self_val, sort_keys=True) != json.dumps(other_val, sort_keys=True):
                changed[key] = (self_val, other_val)

        return MemoryDiff(added=added, removed=removed, changed=changed)

    # =========================================================================
    # Iteration
    # =========================================================================

    def keys(self) -> List[str]:
        """
        Return all visible keys (this scope + parent chain).

        Keys are returned in sorted order for determinism.

        Returns:
            Sorted list of all visible keys
        """
        all_keys: Set[str] = set()

        # Collect from parent first
        if self._parent is not None:
            all_keys.update(self._parent.keys())

        # Add/override with this scope's keys
        all_keys.update(self._values.keys())

        return sorted(all_keys)

    def local_keys(self) -> List[str]:
        """
        Return keys defined in THIS scope only (not parent).

        Returns:
            Sorted list of local keys
        """
        return sorted(self._values.keys())

    def values(self) -> List[Any]:
        """
        Return all visible values in key order.

        Returns:
            List of values for all visible keys
        """
        return [self.get(k) for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        """
        Return all visible key-value pairs.

        Returns:
            List of (key, value) tuples in sorted key order
        """
        return [(k, self.get(k)) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        """Iterate over visible keys."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Return number of visible keys."""
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        """Check if key is visible (supports 'in' operator)."""
        return self.has(key)

    def __getitem__(self, key: str) -> Any:
        """Get value by key (supports scope[key] syntax)."""
        return self.get_strict(key)

    # =========================================================================
    # Scope ID Computation
    # =========================================================================

    def _compute_scope_id(self) -> str:
        """
        Compute deterministic content-based hash.

        Uses SHA-256 of the JSON representation.
        Includes parent scope_id if present.
        """
        content = {
            "values": dict(sorted(self._values.items())),
        }
        if self._parent is not None:
            content["parent_scope_id"] = self._parent.scope_id

        json_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        hash_bytes = hashlib.sha256(json_str.encode('utf-8')).digest()

        return hash_bytes.hex()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def scope_id(self) -> str:
        """Deterministic content-based hash of this scope."""
        return self._scope_id

    @property
    def parent(self) -> Optional["MemoryScope"]:
        """Parent scope, or None if this is a root scope."""
        return self._parent

    @property
    def depth(self) -> int:
        """
        Depth of this scope in the parent chain.

        Root scope (no parent) has depth 0.
        """
        d = 0
        current = self._parent
        while current is not None:
            d += 1
            current = current._parent
        return d

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Keys are sorted for deterministic output.
        """
        result = {
            "scope_id": self._scope_id,
            "values": dict(sorted(self._values.items())),
        }

        if self._parent is not None:
            result["parent_scope_id"] = self._parent.scope_id

        return dict(sorted(result.items()))

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """
        Serialize to JSON string.

        Guarantees deterministic output.

        Args:
            indent: Optional indentation for pretty-printing

        Returns:
            JSON string representation
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Return all visible key-value pairs as a flat dictionary.

        Resolves parent inheritance. Useful for snapshots.

        Returns:
            Dictionary with all visible keys and their values
        """
        return {k: self.get(k) for k in self.keys()}

    @classmethod
    def from_json(
        cls,
        json_str: str,
        parent: Optional["MemoryScope"] = None,
    ) -> "MemoryScope":
        """
        Reconstruct a MemoryScope from a JSON string.

        Args:
            json_str: JSON string representation
            parent: Optional parent scope

        Returns:
            New MemoryScope instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data, parent=parent)

    # =========================================================================
    # Equality and Hashing
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        """Two scopes are equal if their scope_ids are identical."""
        if not isinstance(other, MemoryScope):
            return NotImplemented
        return self._scope_id == other._scope_id

    def __hash__(self) -> int:
        """Hash based on scope_id."""
        return hash(self._scope_id)

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        parent_info = f", parent={self._parent.scope_id[:8]}..." if self._parent else ""
        return f"MemoryScope(id={self._scope_id[:16]}..., keys={len(self)}{parent_info})"

    def __str__(self) -> str:
        """Human-readable representation."""
        local = len(self._values)
        inherited = len(self) - local
        lines = [f"MemoryScope ({local} local, {inherited} inherited):"]
        for k, v in self.items():
            source = ">" if k in self._values else " "
            lines.append(f"  {source} {k}: {v!r}")
        return "\n".join(lines)

    # =========================================================================
    # Chain Navigation
    # =========================================================================

    def chain(self) -> List["MemoryScope"]:
        """
        Return the full chain of scopes from root to this scope.

        Returns:
            List of scopes, oldest (root) first
        """
        chain = [self]
        current = self._parent
        while current is not None:
            chain.append(current)
            current = current._parent
        return list(reversed(chain))

    def root(self) -> "MemoryScope":
        """
        Return the root scope (topmost parent).

        Returns:
            The root scope (self if no parent)
        """
        current = self
        while current._parent is not None:
            current = current._parent
        return current
