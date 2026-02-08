"""
test_memory_scope.py

Unit tests for the Ntive MemoryScope primitive.

Tests prove:
- Inheritance (child → parent lookup)
- Immutability (cannot modify after creation)
- Deterministic hashing (same content = same scope_id)
- Diff correctness (added, removed, changed)
- Validation (invalid keys/values rejected)
"""

import pytest
import json
from ntive.memory import (
    MemoryScope,
    MemoryDiff,
    MemoryValidationError,
    InvalidMemoryKeyError,
    InvalidMemoryValueError,
    MemoryImmutabilityError,
    KeyNotFoundError,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def root_scope():
    """Root scope with some initial values."""
    return MemoryScope(values={
        "user": "alice",
        "mode": "read",
        "count": 42,
    })


@pytest.fixture
def child_scope(root_scope):
    """Child scope that overrides mode."""
    return MemoryScope(
        values={"mode": "write", "new_key": "new_value"},
        parent=root_scope,
    )


# =============================================================================
# SECTION 1: Inheritance Tests
# =============================================================================

class TestInheritance:
    """Test parent-child scope inheritance."""
    
    def test_child_inherits_parent_values(self, root_scope):
        """Child scope can access parent values."""
        child = MemoryScope(values={"new": "value"}, parent=root_scope)
        
        assert child.get("user") == "alice"  # From parent
        assert child.get("mode") == "read"   # From parent
        assert child.get("new") == "value"   # From child
    
    def test_child_overrides_parent_values(self, root_scope):
        """Child scope can override parent values."""
        child = MemoryScope(values={"mode": "write"}, parent=root_scope)
        
        assert child.get("mode") == "write"  # Child overrides
        assert root_scope.get("mode") == "read"  # Parent unchanged
    
    def test_deep_inheritance_chain(self):
        """Deep inheritance chains work correctly."""
        root = MemoryScope(values={"a": 1})
        level1 = MemoryScope(values={"b": 2}, parent=root)
        level2 = MemoryScope(values={"c": 3}, parent=level1)
        level3 = MemoryScope(values={"d": 4}, parent=level2)
        
        # All values accessible from deepest scope
        assert level3.get("a") == 1
        assert level3.get("b") == 2
        assert level3.get("c") == 3
        assert level3.get("d") == 4
    
    def test_override_at_different_levels(self):
        """Values can be overridden at different levels."""
        root = MemoryScope(values={"x": "root"})
        mid = MemoryScope(values={"x": "mid"}, parent=root)
        leaf = MemoryScope(values={"x": "leaf"}, parent=mid)
        
        assert root.get("x") == "root"
        assert mid.get("x") == "mid"
        assert leaf.get("x") == "leaf"
    
    def test_parent_is_read_only(self, root_scope, child_scope):
        """Setting on child doesn't affect parent."""
        new_child = child_scope.set("user", "bob")
        
        assert new_child.get("user") == "bob"
        assert child_scope.get("user") == "alice"  # Unchanged
        assert root_scope.get("user") == "alice"   # Unchanged
    
    def test_has_with_inheritance(self, child_scope):
        """has() checks both child and parent."""
        assert child_scope.has("mode")     # In child
        assert child_scope.has("user")     # In parent
        assert child_scope.has("new_key")  # In child
        assert not child_scope.has("nonexistent")
    
    def test_has_local_only_checks_this_scope(self, child_scope):
        """has_local() only checks this scope, not parent."""
        assert child_scope.has_local("mode")      # In child
        assert child_scope.has_local("new_key")   # In child
        assert not child_scope.has_local("user")  # In parent only
    
    def test_keys_includes_parent_keys(self, child_scope):
        """keys() returns all visible keys including parent."""
        keys = child_scope.keys()
        
        assert "user" in keys    # From parent
        assert "mode" in keys    # Overridden in child
        assert "count" in keys   # From parent
        assert "new_key" in keys # From child
    
    def test_local_keys_excludes_parent(self, child_scope):
        """local_keys() returns only this scope's keys."""
        local = child_scope.local_keys()
        
        assert "mode" in local
        assert "new_key" in local
        assert "user" not in local
        assert "count" not in local
    
    def test_items_resolves_inheritance(self, child_scope):
        """items() returns resolved key-value pairs."""
        items = dict(child_scope.items())
        
        assert items["user"] == "alice"      # From parent
        assert items["mode"] == "write"      # Child override
        assert items["new_key"] == "new_value"


class TestSetOperation:
    """Test the set() operation."""
    
    def test_set_returns_new_scope(self, root_scope):
        """set() returns a new scope, original unchanged."""
        original_id = root_scope.scope_id
        
        new_scope = root_scope.set("new_key", "new_value")
        
        assert new_scope is not root_scope
        assert new_scope.get("new_key") == "new_value"
        assert root_scope.scope_id == original_id  # Unchanged
        assert not root_scope.has("new_key")
    
    def test_set_creates_child_with_parent(self, root_scope):
        """set() creates a child scope with self as parent."""
        new_scope = root_scope.set("key", "value")
        
        assert new_scope.parent is root_scope
    
    def test_set_many_returns_new_scope(self, root_scope):
        """set_many() returns a new scope with multiple keys."""
        new_scope = root_scope.set_many({"a": 1, "b": 2, "c": 3})
        
        assert new_scope.get("a") == 1
        assert new_scope.get("b") == 2
        assert new_scope.get("c") == 3
        assert new_scope.parent is root_scope
    
    def test_set_many_empty_returns_self(self, root_scope):
        """set_many({}) returns self unchanged."""
        result = root_scope.set_many({})
        assert result is root_scope
    
    def test_chained_set_operations(self):
        """Multiple set() calls can be chained."""
        scope = (
            MemoryScope.empty()
            .set("a", 1)
            .set("b", 2)
            .set("c", 3)
        )
        
        assert scope.get("a") == 1
        assert scope.get("b") == 2
        assert scope.get("c") == 3
        assert scope.depth == 3  # 3 levels deep


class TestRemoveOperation:
    """Test the remove() operation."""
    
    def test_remove_key_creates_new_scope(self, root_scope):
        """remove() creates a new scope without the key."""
        new_scope = root_scope.remove("user")
        
        assert not new_scope.has("user")
        assert new_scope.has("mode")
        assert new_scope.has("count")
    
    def test_remove_preserves_other_keys(self, root_scope):
        """remove() preserves other keys."""
        new_scope = root_scope.remove("user")
        
        assert new_scope.get("mode") == "read"
        assert new_scope.get("count") == 42
    
    def test_remove_nonexistent_key_is_safe(self, root_scope):
        """remove() on nonexistent key is safe."""
        new_scope = root_scope.remove("nonexistent")
        
        assert len(new_scope) == len(root_scope)
    
    def test_remove_inherited_key(self, child_scope):
        """remove() works on inherited keys."""
        new_scope = child_scope.remove("user")  # Inherited from parent
        
        assert not new_scope.has("user")
        assert new_scope.has("mode")


# =============================================================================
# SECTION 2: Immutability Tests
# =============================================================================

class TestImmutability:
    """Test that MemoryScope is immutable after creation."""
    
    def test_cannot_set_attribute(self, root_scope):
        """Setting any attribute raises MemoryImmutabilityError."""
        with pytest.raises(MemoryImmutabilityError):
            root_scope._values = {}
    
    def test_cannot_delete_attribute(self, root_scope):
        """Deleting any attribute raises MemoryImmutabilityError."""
        with pytest.raises(MemoryImmutabilityError):
            del root_scope._scope_id
    
    def test_cannot_add_attribute(self, root_scope):
        """Adding new attribute raises MemoryImmutabilityError."""
        with pytest.raises(MemoryImmutabilityError):
            root_scope.new_field = "value"
    
    def test_get_returns_deep_copy(self, root_scope):
        """get() returns a copy, modifying it doesn't affect scope."""
        scope = MemoryScope(values={"list": [1, 2, 3]})
        
        retrieved = scope.get("list")
        retrieved.append(4)
        
        # Original should be unchanged
        assert scope.get("list") == [1, 2, 3]
    
    def test_original_dict_modification_doesnt_affect_scope(self):
        """Modifying the original dict doesn't affect scope."""
        original = {"key": "value"}
        scope = MemoryScope(values=original)
        
        original["key"] = "changed"
        original["new"] = "added"
        
        assert scope.get("key") == "value"
        assert not scope.has("new")
    
    def test_memory_diff_is_immutable(self):
        """MemoryDiff is immutable."""
        diff = MemoryDiff(
            added={"a": 1},
            removed={"b": 2},
            changed={"c": (1, 2)},
        )
        
        with pytest.raises(MemoryImmutabilityError):
            diff.added = {}


# =============================================================================
# SECTION 3: Deterministic Hashing Tests
# =============================================================================

class TestDeterministicHashing:
    """Test that scope_id is content-based and deterministic."""
    
    def test_identical_values_produce_identical_scope_id(self):
        """Same values = same scope_id."""
        scope1 = MemoryScope(values={"a": 1, "b": 2})
        scope2 = MemoryScope(values={"a": 1, "b": 2})
        
        assert scope1.scope_id == scope2.scope_id
    
    def test_value_order_doesnt_affect_scope_id(self):
        """Key order doesn't affect scope_id."""
        scope1 = MemoryScope(values={"z": 1, "a": 2, "m": 3})
        scope2 = MemoryScope(values={"a": 2, "m": 3, "z": 1})
        
        assert scope1.scope_id == scope2.scope_id
    
    def test_different_values_produce_different_scope_id(self):
        """Different values = different scope_id."""
        scope1 = MemoryScope(values={"a": 1})
        scope2 = MemoryScope(values={"a": 2})
        
        assert scope1.scope_id != scope2.scope_id
    
    def test_different_parent_produces_different_scope_id(self):
        """Different parent = different scope_id."""
        parent1 = MemoryScope(values={"p": 1})
        parent2 = MemoryScope(values={"p": 2})
        
        child1 = MemoryScope(values={"c": 1}, parent=parent1)
        child2 = MemoryScope(values={"c": 1}, parent=parent2)
        
        assert child1.scope_id != child2.scope_id
    
    def test_parent_vs_no_parent_different_scope_id(self):
        """Having parent vs no parent = different scope_id."""
        parent = MemoryScope(values={"p": 1})
        
        with_parent = MemoryScope(values={"c": 1}, parent=parent)
        without_parent = MemoryScope(values={"c": 1}, parent=None)
        
        assert with_parent.scope_id != without_parent.scope_id
    
    def test_scope_id_is_hex_string(self):
        """scope_id is a hex string (SHA-256)."""
        scope = MemoryScope(values={"a": 1})
        
        assert len(scope.scope_id) == 64
        assert all(c in '0123456789abcdef' for c in scope.scope_id)
    
    def test_json_is_deterministic(self, root_scope):
        """Multiple to_json() calls return identical strings."""
        json1 = root_scope.to_json()
        json2 = root_scope.to_json()
        json3 = root_scope.to_json()
        
        assert json1 == json2 == json3
    
    def test_json_keys_are_sorted(self, root_scope):
        """JSON output has sorted keys."""
        json_str = root_scope.to_json()
        data = json.loads(json_str)
        keys = list(data.keys())
        
        assert keys == sorted(keys)
    
    def test_roundtrip_preserves_scope_id(self, root_scope):
        """from_json(to_json()) produces same scope_id."""
        json_str = root_scope.to_json()
        restored = MemoryScope.from_json(json_str)
        
        # Note: parent is not serialized, so we compare values only
        assert restored.scope_id == MemoryScope(values=root_scope.to_flat_dict()).scope_id
    
    def test_empty_scopes_have_same_id(self):
        """Empty scopes have identical scope_id."""
        scope1 = MemoryScope.empty()
        scope2 = MemoryScope.empty()
        
        assert scope1.scope_id == scope2.scope_id


# =============================================================================
# SECTION 4: Diff Correctness Tests
# =============================================================================

class TestDiff:
    """Test MemoryDiff computation."""
    
    def test_diff_detects_added_keys(self):
        """diff() detects added keys."""
        old = MemoryScope(values={"a": 1})
        new = MemoryScope(values={"a": 1, "b": 2})
        
        diff = old.diff(new)
        
        assert diff.added == {"b": 2}
        assert diff.removed == {}
        assert diff.changed == {}
    
    def test_diff_detects_removed_keys(self):
        """diff() detects removed keys."""
        old = MemoryScope(values={"a": 1, "b": 2})
        new = MemoryScope(values={"a": 1})
        
        diff = old.diff(new)
        
        assert diff.added == {}
        assert diff.removed == {"b": 2}
        assert diff.changed == {}
    
    def test_diff_detects_changed_values(self):
        """diff() detects changed values."""
        old = MemoryScope(values={"a": 1})
        new = MemoryScope(values={"a": 2})
        
        diff = old.diff(new)
        
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {"a": (1, 2)}
    
    def test_diff_complex_changes(self):
        """diff() handles complex changes."""
        old = MemoryScope(values={"a": 1, "b": 2, "c": 3})
        new = MemoryScope(values={"a": 100, "c": 3, "d": 4})
        
        diff = old.diff(new)
        
        assert diff.added == {"d": 4}
        assert diff.removed == {"b": 2}
        assert diff.changed == {"a": (1, 100)}
    
    def test_diff_identical_scopes(self):
        """diff() between identical scopes is empty."""
        scope1 = MemoryScope(values={"a": 1, "b": 2})
        scope2 = MemoryScope(values={"a": 1, "b": 2})
        
        diff = scope1.diff(scope2)
        
        assert diff.is_empty
        assert not diff  # Boolean is False
    
    def test_diff_non_empty_is_truthy(self):
        """Non-empty diff is truthy."""
        old = MemoryScope(values={"a": 1})
        new = MemoryScope(values={"a": 2})
        
        diff = old.diff(new)
        
        assert not diff.is_empty
        assert diff  # Boolean is True
    
    def test_diff_deep_equality(self):
        """diff() uses deep equality for nested values."""
        old = MemoryScope(values={"data": {"x": 1, "y": 2}})
        new = MemoryScope(values={"data": {"x": 1, "y": 2}})
        
        diff = old.diff(new)
        
        assert diff.is_empty
    
    def test_diff_detects_nested_changes(self):
        """diff() detects changes in nested values."""
        old = MemoryScope(values={"data": {"x": 1}})
        new = MemoryScope(values={"data": {"x": 2}})
        
        diff = old.diff(new)
        
        assert "data" in diff.changed
    
    def test_diff_with_inheritance(self, root_scope, child_scope):
        """diff() considers inherited values."""
        # child_scope has: user=alice (inherited), mode=write (override), 
        # count=42 (inherited), new_key=new_value (local)
        
        other = MemoryScope(values={
            "user": "bob",  # Changed from alice
            "mode": "write",  # Same
            "count": 42,  # Same
            # new_key removed
        })
        
        diff = child_scope.diff(other)
        
        assert diff.added == {}
        assert diff.removed == {"new_key": "new_value"}
        assert diff.changed == {"user": ("alice", "bob")}
    
    def test_diff_to_dict(self):
        """diff().to_dict() produces serializable output."""
        old = MemoryScope(values={"a": 1, "b": 2})
        new = MemoryScope(values={"a": 100, "c": 3})
        
        diff = old.diff(new)
        data = diff.to_dict()
        
        assert "added" in data
        assert "removed" in data
        assert "changed" in data
        
        # Ensure serializable
        json.dumps(data)


# =============================================================================
# SECTION 5: Validation Tests
# =============================================================================

class TestValidation:
    """Test key and value validation."""
    
    def test_non_string_key_rejected(self):
        """Non-string keys are rejected."""
        with pytest.raises(InvalidMemoryKeyError) as exc_info:
            MemoryScope(values={123: "value"})
        
        assert exc_info.value.key == 123
        assert "M001" in str(exc_info.value)
    
    def test_empty_string_key_rejected(self):
        """Empty string keys are rejected."""
        with pytest.raises(InvalidMemoryKeyError):
            MemoryScope(values={"": "value"})
    
    def test_whitespace_only_key_rejected(self):
        """Whitespace-only keys are rejected."""
        with pytest.raises(InvalidMemoryKeyError):
            MemoryScope(values={"   ": "value"})
    
    def test_function_value_rejected(self):
        """Function values are rejected."""
        with pytest.raises(InvalidMemoryValueError) as exc_info:
            MemoryScope(values={"func": lambda x: x})
        
        assert exc_info.value.key == "func"
        assert "M002" in str(exc_info.value)
    
    def test_nan_value_rejected(self):
        """NaN values are rejected."""
        with pytest.raises(InvalidMemoryValueError):
            MemoryScope(values={"num": float('nan')})
    
    def test_infinity_value_rejected(self):
        """Infinity values are rejected."""
        with pytest.raises(InvalidMemoryValueError):
            MemoryScope(values={"num": float('inf')})
    
    def test_nested_dict_with_non_string_key_rejected(self):
        """Nested dicts with non-string keys are rejected."""
        with pytest.raises(InvalidMemoryValueError):
            MemoryScope(values={"data": {123: "value"}})
    
    def test_valid_primitive_types(self):
        """Valid primitive types are accepted."""
        scope = MemoryScope(values={
            "none": None,
            "bool_true": True,
            "bool_false": False,
            "int": 42,
            "float": 3.14,
            "string": "hello",
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        })
        
        assert scope.get("none") is None
        assert scope.get("bool_true") is True
        assert scope.get("int") == 42
        assert scope.get("float") == 3.14
    
    def test_deeply_nested_values(self):
        """Deeply nested values are validated."""
        deep = {"level1": {"level2": {"level3": [1, 2, {"level4": "deep"}]}}}
        scope = MemoryScope(values={"deep": deep})
        
        assert scope.get("deep")["level1"]["level2"]["level3"][2]["level4"] == "deep"
    
    def test_invalid_parent_type_rejected(self):
        """Non-MemoryScope parent is rejected."""
        with pytest.raises(MemoryValidationError) as exc_info:
            MemoryScope(values={}, parent="not a scope")
        
        assert "M003" in str(exc_info.value)
    
    def test_invalid_values_type_rejected(self):
        """Non-dict values is rejected."""
        with pytest.raises(MemoryValidationError) as exc_info:
            MemoryScope(values=["not", "a", "dict"])
        
        assert "M004" in str(exc_info.value)


class TestKeyNotFound:
    """Test KeyNotFoundError behavior."""
    
    def test_get_strict_raises_on_missing_key(self, root_scope):
        """get_strict() raises KeyNotFoundError for missing keys."""
        with pytest.raises(KeyNotFoundError) as exc_info:
            root_scope.get_strict("nonexistent")
        
        assert exc_info.value.key == "nonexistent"
    
    def test_get_returns_default_on_missing_key(self, root_scope):
        """get() returns default for missing keys."""
        result = root_scope.get("nonexistent", "default")
        assert result == "default"
    
    def test_get_returns_none_by_default(self, root_scope):
        """get() returns None by default for missing keys."""
        result = root_scope.get("nonexistent")
        assert result is None
    
    def test_getitem_raises_on_missing_key(self, root_scope):
        """scope[key] raises KeyNotFoundError for missing keys."""
        with pytest.raises(KeyNotFoundError):
            _ = root_scope["nonexistent"]


# =============================================================================
# SECTION 6: Iteration and Container Tests
# =============================================================================

class TestIterationAndContainer:
    """Test iteration and container protocol."""
    
    def test_len_returns_visible_key_count(self, child_scope):
        """len() returns count of all visible keys."""
        # child has: user, mode, count (from parent) + new_key (local)
        # mode is overridden, so 4 unique keys
        assert len(child_scope) == 4
    
    def test_iter_yields_keys(self, root_scope):
        """Iterating yields keys."""
        keys = list(root_scope)
        assert set(keys) == {"user", "mode", "count"}
    
    def test_in_operator(self, root_scope):
        """'in' operator works."""
        assert "user" in root_scope
        assert "nonexistent" not in root_scope
    
    def test_getitem(self, root_scope):
        """scope[key] works."""
        assert root_scope["user"] == "alice"
        assert root_scope["count"] == 42
    
    def test_keys_are_sorted(self, root_scope):
        """keys() returns sorted list."""
        keys = root_scope.keys()
        assert keys == sorted(keys)
    
    def test_values_in_key_order(self, root_scope):
        """values() returns values in key order."""
        keys = root_scope.keys()
        values = root_scope.values()
        
        for k, v in zip(keys, values):
            assert root_scope.get(k) == v
    
    def test_items_as_tuples(self, root_scope):
        """items() returns (key, value) tuples."""
        items = root_scope.items()
        
        for k, v in items:
            assert root_scope.get(k) == v


# =============================================================================
# SECTION 7: Chain Navigation Tests
# =============================================================================

class TestChainNavigation:
    """Test chain and navigation methods."""
    
    def test_chain_returns_all_ancestors(self):
        """chain() returns all scopes from root to self."""
        root = MemoryScope(values={"level": 0})
        level1 = MemoryScope(values={"level": 1}, parent=root)
        level2 = MemoryScope(values={"level": 2}, parent=level1)
        
        chain = level2.chain()
        
        assert len(chain) == 3
        assert chain[0] is root
        assert chain[1] is level1
        assert chain[2] is level2
    
    def test_chain_single_scope(self):
        """chain() on root returns just itself."""
        scope = MemoryScope(values={"a": 1})
        assert scope.chain() == [scope]
    
    def test_root_returns_topmost_parent(self):
        """root() returns the root scope."""
        root = MemoryScope(values={"level": 0})
        level1 = MemoryScope(values={"level": 1}, parent=root)
        level2 = MemoryScope(values={"level": 2}, parent=level1)
        
        assert level2.root() is root
        assert level1.root() is root
        assert root.root() is root
    
    def test_depth_property(self):
        """depth property returns correct depth."""
        root = MemoryScope(values={"level": 0})
        level1 = MemoryScope(values={"level": 1}, parent=root)
        level2 = MemoryScope(values={"level": 2}, parent=level1)
        
        assert root.depth == 0
        assert level1.depth == 1
        assert level2.depth == 2
    
    def test_parent_property(self, root_scope, child_scope):
        """parent property returns parent scope."""
        assert child_scope.parent is root_scope
        assert root_scope.parent is None


# =============================================================================
# SECTION 8: Serialization Tests
# =============================================================================

class TestSerialization:
    """Test serialization and deserialization."""
    
    def test_to_dict_includes_scope_id(self, root_scope):
        """to_dict() includes scope_id."""
        data = root_scope.to_dict()
        assert "scope_id" in data
        assert data["scope_id"] == root_scope.scope_id
    
    def test_to_dict_includes_values(self, root_scope):
        """to_dict() includes values."""
        data = root_scope.to_dict()
        assert "values" in data
    
    def test_to_dict_includes_parent_id_when_present(self, child_scope):
        """to_dict() includes parent_scope_id when parent exists."""
        data = child_scope.to_dict()
        assert "parent_scope_id" in data
    
    def test_to_dict_no_parent_id_when_absent(self, root_scope):
        """to_dict() omits parent_scope_id when no parent."""
        data = root_scope.to_dict()
        assert "parent_scope_id" not in data
    
    def test_to_json_with_indent(self, root_scope):
        """to_json() accepts indent parameter."""
        json_str = root_scope.to_json(indent=2)
        assert "\n" in json_str
        assert "  " in json_str
    
    def test_to_flat_dict(self, child_scope):
        """to_flat_dict() returns all visible keys."""
        flat = child_scope.to_flat_dict()
        
        assert flat["user"] == "alice"      # Inherited
        assert flat["mode"] == "write"      # Overridden
        assert flat["count"] == 42          # Inherited
        assert flat["new_key"] == "new_value"  # Local
    
    def test_from_dict(self):
        """from_dict() reconstructs scope."""
        data = {"values": {"a": 1, "b": 2}}
        scope = MemoryScope.from_dict(data)
        
        assert scope.get("a") == 1
        assert scope.get("b") == 2
    
    def test_from_json(self):
        """from_json() reconstructs scope."""
        json_str = '{"values": {"x": 100}}'
        scope = MemoryScope.from_json(json_str)
        
        assert scope.get("x") == 100


# =============================================================================
# SECTION 9: Equality and Hashing Tests
# =============================================================================

class TestEqualityAndHashing:
    """Test equality and hash behavior."""
    
    def test_equal_scopes_are_equal(self):
        """Scopes with same content are equal."""
        scope1 = MemoryScope(values={"a": 1, "b": 2})
        scope2 = MemoryScope(values={"a": 1, "b": 2})
        
        assert scope1 == scope2
    
    def test_different_scopes_not_equal(self):
        """Scopes with different content are not equal."""
        scope1 = MemoryScope(values={"a": 1})
        scope2 = MemoryScope(values={"a": 2})
        
        assert scope1 != scope2
    
    def test_scope_is_hashable(self, root_scope):
        """MemoryScope can be used in sets."""
        s = {root_scope}
        assert root_scope in s
    
    def test_equal_scopes_have_same_hash(self):
        """Equal scopes have the same hash."""
        scope1 = MemoryScope(values={"a": 1})
        scope2 = MemoryScope(values={"a": 1})
        
        assert hash(scope1) == hash(scope2)
    
    def test_equal_scopes_deduplicate_in_set(self):
        """Equal scopes deduplicate in sets."""
        scope1 = MemoryScope(values={"a": 1})
        scope2 = MemoryScope(values={"a": 1})
        
        s = {scope1, scope2}
        assert len(s) == 1


# =============================================================================
# SECTION 10: String Representation Tests
# =============================================================================

class TestStringRepresentations:
    """Test __repr__ and __str__."""
    
    def test_repr_includes_scope_id(self, root_scope):
        """__repr__ includes scope_id prefix."""
        r = repr(root_scope)
        assert "MemoryScope(" in r
        assert root_scope.scope_id[:16] in r
    
    def test_repr_includes_key_count(self, root_scope):
        """__repr__ includes key count."""
        r = repr(root_scope)
        assert f"keys={len(root_scope)}" in r
    
    def test_repr_includes_parent_when_present(self, child_scope):
        """__repr__ includes parent info when present."""
        r = repr(child_scope)
        assert "parent=" in r
    
    def test_str_is_human_readable(self, root_scope):
        """__str__ is human-readable."""
        s = str(root_scope)
        assert "MemoryScope" in s
        assert "local" in s


# =============================================================================
# SECTION 11: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_scope(self):
        """Empty scope works correctly."""
        scope = MemoryScope.empty()
        
        assert len(scope) == 0
        assert scope.keys() == []
        assert scope.get("anything") is None
    
    def test_unicode_keys_and_values(self):
        """Unicode keys and values work."""
        scope = MemoryScope(values={
            "日本語": "Japanese",
            "emoji": "🎉",
            "中文": {"nested": "嵌套"},
        })
        
        assert scope.get("日本語") == "Japanese"
        assert scope.get("emoji") == "🎉"
    
    def test_very_long_values(self):
        """Very long string values work."""
        long_string = "x" * 100000
        scope = MemoryScope(values={"long": long_string})
        
        assert len(scope.get("long")) == 100000
    
    def test_deeply_nested_parent_chain(self):
        """Deep parent chain works."""
        scope = MemoryScope(values={"level": 0})
        
        for i in range(1, 100):
            scope = MemoryScope(values={"level": i}, parent=scope)
        
        assert scope.depth == 99
        assert scope.get("level") == 99
    
    def test_many_keys(self):
        """Many keys work."""
        values = {f"key_{i:04d}": i for i in range(1000)}
        scope = MemoryScope(values=values)
        
        assert len(scope) == 1000
        assert scope.get("key_0500") == 500
    
    def test_diff_with_non_scope_raises_error(self, root_scope):
        """diff() with non-MemoryScope raises error."""
        with pytest.raises(MemoryValidationError):
            root_scope.diff({"not": "a scope"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
