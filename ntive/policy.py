"""
policy.py

Ntive Policy Primitive — Semantic Decision Specification v1.0.0

A Policy is a declarative modifier of decision behavior.
It exists ONLY to answer: "Under which rules was this decision allowed or forbidden?"

Design Invariants:
- Immutable after creation
- Pure data structure (no side effects)
- Deterministic serialization (sorted keys, content-based hash)
- No evaluation engine
- No runtime condition execution
- No Decision or Memory access
- Policies constrain possibility space; they never choose
"""

import hashlib
import json
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# =============================================================================
# Enums
# =============================================================================

class PolicyEffect(Enum):
    """
    Effect types for policy rules.

    - REQUIRE: The target must be selected
    - FORBID: The target must not be selected
    - PREFER: The target should be preferred (soft constraint)
    - DEFAULT: Use this target if no other applies
    """
    REQUIRE = "require"
    FORBID = "forbid"
    PREFER = "prefer"
    DEFAULT = "default"

    @classmethod
    def from_string(cls, value: str) -> "PolicyEffect":
        """Convert string to PolicyEffect."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [e.value for e in cls]
            raise InvalidPolicyEffectError(value, valid)


class ConflictResolutionStrategy(Enum):
    """
    Strategies for resolving conflicting policy rules.

    - MOST_RESTRICTIVE: Forbid > Require > Prefer > Default
    - EXPLICIT_PRIORITY: Higher priority wins
    - REJECT_CONFLICT: Raise error on conflict
    """
    MOST_RESTRICTIVE = "most_restrictive"
    EXPLICIT_PRIORITY = "explicit_priority"
    REJECT_CONFLICT = "reject_conflict"

    @classmethod
    def from_string(cls, value: str) -> "ConflictResolutionStrategy":
        """Convert string to ConflictResolutionStrategy."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [s.value for s in cls]
            raise PolicyValidationError(
                f"Invalid conflict resolution strategy: {value!r}. "
                f"Valid strategies: {valid}",
                error_code="P006",
            )


# =============================================================================
# Policy Errors
# =============================================================================

class PolicyValidationError(Exception):
    """
    Raised when a Policy cannot be constructed due to validation failure.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "P000",
    ):
        self.message = message
        self.error_code = error_code
        super().__init__(self.format())

    def format(self) -> str:
        """Format as human-readable error message."""
        return f"[{self.error_code}] Policy validation failed: {self.message}"


class InvalidPolicyRuleError(PolicyValidationError):
    """Raised when a PolicyRule structure is invalid."""

    def __init__(self, rule_id: str, reason: str):
        self.rule_id = rule_id
        self.reason = reason
        super().__init__(
            message=f"Invalid rule '{rule_id}': {reason}",
            error_code="P001",
        )


class DuplicateRuleIdError(PolicyValidationError):
    """Raised when multiple rules have the same ID."""

    def __init__(self, rule_id: str):
        self.rule_id = rule_id
        super().__init__(
            message=f"Duplicate rule ID: '{rule_id}'",
            error_code="P002",
        )


class InvalidPolicyEffectError(PolicyValidationError):
    """Raised when an invalid effect type is used."""

    def __init__(self, effect: str, valid_effects: List[str]):
        self.effect = effect
        self.valid_effects = valid_effects
        super().__init__(
            message=f"Invalid effect type: '{effect}'. "
                    f"Valid effects: {valid_effects}",
            error_code="P003",
        )


class CyclicPolicyInheritanceError(PolicyValidationError):
    """Raised when policy inheritance creates a cycle."""

    def __init__(self, cycle_path: List[str]):
        self.cycle_path = cycle_path
        path_str = " -> ".join(cycle_path)
        super().__init__(
            message=f"Cyclic policy inheritance detected: {path_str}",
            error_code="P004",
        )


class PolicyConflictError(PolicyValidationError):
    """Raised when conflicting rules exist and strategy is REJECT_CONFLICT."""

    def __init__(self, target: str, conflicting_effects: List[str]):
        self.target = target
        self.conflicting_effects = conflicting_effects
        super().__init__(
            message=f"Conflicting rules for target '{target}': {conflicting_effects}",
            error_code="P005",
        )


class PolicyImmutabilityError(Exception):
    """Raised when attempting to mutate an immutable Policy."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: Policy is immutable after creation"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_string(value: Any, field_name: str) -> str:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str):
        raise PolicyValidationError(
            f"{field_name} must be a string, got {type(value).__name__}",
            error_code="P007",
        )
    if not value.strip():
        raise PolicyValidationError(
            f"{field_name} cannot be empty or whitespace-only",
            error_code="P007",
        )
    return value


def _validate_condition(condition: Any) -> Dict[str, Any]:
    """
    Validate that a condition is a JSON-serializable dict.
    Conditions are opaque predicates — we don't execute them.
    """
    if condition is None:
        return {}

    if not isinstance(condition, dict):
        raise PolicyValidationError(
            f"Condition must be a dict, got {type(condition).__name__}",
            error_code="P008",
        )

    # Verify JSON-serializable by attempting to serialize
    try:
        json.dumps(condition, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise PolicyValidationError(
            f"Condition must be JSON-serializable: {e}",
            error_code="P008",
        )

    return condition


def _deep_copy_json(value: Any) -> Any:
    """Create a deep copy of a JSON-serializable value."""
    return json.loads(json.dumps(value, sort_keys=True))


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of JSON-serialized data."""
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# =============================================================================
# PolicyRule
# =============================================================================

class PolicyRule:
    """
    A single rule within a Policy.

    Attributes:
        rule_id: Unique identifier within the policy
        condition: Opaque predicate dict (not executed)
        effect: One of require, forbid, prefer, default
        target: String identifier of what this rule affects
        weight: Optional numeric weight for prefer/default
    """

    __slots__ = ('_rule_id', '_condition', '_effect', '_target', '_weight', '_frozen')

    def __init__(
        self,
        *,
        rule_id: str,
        effect: str | PolicyEffect,
        target: str,
        condition: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = None,
    ):
        # Parse effect
        if isinstance(effect, str):
            effect_enum = PolicyEffect.from_string(effect)
        elif isinstance(effect, PolicyEffect):
            effect_enum = effect
        else:
            raise PolicyValidationError(
                f"effect must be string or PolicyEffect, got {type(effect).__name__}",
                error_code="P003",
            )

        # Validate rule_id
        rule_id = _validate_string(rule_id, "rule_id")

        # Validate target
        target = _validate_string(target, "target")

        # Validate condition
        validated_condition = _validate_condition(condition)

        # Validate weight
        if weight is not None:
            if not isinstance(weight, (int, float)):
                raise InvalidPolicyRuleError(
                    rule_id,
                    f"weight must be numeric, got {type(weight).__name__}",
                )
            if not (-1e10 < weight < 1e10):
                raise InvalidPolicyRuleError(
                    rule_id,
                    f"weight must be finite, got {weight}",
                )
            # Reject NaN and Infinity
            if weight != weight or weight == float('inf') or weight == float('-inf'):
                raise InvalidPolicyRuleError(
                    rule_id,
                    f"weight must be finite, got {weight}",
                )

        object.__setattr__(self, '_rule_id', rule_id)
        object.__setattr__(self, '_condition', _deep_copy_json(validated_condition))
        object.__setattr__(self, '_effect', effect_enum)
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_weight', weight)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise PolicyImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise PolicyImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def rule_id(self) -> str:
        return self._rule_id

    @property
    def condition(self) -> Dict[str, Any]:
        """Return a copy of the condition dict."""
        return _deep_copy_json(self._condition)

    @property
    def effect(self) -> PolicyEffect:
        return self._effect

    @property
    def target(self) -> str:
        return self._target

    @property
    def weight(self) -> Optional[float]:
        return self._weight

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolicyRule):
            return NotImplemented
        return (
            self._rule_id == other._rule_id
            and self._condition == other._condition
            and self._effect == other._effect
            and self._target == other._target
            and self._weight == other._weight
        )

    def __hash__(self) -> int:
        return hash((
            self._rule_id,
            json.dumps(self._condition, sort_keys=True),
            self._effect,
            self._target,
            self._weight,
        ))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "condition": self._condition,
            "effect": self._effect.value,
            "rule_id": self._rule_id,
            "target": self._target,
        }
        if self._weight is not None:
            result["weight"] = self._weight
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRule":
        """Construct from a dictionary."""
        return cls(
            rule_id=data["rule_id"],
            effect=data["effect"],
            target=data["target"],
            condition=data.get("condition"),
            weight=data.get("weight"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PolicyRule":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        weight_str = f", weight={self._weight}" if self._weight is not None else ""
        return (
            f"PolicyRule(rule_id={self._rule_id!r}, "
            f"effect={self._effect.value!r}, "
            f"target={self._target!r}{weight_str})"
        )

    def __str__(self) -> str:
        return f"{self._effect.value.upper()} {self._target}"


# =============================================================================
# PolicyEffectResult
# =============================================================================

class PolicyEffectResult:
    """
    Result of a policy rule application.

    This is a record of what happened when a rule was considered.
    It does NOT evaluate — it only records.

    Attributes:
        policy_id: ID of the policy containing the rule
        rule_id: ID of the rule
        effect: The effect type
        applied: Whether this rule was applied
        reason: Human-readable explanation
    """

    __slots__ = ('_policy_id', '_rule_id', '_effect', '_applied', '_reason', '_frozen')

    def __init__(
        self,
        *,
        policy_id: str,
        rule_id: str,
        effect: str | PolicyEffect,
        applied: bool,
        reason: str,
    ):
        # Validate policy_id
        policy_id = _validate_string(policy_id, "policy_id")

        # Validate rule_id
        rule_id = _validate_string(rule_id, "rule_id")

        # Parse effect
        if isinstance(effect, str):
            effect_enum = PolicyEffect.from_string(effect)
        elif isinstance(effect, PolicyEffect):
            effect_enum = effect
        else:
            raise PolicyValidationError(
                f"effect must be string or PolicyEffect, got {type(effect).__name__}",
                error_code="P003",
            )

        # Validate applied
        if not isinstance(applied, bool):
            raise PolicyValidationError(
                f"applied must be bool, got {type(applied).__name__}",
                error_code="P009",
            )

        # Validate reason
        if not isinstance(reason, str):
            raise PolicyValidationError(
                f"reason must be string, got {type(reason).__name__}",
                error_code="P009",
            )

        object.__setattr__(self, '_policy_id', policy_id)
        object.__setattr__(self, '_rule_id', rule_id)
        object.__setattr__(self, '_effect', effect_enum)
        object.__setattr__(self, '_applied', applied)
        object.__setattr__(self, '_reason', reason)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise PolicyImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise PolicyImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def rule_id(self) -> str:
        return self._rule_id

    @property
    def effect(self) -> PolicyEffect:
        return self._effect

    @property
    def applied(self) -> bool:
        return self._applied

    @property
    def reason(self) -> str:
        return self._reason

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolicyEffectResult):
            return NotImplemented
        return (
            self._policy_id == other._policy_id
            and self._rule_id == other._rule_id
            and self._effect == other._effect
            and self._applied == other._applied
            and self._reason == other._reason
        )

    def __hash__(self) -> int:
        return hash((
            self._policy_id,
            self._rule_id,
            self._effect,
            self._applied,
            self._reason,
        ))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "applied": self._applied,
            "effect": self._effect.value,
            "policy_id": self._policy_id,
            "reason": self._reason,
            "rule_id": self._rule_id,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyEffectResult":
        """Construct from a dictionary."""
        return cls(
            policy_id=data["policy_id"],
            rule_id=data["rule_id"],
            effect=data["effect"],
            applied=data["applied"],
            reason=data["reason"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PolicyEffectResult":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        applied_str = "applied" if self._applied else "not applied"
        return (
            f"PolicyEffectResult(policy={self._policy_id!r}, "
            f"rule={self._rule_id!r}, "
            f"effect={self._effect.value!r}, {applied_str})"
        )

    def __str__(self) -> str:
        status = "APPLIED" if self._applied else "SKIPPED"
        return f"[{status}] {self._effect.value.upper()} (rule: {self._rule_id})"


# =============================================================================
# Policy
# =============================================================================

class Policy:
    """
    A declarative policy that constrains decision behavior.

    Policies are DATA, not code. They do not evaluate conditions —
    they only define what rules exist and how conflicts are resolved.

    Attributes:
        policy_id: Content-based hash (computed, not user-provided)
        name: Human-readable name
        version: Semantic version string
        rules: List of PolicyRule objects
        priority: Numeric priority for conflict resolution
        parent: Optional parent policy for inheritance
        conflict_strategy: How to resolve conflicting rules
    """

    __slots__ = (
        '_policy_id',
        '_name',
        '_version',
        '_rules',
        '_rules_by_id',
        '_priority',
        '_parent',
        '_conflict_strategy',
        '_frozen',
    )

    def __init__(
        self,
        *,
        name: str,
        version: str = "1.0.0",
        rules: Optional[List[PolicyRule | Dict[str, Any]]] = None,
        priority: int = 0,
        parent: Optional["Policy"] = None,
        conflict_strategy: str | ConflictResolutionStrategy = ConflictResolutionStrategy.MOST_RESTRICTIVE,
    ):
        # Validate name
        name = _validate_string(name, "name")

        # Validate version
        version = _validate_string(version, "version")

        # Validate priority
        if not isinstance(priority, int):
            raise PolicyValidationError(
                f"priority must be int, got {type(priority).__name__}",
                error_code="P010",
            )

        # Validate parent
        if parent is not None and not isinstance(parent, Policy):
            raise PolicyValidationError(
                f"parent must be Policy or None, got {type(parent).__name__}",
                error_code="P011",
            )

        # Check for cyclic inheritance
        if parent is not None:
            self._check_cyclic_inheritance(name, parent)

        # Parse conflict strategy
        if isinstance(conflict_strategy, str):
            strategy = ConflictResolutionStrategy.from_string(conflict_strategy)
        elif isinstance(conflict_strategy, ConflictResolutionStrategy):
            strategy = conflict_strategy
        else:
            raise PolicyValidationError(
                f"conflict_strategy must be string or ConflictResolutionStrategy, "
                f"got {type(conflict_strategy).__name__}",
                error_code="P006",
            )

        # Parse and validate rules
        parsed_rules: List[PolicyRule] = []
        rules_by_id: Dict[str, PolicyRule] = {}

        if rules:
            for rule in rules:
                if isinstance(rule, dict):
                    pr = PolicyRule.from_dict(rule)
                elif isinstance(rule, PolicyRule):
                    pr = rule
                else:
                    raise PolicyValidationError(
                        f"rules must be PolicyRule or dict, got {type(rule).__name__}",
                        error_code="P001",
                    )

                # Check for duplicate rule IDs
                if pr.rule_id in rules_by_id:
                    raise DuplicateRuleIdError(pr.rule_id)

                parsed_rules.append(pr)
                rules_by_id[pr.rule_id] = pr

        # Validate conflicts if strategy is REJECT_CONFLICT
        if strategy == ConflictResolutionStrategy.REJECT_CONFLICT:
            self._check_conflicts(parsed_rules, parent)

        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_version', version)
        object.__setattr__(self, '_rules', tuple(parsed_rules))
        object.__setattr__(self, '_rules_by_id', rules_by_id)
        object.__setattr__(self, '_priority', priority)
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_conflict_strategy', strategy)

        # Compute content-based policy_id
        policy_id = self._compute_policy_id()
        object.__setattr__(self, '_policy_id', policy_id)

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise PolicyImmutabilityError(f"set attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, '_frozen', False):
            raise PolicyImmutabilityError(f"delete attribute '{name}'")
        object.__delattr__(self, name)

    # -------------------------------------------------------------------------
    # Internal Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _check_cyclic_inheritance(name: str, parent: "Policy") -> None:
        """Check for cycles in policy inheritance."""
        visited: Set[str] = {name}
        path: List[str] = [name]

        current = parent
        while current is not None:
            if current.name in visited:
                path.append(current.name)
                raise CyclicPolicyInheritanceError(path)
            visited.add(current.name)
            path.append(current.name)
            current = current.parent

    @staticmethod
    def _check_conflicts(
        rules: List[PolicyRule],
        parent: Optional["Policy"],
    ) -> None:
        """
        Check for conflicting rules (only when strategy is REJECT_CONFLICT).

        Conflicts are when the same target has both REQUIRE and FORBID.
        """
        # Collect effects by target
        effects_by_target: Dict[str, Set[PolicyEffect]] = {}

        # Add effects from this policy's rules
        for rule in rules:
            if rule.target not in effects_by_target:
                effects_by_target[rule.target] = set()
            effects_by_target[rule.target].add(rule.effect)

        # Add effects from parent chain
        current = parent
        while current is not None:
            for rule in current.local_rules:
                if rule.target not in effects_by_target:
                    effects_by_target[rule.target] = set()
                effects_by_target[rule.target].add(rule.effect)
            current = current.parent

        # Check for REQUIRE + FORBID conflicts
        for target, effects in effects_by_target.items():
            if PolicyEffect.REQUIRE in effects and PolicyEffect.FORBID in effects:
                raise PolicyConflictError(
                    target,
                    [e.value for e in effects],
                )

    def _compute_policy_id(self) -> str:
        """Compute content-based hash for policy_id."""
        # Note: Use 'is not None' instead of 'if self._parent' because
        # Policy objects with no rules have __len__=0 and are falsy
        parent_policy_id = None
        if self._parent is not None:
            parent_policy_id = self._parent.policy_id

        data = {
            "conflict_strategy": self._conflict_strategy.value,
            "name": self._name,
            "parent_policy_id": parent_policy_id,
            "priority": self._priority,
            "rules": [r.to_dict() for r in self._rules],
            "version": self._version,
        }
        return _compute_hash(data)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def policy_id(self) -> str:
        """Content-based hash of this policy."""
        return self._policy_id

    @property
    def name(self) -> str:
        """Human-readable policy name."""
        return self._name

    @property
    def version(self) -> str:
        """Semantic version string."""
        return self._version

    @property
    def priority(self) -> int:
        """Numeric priority for conflict resolution."""
        return self._priority

    @property
    def parent(self) -> Optional["Policy"]:
        """Parent policy for inheritance."""
        return self._parent

    @property
    def conflict_strategy(self) -> ConflictResolutionStrategy:
        """Conflict resolution strategy."""
        return self._conflict_strategy

    @property
    def local_rules(self) -> Tuple[PolicyRule, ...]:
        """Rules defined directly on this policy (not inherited)."""
        return self._rules

    @property
    def depth(self) -> int:
        """Depth in the inheritance chain (0 for root)."""
        if self._parent is None:
            return 0
        return self._parent.depth + 1

    # -------------------------------------------------------------------------
    # Rule Access Methods
    # -------------------------------------------------------------------------

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        """
        Get a rule by ID, checking this policy and parents.

        Returns:
            PolicyRule if found, None otherwise.
        """
        # Check local rules first
        if rule_id in self._rules_by_id:
            return self._rules_by_id[rule_id]

        # Check parent chain
        if self._parent is not None:
            return self._parent.get_rule(rule_id)

        return None

    def has_rule(self, rule_id: str) -> bool:
        """Check if a rule exists (including inherited)."""
        return self.get_rule(rule_id) is not None

    def has_local_rule(self, rule_id: str) -> bool:
        """Check if a rule is defined locally (not inherited)."""
        return rule_id in self._rules_by_id

    def rules(self) -> List[PolicyRule]:
        """
        Get all visible rules, with inheritance resolved.

        Child rules override parent rules with the same ID.

        Returns:
            List of PolicyRule sorted by rule_id.
        """
        # Collect all rules, child overrides parent
        all_rules: Dict[str, PolicyRule] = {}

        # Start from root, then override with children
        for policy in self.chain():
            for rule in policy.local_rules:
                all_rules[rule.rule_id] = rule

        # Return sorted by rule_id
        return [all_rules[k] for k in sorted(all_rules.keys())]

    def rules_for_target(self, target: str) -> List[PolicyRule]:
        """
        Get all rules that affect a specific target.

        Returns:
            List of PolicyRule for the target, sorted by rule_id.
        """
        return [r for r in self.rules() if r.target == target]

    # -------------------------------------------------------------------------
    # Inheritance Chain Methods
    # -------------------------------------------------------------------------

    def chain(self) -> List["Policy"]:
        """
        Get the full inheritance chain from root to this policy.

        Returns:
            List of Policy objects, root first.
        """
        if self._parent is None:
            return [self]
        return self._parent.chain() + [self]

    def root(self) -> "Policy":
        """Get the root policy in the inheritance chain."""
        if self._parent is None:
            return self
        return self._parent.root()

    # -------------------------------------------------------------------------
    # Conflict Resolution Methods
    # -------------------------------------------------------------------------

    def resolve_effect_for_target(self, target: str) -> Optional[PolicyEffect]:
        """
        Resolve the effective policy effect for a target.

        This does NOT evaluate conditions — it only considers the rules
        that exist and applies the conflict resolution strategy.

        For full evaluation considering conditions, a separate evaluation
        layer must be used externally.

        Returns:
            The resolved PolicyEffect, or None if no rules match.
        """
        rules = self.rules_for_target(target)
        if not rules:
            return None

        effects = [r.effect for r in rules]

        if self._conflict_strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE:
            # Order: FORBID > REQUIRE > PREFER > DEFAULT
            if PolicyEffect.FORBID in effects:
                return PolicyEffect.FORBID
            if PolicyEffect.REQUIRE in effects:
                return PolicyEffect.REQUIRE
            if PolicyEffect.PREFER in effects:
                return PolicyEffect.PREFER
            return PolicyEffect.DEFAULT

        elif self._conflict_strategy == ConflictResolutionStrategy.EXPLICIT_PRIORITY:
            # Find rule with highest priority (from highest priority policy)
            best_rule = None
            best_priority = float('-inf')

            for policy in self.chain():
                for rule in policy.local_rules:
                    if rule.target == target:
                        if policy.priority > best_priority:
                            best_priority = policy.priority
                            best_rule = rule

            return best_rule.effect if best_rule else None

        else:  # REJECT_CONFLICT
            # Already validated in constructor, just return first
            return effects[0] if effects else None

    def get_all_targets(self) -> List[str]:
        """
        Get all unique targets across all rules.

        Returns:
            Sorted list of target strings.
        """
        targets = {r.target for r in self.rules()}
        return sorted(targets)

    # -------------------------------------------------------------------------
    # Policy Modification (returns new Policy)
    # -------------------------------------------------------------------------

    def add_rule(self, rule: PolicyRule | Dict[str, Any]) -> "Policy":
        """
        Create a new Policy with an additional rule.

        Returns:
            New Policy with the rule added.
        """
        if isinstance(rule, dict):
            rule = PolicyRule.from_dict(rule)

        new_rules = list(self._rules) + [rule]

        return Policy(
            name=self._name,
            version=self._version,
            rules=new_rules,
            priority=self._priority,
            parent=self._parent,
            conflict_strategy=self._conflict_strategy,
        )

    def with_priority(self, priority: int) -> "Policy":
        """Create a new Policy with a different priority."""
        return Policy(
            name=self._name,
            version=self._version,
            rules=list(self._rules),
            priority=priority,
            parent=self._parent,
            conflict_strategy=self._conflict_strategy,
        )

    def with_parent(self, parent: Optional["Policy"]) -> "Policy":
        """Create a new Policy with a different parent."""
        return Policy(
            name=self._name,
            version=self._version,
            rules=list(self._rules),
            priority=self._priority,
            parent=parent,
            conflict_strategy=self._conflict_strategy,
        )

    def with_strategy(
        self,
        strategy: str | ConflictResolutionStrategy,
    ) -> "Policy":
        """Create a new Policy with a different conflict strategy."""
        return Policy(
            name=self._name,
            version=self._version,
            rules=list(self._rules),
            priority=self._priority,
            parent=self._parent,
            conflict_strategy=strategy,
        )

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def empty(cls, name: str = "empty") -> "Policy":
        """Create an empty policy with no rules."""
        return cls(name=name, rules=[])

    # -------------------------------------------------------------------------
    # Comparison Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Policy):
            return NotImplemented
        return self._policy_id == other._policy_id

    def __hash__(self) -> int:
        return hash(self._policy_id)

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return count of all visible rules."""
        return len(self.rules())

    def __iter__(self) -> Iterator[PolicyRule]:
        """Iterate over all visible rules."""
        return iter(self.rules())

    def __contains__(self, rule_id: str) -> bool:
        """Check if a rule ID exists."""
        return self.has_rule(rule_id)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = {
            "conflict_strategy": self._conflict_strategy.value,
            "name": self._name,
            "policy_id": self._policy_id,
            "priority": self._priority,
            "rules": [r.to_dict() for r in self._rules],
            "version": self._version,
        }
        if self._parent is not None:
            result["parent_policy_id"] = self._parent.policy_id
        return result

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        """
        Construct from a dictionary.

        Note: Parent policy is not restored from dict. Use with_parent() after.
        """
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            rules=[PolicyRule.from_dict(r) for r in data.get("rules", [])],
            priority=data.get("priority", 0),
            conflict_strategy=data.get("conflict_strategy", "most_restrictive"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Policy":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parent_str = f", parent={self._parent.name!r}" if self._parent else ""
        return (
            f"Policy(name={self._name!r}, version={self._version!r}, "
            f"rules={len(self)}, priority={self._priority}{parent_str})"
        )

    def __str__(self) -> str:
        return (
            f"Policy '{self._name}' v{self._version} "
            f"[{len(self)} rules, {self._conflict_strategy.value}]"
        )
