"""
test_runtime.py

Tests for the Ntive runtime.
Validates transformation, argument substitution, validation, and determinism.
"""

import pytest

from ntive.parser import parse
from ntive.runtime import (
    run, run_all, transform_task,
    RuntimeError, ValidationError,
    validate_step, validate_output
)


class TestBasicTransformation:
    """Test basic AST to Human IR transformation."""

    def test_simple_press(self):
        """Press command transforms correctly."""
        ast = parse("""
task test:
    press enter
end
""")
        result = run(ast, "test")
        
        assert result["task"] == "test"
        assert result["goal"] == "test"
        assert len(result["steps"]) == 1
        assert result["steps"][0] == {"action": "press", "key": "enter"}

    def test_press_with_modifiers(self):
        """Press with modifiers separates key and modifiers."""
        ast = parse("""
task test:
    press ctrl + shift + s
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["action"] == "press"
        assert step["key"] == "s"
        assert step["modifiers"] == ["ctrl", "shift"]

    def test_type_command(self):
        """Type command transforms correctly."""
        ast = parse("""
task test:
    type "hello world"
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["action"] == "type"
        assert step["text"] == "hello world"
        assert "delay_ms" not in step  # Omitted when 0

    def test_type_with_delay(self):
        """Type with delay includes delay_ms."""
        ast = parse("""
task test:
    type "text" delay 100
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["delay_ms"] == 100

    def test_wait_command(self):
        """Wait command transforms correctly."""
        ast = parse("""
task test:
    wait 500
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step == {"action": "wait", "ms": 500}

    def test_move_command(self):
        """Move command transforms correctly."""
        ast = parse("""
task test:
    move 100 200
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["action"] == "move"
        assert step["x"] == 100
        assert step["y"] == 200
        assert "relative" not in step  # Omitted when False
        assert "duration_ms" not in step  # Omitted when 0

    def test_move_with_options(self):
        """Move with relative and duration includes them."""
        ast = parse("""
task test:
    move 50 60 relative over 200
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["relative"] is True
        assert step["duration_ms"] == 200

    def test_click_command(self):
        """Click command transforms correctly."""
        ast = parse("""
task test:
    click
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["action"] == "click"
        assert "button" not in step  # Omitted when 'left' (default)
        assert "count" not in step   # Omitted when 1 (default)

    def test_click_with_options(self):
        """Click with all options includes them."""
        ast = parse("""
task test:
    click right at 100 200 2 times
end
""")
        result = run(ast, "test")
        
        step = result["steps"][0]
        assert step["button"] == "right"
        assert step["x"] == 100
        assert step["y"] == 200
        assert step["count"] == 2


class TestArgumentSubstitution:
    """Test argument substitution in commands."""

    def test_simple_substitution(self):
        """Arguments substitute in type text."""
        ast = parse("""
task login(username):
    type "{username}"
end
""")
        result = run(ast, "login", {"username": "alice"})
        
        step = result["steps"][0]
        assert step["text"] == "alice"

    def test_multiple_arguments(self):
        """Multiple arguments substitute correctly."""
        ast = parse("""
task fill(user, pass):
    type "{user}"
    type "{pass}"
end
""")
        result = run(ast, "fill", {"user": "bob", "pass": "secret123"})
        
        assert result["steps"][0]["text"] == "bob"
        assert result["steps"][1]["text"] == "secret123"

    def test_missing_argument_raises_error(self):
        """Missing required argument raises RuntimeError."""
        ast = parse("""
task greet(name):
    type "Hello {name}"
end
""")
        with pytest.raises(RuntimeError) as exc_info:
            run(ast, "greet", {})
        
        assert "missing argument 'name'" in str(exc_info.value)

    def test_extra_arguments_ignored(self):
        """Extra arguments don't cause errors."""
        ast = parse("""
task test:
    type "hello"
end
""")
        result = run(ast, "test", {"unused": "value"})
        assert result["steps"][0]["text"] == "hello"

    def test_partial_substitution(self):
        """Arguments in mixed text substitute correctly."""
        ast = parse("""
task search(query):
    type "Search: {query} - end"
end
""")
        result = run(ast, "search", {"query": "python"})
        assert result["steps"][0]["text"] == "Search: python - end"


class TestValidation:
    """Test Human IR schema validation."""

    def test_valid_press_step(self):
        """Valid press step passes validation."""
        step = {"action": "press", "key": "enter", "modifiers": ["ctrl"]}
        validate_step(step, "test", 1)  # Should not raise

    def test_invalid_key_rejected(self):
        """Invalid key fails validation."""
        step = {"action": "press", "key": "invalidkey"}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "invalid key" in str(exc_info.value)

    def test_duplicate_modifiers_rejected(self):
        """Duplicate modifiers fail validation."""
        step = {"action": "press", "key": "s", "modifiers": ["ctrl", "ctrl"]}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "duplicate modifiers" in str(exc_info.value)

    def test_empty_text_rejected(self):
        """Empty text fails validation."""
        step = {"action": "type", "text": ""}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "non-empty string" in str(exc_info.value)

    def test_negative_delay_rejected(self):
        """Negative delay fails validation."""
        step = {"action": "type", "text": "hello", "delay_ms": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "delay_ms must be >= 0" in str(exc_info.value)

    def test_zero_wait_rejected(self):
        """Zero wait duration fails validation."""
        step = {"action": "wait", "ms": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "ms must be > 0" in str(exc_info.value)

    def test_invalid_button_rejected(self):
        """Invalid button fails validation."""
        step = {"action": "click", "button": "extra"}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "invalid button" in str(exc_info.value)

    def test_click_count_out_of_range(self):
        """Click count > 3 fails validation."""
        step = {"action": "click", "count": 5}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "count must be 1-3" in str(exc_info.value)

    def test_move_missing_coordinate(self):
        """Move with x but no y fails validation."""
        step = {"action": "move", "x": 100}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "missing x or y" in str(exc_info.value)

    def test_negative_absolute_coordinates_rejected(self):
        """Negative absolute move coordinates fail validation."""
        step = {"action": "move", "x": -10, "y": 100, "relative": False}
        with pytest.raises(ValidationError) as exc_info:
            validate_step(step, "test", 1)
        assert "must be >= 0" in str(exc_info.value)

    def test_negative_relative_coordinates_allowed(self):
        """Negative relative move coordinates pass validation."""
        step = {"action": "move", "x": -10, "y": -20, "relative": True}
        validate_step(step, "test", 1)  # Should not raise

    def test_output_missing_goal_rejected(self):
        """Output without goal fails validation."""
        output = {"steps": []}
        with pytest.raises(ValidationError) as exc_info:
            validate_output(output)
        assert "missing 'goal'" in str(exc_info.value)


class TestDeterminism:
    """Test that runtime is deterministic."""

    def test_same_input_same_output(self):
        """Same AST + args produces identical output."""
        source = """
task demo:
    press ctrl + s
    type "hello" delay 50
    wait 1000
    move 100 200 relative
    click right at 50 60
end
"""
        ast = parse(source)
        
        result1 = run(ast, "demo")
        result2 = run(ast, "demo")
        
        assert result1 == result2

    def test_same_args_same_output(self):
        """Same arguments produce identical substitution."""
        source = """
task login(user, pass):
    type "{user}"
    type "{pass}"
end
"""
        ast = parse(source)
        args = {"user": "alice", "pass": "secret"}
        
        result1 = run(ast, "login", args)
        result2 = run(ast, "login", args)
        
        assert result1 == result2

    def test_multiple_runs_identical(self):
        """Multiple runs are all identical."""
        source = """
task test:
    press enter
    wait 500
    type "test"
end
"""
        ast = parse(source)
        
        results = [run(ast, "test") for _ in range(10)]
        
        assert all(r == results[0] for r in results)

    def test_order_preserved(self):
        """Step order is preserved exactly."""
        source = """
task ordered:
    press a
    press b
    press c
    press d
    press e
end
"""
        ast = parse(source)
        result = run(ast, "ordered")
        
        keys = [s["key"] for s in result["steps"]]
        assert keys == ["a", "b", "c", "d", "e"]


class TestRunAll:
    """Test running all tasks from a script."""

    def test_run_all_tasks(self):
        """run_all executes all tasks in order."""
        source = """
task first:
    press a
end

task second:
    press b
end

task third:
    press c
end
"""
        ast = parse(source)
        results = run_all(ast)
        
        assert len(results) == 3
        assert results[0]["task"] == "first"
        assert results[1]["task"] == "second"
        assert results[2]["task"] == "third"

    def test_run_all_with_args(self):
        """run_all provides arguments per task."""
        source = """
task greet(name):
    type "Hello {name}"
end

task bye(name):
    type "Goodbye {name}"
end
"""
        ast = parse(source)
        results = run_all(ast, {
            "greet": {"name": "Alice"},
            "bye": {"name": "Bob"},
        })
        
        assert results[0]["steps"][0]["text"] == "Hello Alice"
        assert results[1]["steps"][0]["text"] == "Goodbye Bob"


class TestTaskNotFound:
    """Test error handling for missing tasks."""

    def test_task_not_found(self):
        """Running non-existent task raises error."""
        source = """
task exists:
    press enter
end
"""
        ast = parse(source)
        
        with pytest.raises(RuntimeError) as exc_info:
            run(ast, "nonexistent")
        
        assert "task 'nonexistent' not found" in str(exc_info.value)


class TestComplexScripts:
    """Test complex real-world scripts."""

    def test_login_flow(self):
        """Complete login flow transforms correctly."""
        source = """
task login(username, password, url):
    type "{url}"
    press enter
    wait 2000
    click at 100 200
    type "{username}" delay 30
    press tab
    type "{password}" delay 30
    press enter
    wait 3000
end
"""
        ast = parse(source)
        result = run(ast, "login", {
            "username": "testuser",
            "password": "testpass",
            "url": "https://example.com"
        })
        
        assert len(result["steps"]) == 9
        assert result["steps"][0]["text"] == "https://example.com"
        assert result["steps"][4]["text"] == "testuser"
        assert result["steps"][6]["text"] == "testpass"

    def test_empty_task(self):
        """Empty task produces empty steps."""
        source = """
task empty:
end
"""
        ast = parse(source)
        result = run(ast, "empty")
        
        assert result["steps"] == []
