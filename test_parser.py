"""
test_parser.py

Tests for the Ntive parser.
Validates parsing of valid and invalid scripts.
"""

import pytest

from ntive.parser import parse, ParseError
from ntive.ast_nodes import (
    Script, Task, PressCommand, TypeCommand, WaitCommand, MoveCommand, ClickCommand
)


class TestValidScripts:
    """Test parsing of valid Ntive scripts."""

    def test_simple_task(self):
        """Parse a simple task with one command."""
        source = """
task hello:
    press enter
end
"""
        ast = parse(source)
        
        assert isinstance(ast, Script)
        assert len(ast.tasks) == 1
        
        task = ast.tasks[0]
        assert task.name == "hello"
        assert task.args == []
        assert len(task.commands) == 1
        
        cmd = task.commands[0]
        assert isinstance(cmd, PressCommand)
        assert cmd.keys == ["enter"]

    def test_task_with_arguments(self):
        """Parse a task with arguments."""
        source = """
task login(username, password):
    type "user" delay 50
    press tab
    type "pass"
    press enter
end
"""
        ast = parse(source)
        
        task = ast.tasks[0]
        assert task.name == "login"
        assert task.args == ["username", "password"]
        assert len(task.commands) == 4

    def test_all_command_types(self):
        """Parse script with all command types."""
        source = """
task demo:
    press ctrl + s
    type "hello world"
    wait 1000
    move 100 200
    click left at 50 60 2 times
end
"""
        ast = parse(source)
        
        commands = ast.tasks[0].commands
        assert len(commands) == 5
        
        assert isinstance(commands[0], PressCommand)
        assert commands[0].keys == ["ctrl", "s"]
        
        assert isinstance(commands[1], TypeCommand)
        assert commands[1].text == "hello world"
        assert commands[1].delay_ms == 0
        
        assert isinstance(commands[2], WaitCommand)
        assert commands[2].ms == 1000
        
        assert isinstance(commands[3], MoveCommand)
        assert commands[3].x == 100
        assert commands[3].y == 200
        assert commands[3].relative is False
        
        assert isinstance(commands[4], ClickCommand)
        assert commands[4].button == "left"
        assert commands[4].x == 50
        assert commands[4].y == 60
        assert commands[4].count == 2

    def test_multiple_tasks(self):
        """Parse script with multiple tasks."""
        source = """
task first:
    press a
end

task second:
    press b
end
"""
        ast = parse(source)
        
        assert len(ast.tasks) == 2
        assert ast.tasks[0].name == "first"
        assert ast.tasks[1].name == "second"

    def test_comments_ignored(self):
        """Comments are properly ignored."""
        source = """
# This is a comment
task example:
    # Another comment
    press enter  # Inline comment
end
"""
        ast = parse(source)
        
        assert len(ast.tasks) == 1
        assert len(ast.tasks[0].commands) == 1

    def test_move_with_options(self):
        """Parse move command with all options."""
        source = """
task moving:
    move 10 20 relative
    move 100 200 over 500
    move 50 60 relative over 100
end
"""
        ast = parse(source)
        
        cmds = ast.tasks[0].commands
        
        assert cmds[0].relative is True
        assert cmds[0].duration_ms == 0
        
        assert cmds[1].relative is False
        assert cmds[1].duration_ms == 500
        
        assert cmds[2].relative is True
        assert cmds[2].duration_ms == 100

    def test_key_combinations(self):
        """Parse multi-key combinations."""
        source = """
task shortcuts:
    press ctrl + shift + s
    press alt + f4
    press ctrl + alt + delete
end
"""
        ast = parse(source)
        
        cmds = ast.tasks[0].commands
        assert cmds[0].keys == ["ctrl", "shift", "s"]
        assert cmds[1].keys == ["alt", "f4"]
        assert cmds[2].keys == ["ctrl", "alt", "delete"]

    def test_escape_sequences_in_strings(self):
        """Parse escape sequences in type command."""
        source = r'''
task escapes:
    type "line1\nline2"
    type "tab\there"
    type "quote: \"yes\""
end
'''
        ast = parse(source)
        
        cmds = ast.tasks[0].commands
        assert cmds[0].text == "line1\nline2"
        assert cmds[1].text == "tab\there"
        assert cmds[2].text == 'quote: "yes"'

    def test_empty_task_body(self):
        """Task with no commands is valid."""
        source = """
task empty:
end
"""
        ast = parse(source)
        
        assert len(ast.tasks) == 1
        assert ast.tasks[0].commands == []

    def test_to_dict_serialization(self):
        """AST can be serialized to dict."""
        source = """
task test(arg1):
    press enter
    type "hello" delay 100
end
"""
        ast = parse(source)
        d = ast.to_dict()
        
        assert d == {
            "tasks": [{
                "name": "test",
                "args": ["arg1"],
                "commands": [
                    {"type": "press", "keys": ["enter"]},
                    {"type": "type", "text": "hello", "delay_ms": 100}
                ]
            }]
        }


class TestInvalidScripts:
    """Test parsing of invalid scripts produces proper errors."""

    def test_missing_end(self):
        """Missing 'end' keyword."""
        source = """
task incomplete:
    press enter
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "expected 'end'" in str(exc_info.value)

    def test_unknown_key(self):
        """Invalid key name."""
        source = """
task bad:
    press superkey
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "unknown key 'superkey'" in str(exc_info.value)
        assert "line 3" in str(exc_info.value)

    def test_unterminated_string(self):
        """Unterminated string literal."""
        source = """
task bad:
    type "hello
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "unterminated string" in str(exc_info.value)

    def test_missing_string_quote(self):
        """Type without quoted string."""
        source = """
task bad:
    type hello world
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "expected quoted string" in str(exc_info.value)

    def test_loops_not_supported(self):
        """For loop rejected."""
        source = """
task bad:
    for i in range:
        press enter
    end
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "loops not supported" in str(exc_info.value)

    def test_conditionals_not_supported(self):
        """If statement rejected."""
        source = """
task bad:
    if condition:
        press enter
    end
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "conditionals not supported" in str(exc_info.value)

    def test_imports_not_supported(self):
        """Import rejected."""
        source = """
import utils
task main:
    press enter
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "expected 'task'" in str(exc_info.value)

    def test_wait_requires_positive_duration(self):
        """Wait duration must be > 0."""
        source = """
task bad:
    wait 0
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "must be > 0" in str(exc_info.value)

    def test_click_count_out_of_range(self):
        """Click count must be 1-3."""
        source = """
task bad:
    click left 5 times
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "click count must be 1-3" in str(exc_info.value)

    def test_move_missing_y(self):
        """Move requires both x and y."""
        source = """
task bad:
    move 100
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "requires y coordinate" in str(exc_info.value)

    def test_reserved_word_as_task_name(self):
        """Cannot use reserved word as task name."""
        source = """
task press:
    press enter
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "reserved word" in str(exc_info.value)

    def test_too_many_keys(self):
        """Maximum 4 keys in combination."""
        source = """
task bad:
    press ctrl + alt + shift + meta + a
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        assert "maximum 4 keys" in str(exc_info.value)


class TestErrorLineNumbers:
    """Test that errors include correct line numbers."""

    def test_error_on_correct_line(self):
        """Error reports correct line number."""
        source = """
task test:
    press enter
    wait 0
    press a
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        error = exc_info.value
        assert error.line == 4

    def test_multiline_error_tracking(self):
        """Line numbers remain accurate across multiple lines."""
        source = """
task one:
    press a
end

task two:
    press badkey
end
"""
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        
        error = exc_info.value
        assert error.line == 7
        assert "badkey" in error.message


class TestDeterminism:
    """Test that parsing is deterministic."""

    def test_same_input_same_output(self):
        """Same source produces identical AST."""
        source = """
task demo:
    press ctrl + s
    type "hello"
    wait 100
end
"""
        ast1 = parse(source)
        ast2 = parse(source)
        
        assert ast1.to_dict() == ast2.to_dict()

    def test_whitespace_insensitive(self):
        """Indentation doesn't affect parsing."""
        source1 = """
task test:
press enter
type "hello"
end
"""
        source2 = """
task test:
    press enter
        type "hello"
end
"""
        ast1 = parse(source1)
        ast2 = parse(source2)
        
        assert ast1.to_dict() == ast2.to_dict()
