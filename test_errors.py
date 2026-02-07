"""
test_errors.py

Tests for Ntive's human-friendly error system.

Validates:
- Error messages are clear and helpful
- Line numbers are always included
- Suggestions are provided when possible
- Internal stack traces are never exposed
"""

import pytest

from ntive.errors import (
    NtiveError,
    ErrorLocation,
    create_location,
    format_error_for_user,
    # Parse errors
    ParseError,
    UnknownCommandError,
    UnterminatedStringError,
    UnexpectedTokenError,
    MissingTokenError,
    ReservedWordError,
    UnsupportedFeatureError,
    # Value errors
    InvalidValueError,
    InvalidKeyError,
    InvalidDurationError,
    ValueOutOfRangeError,
    # Runtime errors
    RuntimeError,
    MissingArgumentError,
    TaskNotFoundError,
    ValidationError,
)


class TestErrorLocation:
    """Test ErrorLocation formatting."""

    def test_line_only(self):
        """Location with just line number."""
        loc = ErrorLocation(line=10)
        assert loc.format() == "line 10"

    def test_line_and_column(self):
        """Location with line and column."""
        loc = ErrorLocation(line=10, column=5)
        assert loc.format() == "line 10, column 5"

    def test_with_source_line(self):
        """Location with source context."""
        loc = ErrorLocation(line=3, column=1, source_line="    press enter")
        assert loc.source_line == "    press enter"


class TestNtiveErrorBase:
    """Test base NtiveError class."""

    def test_short_format_with_location(self):
        """Short format includes error code and line."""
        loc = ErrorLocation(line=5)
        err = NtiveError("Something went wrong", location=loc, error_code="E100")
        
        short = err.format_short()
        assert "[E100]" in short
        assert "Line 5" in short
        assert "Something went wrong" in short

    def test_short_format_without_location(self):
        """Short format works without location."""
        err = NtiveError("General error", error_code="E000")
        short = err.format_short()
        assert "[E000]" in short
        assert "General error" in short

    def test_full_format_includes_explanation(self):
        """Full format includes explanation."""
        loc = ErrorLocation(line=5)
        err = NtiveError(
            "Something went wrong",
            location=loc,
            explanation="This happened because of X.",
            error_code="E100",
        )
        
        full = err.format_full()
        assert "This happened because of X" in full

    def test_full_format_includes_suggestions(self):
        """Full format includes suggestions."""
        err = NtiveError(
            "Something went wrong",
            suggestions=["Try doing A", "Or try B"],
            error_code="E100",
        )
        
        full = err.format_full()
        assert "Try doing A" in full
        assert "Or try B" in full

    def test_full_format_shows_source_context(self):
        """Full format shows the offending source line."""
        loc = ErrorLocation(line=3, column=5, source_line="    bad_command")
        err = NtiveError("Unknown command", location=loc, error_code="E101")
        
        full = err.format_full()
        assert "3 | " in full
        assert "bad_command" in full

    def test_str_uses_short_format(self):
        """str() returns short format."""
        err = NtiveError("Test error", error_code="E100")
        assert str(err) == err.format_short()


class TestUnknownCommandError:
    """Test unknown command error messages."""

    def test_basic_message(self):
        """Error includes the unknown command name."""
        loc = ErrorLocation(line=5, column=1, source_line="    jump")
        err = UnknownCommandError("jump", loc)
        
        assert "Unknown command 'jump'" in str(err)
        assert "E101" in str(err)

    def test_suggests_valid_commands(self):
        """Error suggests valid commands."""
        loc = ErrorLocation(line=5)
        err = UnknownCommandError("jump", loc)
        
        full = err.format_full()
        assert "press" in full
        assert "type" in full
        assert "wait" in full

    def test_suggests_similar_command(self):
        """Error suggests similar command if close match."""
        loc = ErrorLocation(line=5)
        err = UnknownCommandError("pres", loc)  # typo of "press"
        
        full = err.format_full()
        assert "press" in full


class TestUnterminatedStringError:
    """Test unterminated string error messages."""

    def test_basic_message(self):
        """Error explains the problem."""
        loc = ErrorLocation(line=3, source_line='    type "hello')
        err = UnterminatedStringError(loc)
        
        assert "Unterminated string" in str(err)
        assert "E102" in str(err)

    def test_suggests_closing_quote(self):
        """Error suggests adding closing quote."""
        loc = ErrorLocation(line=3)
        err = UnterminatedStringError(loc)
        
        full = err.format_full()
        assert 'closing quote' in full.lower()


class TestReservedWordError:
    """Test reserved word error messages."""

    def test_basic_message(self):
        """Error identifies the reserved word."""
        loc = ErrorLocation(line=2, source_line="task press:")
        err = ReservedWordError("press", loc)
        
        assert "'press' is a reserved word" in str(err)
        assert "E105" in str(err)

    def test_suggests_alternatives(self):
        """Error suggests alternative names."""
        loc = ErrorLocation(line=2)
        err = ReservedWordError("end", loc)
        
        full = err.format_full()
        assert "_task" in full or "my_" in full


class TestUnsupportedFeatureError:
    """Test unsupported feature error messages."""

    def test_loop_not_supported(self):
        """Error explains why loops aren't supported."""
        loc = ErrorLocation(line=5, source_line="    for i in range(10):")
        err = UnsupportedFeatureError("for", loc)
        
        assert "'for' is not supported" in str(err)
        full = err.format_full()
        assert "deterministic" in full.lower()

    def test_conditional_not_supported(self):
        """Error explains why conditionals aren't supported."""
        loc = ErrorLocation(line=5)
        err = UnsupportedFeatureError("if", loc)
        
        full = err.format_full()
        assert "deterministic" in full.lower()

    def test_import_not_supported(self):
        """Error explains why imports aren't supported."""
        loc = ErrorLocation(line=1)
        err = UnsupportedFeatureError("import", loc)
        
        full = err.format_full()
        assert "self-contained" in full.lower()


class TestInvalidKeyError:
    """Test invalid key error messages."""

    def test_basic_message(self):
        """Error identifies the unknown key."""
        loc = ErrorLocation(line=5, source_line="    press superkey")
        err = InvalidKeyError("superkey", loc)
        
        assert "Unknown key 'superkey'" in str(err)
        assert "E202" in str(err)

    def test_suggests_common_keys(self):
        """Error provides examples of valid keys."""
        loc = ErrorLocation(line=5)
        err = InvalidKeyError("badkey", loc)
        
        full = err.format_full()
        assert "enter" in full.lower() or "tab" in full.lower()


class TestInvalidDurationError:
    """Test invalid duration error messages."""

    def test_word_instead_of_number(self):
        """Error helps when user types word instead of number."""
        loc = ErrorLocation(line=5, source_line='    wait "fast"')
        err = InvalidDurationError("fast", loc)
        
        full = err.format_full()
        assert "100" in full or "500" in full  # suggests numeric alternative
        assert "fast" in full

    def test_with_unit_suffix(self):
        """Error helps when user includes unit suffix."""
        loc = ErrorLocation(line=5)
        err = InvalidDurationError("100ms", loc)
        
        full = err.format_full()
        assert "100" in full  # suggests removing suffix


class TestValueOutOfRangeError:
    """Test value out of range error messages."""

    def test_below_minimum(self):
        """Error explains value is below minimum."""
        loc = ErrorLocation(line=5)
        err = ValueOutOfRangeError(0, min_val=1, context="wait duration", location=loc)
        
        assert "out of range" in str(err).lower()
        assert "at least 1" in err.format_full()

    def test_above_maximum(self):
        """Error explains value is above maximum."""
        loc = ErrorLocation(line=5)
        err = ValueOutOfRangeError(5, max_val=3, context="click count", location=loc)
        
        assert "at most 3" in err.format_full()

    def test_both_bounds(self):
        """Error explains value must be in range."""
        loc = ErrorLocation(line=5)
        err = ValueOutOfRangeError(10, min_val=1, max_val=3, context="click count", location=loc)
        
        assert "between 1 and 3" in err.format_full()


class TestMissingArgumentError:
    """Test missing argument error messages."""

    def test_basic_message(self):
        """Error identifies the missing argument."""
        err = MissingArgumentError("username", task="login", step=2)
        
        assert "Missing required argument 'username'" in str(err)
        assert "E302" in str(err)

    def test_suggests_providing_argument(self):
        """Error suggests providing the argument."""
        err = MissingArgumentError("password", task="login")
        
        full = err.format_full()
        assert "password" in full
        assert "argument" in full.lower()


class TestTaskNotFoundError:
    """Test task not found error messages."""

    def test_basic_message(self):
        """Error identifies the missing task."""
        err = TaskNotFoundError("loign")  # typo
        
        assert "Task 'loign' not found" in str(err)
        assert "E303" in str(err)

    def test_suggests_similar_task(self):
        """Error suggests similar task name."""
        err = TaskNotFoundError("loign", available_tasks=["login", "logout"])
        
        full = err.format_full()
        assert "login" in full

    def test_lists_available_tasks(self):
        """Error lists available tasks when few."""
        err = TaskNotFoundError("unknown", available_tasks=["task1", "task2"])
        
        full = err.format_full()
        assert "task1" in full
        assert "task2" in full


class TestValidationError:
    """Test validation error messages."""

    def test_basic_message(self):
        """Error includes validation message."""
        err = ValidationError("count must be 1-3", task="click_test", step=1)
        
        assert "count must be 1-3" in str(err)
        assert "E304" in str(err)

    def test_includes_task_info(self):
        """Error includes task context."""
        err = ValidationError("invalid value", task="my_task", step=3)
        
        full = err.format_full()
        assert "my_task" in full
        assert "step 3" in full


class TestFormatErrorForUser:
    """Test the format_error_for_user utility."""

    def test_ntive_error_uses_full_format(self):
        """NtiveError uses full format."""
        loc = ErrorLocation(line=5)
        err = NtiveError("Test error", location=loc, error_code="E100")
        
        result = format_error_for_user(err)
        assert "Error E100" in result
        assert "Test error" in result

    def test_generic_exception_hidden(self):
        """Generic exceptions don't expose internals."""
        err = ValueError("Internal implementation detail")
        
        result = format_error_for_user(err)
        assert "Internal implementation detail" not in result
        assert "unexpected error" in result.lower()


class TestCreateLocation:
    """Test the create_location utility."""

    def test_basic_location(self):
        """Create location with line and column."""
        loc = create_location(5, 10)
        assert loc.line == 5
        assert loc.column == 10

    def test_extracts_source_line(self):
        """Create location extracts source line from source."""
        source = "line 1\nline 2\nline 3\nline 4"
        loc = create_location(2, 1, source)
        
        assert loc.source_line == "line 2"

    def test_empty_source_ok(self):
        """Create location handles empty source."""
        loc = create_location(5, 1, "")
        assert loc.source_line == ""


class TestNoStackTraces:
    """Verify that stack traces are never exposed."""

    def test_ntive_error_str_no_traceback(self):
        """NtiveError str has no traceback."""
        err = NtiveError("Test", error_code="E100")
        
        result = str(err)
        assert "Traceback" not in result
        assert "File" not in result
        assert ".py" not in result

    def test_full_format_no_traceback(self):
        """Full format has no traceback."""
        loc = ErrorLocation(line=5, source_line="    press enter")
        err = ParseError(
            "Some parse error",
            location=loc,
            error_code="E100",
        )
        
        result = err.format_full()
        assert "Traceback" not in result
        assert "at 0x" not in result  # memory addresses


class TestExampleErrorMessages:
    """Test the example scenarios from the requirements."""

    def test_unknown_command_example(self):
        """Example: Unknown command."""
        loc = create_location(5, 1, "task demo:\n    jump\nend")
        err = UnknownCommandError("jump", loc)
        
        full = err.format_full()
        
        # Must include line number
        assert "line 5" in full.lower() or "Line 5" in full
        # Must explain what went wrong
        assert "Unknown command" in full
        assert "jump" in full
        # Must suggest fixes
        assert "press" in full or "type" in full

    def test_missing_argument_example(self):
        """Example: Missing argument."""
        err = MissingArgumentError(
            "username",
            task="login",
            available_args=["password"],
        )
        
        full = err.format_full()
        
        # Must explain what went wrong
        assert "Missing" in full
        assert "username" in full
        # Must suggest fixes
        assert "argument" in full.lower()

    def test_invalid_value_example(self):
        """Example: Invalid value (wait "fast")."""
        loc = ErrorLocation(line=3, source_line='    wait "fast"')
        err = InvalidDurationError("fast", loc)
        
        full = err.format_full()
        
        # Must include line number
        assert "3" in full
        # Must explain what went wrong
        assert "Invalid" in full or "fast" in full
        # Must suggest fixes
        assert "milliseconds" in full.lower() or "number" in full.lower()
