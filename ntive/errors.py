"""
errors.py

Human-friendly error system for Ntive.

Design principles:
- Never expose internal stack traces
- Always include line number
- Explain what went wrong in plain language
- Suggest fixes when possible
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ErrorLocation:
    """Source location for an error."""
    line: int
    column: int = 0
    source_line: str = ""
    
    def format(self) -> str:
        """Format location as 'line N' or 'line N, column M'."""
        if self.column > 0:
            return f"line {self.line}, column {self.column}"
        return f"line {self.line}"


class NtiveError(Exception):
    """
    Base class for all Ntive errors.
    
    All Ntive errors are designed to be user-friendly:
    - Clear explanation of what went wrong
    - Exact source location
    - Actionable suggestions for fixes
    
    Internal stack traces are never shown to users.
    """
    
    def __init__(
        self,
        message: str,
        location: Optional[ErrorLocation] = None,
        *,
        explanation: str = "",
        suggestions: Optional[List[str]] = None,
        error_code: str = "E000",
    ):
        self.message = message
        self.location = location
        self.explanation = explanation
        self.suggestions = suggestions or []
        self.error_code = error_code
        super().__init__(self.format_short())
    
    def format_short(self) -> str:
        """Format as single-line error message."""
        parts = [f"[{self.error_code}]"]
        if self.location:
            parts.append(f"Line {self.location.line}:")
        parts.append(self.message)
        return " ".join(parts)
    
    def format_full(self) -> str:
        """Format as multi-line human-readable error."""
        lines = []
        
        # Header with error code and location
        header = f"Error {self.error_code}"
        if self.location:
            header += f" at {self.location.format()}"
        lines.append(header)
        lines.append("")
        
        # Main message
        lines.append(f"  {self.message}")
        
        # Source context if available
        if self.location and self.location.source_line:
            lines.append("")
            lines.append(f"    {self.location.line} | {self.location.source_line}")
            if self.location.column > 0:
                # Pointer to the error location
                pointer_offset = len(str(self.location.line)) + 3 + self.location.column - 1
                lines.append(" " * pointer_offset + "^")
        
        # Explanation
        if self.explanation:
            lines.append("")
            lines.append(f"  {self.explanation}")
        
        # Suggestions
        if self.suggestions:
            lines.append("")
            if len(self.suggestions) == 1:
                lines.append(f"  Suggestion: {self.suggestions[0]}")
            else:
                lines.append("  Suggestions:")
                for suggestion in self.suggestions:
                    lines.append(f"    - {suggestion}")
        
        return "\n".join(lines)


# === Parse Errors (E1xx) ===

class ParseError(NtiveError):
    """Base class for parsing errors."""
    pass


class UnknownCommandError(ParseError):
    """Raised when an unknown command is encountered."""
    
    def __init__(
        self,
        command: str,
        location: ErrorLocation,
        valid_commands: Optional[List[str]] = None,
    ):
        valid = valid_commands or ["press", "type", "wait", "move", "click"]
        
        # Find similar commands for suggestion
        suggestions = [f"Valid commands are: {', '.join(valid)}"]
        similar = self._find_similar(command, valid)
        if similar:
            suggestions.insert(0, f"Did you mean '{similar}'?")
        
        super().__init__(
            message=f"Unknown command '{command}'",
            location=location,
            explanation="Ntive only supports a fixed set of commands.",
            suggestions=suggestions,
            error_code="E101",
        )
        self.command = command
    
    @staticmethod
    def _find_similar(word: str, candidates: List[str]) -> Optional[str]:
        """Find a similar word using simple prefix matching."""
        word_lower = word.lower()
        for candidate in candidates:
            if candidate.startswith(word_lower[:2]):
                return candidate
            if word_lower.startswith(candidate[:2]):
                return candidate
        return None


class UnterminatedStringError(ParseError):
    """Raised when a string literal is not closed."""
    
    def __init__(self, location: ErrorLocation):
        super().__init__(
            message="Unterminated string literal",
            location=location,
            explanation="String literals must be closed with a matching quote.",
            suggestions=[
                "Add a closing quote (\") at the end of the string",
                "Check for unescaped quotes inside the string",
            ],
            error_code="E102",
        )


class UnexpectedTokenError(ParseError):
    """Raised when an unexpected token is found."""
    
    def __init__(
        self,
        found: str,
        expected: Optional[str] = None,
        location: Optional[ErrorLocation] = None,
    ):
        message = f"Unexpected token '{found}'"
        if expected:
            message = f"Expected {expected}, found '{found}'"
        
        suggestions = []
        if expected:
            suggestions.append(f"Replace '{found}' with {expected}")
        
        super().__init__(
            message=message,
            location=location,
            suggestions=suggestions,
            error_code="E103",
        )
        self.found = found
        self.expected = expected


class MissingTokenError(ParseError):
    """Raised when a required token is missing."""
    
    def __init__(
        self,
        expected: str,
        location: Optional[ErrorLocation] = None,
        context: str = "",
    ):
        explanation = ""
        if context:
            explanation = f"This is required {context}."
        
        super().__init__(
            message=f"Missing {expected}",
            location=location,
            explanation=explanation,
            suggestions=[f"Add {expected} here"],
            error_code="E104",
        )
        self.expected = expected


class ReservedWordError(ParseError):
    """Raised when a reserved word is used as an identifier."""
    
    def __init__(self, word: str, location: ErrorLocation):
        super().__init__(
            message=f"'{word}' is a reserved word and cannot be used as a name",
            location=location,
            explanation="Reserved words have special meaning in Ntive.",
            suggestions=[
                f"Rename to '{word}_task' or 'my_{word}'",
                "Choose a different name",
            ],
            error_code="E105",
        )
        self.word = word


class UnsupportedFeatureError(ParseError):
    """Raised when user tries to use an unsupported language feature."""
    
    FEATURE_EXPLANATIONS = {
        "loop": "Ntive scripts are deterministic. Loops would make execution unpredictable.",
        "for": "Ntive scripts are deterministic. Loops would make execution unpredictable.",
        "while": "Ntive scripts are deterministic. Loops would make execution unpredictable.",
        "if": "Ntive scripts are deterministic. Conditionals would make execution unpredictable.",
        "else": "Ntive scripts are deterministic. Conditionals would make execution unpredictable.",
        "import": "Ntive scripts are self-contained. Imports are not supported.",
        "from": "Ntive scripts are self-contained. Imports are not supported.",
    }
    
    def __init__(self, feature: str, location: ErrorLocation):
        explanation = self.FEATURE_EXPLANATIONS.get(
            feature.lower(),
            f"The '{feature}' feature is not part of the Ntive language."
        )
        
        super().__init__(
            message=f"'{feature}' is not supported",
            location=location,
            explanation=explanation,
            suggestions=["Remove this line", "Ntive uses a simple task-based structure"],
            error_code="E106",
        )
        self.feature = feature


# === Value Errors (E2xx) ===

class InvalidValueError(NtiveError):
    """Raised when a value is invalid for its context."""
    
    def __init__(
        self,
        value: str,
        expected_type: str,
        location: Optional[ErrorLocation] = None,
        valid_values: Optional[List[str]] = None,
    ):
        suggestions = []
        if valid_values:
            if len(valid_values) <= 5:
                suggestions.append(f"Use one of: {', '.join(valid_values)}")
            else:
                suggestions.append(f"Examples: {', '.join(valid_values[:3])}, ...")
        else:
            suggestions.append(f"Provide a valid {expected_type}")
        
        super().__init__(
            message=f"Invalid value '{value}' (expected {expected_type})",
            location=location,
            suggestions=suggestions,
            error_code="E201",
        )
        self.value = value
        self.expected_type = expected_type


class InvalidKeyError(InvalidValueError):
    """Raised when an invalid key name is used."""
    
    COMMON_KEYS = ["enter", "tab", "escape", "space", "backspace", "up", "down", "left", "right"]
    
    def __init__(self, key: str, location: Optional[ErrorLocation] = None):
        super(InvalidValueError, self).__init__(
            message=f"Unknown key '{key}'",
            location=location,
            explanation="Key names must match standard keyboard keys.",
            suggestions=[
                f"Common keys: {', '.join(self.COMMON_KEYS[:5])}",
                "For letters, use lowercase: a, b, c, ...",
                "For function keys: f1, f2, ..., f12",
            ],
            error_code="E202",
        )
        self.key = key


class InvalidDurationError(InvalidValueError):
    """Raised when an invalid duration value is used."""
    
    def __init__(self, value: str, location: Optional[ErrorLocation] = None):
        # Check for common mistakes
        suggestions = ["Duration must be a positive integer (milliseconds)"]
        
        if value in ("fast", "slow", "quick", "long"):
            suggestions.insert(0, f"Instead of '{value}', use a number like 100 or 500")
        elif value.endswith("s") or value.endswith("ms"):
            numeric = "".join(c for c in value if c.isdigit())
            if numeric:
                suggestions.insert(0, f"Remove the unit suffix, just use: {numeric}")
        
        super(InvalidValueError, self).__init__(
            message=f"Invalid duration '{value}'",
            location=location,
            explanation="Durations are specified in milliseconds as whole numbers.",
            suggestions=suggestions,
            error_code="E203",
        )
        self.value = value


class ValueOutOfRangeError(NtiveError):
    """Raised when a numeric value is out of acceptable range."""
    
    def __init__(
        self,
        value: int,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        context: str = "value",
        location: Optional[ErrorLocation] = None,
    ):
        if min_val is not None and max_val is not None:
            range_str = f"between {min_val} and {max_val}"
        elif min_val is not None:
            range_str = f"at least {min_val}"
        elif max_val is not None:
            range_str = f"at most {max_val}"
        else:
            range_str = "within valid range"
        
        super().__init__(
            message=f"{context.capitalize()} {value} is out of range (must be {range_str})",
            location=location,
            suggestions=[f"Use a {context} {range_str}"],
            error_code="E204",
        )
        self.value = value
        self.min_val = min_val
        self.max_val = max_val


# === Runtime Errors (E3xx) ===

class RuntimeError(NtiveError):
    """Base class for runtime errors."""
    
    def __init__(
        self,
        message: str,
        task: str = "",
        step: int = 0,
        **kwargs,
    ):
        self.task_name = task
        self.step_number = step
        
        location_str = ""
        if task:
            location_str = f"in task '{task}'"
            if step > 0:
                location_str += f" at step {step}"
        
        if location_str and "explanation" not in kwargs:
            kwargs["explanation"] = f"Error occurred {location_str}."
        
        error_code = kwargs.pop("error_code", "E301")
        super().__init__(message, error_code=error_code, **kwargs)


class MissingArgumentError(RuntimeError):
    """Raised when a required argument is not provided."""
    
    def __init__(
        self,
        argument: str,
        task: str = "",
        step: int = 0,
        available_args: Optional[List[str]] = None,
    ):
        suggestions = [f"Provide the '{argument}' argument when calling this task"]
        
        if available_args:
            # Check for typos
            for arg in available_args:
                if arg.lower() in argument.lower() or argument.lower() in arg.lower():
                    suggestions.insert(0, f"Did you mean '{arg}'?")
                    break
        
        super().__init__(
            message=f"Missing required argument '{argument}'",
            task=task,
            step=step,
            suggestions=suggestions,
            error_code="E302",
        )
        self.argument = argument


class TaskNotFoundError(RuntimeError):
    """Raised when a referenced task does not exist."""
    
    def __init__(self, task_name: str, available_tasks: Optional[List[str]] = None):
        suggestions = []
        
        if available_tasks:
            # Find similar task names
            for available in available_tasks:
                if task_name.lower() in available.lower() or available.lower() in task_name.lower():
                    suggestions.append(f"Did you mean '{available}'?")
                    break
            
            if len(available_tasks) <= 5:
                suggestions.append(f"Available tasks: {', '.join(available_tasks)}")
        else:
            suggestions.append("Check that the task name is spelled correctly")
        
        super().__init__(
            message=f"Task '{task_name}' not found",
            task=task_name,
            suggestions=suggestions,
            error_code="E303",
        )


class ValidationError(RuntimeError):
    """Raised when output fails Human IR schema validation."""
    
    def __init__(
        self,
        message: str,
        task: str = "",
        step: int = 0,
        field: str = "",
    ):
        explanation = "The generated step does not conform to the Human IR schema."
        if task:
            explanation = f"In task '{task}'"
            if step > 0:
                explanation += f" at step {step}"
            explanation += ": the generated step does not conform to the Human IR schema."
        if field:
            explanation = f"The '{field}' field has an invalid value."
        
        super().__init__(
            message=message,
            task=task,
            step=step,
            explanation=explanation,
            suggestions=["Check the command syntax", "Refer to the Ntive documentation"],
            error_code="E304",
        )
        self.field = field


# === Utility Functions ===

def format_error_for_user(error: BaseException) -> str:
    """
    Format any exception for user display.
    
    For NtiveError instances, returns the human-friendly format.
    For other exceptions, returns a generic message without stack trace.
    """
    if isinstance(error, NtiveError):
        return error.format_full()
    
    # Generic fallback - never expose internal details
    return (
        "Error E000\n"
        "\n"
        "  An unexpected error occurred.\n"
        "\n"
        "  If this persists, please report it as a bug."
    )


def create_location(line: int, column: int = 0, source: str = "") -> ErrorLocation:
    """Create an ErrorLocation, optionally extracting the source line."""
    source_line = ""
    if source and line > 0:
        lines = source.split("\n")
        if line <= len(lines):
            source_line = lines[line - 1]
    return ErrorLocation(line=line, column=column, source_line=source_line)
