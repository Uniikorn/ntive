"""
parser.py — INTERNAL MODULE (unstable)
========================================

.. warning::

    This module is **internal and experimental**. It is NOT part of the
    Ntive Core public API and may change or be removed without notice.

    Do not import from ``ntive.parser`` directly.

Ntive script parser. Converts source text to AST.
No execution, no side effects, deterministic output.
"""

from dataclasses import dataclass
from typing import List, Optional

from ntive.ast_nodes import (
    ClickCommand,
    Command,
    MoveCommand,
    PressCommand,
    Script,
    SourceLocation,
    Task,
    TypeCommand,
    WaitCommand,
)


class ParseError(Exception):
    """Parsing error with location information."""
    def __init__(self, message: str, line: int, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"line {line}: {message}")


# Valid keys for press command
VALID_KEYS = {
    # Letters
    *"abcdefghijklmnopqrstuvwxyz",
    # Digits
    *"0123456789",
    # Special keys
    "enter", "tab", "escape", "space", "backspace", "delete",
    "up", "down", "left", "right",
    "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "ctrl", "alt", "shift", "meta",
    "capslock", "numlock", "scrolllock", "printscreen", "pause", "insert",
}

VALID_BUTTONS = {"left", "right", "middle"}

RESERVED_WORDS = {
    "task", "end", "press", "type", "wait", "move", "click",
    "at", "over", "relative", "times", "delay",
} | VALID_KEYS | VALID_BUTTONS


@dataclass
class Token:
    """A lexical token."""
    type: str
    value: str
    line: int
    column: int


class Lexer:
    """Tokenizes Ntive source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """Convert source to token list."""
        while self.pos < len(self.source):
            self._skip_whitespace()
            if self.pos >= len(self.source):
                break

            ch = self.source[self.pos]

            # Comment
            if ch == '#':
                self._skip_comment()
                continue

            # Newline
            if ch == '\n':
                self.tokens.append(Token('NEWLINE', '\n', self.line, self.column))
                self._advance()
                continue

            if ch == '\r':
                self._advance()
                if self.pos < len(self.source) and self.source[self.pos] == '\n':
                    self._advance()
                self.tokens.append(Token('NEWLINE', '\n', self.line - 1, self.column))
                continue

            # String literal
            if ch == '"':
                self.tokens.append(self._read_string())
                continue

            # Symbols
            if ch == ':':
                self.tokens.append(Token('COLON', ':', self.line, self.column))
                self._advance()
                continue

            if ch == '(':
                self.tokens.append(Token('LPAREN', '(', self.line, self.column))
                self._advance()
                continue

            if ch == ')':
                self.tokens.append(Token('RPAREN', ')', self.line, self.column))
                self._advance()
                continue

            if ch == ',':
                self.tokens.append(Token('COMMA', ',', self.line, self.column))
                self._advance()
                continue

            if ch == '+':
                self.tokens.append(Token('PLUS', '+', self.line, self.column))
                self._advance()
                continue

            # Number
            if ch.isdigit():
                self.tokens.append(self._read_number())
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == '_':
                self.tokens.append(self._read_identifier())
                continue

            raise ParseError(f"unexpected character '{ch}'", self.line, self.column)

        self.tokens.append(Token('EOF', '', self.line, self.column))
        return self.tokens

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in ' \t':
            self._advance()

    def _skip_comment(self):
        while self.pos < len(self.source) and self.source[self.pos] != '\n':
            self._advance()

    def _read_string(self) -> Token:
        start_line = self.line
        start_col = self.column
        self._advance()  # skip opening quote

        chars = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]

            if ch == '"':
                self._advance()
                return Token('STRING', ''.join(chars), start_line, start_col)

            if ch == '\n':
                raise ParseError("unterminated string literal", start_line, start_col)

            if ch == '\\' and self.pos + 1 < len(self.source):
                self._advance()
                next_ch = self.source[self.pos]
                if next_ch == 'n':
                    chars.append('\n')
                elif next_ch == 't':
                    chars.append('\t')
                elif next_ch == '"':
                    chars.append('"')
                elif next_ch == '\\':
                    chars.append('\\')
                else:
                    chars.append(next_ch)
                self._advance()
                continue

            chars.append(ch)
            self._advance()

        raise ParseError("unterminated string literal", start_line, start_col)

    def _read_number(self) -> Token:
        start_line = self.line
        start_col = self.column
        chars = []

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            chars.append(self._advance())

        return Token('NUMBER', ''.join(chars), start_line, start_col)

    def _read_identifier(self) -> Token:
        start_line = self.line
        start_col = self.column
        chars = []

        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch.isalnum() or ch == '_':
                chars.append(self._advance())
            else:
                break

        value = ''.join(chars)
        return Token('IDENT', value, start_line, start_col)


class Parser:
    """Parses Ntive tokens into AST."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Script:
        """Parse tokens into Script AST."""
        tasks = []

        self._skip_newlines()

        while not self._at_end():
            if self._check('IDENT') and self._peek().value == 'task':
                tasks.append(self._parse_task())
            else:
                raise ParseError(
                    f"expected 'task', got '{self._peek().value}'",
                    self._peek().line, self._peek().column
                )
            self._skip_newlines()

        return Script(tasks=tasks)

    def _parse_task(self) -> Task:
        """Parse a task definition."""
        task_token = self._consume('IDENT', "expected 'task'")
        if task_token.value != 'task':
            raise ParseError("expected 'task'", task_token.line, task_token.column)

        loc = SourceLocation(task_token.line, task_token.column)

        # Task name
        name_token = self._consume('IDENT', "expected task name")
        if name_token.value in RESERVED_WORDS:
            raise ParseError(
                f"'{name_token.value}' is a reserved word",
                name_token.line, name_token.column
            )

        # Optional arguments
        args = []
        if self._check('LPAREN'):
            self._advance()
            if not self._check('RPAREN'):
                args = self._parse_arg_list()
            self._consume('RPAREN', "expected ')' after arguments")

        self._consume('COLON', "expected ':' after task declaration")
        self._consume('NEWLINE', "expected newline after ':'")

        # Body
        commands = self._parse_body()

        # End
        end_token = self._consume('IDENT', f"expected 'end' to close task '{name_token.value}'")
        if end_token.value != 'end':
            raise ParseError(
                f"expected 'end', got '{end_token.value}'",
                end_token.line, end_token.column
            )

        return Task(name=name_token.value, args=args, commands=commands, loc=loc)

    def _parse_arg_list(self) -> List[str]:
        """Parse comma-separated argument list."""
        args = []

        arg = self._consume('IDENT', "expected argument name")
        args.append(arg.value)

        while self._check('COMMA'):
            self._advance()
            arg = self._consume('IDENT', "expected argument name after ','")
            args.append(arg.value)

        return args

    def _parse_body(self) -> List[Command]:
        """Parse task body (list of commands)."""
        commands = []
        self._skip_newlines()

        while not self._at_end():
            if self._check('IDENT') and self._peek().value == 'end':
                break

            cmd = self._parse_command()
            if cmd:
                commands.append(cmd)

            if not self._at_end() and not self._check('NEWLINE'):
                if not (self._check('IDENT') and self._peek().value == 'end'):
                    raise ParseError(
                        "expected newline after command",
                        self._peek().line, self._peek().column
                    )

            self._skip_newlines()

        return commands

    def _parse_command(self) -> Optional[Command]:
        """Parse a single command."""
        if not self._check('IDENT'):
            return None

        token = self._peek()
        cmd_name = token.value.lower()

        if cmd_name == 'press':
            return self._parse_press()
        elif cmd_name == 'type':
            return self._parse_type()
        elif cmd_name == 'wait':
            return self._parse_wait()
        elif cmd_name == 'move':
            return self._parse_move()
        elif cmd_name == 'click':
            return self._parse_click()
        elif cmd_name in ('for', 'while'):
            raise ParseError("loops not supported", token.line, token.column)
        elif cmd_name == 'if':
            raise ParseError("conditionals not supported", token.line, token.column)
        elif cmd_name == 'import':
            raise ParseError("imports not supported", token.line, token.column)
        else:
            raise ParseError(f"unknown command '{cmd_name}'", token.line, token.column)

    def _parse_press(self) -> PressCommand:
        """Parse: press <key> [+ <key>]..."""
        token = self._advance()
        loc = SourceLocation(token.line, token.column)

        keys = []

        # First key
        key_token = self._consume('IDENT', "expected key after 'press'")
        key = key_token.value.lower()
        if key not in VALID_KEYS:
            raise ParseError(f"unknown key '{key}'", key_token.line, key_token.column)
        keys.append(key)

        # Additional keys with +
        while self._check('PLUS'):
            self._advance()
            key_token = self._consume('IDENT', "expected key after '+'")
            key = key_token.value.lower()
            if key not in VALID_KEYS:
                raise ParseError(f"unknown key '{key}'", key_token.line, key_token.column)
            keys.append(key)

        if len(keys) > 4:
            raise ParseError("maximum 4 keys in combination", loc.line, loc.column)

        return PressCommand(keys=keys, loc=loc)

    def _parse_type(self) -> TypeCommand:
        """Parse: type "<text>" [delay <ms>]"""
        token = self._advance()
        loc = SourceLocation(token.line, token.column)

        str_token = self._consume('STRING', "expected quoted string after 'type'")
        text = str_token.value

        if not text:
            raise ParseError("type text cannot be empty", str_token.line, str_token.column)

        delay_ms = 0
        if self._check('IDENT') and self._peek().value == 'delay':
            self._advance()
            num_token = self._consume('NUMBER', "expected number after 'delay'")
            delay_ms = int(num_token.value)
            if delay_ms < 0:
                raise ParseError("delay must be >= 0", num_token.line, num_token.column)

        return TypeCommand(text=text, delay_ms=delay_ms, loc=loc)

    def _parse_wait(self) -> WaitCommand:
        """Parse: wait <ms>"""
        token = self._advance()
        loc = SourceLocation(token.line, token.column)

        num_token = self._consume('NUMBER', "expected duration after 'wait'")
        ms = int(num_token.value)

        if ms <= 0:
            raise ParseError("wait duration must be > 0", num_token.line, num_token.column)

        return WaitCommand(ms=ms, loc=loc)

    def _parse_move(self) -> MoveCommand:
        """Parse: move <x> <y> [relative] [over <ms>]"""
        token = self._advance()
        loc = SourceLocation(token.line, token.column)

        x_token = self._consume('NUMBER', "'move' requires x coordinate")
        y_token = self._consume('NUMBER', "'move' requires y coordinate")

        x = int(x_token.value)
        y = int(y_token.value)
        relative = False
        duration_ms = 0

        # Optional: relative
        if self._check('IDENT') and self._peek().value == 'relative':
            self._advance()
            relative = True

        # Optional: over <ms>
        if self._check('IDENT') and self._peek().value == 'over':
            self._advance()
            dur_token = self._consume('NUMBER', "expected duration after 'over'")
            duration_ms = int(dur_token.value)
            if duration_ms < 0:
                raise ParseError("duration must be >= 0", dur_token.line, dur_token.column)

        return MoveCommand(x=x, y=y, relative=relative, duration_ms=duration_ms, loc=loc)

    def _parse_click(self) -> ClickCommand:
        """Parse: click [button] [at <x> <y>] [<n> times]"""
        token = self._advance()
        loc = SourceLocation(token.line, token.column)

        button = "left"
        x = None
        y = None
        count = 1

        # Optional: button
        if self._check('IDENT') and self._peek().value in VALID_BUTTONS:
            button = self._advance().value

        # Optional: at <x> <y>
        if self._check('IDENT') and self._peek().value == 'at':
            self._advance()
            x_token = self._consume('NUMBER', "expected x coordinate after 'at'")
            y_token = self._consume('NUMBER', "expected y coordinate after x")
            x = int(x_token.value)
            y = int(y_token.value)

        # Optional: <n> times
        if self._check('NUMBER'):
            count_token = self._advance()
            count = int(count_token.value)
            times_token = self._consume('IDENT', "expected 'times' after count")
            if times_token.value != 'times':
                raise ParseError(
                    f"expected 'times', got '{times_token.value}'",
                    times_token.line, times_token.column
                )
            if count < 1 or count > 3:
                raise ParseError(
                    f"click count must be 1-3, got {count}",
                    count_token.line, count_token.column
                )

        return ClickCommand(button=button, x=x, y=y, count=count, loc=loc)

    # === Helper methods ===

    def _at_end(self) -> bool:
        return self.pos >= len(self.tokens) or self.tokens[self.pos].type == 'EOF'

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _check(self, type_: str) -> bool:
        if self._at_end():
            return False
        return self.tokens[self.pos].type == type_

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _consume(self, type_: str, error_msg: str) -> Token:
        if self._check(type_):
            return self._advance()
        token = self._peek()
        raise ParseError(error_msg, token.line, token.column)

    def _skip_newlines(self):
        while self._check('NEWLINE'):
            self._advance()


def parse(source: str) -> Script:
    """
    Parse Ntive source code into an AST.

    Args:
        source: Ntive source code string

    Returns:
        Script AST

    Raises:
        ParseError: If parsing fails
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
