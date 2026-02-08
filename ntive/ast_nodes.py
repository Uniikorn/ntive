"""
ast_nodes.py — INTERNAL MODULE (unstable)
=========================================

.. warning::

    This module is **internal and experimental**. It is NOT part of the
    Ntive Core public API and may change or be removed without notice.

    Do not import from ``ntive.ast_nodes`` directly.

AST node definitions for Ntive scripts.
Pure data structures, no execution logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass(frozen=True)
class SourceLocation:
    """Location in source code for error reporting."""
    line: int
    column: int = 0

    def __str__(self) -> str:
        return f"line {self.line}"


# === Command Nodes ===

@dataclass(frozen=True)
class PressCommand:
    """press <key> [+ <key>]..."""
    keys: List[str]  # e.g., ["ctrl", "shift", "s"]
    loc: SourceLocation


@dataclass(frozen=True)
class TypeCommand:
    """type "<text>" [delay <ms>]"""
    text: str
    delay_ms: int = 0
    loc: SourceLocation = field(default_factory=lambda: SourceLocation(0))


@dataclass(frozen=True)
class WaitCommand:
    """wait <ms>"""
    ms: int
    loc: SourceLocation


@dataclass(frozen=True)
class MoveCommand:
    """move <x> <y> [relative] [over <ms>]"""
    x: int
    y: int
    relative: bool = False
    duration_ms: int = 0
    loc: SourceLocation = field(default_factory=lambda: SourceLocation(0))


@dataclass(frozen=True)
class ClickCommand:
    """click [button] [at <x> <y>] [<n> times]"""
    button: str = "left"
    x: Optional[int] = None
    y: Optional[int] = None
    count: int = 1
    loc: SourceLocation = field(default_factory=lambda: SourceLocation(0))


Command = Union[PressCommand, TypeCommand, WaitCommand, MoveCommand, ClickCommand]


# === Task Node ===

@dataclass(frozen=True)
class Task:
    """A named task with optional arguments and commands."""
    name: str
    args: List[str]
    commands: List[Command]
    loc: SourceLocation


# === Script Node (root) ===

@dataclass(frozen=True)
class Script:
    """Root AST node containing all tasks."""
    tasks: List[Task]

    def to_dict(self) -> dict:
        """Convert AST to dict for serialization."""
        def cmd_to_dict(cmd: Command) -> dict:
            if isinstance(cmd, PressCommand):
                return {"type": "press", "keys": list(cmd.keys)}
            elif isinstance(cmd, TypeCommand):
                return {"type": "type", "text": cmd.text, "delay_ms": cmd.delay_ms}
            elif isinstance(cmd, WaitCommand):
                return {"type": "wait", "ms": cmd.ms}
            elif isinstance(cmd, MoveCommand):
                return {
                    "type": "move", "x": cmd.x, "y": cmd.y,
                    "relative": cmd.relative, "duration_ms": cmd.duration_ms
                }
            elif isinstance(cmd, ClickCommand):
                d = {"type": "click", "button": cmd.button, "count": cmd.count}
                if cmd.x is not None:
                    d["x"] = cmd.x
                    d["y"] = cmd.y
                return d
            return {}

        return {
            "tasks": [
                {
                    "name": t.name,
                    "args": list(t.args),
                    "commands": [cmd_to_dict(c) for c in t.commands]
                }
                for t in self.tasks
            ]
        }
