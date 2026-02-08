"""
runtime.py — INTERNAL MODULE (unstable)
=======================================

.. warning::

    This module is **internal and experimental**. It is NOT part of the
    Ntive Core public API and may change or be removed without notice.

    Do not import from ``ntive.runtime`` directly.

Pure Ntive runtime. Transforms AST into Human IR steps.
No IO, no OS access, no sleep, no randomness, no time dependency.
"""

from typing import Any, Dict, List, Optional

from ntive.ast_nodes import (
    ClickCommand,
    Command,
    MoveCommand,
    PressCommand,
    Script,
    Task,
    TypeCommand,
    WaitCommand,
)


class RuntimeError(Exception):
    """Runtime error during AST transformation."""
    def __init__(self, message: str, task: str = "", step: int = 0):
        self.message = message
        self.task = task
        self.step = step
        location = f"task '{task}'" if task else "runtime"
        if step > 0:
            location += f", step {step}"
        super().__init__(f"{location}: {message}")


class ValidationError(RuntimeError):
    """Validation error for Human IR schema."""
    pass


# === Schema Constants ===

VALID_KEYS = {
    *"abcdefghijklmnopqrstuvwxyz",
    *"0123456789",
    "enter", "tab", "escape", "space", "backspace", "delete",
    "up", "down", "left", "right",
    "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "ctrl", "alt", "shift", "meta",
    "capslock", "numlock", "scrolllock", "printscreen", "pause", "insert",
}

MODIFIER_KEYS = {"ctrl", "alt", "shift", "meta"}

VALID_BUTTONS = {"left", "right", "middle"}


# === Validation Functions ===

def validate_press_step(step: Dict[str, Any], task: str, idx: int) -> None:
    """Validate a press step against Human IR schema."""
    if "key" not in step:
        raise ValidationError("press step missing 'key'", task, idx)

    key = step["key"]
    if key not in VALID_KEYS:
        raise ValidationError(f"invalid key '{key}'", task, idx)

    modifiers = step.get("modifiers", [])
    if not isinstance(modifiers, list):
        raise ValidationError("modifiers must be a list", task, idx)

    for mod in modifiers:
        if mod not in MODIFIER_KEYS:
            raise ValidationError(f"invalid modifier '{mod}'", task, idx)

    if len(modifiers) != len(set(modifiers)):
        raise ValidationError("duplicate modifiers", task, idx)


def validate_type_step(step: Dict[str, Any], task: str, idx: int) -> None:
    """Validate a type step against Human IR schema."""
    if "text" not in step:
        raise ValidationError("type step missing 'text'", task, idx)

    text = step["text"]
    if not isinstance(text, str) or not text:
        raise ValidationError("text must be non-empty string", task, idx)

    delay_ms = step.get("delay_ms", 0)
    if not isinstance(delay_ms, int) or delay_ms < 0:
        raise ValidationError("delay_ms must be >= 0", task, idx)


def validate_wait_step(step: Dict[str, Any], task: str, idx: int) -> None:
    """Validate a wait step against Human IR schema."""
    if "ms" not in step:
        raise ValidationError("wait step missing 'ms'", task, idx)

    ms = step["ms"]
    if not isinstance(ms, int) or ms <= 0:
        raise ValidationError("ms must be > 0", task, idx)


def validate_move_step(step: Dict[str, Any], task: str, idx: int) -> None:
    """Validate a move step against Human IR schema."""
    if "x" not in step or "y" not in step:
        raise ValidationError("move step missing x or y", task, idx)

    x, y = step["x"], step["y"]
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValidationError("x and y must be integers", task, idx)

    relative = step.get("relative", False)
    if not relative and (x < 0 or y < 0):
        raise ValidationError("absolute coordinates must be >= 0", task, idx)

    duration_ms = step.get("duration_ms", 0)
    if not isinstance(duration_ms, int) or duration_ms < 0:
        raise ValidationError("duration_ms must be >= 0", task, idx)


def validate_click_step(step: Dict[str, Any], task: str, idx: int) -> None:
    """Validate a click step against Human IR schema."""
    button = step.get("button", "left")
    if button not in VALID_BUTTONS:
        raise ValidationError(f"invalid button '{button}'", task, idx)

    count = step.get("count", 1)
    if not isinstance(count, int) or count < 1 or count > 3:
        raise ValidationError("count must be 1-3", task, idx)

    has_x = "x" in step
    has_y = "y" in step
    if has_x != has_y:
        raise ValidationError("x and y must both be present or both absent", task, idx)

    if has_x:
        x, y = step["x"], step["y"]
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValidationError("x and y must be integers", task, idx)
        if x < 0 or y < 0:
            raise ValidationError("click coordinates must be >= 0", task, idx)


def validate_step(step: Dict[str, Any], task: str, idx: int) -> None:
    """Validate a step against Human IR schema."""
    if "action" not in step:
        raise ValidationError("step missing 'action'", task, idx)

    action = step["action"]

    validators = {
        "press": validate_press_step,
        "type": validate_type_step,
        "wait": validate_wait_step,
        "move": validate_move_step,
        "click": validate_click_step,
    }

    if action not in validators:
        raise ValidationError(f"unknown action '{action}'", task, idx)

    validators[action](step, task, idx)


def validate_output(output: Dict[str, Any]) -> None:
    """Validate complete Human IR output."""
    if "goal" not in output:
        raise ValidationError("output missing 'goal'")

    if not isinstance(output["goal"], str) or not output["goal"]:
        raise ValidationError("goal must be non-empty string")

    if "steps" not in output:
        raise ValidationError("output missing 'steps'")

    if not isinstance(output["steps"], list):
        raise ValidationError("steps must be a list")

    task_name = output.get("task", "")
    for i, step in enumerate(output["steps"], 1):
        validate_step(step, task_name, i)


# === AST to IR Transformation ===

def transform_press(cmd: PressCommand) -> Dict[str, Any]:
    """Transform PressCommand to Human IR step."""
    keys = list(cmd.keys)

    # Separate modifiers and final key
    modifiers = [k for k in keys[:-1] if k in MODIFIER_KEYS]
    final_key = keys[-1]

    # If final key is also a modifier (e.g., "press ctrl"), treat it as the key
    step = {
        "action": "press",
        "key": final_key,
    }

    if modifiers:
        step["modifiers"] = modifiers

    return step


def transform_type(cmd: TypeCommand, args: Dict[str, str]) -> Dict[str, Any]:
    """Transform TypeCommand to Human IR step with argument substitution."""
    text = substitute_args(cmd.text, args)

    step = {
        "action": "type",
        "text": text,
    }

    if cmd.delay_ms > 0:
        step["delay_ms"] = cmd.delay_ms

    return step


def transform_wait(cmd: WaitCommand) -> Dict[str, Any]:
    """Transform WaitCommand to Human IR step."""
    return {
        "action": "wait",
        "ms": cmd.ms,
    }


def transform_move(cmd: MoveCommand) -> Dict[str, Any]:
    """Transform MoveCommand to Human IR step."""
    step = {
        "action": "move",
        "x": cmd.x,
        "y": cmd.y,
    }

    if cmd.relative:
        step["relative"] = True

    if cmd.duration_ms > 0:
        step["duration_ms"] = cmd.duration_ms

    return step


def transform_click(cmd: ClickCommand) -> Dict[str, Any]:
    """Transform ClickCommand to Human IR step."""
    step = {
        "action": "click",
    }

    if cmd.button != "left":
        step["button"] = cmd.button

    if cmd.x is not None:
        step["x"] = cmd.x
        step["y"] = cmd.y

    if cmd.count != 1:
        step["count"] = cmd.count

    return step


def transform_command(cmd: Command, args: Dict[str, str]) -> Dict[str, Any]:
    """Transform a command AST node to Human IR step."""
    if isinstance(cmd, PressCommand):
        return transform_press(cmd)
    elif isinstance(cmd, TypeCommand):
        return transform_type(cmd, args)
    elif isinstance(cmd, WaitCommand):
        return transform_wait(cmd)
    elif isinstance(cmd, MoveCommand):
        return transform_move(cmd)
    elif isinstance(cmd, ClickCommand):
        return transform_click(cmd)
    else:
        raise RuntimeError(f"unknown command type: {type(cmd).__name__}")


def substitute_args(text: str, args: Dict[str, str]) -> str:
    """
    Substitute argument placeholders in text.

    Placeholders use {arg_name} syntax.
    """
    result = text
    for name, value in args.items():
        result = result.replace(f"{{{name}}}", value)
    return result


# === Main Runtime Functions ===

def transform_task(
    task: Task,
    args: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Transform a Task AST node to Human IR output.

    Args:
        task: Task AST node
        args: Optional dict of argument values to substitute

    Returns:
        Dict with 'task', 'goal', and 'steps'

    Raises:
        RuntimeError: If transformation fails
        ValidationError: If output violates schema
    """
    args = args or {}

    # Check all required args are provided
    for arg_name in task.args:
        if arg_name not in args:
            raise RuntimeError(f"missing argument '{arg_name}'", task.name)

    # Transform each command
    steps = []
    for i, cmd in enumerate(task.commands, 1):
        try:
            step = transform_command(cmd, args)
            steps.append(step)
        except Exception as e:
            raise RuntimeError(str(e), task.name, i)

    output = {
        "task": task.name,
        "goal": task.name,  # Use task name as goal
        "steps": steps,
    }

    # Validate output
    validate_output(output)

    return output


def run(
    script: Script,
    task_name: str,
    args: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Run a task from a script.

    Args:
        script: Script AST
        task_name: Name of task to run
        args: Optional dict of argument values

    Returns:
        Human IR output dict

    Raises:
        RuntimeError: If task not found or transformation fails
    """
    # Find task
    task = None
    for t in script.tasks:
        if t.name == task_name:
            task = t
            break

    if task is None:
        raise RuntimeError(f"task '{task_name}' not found")

    return transform_task(task, args)


def run_all(
    script: Script,
    task_args: Optional[Dict[str, Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Run all tasks from a script.

    Args:
        script: Script AST
        task_args: Optional dict mapping task names to their arguments

    Returns:
        List of Human IR outputs for each task
    """
    task_args = task_args or {}
    results = []

    for task in script.tasks:
        args = task_args.get(task.name, {})
        result = transform_task(task, args)
        results.append(result)

    return results
