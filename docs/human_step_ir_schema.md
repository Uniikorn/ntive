# Human Step IR Schema

A minimal, deterministic intermediate representation for human-like actions.

## Design Principles

- **Pure data**: JSON-serializable, no side effects
- **Deterministic**: Same IR always produces same behavior
- **Executor-agnostic**: Schema defines *what*, not *how*
- **Auditable**: Every step traces back to an explicit constraint

---

## Top-Level Structure

```json
{
  "goal": "string",
  "steps": [ HumanStep, ... ]
}
```

| Field   | Type     | Required | Description                          |
|---------|----------|----------|--------------------------------------|
| `goal`  | string   | yes      | Human-readable intent                |
| `steps` | array    | yes      | Ordered list of HumanStep objects    |

---

## Step Types

### 1. `press` — Single Key Press

Press and release a single key or key combination.

```json
{
  "action": "press",
  "key": "enter",
  "modifiers": ["ctrl"]
}
```

| Field       | Type     | Required | Description                              |
|-------------|----------|----------|------------------------------------------|
| `action`    | string   | yes      | Must be `"press"`                        |
| `key`       | string   | yes      | Key identifier (see Key Names below)     |
| `modifiers` | string[] | no       | Modifier keys: `ctrl`, `alt`, `shift`, `meta` |

**Validation**:
- `key` must be a valid key name
- `modifiers` must only contain allowed modifier names
- `modifiers` must not duplicate

---

### 2. `type` — Text Input

Type a sequence of characters as keyboard input.

```json
{
  "action": "type",
  "text": "hello world",
  "delay_ms": 50
}
```

| Field      | Type   | Required | Default | Description                        |
|------------|--------|----------|---------|------------------------------------|
| `action`   | string | yes      |         | Must be `"type"`                   |
| `text`     | string | yes      |         | Text to type (UTF-8)               |
| `delay_ms` | int    | no       | 0       | Delay between keystrokes (ms)      |

**Validation**:
- `text` must be non-empty string
- `delay_ms` must be >= 0

---

### 3. `wait` — Pause Execution

Wait for a fixed duration.

```json
{
  "action": "wait",
  "ms": 1000
}
```

| Field    | Type   | Required | Description                  |
|----------|--------|----------|------------------------------|
| `action` | string | yes      | Must be `"wait"`             |
| `ms`     | int    | yes      | Duration in milliseconds     |

**Validation**:
- `ms` must be > 0

---

### 4. `move` — Mouse Movement

Move mouse cursor to absolute or relative position.

```json
{
  "action": "move",
  "x": 500,
  "y": 300,
  "relative": false,
  "duration_ms": 100
}
```

| Field         | Type   | Required | Default | Description                          |
|---------------|--------|----------|---------|--------------------------------------|
| `action`      | string | yes      |         | Must be `"move"`                     |
| `x`           | int    | yes      |         | X coordinate (pixels)                |
| `y`           | int    | yes      |         | Y coordinate (pixels)                |
| `relative`    | bool   | no       | false   | If true, coordinates are relative    |
| `duration_ms` | int    | no       | 0       | Movement duration (0 = instant)      |

**Validation**:
- `x` and `y` must be integers
- `duration_ms` must be >= 0
- If `relative: false`, `x` and `y` should be >= 0

---

### 5. `click` — Mouse Click

Perform a mouse click at current or specified position.

```json
{
  "action": "click",
  "button": "left",
  "count": 1,
  "x": 500,
  "y": 300
}
```

| Field    | Type   | Required | Default   | Description                          |
|----------|--------|----------|-----------|--------------------------------------|
| `action` | string | yes      |           | Must be `"click"`                    |
| `button` | string | no       | `"left"`  | Button: `left`, `right`, `middle`    |
| `count`  | int    | no       | 1         | Click count (1=single, 2=double)     |
| `x`      | int    | no       |           | X coordinate (if provided, move first) |
| `y`      | int    | no       |           | Y coordinate (if provided, move first) |

**Validation**:
- `button` must be one of: `left`, `right`, `middle`
- `count` must be >= 1 and <= 3
- If `x` provided, `y` must also be provided (and vice versa)

---

## Key Names

Standard key identifiers (case-insensitive):

### Alphanumeric
`a`-`z`, `0`-`9`

### Navigation
`up`, `down`, `left`, `right`, `home`, `end`, `pageup`, `pagedown`

### Editing
`backspace`, `delete`, `insert`, `enter`, `tab`, `escape`, `space`

### Function Keys
`f1`-`f12`

### Modifiers (used in `modifiers` array only)
`ctrl`, `alt`, `shift`, `meta`

### Special
`capslock`, `numlock`, `scrolllock`, `printscreen`, `pause`

---

## Validation Rules Summary

| Rule ID | Applies To | Rule                                              |
|---------|------------|---------------------------------------------------|
| V1      | all        | `action` is required and must be valid type       |
| V2      | press      | `key` must be valid key name                      |
| V3      | press      | `modifiers` must not duplicate                    |
| V4      | type       | `text` must be non-empty                          |
| V5      | type       | `delay_ms` >= 0                                   |
| V6      | wait       | `ms` > 0                                          |
| V7      | move       | `x` and `y` required                              |
| V8      | move       | `duration_ms` >= 0                                |
| V9      | click      | `button` in [left, right, middle]                 |
| V10     | click      | `count` in [1, 2, 3]                              |
| V11     | click      | if `x` provided, `y` required                     |

---

## Example: Complete Task IR

**Task**: Open Notepad, type a message, save the file.

```json
{
  "goal": "Create and save a text file in Notepad",
  "steps": [
    {
      "action": "press",
      "key": "r",
      "modifiers": ["meta"]
    },
    {
      "action": "wait",
      "ms": 500
    },
    {
      "action": "type",
      "text": "notepad",
      "delay_ms": 30
    },
    {
      "action": "press",
      "key": "enter"
    },
    {
      "action": "wait",
      "ms": 1000
    },
    {
      "action": "type",
      "text": "Hello from Ntive!\nThis is a deterministic trace.",
      "delay_ms": 20
    },
    {
      "action": "press",
      "key": "s",
      "modifiers": ["ctrl"]
    },
    {
      "action": "wait",
      "ms": 500
    },
    {
      "action": "type",
      "text": "ntive_output.txt"
    },
    {
      "action": "press",
      "key": "enter"
    }
  ]
}
```

---

## Determinism Guarantees

1. **Order preservation**: Steps execute in array order
2. **No implicit waits**: All timing is explicit via `wait` or `duration_ms`
3. **No ambient state**: Each step is self-contained
4. **No runtime decisions**: IR contains all information needed for execution

---

## Extension Points (Future)

Reserved but not yet defined:

| Action     | Purpose                              |
|------------|--------------------------------------|
| `scroll`   | Mouse wheel scrolling                |
| `drag`     | Click-and-drag operation             |
| `hold`     | Hold key/button down                 |
| `release`  | Release held key/button              |
| `screenshot` | Capture screen state (for assertions) |

---

## Mapping to SemanticIR

Human Step IR integrates with the existing trace engine:

```python
# Each HumanStep becomes an IRStep
IRStep(action="press", params={"key": "enter", "modifiers": []})
IRStep(action="type", params={"text": "hello", "delay_ms": 0})
IRStep(action="wait", params={"ms": 1000})
IRStep(action="move", params={"x": 100, "y": 200, "relative": False})
IRStep(action="click", params={"button": "left", "count": 1})
```

The executor interprets these actions; the IR remains pure data.
