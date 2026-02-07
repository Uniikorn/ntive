# Ntive Syntax Specification v0.1

Minimal syntax for deterministic human automation scripts.

---

## Design Principles

1. **Sequential only** — no control flow
2. **Declarative** — describes *what*, not *how*
3. **Explicit** — no defaults, no inference
4. **Portable** — compiles to Human Step IR

---

## Formal Grammar (EBNF)

```ebnf
(* === Top-Level Structure === *)

script      = { task } ;

task        = "task" , identifier , [ params ] , ":" , newline , body , "end" ;

params      = "(" , [ param_list ] , ")" ;
param_list  = identifier , { "," , identifier } ;

body        = { statement } ;

statement   = command , newline ;


(* === Commands === *)

command     = press_cmd
            | type_cmd
            | wait_cmd
            | move_cmd
            | click_cmd
            ;

press_cmd   = "press" , key , [ "+" , key ] , { "+" , key } ;

type_cmd    = "type" , string_literal , [ "delay" , integer ] ;

wait_cmd    = "wait" , integer ;

move_cmd    = "move" , integer , integer , [ "relative" ] , [ "over" , integer ] ;

click_cmd   = "click" , [ button ] , [ "at" , integer , integer ] , [ integer , "times" ] ;


(* === Terminals === *)

key         = letter | digit | special_key ;
special_key = "enter" | "tab" | "escape" | "space" | "backspace" | "delete"
            | "up" | "down" | "left" | "right"
            | "home" | "end" | "pageup" | "pagedown"
            | "f1" | "f2" | "f3" | "f4" | "f5" | "f6"
            | "f7" | "f8" | "f9" | "f10" | "f11" | "f12"
            | "ctrl" | "alt" | "shift" | "meta"
            ;

button      = "left" | "right" | "middle" ;

identifier  = letter , { letter | digit | "_" } ;

string_literal = '"' , { character } , '"' ;

integer     = digit , { digit } ;

letter      = "a" | ... | "z" | "A" | ... | "Z" ;
digit       = "0" | ... | "9" ;
character   = (* any UTF-8 character except unescaped quote *) ;

newline     = "\n" | "\r\n" ;


(* === Comments === *)

comment     = "#" , { character } , newline ;

(* Comments are ignored by the parser and can appear on any line *)
```

---

## Lexical Rules

| Element         | Rule                                           |
|-----------------|------------------------------------------------|
| Whitespace      | Spaces/tabs ignored except as token separator  |
| Indentation     | Not significant (style only)                   |
| Line endings    | Statement terminator                           |
| Comments        | `#` to end of line                             |
| Strings         | Double-quoted, `\"` for escaped quote          |
| Integers        | Decimal only, no sign                          |
| Identifiers     | Start with letter, then alphanumeric or `_`    |

---

## Command Reference

### `press`

Press one or more keys simultaneously.

```
press <key>
press <modifier> + <key>
press <modifier> + <modifier> + <key>
```

**Rules:**
- Keys are combined with `+`
- Order matters: modifiers before final key
- Maximum 4 keys in combination

**Examples:**
```
press enter
press ctrl + s
press ctrl + shift + escape
```

---

### `type`

Type a string of characters.

```
type "<text>"
type "<text>" delay <ms>
```

**Rules:**
- Text is double-quoted
- `delay` is inter-keystroke delay in milliseconds
- Supports escape sequences: `\"`, `\\`, `\n`, `\t`

**Examples:**
```
type "hello world"
type "username" delay 50
type "line1\nline2"
```

---

### `wait`

Pause for a duration.

```
wait <ms>
```

**Rules:**
- Duration in milliseconds
- Must be > 0

**Examples:**
```
wait 1000
wait 500
```

---

### `move`

Move mouse cursor.

```
move <x> <y>
move <x> <y> relative
move <x> <y> over <ms>
move <x> <y> relative over <ms>
```

**Rules:**
- Coordinates in pixels
- `relative` = offset from current position
- `over` = animation duration (instant if omitted)

**Examples:**
```
move 100 200
move -50 0 relative
move 500 300 over 200
```

---

### `click`

Perform mouse click.

```
click
click <button>
click at <x> <y>
click <button> at <x> <y>
click <button> <n> times
click <button> at <x> <y> <n> times
```

**Rules:**
- `button` defaults to `left`
- `at` moves before clicking
- `times` defaults to 1, max 3

**Examples:**
```
click
click right
click at 100 200
click left at 500 300 2 times
```

---

## Task Definition

Tasks group commands into named units.

```
task <name>:
  <command>
  <command>
  ...
end

task <name>(<arg1>, <arg2>):
  <command>
  ...
end
```

**Rules:**
- Name must be valid identifier
- Arguments are identifiers (not yet bound — schema only)
- Body contains zero or more commands
- Ends with `end` keyword

---

## Valid Script Examples

### Example 1: Open Notepad and Type

```ntive
# Open Notepad via Run dialog
task open_notepad:
  press meta + r
  wait 500
  type "notepad"
  press enter
  wait 1000
end
```

### Example 2: Save File

```ntive
task save_file(filename):
  press ctrl + s
  wait 300
  type filename
  press enter
end
```

### Example 3: Click Button at Position

```ntive
# Navigate and click submit button
task click_submit:
  move 800 600 over 100
  wait 100
  click left
end
```

### Example 4: Form Fill

```ntive
task fill_form(user, pass):
  # Username field
  click at 200 150
  type user delay 30
  
  # Password field  
  press tab
  type pass delay 30
  
  # Submit
  press enter
end
```

### Example 5: Multi-Key Shortcut

```ntive
task force_quit:
  press ctrl + alt + delete
  wait 500
  press alt + t
end
```

---

## Invalid Script Examples

### Error: Missing `end`

```ntive
task incomplete:
  press enter
# ERROR: Expected 'end', got EOF
```

**Error:** `SyntaxError: line 3: expected 'end' to close task 'incomplete'`

---

### Error: Invalid Key Name

```ntive
task bad_key:
  press superkey
end
# ERROR: 'superkey' is not a valid key
```

**Error:** `ParseError: line 2: unknown key 'superkey'`

---

### Error: String Not Quoted

```ntive
task no_quote:
  type hello world
end
# ERROR: Expected string literal
```

**Error:** `SyntaxError: line 2: expected quoted string after 'type'`

---

### Error: Negative Wait

```ntive
task bad_wait:
  wait -100
end
# ERROR: Duration must be positive
```

**Error:** `ValueError: line 2: wait duration must be > 0`

---

### Error: Invalid Click Count

```ntive
task bad_click:
  click left 5 times
end
# ERROR: Click count exceeds maximum
```

**Error:** `ValueError: line 2: click count must be 1-3, got 5`

---

### Error: Loop Attempted

```ntive
task with_loop:
  for i in 1..5:
    press enter
  end
end
# ERROR: Loops not supported
```

**Error:** `SyntaxError: line 2: unexpected token 'for' (loops not supported)`

---

### Error: Conditional Attempted

```ntive
task with_if:
  if condition:
    press enter
  end
end
# ERROR: Conditionals not supported
```

**Error:** `SyntaxError: line 2: unexpected token 'if' (conditionals not supported)`

---

### Error: Import Attempted

```ntive
import utils

task main:
  press enter
end
# ERROR: Imports not supported
```

**Error:** `SyntaxError: line 1: unexpected token 'import' (imports not supported)`

---

### Error: Empty Task Name

```ntive
task :
  press enter
end
# ERROR: Task name required
```

**Error:** `SyntaxError: line 1: expected identifier after 'task'`

---

### Error: Unclosed String

```ntive
task unclosed:
  type "hello
end
```

**Error:** `SyntaxError: line 2: unterminated string literal`

---

### Error: Missing Coordinates

```ntive
task bad_move:
  move 100
end
```

**Error:** `SyntaxError: line 2: 'move' requires x and y coordinates`

---

## Reserved Words

Cannot be used as identifiers:

```
task end press type wait move click
at over relative times delay
left right middle
ctrl alt shift meta
enter tab escape space backspace delete
up down left right home end pageup pagedown
f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12
```

---

## Compilation Target

Ntive scripts compile to Human Step IR (see `docs/human_step_ir_schema.md`).

```
task example:           →  { "goal": "example",
  press ctrl + s              "steps": [
  wait 500                      {"action": "press", "key": "s", "modifiers": ["ctrl"]},
end                             {"action": "wait", "ms": 500}
                              ]
                            }
```

---

## Version

- Specification: 0.1
- Status: Draft
- Execution: Not implemented (schema only)
