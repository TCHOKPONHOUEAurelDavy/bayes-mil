# Understanding `if __name__ == "__main__":`

In Python, every module has a built-in variable named `__name__`. When a file is executed directly (for example with `python my_script.py`), Python sets `__name__` to the string `"__main__"`. When the same file is imported as a module from another script, `__name__` instead takes the module's import name.

Wrapping code in

```python
if __name__ == "__main__":
    ...
```

ensures that the indented block only runs when the file is executed directly. This is commonly used to:

- Provide a command-line entry point while keeping reusable functions/classes importable without side effects.
- Run quick tests or demonstrations when the script is invoked alone, but avoid running them when the module is imported elsewhere.
- Protect code that should not execute during module import (for example, costly training loops or dataset downloads).

In the context of this repository, you might place training or evaluation logic inside this guard so that the module can be imported by other scripts (e.g., for experimentation or unit testing) without immediately launching a full experiment.
