"""Project-local file access; importing this module performs no file IO."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def checked_root(root):
    value = Path(root).resolve(strict=False)
    if not value.is_relative_to(PROJECT_ROOT):
        raise ValueError("Project root escapes the authorized repository")
    if not value.is_dir():
        raise ValueError("Project root must be an existing directory")
    return value


def contained_path(root, path, *, exists=False):
    root = checked_root(root)
    value = Path(path)
    value = (value if value.is_absolute() else root / value).resolve(strict=False)
    if not value.is_relative_to(root):
        raise ValueError("Input/output path escapes the project root")
    if exists:
        value = value.resolve(strict=True)
        if not value.is_relative_to(root):
            raise ValueError("Resolved path escapes the project root")
    return value
