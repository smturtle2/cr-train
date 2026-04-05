from __future__ import annotations

import shutil
import sys
from typing import Any

_DEFAULT_PROGRESS_COLUMNS = 80
_PROGRESS_WIDTH_PADDING = 1


def resolve_progress_bar_ncols(*, file: Any | None = None) -> int | None:
    stream = sys.stderr if file is None else file
    isatty = getattr(stream, "isatty", None)
    if callable(isatty) and not isatty():
        return None

    columns = shutil.get_terminal_size(
        fallback=(_DEFAULT_PROGRESS_COLUMNS, 24),
    ).columns
    return max(1, columns - _PROGRESS_WIDTH_PADDING)


def set_progress_postfix_str(progress: Any, text: str) -> None:
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(text)
        return
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(text)
