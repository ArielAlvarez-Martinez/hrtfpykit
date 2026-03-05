from .conventions import CONVENTIONS
import time
from functools import wraps

def available_conventions() -> None:
    if len(CONVENTIONS) is False:
        raise ValueError("There is no conventions available yet") 

    rows = [("AVAILABLE CONVENTIONS","VERSION")]
    for convention, versions in sorted(CONVENTIONS.items()):
        version_list = ", ".join(sorted(versions.keys()))
        rows.append((convention, version_list))
    table = _format_table(rows)
    print(table)


def _format_table(rows: list[tuple[str, str]]) -> str:
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    separator = f"-{'-' * (label_width + 2)}-{'-' * (value_width + 2)}-"
    lines = [separator]
    for label, value in rows:
        lines.append(f"| {label.ljust(label_width)} | {value.ljust(value_width)} |")
        lines.append(separator)
    return "\n".join(lines)

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}() took {end - start:.6f} seconds")
        return result
    return wrapper

def print_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result!r}")
        return result
    return wrapper
