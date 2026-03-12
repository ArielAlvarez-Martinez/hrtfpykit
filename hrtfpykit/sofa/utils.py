import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar
from .conventions import CONVENTIONS

P = ParamSpec("P")
T = TypeVar("T")


def available_conventions() -> None:
    """Print the list of available SOFA conventions and versions.

    The table is built from the local ``CONVENTIONS`` registry.

    Raises
    ------
    ValueError
        If no conventions are registered.

    """
    if len(CONVENTIONS) is False:
        raise ValueError("There is no conventions available yet")

    rows = [("AVAILABLE CONVENTIONS", "VERSION")]
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



def time_it(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that prints the execution time of the wrapped function.

    Parameters
    ----------
    func : Callable[..., T]
        Function to wrap.

    Returns
    -------
    Callable[..., T]
        Wrapped function that prints elapsed time and returns the result.

    Examples
    --------
    >>> from hrtfpykit.sofa.utils import time_it
    >>> @time_it
    ... def add(a, b):
    ...     return a + b
    >>> _ = add(1, 2)
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}() took {end - start:.6f} seconds")
        return result
    return wrapper



def print_return(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that prints the return value of the wrapped function.

    Parameters
    ----------
    func : Callable[..., T]
        Function to wrap.

    Returns
    -------
    Callable[..., T]
        Wrapped function that prints and returns the result.

    Examples
    --------
    >>> from hrtfpykit.sofa.utils import print_return
    >>> @print_return
    ... def greet(name):
    ...     return f"hello {name}"
    >>> _ = greet("SOFA")
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result!r}")
        return result
    return wrapper
