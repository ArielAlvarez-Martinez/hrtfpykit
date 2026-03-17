import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

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
