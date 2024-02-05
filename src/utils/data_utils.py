import inspect

from typing import Callable

def get_source_code(fns: list[Callable]) -> str:
    """Get source code of specified functions

    Args:
        fns (list[Callable]): List of functions

    Returns:
        str: Source code of specified functions
    """
    res = []
    for fn in fns:
        res.append(inspect.getsource(fn))

    return '\n'.join(res)


def get_source_code_of_file(script_path: str) -> str:
    """Get source code of file

    Args:
        script_path (str): File paths

    Returns:
        str: source code of file
    """
    source_code = None
    with open(script_path, 'r') as f:
        source_code = f.read()

    return source_code
