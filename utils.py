import os


def expanded_join(path: str, *paths: str) -> str:
    """Path concatenation utility function.
    Automatically handle slash and backslash depending on the OS but also relative path user.

    :param path: Most left path.
    :param paths: Most right parts of path to concatenate.
    :return: A string which contains the absolute path.
    """
    return os.path.expanduser(os.path.join(path, *paths))
