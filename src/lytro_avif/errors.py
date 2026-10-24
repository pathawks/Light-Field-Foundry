"""Errors safe to show at the command line without a traceback."""


class LytroError(Exception):
    """A failed input, native-tool, or output contract."""
