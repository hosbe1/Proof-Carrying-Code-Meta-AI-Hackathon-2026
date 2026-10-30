"""Palindrome fixture for the equivalence differential check."""


def is_palindrome(s: str) -> bool:
    """Return True when ``s`` reads the same forwards and backwards."""
    return s == s[::-1]
