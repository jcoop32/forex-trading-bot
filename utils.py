"""
Shared forex utilities to avoid duplication across modules.
"""


def pip_unit(instrument: str) -> float:
    """Return the pip value for a given instrument (0.01 for JPY pairs, 0.0001 for everything else)."""
    return 0.01 if "JPY" in instrument else 0.0001


def price_precision(instrument: str) -> int:
    """Return the number of decimal places for price formatting on a given instrument."""
    return 3 if "JPY" in instrument else 5


def format_price(price: float, instrument: str) -> str:
    """Format a price to the correct number of decimal places for OANDA."""
    precision = price_precision(instrument)
    return f"{price:.{precision}f}"
