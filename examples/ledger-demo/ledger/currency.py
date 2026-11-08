"""Currency conversion with a fixed rate table.

RATES maps a currency to how many USD one unit of it is worth (1 EUR = 1.10 USD).
"""
from decimal import Decimal, ROUND_HALF_UP

RATES = {"USD": Decimal("1"), "EUR": Decimal("1.10"), "GBP": Decimal("1.25"), "JPY": Decimal("0.0067")}


def convert(amount, src, dst):
    """Convert `amount` from currency `src` to `dst`, rounded to cents."""
    if src not in RATES or dst not in RATES:
        raise ValueError(f"unknown currency: {src if src not in RATES else dst}")
    usd = amount / RATES[src]
    result = usd * RATES[dst]
    return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
