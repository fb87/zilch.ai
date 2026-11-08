"""Aggregations over transactions."""
from collections import defaultdict
from decimal import Decimal

from .currency import convert


def total(txns, currency="USD"):
    return sum((convert(t.amount, t.currency, currency) for t in txns), Decimal("0.00"))


def by_category(txns, currency="USD"):
    out = defaultdict(lambda: Decimal("0.00"))
    for t in txns:
        out[t.category] += convert(t.amount, t.currency, currency)
    return dict(out)


def monthly_totals(txns, currency="USD"):
    """Return {"YYYY-MM": {"income": Decimal, "expenses": Decimal, "net": Decimal}} sorted by month.

    income   = sum of positive amounts, expenses = sum of negative amounts (kept negative),
    net = income + expenses. Each transaction is converted with convert() before summing
    (same per-transaction rounding as total()).
    """
    raise NotImplementedError("monthly_totals is not implemented yet (see ISSUES.md #3)")
