import unittest
from datetime import date
from decimal import Decimal

from ledger.models import Transaction as T
from ledger.report import by_category, monthly_totals, total

TXNS = [
    T(date(2026, 1, 3), "salary", Decimal("3200.00"), "EUR"),
    T(date(2026, 1, 5), "groceries", Decimal("-82.40"), "EUR"),
    T(date(2026, 1, 17), "rent", Decimal("-950.00"), "EUR"),
    T(date(2026, 2, 2), "salary", Decimal("3200.00"), "EUR"),
    T(date(2026, 2, 9), "books", Decimal("-12.99"), "GBP"),
    T(date(2026, 2, 21), "groceries", Decimal("-64.15"), "EUR"),
]


class ReportTests(unittest.TestCase):
    def test_total_eur(self):
        self.assertEqual(total(TXNS, "EUR"), Decimal("5288.69"))

    def test_by_category(self):
        cats = by_category(TXNS, "EUR")
        self.assertEqual(cats["groceries"], Decimal("-146.55"))
        self.assertEqual(cats["books"], Decimal("-14.76"))

    def test_monthly_totals(self):
        m = monthly_totals(TXNS, "EUR")
        self.assertEqual(list(m), ["2026-01", "2026-02"])
        self.assertEqual(m["2026-01"], {"income": Decimal("3200.00"), "expenses": Decimal("-1032.40"), "net": Decimal("2167.60")})
        self.assertEqual(m["2026-02"], {"income": Decimal("3200.00"), "expenses": Decimal("-78.91"), "net": Decimal("3121.09")})

    def test_monthly_empty(self):
        self.assertEqual(monthly_totals([], "USD"), {})


if __name__ == "__main__":
    unittest.main()
