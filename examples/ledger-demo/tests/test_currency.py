import unittest
from decimal import Decimal

from ledger.currency import convert


class CurrencyTests(unittest.TestCase):
    def test_to_usd(self):
        self.assertEqual(convert(Decimal("100"), "EUR", "USD"), Decimal("110.00"))
        self.assertEqual(convert(Decimal("1000"), "JPY", "USD"), Decimal("6.70"))

    def test_from_usd(self):
        self.assertEqual(convert(Decimal("110"), "USD", "EUR"), Decimal("100.00"))

    def test_cross(self):
        self.assertEqual(convert(Decimal("10"), "GBP", "EUR"), Decimal("11.36"))

    def test_identity_and_unknown(self):
        self.assertEqual(convert(Decimal("5.555"), "USD", "USD"), Decimal("5.56"))
        with self.assertRaises(ValueError):
            convert(Decimal("1"), "XXX", "USD")


if __name__ == "__main__":
    unittest.main()
