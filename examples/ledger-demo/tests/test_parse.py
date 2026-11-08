import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from ledger.parse import parse_amount, parse_csv

SAMPLE = (Path(__file__).parent.parent / "data" / "sample.csv").read_text()


class ParseTests(unittest.TestCase):
    def test_plain_amounts(self):
        self.assertEqual(parse_amount("-82.40"), Decimal("-82.40"))
        self.assertEqual(parse_amount("12"), Decimal("12"))

    def test_thousands_separator(self):
        self.assertEqual(parse_amount("1,234.50"), Decimal("1234.50"))
        self.assertEqual(parse_amount("-12,000"), Decimal("-12000"))

    def test_sample_file(self):
        txns = parse_csv(SAMPLE)
        self.assertEqual(len(txns), 6)
        self.assertEqual(txns[0].amount, Decimal("3200.00"))
        self.assertEqual(txns[0].day, date(2026, 1, 3))
        self.assertEqual(txns[1].note, "weekly shop, market")
        self.assertEqual(txns[2].note, "")
        self.assertEqual(txns[4].currency, "GBP")

    def test_bad_date_reports_line(self):
        with self.assertRaises(ValueError) as cm:
            parse_csv("date,category,amount,currency,note\n2026-13-01,x,1,USD,\n")
        self.assertIn("line 2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
