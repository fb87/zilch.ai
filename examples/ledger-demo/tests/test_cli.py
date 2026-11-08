import io
import unittest
from pathlib import Path

from ledger.cli import main

SAMPLE = str(Path(__file__).parent.parent / "data" / "sample.csv")


def run(*args):
    buf = io.StringIO()
    code = main([SAMPLE, *args], out=buf)
    return code, buf.getvalue().splitlines()


class CliTests(unittest.TestCase):
    def test_summary(self):
        code, lines = run("--currency", "EUR")
        self.assertEqual(code, 0)
        self.assertEqual(lines[-1], "TOTAL           5288.69 EUR")

    def test_category_filter(self):
        code, lines = run("--currency", "EUR", "--category", "groceries", "--category", "rent")
        self.assertEqual(lines, [
            "groceries       -146.55 EUR",
            "rent            -950.00 EUR",
            "TOTAL          -1096.55 EUR",
        ])

    def test_monthly(self):
        code, lines = run("--currency", "EUR", "--monthly")
        self.assertEqual(lines, [
            "2026-01  income    3200.00  expenses   -1032.40  net    2167.60 EUR",
            "2026-02  income    3200.00  expenses     -78.91  net    3121.09 EUR",
        ])


if __name__ == "__main__":
    unittest.main()
