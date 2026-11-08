# Open issues

All four must be fixed. The test suite (`./run_tests.sh`) encodes the expected behaviour;
it must pass completely when you are done. Don't change the tests.

## 1. CSV parsing breaks on quoted fields (`ledger/parse.py`)
`parse_csv` splits lines on `,`, so quoted fields such as `"3,200.00"` or a note like
`"weekly shop, market"` are torn apart. `parse_amount` also rejects thousands separators
(`"1,234.50"`). Use proper CSV parsing. A malformed row must raise `ValueError` whose message
contains the line number, e.g. `line 2: ...` (the header is line 1).

## 2. Currency conversion is inverted (`ledger/currency.py`)
`RATES[c]` is how many USD one unit of `c` is worth, but `convert` divides where it should
multiply (100 EUR comes out as ~90.91 USD instead of 110.00).

## 3. Implement `monthly_totals` (`ledger/report.py`)
See its docstring: per `YYYY-MM` month, income (positive sum), expenses (negative sum) and net,
months in ascending order, each transaction converted with `convert()` before summing.

## 4. CLI: `--category` filter and `--monthly` view (`ledger/cli.py`) — depends on #3
- `--category NAME` (repeatable): only include those categories in the summary and TOTAL.
- `--monthly`: instead of the category summary, print one line per month:
  `f"{month}  income {income:>10}  expenses {expenses:>10}  net {net:>10} {currency}"`
