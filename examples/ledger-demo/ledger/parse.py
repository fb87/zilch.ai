"""Parse transactions from CSV text.

Format (header required):  date,category,amount,currency,note
  - date: ISO YYYY-MM-DD
  - amount: may use a thousands separator, e.g. "1,234.50" (then the field is quoted)
  - note: optional, may contain commas (then quoted)
"""
from datetime import date
from decimal import Decimal

from .models import Transaction


def parse_amount(text):
    return Decimal(text)


def parse_csv(text):
    lines = text.strip().splitlines()
    out = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        day, category, amount, currency = parts[0], parts[1], parts[2], parts[3]
        note = parts[4] if len(parts) > 4 else ""
        out.append(Transaction(date.fromisoformat(day), category.strip(), parse_amount(amount), currency.strip(), note))
    return out
