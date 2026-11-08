from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Transaction:
    day: date
    category: str
    amount: Decimal  # negative = expense, positive = income
    currency: str
    note: str = ""
