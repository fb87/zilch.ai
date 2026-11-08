"""Command line interface:  python3 -m ledger.cli <file.csv> [--currency USD]"""
import argparse
import sys

from .parse import parse_csv
from .report import by_category, total


def main(argv=None, out=sys.stdout):
    ap = argparse.ArgumentParser(prog="ledger")
    ap.add_argument("file")
    ap.add_argument("--currency", default="USD")
    args = ap.parse_args(argv)
    with open(args.file, encoding="utf-8") as f:
        txns = parse_csv(f.read())
    for cat, amt in sorted(by_category(txns, args.currency).items()):
        print(f"{cat:<12} {amt:>10} {args.currency}", file=out)
    print(f"{'TOTAL':<12} {total(txns, args.currency):>10} {args.currency}", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
