"""Compare per-test outcomes of two pytest -rA logs and summarize failure reasons.

Usage: python compare_suites.py results/suite-base.log results/suite-new.log
"""

import collections
import re
import sys

OUTCOME = re.compile(r"^(PASSED|FAILED|SKIPPED|ERROR) (\S+)")
REASON = re.compile(r"^E\s+(\S.*)")


def parse(path: str) -> tuple[dict[str, str], collections.Counter, str]:
    outcomes, reasons, summary = {}, collections.Counter(), ""
    with open(path) as f:
        for line in f:
            if m := OUTCOME.match(line):
                outcomes[m[2]] = m[1]
            elif (m := REASON.match(line)) and ("Error" in m[1] or "Check failed" in m[1]):
                reasons[re.sub(r"\d{3,}", "N", m[1])[:140]] += 1
            if re.search(r"(passed|failed).* in [0-9.]+s", line):
                summary = line.strip().strip("=").strip()
    return outcomes, reasons, summary


def main() -> None:
    (a_out, a_why, a_sum), (b_out, b_why, b_sum) = parse(sys.argv[1]), parse(sys.argv[2])
    print(f"A: {sys.argv[1]}: {a_sum}\nB: {sys.argv[2]}: {b_sum}\n")
    diffs = sorted(t for t in a_out.keys() | b_out.keys() if a_out.get(t) != b_out.get(t))
    print(f"Tests whose outcome differs: {len(diffs)}")
    for t in diffs:
        print(f"  {a_out.get(t, 'MISSING'):8s} -> {b_out.get(t, 'MISSING'):8s} {t}")
    for label, why in (("A", a_why), ("B", b_why)):
        print(f"\nFailure reasons in {label}:")
        for reason, count in why.most_common():
            print(f"  {count:4d}  {reason}")


if __name__ == "__main__":
    main()
