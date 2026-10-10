"""Merge the coverage.json files of a coverage run (step 7) and list, per file, the lines that never ran.

  python coverage_report.py <coverage.json> [<coverage.json> ...] [--files <path under hy3dpaint/> ...]

A line counts as run when any of the given runs ran it (training fresh and resume, the gates). Paths are
shown relative to hy3dpaint/. Without --files: every file that ran at least one line. "Never ran" is not
"dead": config- and env-switched branches (D6), DDP and SLURM paths and upstream code (D9) stay.
"""
import argparse
import json
import sys


def rel(path):
    i = path.find("/hy3dpaint/")
    return path[i + len("/hy3dpaint/"):] if i >= 0 else path


def ranges(nums):
    out, start, prev = [], None, None
    for n in sorted(nums):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = n
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--files", nargs="*")
    a = ap.parse_args()
    executed, statements = {}, {}
    for j in a.jsons:
        for path, f in json.load(open(j))["files"].items():
            r = rel(path)
            executed.setdefault(r, set()).update(f["executed_lines"])
            statements.setdefault(r, set()).update(f["executed_lines"] + f["missing_lines"])
    files = a.files or sorted(r for r in executed if executed[r])
    for r in files:
        if r not in statements:
            print(f"{r}: not imported by any run")
            continue
        miss = statements[r] - executed[r]
        print(f"{r}: {len(statements[r]) - len(miss)}/{len(statements[r])} statements ran; never: "
              + (", ".join(ranges(miss)) if miss else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
