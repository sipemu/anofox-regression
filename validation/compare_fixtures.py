#!/usr/bin/env python3
"""Compare regenerated fixtures with the committed ones, value by value.

Each `const` of a fixture is parsed into its numbers (comment lines, which
hold the generation date and versions, and string literals are ignored, so
layout does not matter). A value differs when `|new - old| > tol (1 + |old|)`
(default `tol = 1e-8`): looser than the last-digit noise between machines,
BLAS builds and R patch releases (up to ~1e-9), tighter than any test
tolerance. Exits 1 and lists the constants whose values changed.

    python3 validation/compare_fixtures.py REGENERATED_DIR [tests/fixtures] [--tol 1e-8]
"""
import argparse
import pathlib
import re
import sys

NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
CONST = re.compile(r"const\s+(\w+)\s*:[^=]*=\s*(.*?);", re.S)


def constants(path: pathlib.Path) -> dict:
    code = "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("//"))
    out = {}
    for m in CONST.finditer(code):
        body = re.sub(r'"[^"]*"', "", m.group(2))
        out[m.group(1)] = [float(x) for x in NUMBER.findall(body)]
    return out


def differences(old: dict, new: dict, tol: float) -> list:
    diffs = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name), new.get(name)
        if a is None or b is None or len(a) != len(b):
            diffs.append(f"{name}: missing or different length")
            continue
        worst = max((abs(x - y) / (1 + abs(x)) for x, y in zip(a, b)), default=0.0)
        if worst > tol:
            diffs.append(f"{name}: max relative change {worst:.2e}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir", type=pathlib.Path)
    ap.add_argument("old_dir", type=pathlib.Path, nargs="?", default=pathlib.Path("tests/fixtures"))
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()
    files = sorted(args.new_dir.glob("*.rs"))
    changed = 0
    for new in files:
        old = args.old_dir / new.name
        diffs = ["fixture not committed"] if not old.exists() else differences(
            constants(old), constants(new), args.tol)
        if diffs:
            changed += 1
            print(f"{new.name}:")
            for d in diffs:
                print(f"  {d}")
    print(f"{changed} of {len(files)} fixtures changed beyond {args.tol:g}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
