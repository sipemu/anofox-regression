#!/usr/bin/env python3
"""Compare regenerated fixtures with the committed ones, ignoring layout.

Comment lines (generation date, R and package versions) are dropped, all
whitespace is removed and trailing commas before `]` / `)` / `}` are
dropped, so a fixture that `cargo fmt` reflowed still matches its
generator's output. Exits 1 and lists the fixtures whose values changed.

    python3 validation/compare_fixtures.py REGENERATED_DIR [tests/fixtures]
"""
import pathlib
import re
import sys


def normalized(path: pathlib.Path) -> str:
    code = [l for l in path.read_text().splitlines() if not l.lstrip().startswith("//")]
    flat = re.sub(r"\s+", "", "".join(code))
    return re.sub(r",([\]\)\}])", r"\1", flat)


def main() -> int:
    new_dir = pathlib.Path(sys.argv[1])
    old_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures")
    changed = []
    for new in sorted(new_dir.glob("*.rs")):
        old = old_dir / new.name
        if not old.exists() or normalized(old) != normalized(new):
            changed.append(new.name)
    for name in changed:
        print(f"changed: {name}")
    print(f"{len(changed)} of {len(list(new_dir.glob('*.rs')))} fixtures changed")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
