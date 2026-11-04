#!/usr/bin/env python3
"""Statistician-skill gate: the skill's API names must exist in the crate.

`skills/anofox-regression-statistician/` documents the crate's API for an
agent (`references/rust_api.md`, `SKILL.md`, the workflow). This check fails
when the skill names a Rust item that the crate does not have, or when the
skill was written for another crate version, so the skill is updated
together with the API.

Checked, in every backtick span of the skill's markdown files:

* CamelCase identifiers (`OlsRegressor`, `IntervalType`, `HcType`) must be
  public types, traits, enum variants or other items of the crate;
* `snake_case` names used as Rust calls or paths, i.e. after `.` or `::`
  (`.with_intercept(`, `solvers::path`, `fit.hc_inference(`), must be public
  functions, methods, fields, modules or variants of the crate.

Names that are not crate items (Rust std / faer types, R and sklearn
functions, prose) go on the allow-list with a reason (several names may
share a line):

    name, other_name   # reason

`pkg::fn` and `pkg.fn` of known R / Python packages are skipped.

The skill's `references/rust_api.md` must also state the crate version it
was verified against as `**<major>.<minor>.<patch>**` in its first lines, and
the major.minor must equal Cargo.toml's: a release bump fails this check
until the skill has been reviewed for the release.

Usage:
    cargo +nightly rustdoc --lib --all-features --target-dir target/skill-pub -- \
        -Z unstable-options --output-format json
    cargo +nightly rustdoc --lib --all-features --target-dir target/skill -- \
        -Z unstable-options --output-format json --document-private-items
    python3 scripts/skill_api_check.py [--json target/doc/anofox_regression.json]
        [--skill skills/anofox-regression-statistician]
        [--allowlist skills/api-allowlist.txt] [--list]
"""

import argparse
import json
import pathlib
import re
import sys

CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b|\b[A-Z][a-z]+[0-9]*\b")
CALL = re.compile(r"(?:\.|::)([a-z_][a-z0-9_]*)")
# `pkg::fn` / `pkg.fn` of R and Python packages are not crate items.
FOREIGN = re.compile(
    r"\b(?:rms|Hmisc|survival|stats|MASS|nlme|lme4|lmerTest|glmnet|sandwich|quantreg|car|mice|"
    r"greybox|mgcv|metafor|statmod|lmtest|sklearn|scipy|numpy|np|statsmodels|pls|nnls|mediation|"
    r"multcomp|mvtnorm|detectseparation|arm|grpreg|broom|bvls|base|utils)(?:::|\.)[A-Za-z_.][A-Za-z0-9_.]*"
)


def crate_names(path):
    """Every name of an item defined in the crate (public or not part of a
    private module is not distinguished here: rustdoc JSON without
    --document-private-items holds the public API only)."""
    d = json.load(open(path))
    names = set()
    for item in d["index"].values():
        if item.get("crate_id") != 0:
            continue
        name = item.get("name")
        if name:
            names.add(name)
    return names


def members(path):
    """{type name: set(member names)} for every struct, enum and trait of the
    crate (fields, variants, methods of all impls incl. trait impls), with
    type aliases resolved, and {type name: builder type name} from each
    type's `builder()` return type."""
    d = json.load(open(path))
    index = d["index"]

    def name_of(t):
        return t.get("resolved_path", {}).get("path", "").split("::")[-1] if isinstance(t, dict) else ""

    out, builders, aliases = {}, {}, {}
    for item in index.values():
        if item.get("crate_id") != 0 or not item.get("name"):
            continue
        kind = next(iter(item["inner"]))
        inner = item["inner"][kind]
        if kind == "type_alias":
            aliases[item["name"]] = name_of(inner.get("type", {}))
            continue
        if kind not in ("struct", "enum", "trait"):
            continue
        names = set()
        ids = list(inner.get("items", []))  # trait items
        if kind == "struct":
            ids += inner.get("kind", {}).get("plain", {}).get("fields", []) if isinstance(inner.get("kind"), dict) else []
        if kind == "enum":
            ids += inner.get("variants", [])
        for imp in inner.get("impls", []):
            ii = index.get(str(imp))
            if ii:
                ids += ii["inner"]["impl"]["items"]
        for i in ids:
            mi = index.get(str(i))
            if not mi or not mi.get("name"):
                continue
            names.add(mi["name"])
            if mi["name"] == "builder" and "function" in mi["inner"]:
                out_t = mi["inner"]["function"]["sig"]["output"]
                b = name_of(out_t) if out_t else ""
                if b:
                    builders[item["name"]] = b
        out.setdefault(item["name"], set()).update(names)
    for a, target in aliases.items():
        if target in out:
            out[a] = out[target]
            if target in builders:
                builders[a] = builders[target]
    return out, builders


PATH = re.compile(r"\b([A-Z][A-Za-z0-9]*)::([a-z_][a-z0-9_]*|[A-Z][A-Za-z0-9]*)")
CHAIN = re.compile(r"\b([A-Z][A-Za-z0-9]*)::builder\(\)((?:\s*\.\s*[a-z_][a-z0-9_]*\s*\([^()]*(?:\([^()]*\)[^()]*)*\))+)")
OPTION = re.compile(r"`\.?([a-z_][a-z0-9_]*)")


def member_errors(text, mem, builders):
    """`Type::member` paths, builder call chains and builder / accessor table
    cells that name members the type does not have."""
    errs = set()
    code = "\n".join(spans(text))
    for t, m in PATH.findall(code):
        if t in mem and m not in mem[t]:
            errs.add(f"`{t}::{m}`: {t} has no member {m}")
    for t, chain in CHAIN.findall(code):
        b = builders.get(t)
        if not b or b not in mem:
            continue
        before_build = re.split(r"\.\s*build\s*\(", chain)[0]
        for m in re.findall(r"\.\s*([a-z_][a-z0-9_]*)\s*\(", before_build):
            if m not in mem[b]:
                errs.add(f"`{t}::builder().{m}(..)`: {b} has no method {m}")
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.lstrip().startswith("|") else []
        if len(cells) < 2:
            continue
        m = re.match(r"`([A-Z][A-Za-z0-9]*)::builder\(\)", cells[0])
        if m and builders.get(m.group(1)) in mem:
            t, b = m.group(1), builders[m.group(1)]
            for opt in OPTION.findall(cells[1]):
                if opt not in mem[b]:
                    errs.add(f"builder table: {b} has no option `{opt}`")
        for cell in cells[1:]:
            m = re.match(r"`(Fitted[A-Za-z0-9]*)`\s*:", cell)
            if m and m.group(1) in mem:
                f = m.group(1)
                rest = cell[m.end():]
                for acc in OPTION.findall(rest):
                    if acc not in mem[f] and not any(acc in mem[tr] for tr in ("FittedRegressor", "IterativeFit", "HasModelInfo") if tr in mem):
                        errs.add(f"accessor table: {f} has no member `{acc}`")
    return errs


def spans(text):
    """Inline backtick spans and fenced code blocks."""
    out = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
        out.append(block)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    out.extend(re.findall(r"`([^`\n]+)`", text))
    return out


def candidates(span):
    span = re.sub(r'"[^"]*"', "", span)  # string literals
    span = FOREIGN.sub("", span)
    camel = set(m.group(0) for m in CAMEL.finditer(span))
    calls = set(CALL.findall(span))
    return camel, calls


def load_allowlist(path):
    allowed = set()
    for n, line in enumerate(open(path), 1):
        body, _, reason = line.partition("#")
        body = body.strip()
        if not body:
            continue
        if not reason.strip():
            sys.exit(f"{path}:{n}: allow-list entry without a reason: {body}")
        allowed.update(b.strip() for b in body.split(",") if b.strip())
    return allowed


def crate_version():
    text = open("Cargo.toml").read()
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="target/skill-pub/doc/anofox_regression.json")
    ap.add_argument(
        "--private-json",
        default="target/skill/doc/anofox_regression.json",
        help="rustdoc JSON with --document-private-items, to resolve the members "
        "of types that are usable but not re-exported (e.g. builders)",
    )
    ap.add_argument("--skill", default="skills/anofox-regression-statistician")
    ap.add_argument("--allowlist", default="skills/api-allowlist.txt")
    ap.add_argument("--list", action="store_true", help="print every unknown name")
    args = ap.parse_args()

    names = crate_names(args.json) | crate_names(args.private_json)
    mem, builders = members(args.private_json)
    allowed = load_allowlist(args.allowlist)
    skill = pathlib.Path(args.skill)
    files = [skill / "SKILL.md", *sorted((skill / "references").glob("*.md"))]

    unknown, used = {}, set()
    for f in files:
        for span in spans(f.read_text()):
            camel, calls = candidates(span)
            for name in camel | calls:
                if name in names:
                    continue
                if name in allowed:
                    used.add(name)
                    continue
                unknown.setdefault(name, set()).add(f.name)

    errors = []
    for f in files:
        for e in sorted(member_errors(f.read_text(), mem, builders)):
            errors.append(f"{f.name}: {e}")
    for name in sorted(unknown):
        errors.append(f"unknown name `{name}` in {', '.join(sorted(unknown[name]))}")
    for name in sorted(allowed - used):
        errors.append(f"stale allow-list entry: {name}")

    api = (skill / "references" / "rust_api.md").read_text()
    m = re.search(r"\*\*(\d+)\.(\d+)\.(\d+)\*\*", "\n".join(api.splitlines()[:6]))
    want = ".".join(crate_version().split(".")[:2])
    if not m:
        errors.append("references/rust_api.md: no **x.y.z** crate version in its first lines")
    elif f"{m.group(1)}.{m.group(2)}" != want:
        errors.append(
            f"references/rust_api.md documents {m.group(1)}.{m.group(2)}.{m.group(3)}, "
            f"the crate is {crate_version()}: review the skill for this release "
            "and update the version"
        )

    for e in errors:
        print(e)
    n_files = len(files)
    print(f"skill: {n_files} files checked; {len(unknown)} unknown names, "
          f"{len(allowed - used)} stale allow-list entries")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
