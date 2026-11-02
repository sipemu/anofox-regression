#!/usr/bin/env python3
"""JS/WASM parity gate (#287, phase 6).

Lists the public API of the `anofox-regression` crate from its rustdoc JSON
and the exports of the WASM package from its `.d.ts`, and fails when a public
Rust item has no binding and is not on the allow-list.

Checked:

* every public free function has an exported JS function (camelCase name);
* every public type with public inherent methods has a JS class of the same
  name, unless the allow-list maps it to another name or excuses it;
* every public inherent method of a type with a JS class has a JS method
  `camel(name)`, `get` + `Camel(name)` or `set` + `Camel(name)`, or is
  allow-listed.

Usage:
    cargo +nightly rustdoc --lib -- -Z unstable-options --output-format json
    wasm-pack build crates/anofox-regression-js --target web --out-dir pkg
    python3 scripts/js_parity.py [--json target/doc/anofox_regression.json]
        [--dts crates/anofox-regression-js/pkg/anofox_regression_js.d.ts]
        [--allowlist js/parity-allowlist.txt] [--list-missing]

Allow-list format (one entry per line, `#` starts a comment that must give
the reason):

    fn   <rust path>                 # reason        an unbound function
    type <rust path>                 # reason        an unbound type
    type <rust path> => <JsName>     # reason        a renamed class
    method <rust type path>::<name>  # reason        an unbound method
    method <rust type path>::{a,b,c} # reason        several methods (braces
                                                     expand anywhere)
    result <rust type path>::{a,b,c} # where         accessors whose values
                                                     are fields of the JS
                                                     class's getResult()

`result` entries are checked: each name must be a field (`pub name:`) or
an object entry (`("name", …)`) in the source file of the JS crate that
defines the class.

Paths are the crate-relative definition paths shown by `--list-missing`
(e.g. `solvers::ols::OlsRegressor`). Entries may end in `*` to match a
prefix. Stale entries (matching nothing missing) are reported as errors so
the list stays honest.
"""

import argparse
import json
import re
import sys
from fnmatch import fnmatchcase


def camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def kind_of(item):
    return next(iter(item["inner"]))


def rust_api(path):
    """(functions, types) of the crate reachable from its root through public
    modules and `pub use` re-exports: functions as paths, types as
    {path: [public inherent method names]}. Items that are `pub` but sit in
    private modules are not part of the API and are skipped."""
    d = json.load(open(path))
    index, paths = d["index"], d["paths"]
    reachable, seen, stack = set(), set(), [str(d["root"])]
    while stack:
        mid = stack.pop()
        if mid in seen:
            continue
        seen.add(mid)
        module = index[mid]["inner"]["module"]
        for i in module["items"]:
            item = index.get(str(i))
            if not item or item.get("visibility") != "public":
                continue
            kind = kind_of(item)
            if kind == "module":
                stack.append(str(i))
            elif kind == "use":
                use = item["inner"]["use"]
                target = index.get(str(use["id"])) if use["id"] is not None else None
                if target is None:
                    continue
                if use["is_glob"] or kind_of(target) == "module":
                    stack.append(str(use["id"]))
                else:
                    reachable.add(str(use["id"]))
            else:
                reachable.add(str(i))
    functions, types = [], {}
    for iid in reachable:
        p, item = paths.get(iid), index[iid]
        if p is None or p["crate_id"] != 0:
            continue
        rel = "::".join(p["path"][1:])
        if p["kind"] == "function":
            functions.append(rel)
        elif p["kind"] in ("struct", "enum"):
            inner = item["inner"][kind_of(item)]
            methods = []
            for imp in inner.get("impls", []):
                ii = index.get(str(imp))
                if not ii or ii["inner"]["impl"].get("trait") is not None:
                    continue
                for m in ii["inner"]["impl"]["items"]:
                    mi = index.get(str(m))
                    if mi and mi.get("visibility") == "public" and kind_of(mi) == "function":
                        methods.append(mi["name"])
            if methods:
                types[rel] = sorted(set(methods))
    return sorted(set(functions)), types


def js_api(path):
    """(functions, {class: set(methods)}) exported by the `.d.ts`."""
    functions, classes, current = set(), {}, None
    for line in open(path):
        m = re.match(r"export function (\w+)\(", line)
        if m:
            functions.add(m.group(1))
            current = None
            continue
        m = re.match(r"export class (\w+)", line)
        if m:
            current = classes.setdefault(m.group(1), set())
            continue
        if line.startswith("}"):
            current = None
            continue
        if current is not None:
            m = re.match(r"\s+(?:static\s+)?(?:readonly\s+)?(\w+)\s*[(:]", line)
            if m:
                current.add(m.group(1))
            m = re.match(r"\s+constructor\(", line)
            if m:
                current.add("constructor")
    return functions, classes


def expand(pattern):
    """Brace expansion: `a::{b,c}::d` → `a::b::d`, `a::c::d`."""
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end() :]
    return [x for alt in m.group(1).split(",") for x in expand(head + alt.strip() + tail)]


def load_allowlist(path):
    entries = []
    for n, raw in enumerate(open(path), 1):
        line = raw.split("#", 1)
        body = line[0].strip()
        if not body:
            continue
        if len(line) < 2 or not line[1].strip():
            sys.exit(f"{path}:{n}: allow-list entry without a reason: {body}")
        parts = body.split()
        kind, target = parts[0], parts[1]
        rename = parts[3] if len(parts) == 4 and parts[2] == "=>" else None
        if kind not in ("fn", "type", "method", "result") or (len(parts) not in (2, 4)):
            sys.exit(f"{path}:{n}: cannot parse: {body}")
        for t in expand(target):
            entries.append({"kind": kind, "target": t, "rename": rename, "line": n, "used": False})
    return entries


def allowed(entries, kind, target, check=None):
    for e in entries:
        kinds = (kind, "result") if kind == "method" else (kind,)
        if e["kind"] in kinds and e["rename"] is None and fnmatchcase(target, e["target"]):
            if e["kind"] == "result" and check is not None and not check(target):
                continue
            e["used"] = True
            return True
    return False


def class_sources(src_dir):
    """{JS class name: source text of the file defining it}."""
    import pathlib

    out = {}
    for f in pathlib.Path(src_dir).rglob("*.rs"):
        text = f.read_text()
        for m in re.finditer(r"#\[wasm_bindgen[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*pub struct (\w+)", text):
            out[m.group(1)] = text
    return out


def renamed(entries, target):
    for e in entries:
        if e["kind"] == "type" and e["rename"] and e["target"] == target:
            e["used"] = True
            return e["rename"]
    return None


def method_names(name):
    c = camel(name)
    cap = c[:1].upper() + c[1:]
    names = {c, "get" + cap, "set" + cap}
    if name == "new":
        names.add("constructor")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="target/doc/anofox_regression.json")
    ap.add_argument("--dts", default="crates/anofox-regression-js/pkg/anofox_regression_js.d.ts")
    ap.add_argument("--allowlist", default="js/parity-allowlist.txt")
    ap.add_argument("--src", default="crates/anofox-regression-js/src")
    ap.add_argument("--list-missing", action="store_true", help="print missing items as allow-list lines")
    args = ap.parse_args()

    r_funcs, r_types = rust_api(args.json)
    j_funcs, j_classes = js_api(args.dts)
    entries = load_allowlist(args.allowlist)
    sources = class_sources(args.src)
    missing = []

    for f in r_funcs:
        if camel(f.rsplit("::", 1)[-1]) not in j_funcs and not allowed(entries, "fn", f):
            missing.append(("fn", f))

    for t, methods in sorted(r_types.items()):
        js_name = renamed(entries, t) or t.rsplit("::", 1)[-1]
        if js_name not in j_classes:
            if not allowed(entries, "type", t):
                missing.append(("type", t))
            continue
        have = j_classes[js_name]
        for m in methods:
            if have & method_names(m):
                continue
            src = sources.get(js_name, "")

            def in_result(target, src=src):
                name = target.rsplit("::", 1)[-1]
                pat = rf"(pub {name}\s*:|\(\"{name}\",)"
                return re.search(pat, src) is not None

            if not allowed(entries, "method", f"{t}::{m}", in_result):
                missing.append(("method", f"{t}::{m}"))

    stale = [e for e in entries if not e["used"]]
    for kind, target in missing:
        print(f"{kind} {target}" + ("  # TODO" if args.list_missing else "  -- missing JS binding"))
    for e in stale:
        print(f"{args.allowlist}:{e['line']}: stale allow-list entry: {e['kind']} {e['target']}")
    n_items = len(r_funcs) + len(r_types) + sum(len(m) for m in r_types.values())
    print(
        f"parity: {len(r_funcs)} functions, {len(r_types)} types, "
        f"{sum(len(m) for m in r_types.values())} methods checked ({n_items} items); "
        f"{len(missing)} missing, {len(stale)} stale allow-list entries",
        file=sys.stderr,
    )
    return 1 if missing or stale else 0


if __name__ == "__main__":
    sys.exit(main())
