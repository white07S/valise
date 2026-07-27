#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Resolve and download Alpine apks (with so: dependency closure) from APKINDEX.

Usage: fetch_apks.py <apkindex-path> <mirror-base-url> <out-dir> <pkg>...
Downloads each requested package plus the transitive closure of its D: deps,
resolving `so:` / plain-name deps through P:/p: provides entries.
"""

import sys
import os
import urllib.request

def parse_index(path):
    pkgs = {}
    provides = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        block = {}
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if "P" in block:
                    name = block["P"]
                    pkgs[name] = block
                    provides.setdefault(name, name)
                    for p in block.get("p", "").split():
                        pname = p.split("=")[0]
                        provides.setdefault(pname, name)
                block = {}
                continue
            if len(line) > 2 and line[1] == ":":
                block[line[0]] = line[2:]
    return pkgs, provides

def resolve(pkgs, provides, roots):
    out, todo = set(), list(roots)
    while todo:
        want = todo.pop()
        if want.startswith("!"):
            continue
        base = want.split("=")[0].split(">")[0].split("<")[0].split("~")[0]
        real = provides.get(base)
        if real is None:
            print(f"WARN unresolved dep: {want}", file=sys.stderr)
            continue
        if real in out:
            continue
        out.add(real)
        for d in pkgs[real].get("D", "").split():
            todo.append(d)
    return sorted(out)

def main():
    index, base_url, out_dir, *roots = sys.argv[1:]
    pkgs, provides = parse_index(index)
    names = resolve(pkgs, provides, roots)
    os.makedirs(out_dir, exist_ok=True)
    for n in names:
        ver = pkgs[n]["V"]
        fname = f"{n}-{ver}.apk"
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            print(f"have {fname}")
            continue
        url = f"{base_url}/{fname}"
        print(f"GET  {url}")
        urllib.request.urlretrieve(url, dest)

if __name__ == "__main__":
    main()
