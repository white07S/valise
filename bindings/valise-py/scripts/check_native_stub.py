#!/usr/bin/env python3
"""CI gate: assert valise._native's public surface matches python/valise/_native.pyi.

This is a NAME-presence check, not a parameter-type diff. It catches the drift
case where the native layer grows or renames a public method/attr but the
hand-written stub (`_native.pyi`) is not updated. It deliberately does not
couple to pyo3 stub-gen macros (which would conflict with keeping the native
signatures stable as a pure refactor).

For each native pyclass (and the module-level functions and exception classes),
we reflect `dir()` at runtime, drop dunders and private (`_`-prefixed) names,
and assert the set of public names exactly matches what the `.pyi` declares.
Exit non-zero on any missing or extra public name.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import valise._native as native

_HERE = Path(__file__).resolve().parent
_PYI = _HERE.parent / "python" / "valise" / "_native.pyi"

# The native classes whose method/attr surface we check.
_CLASSES = [
    "Schema",
    "Space",
    "Record",
    "Search",
    "Stored",
    "Reader",
    "Writer",
    "Store",
    "Partitioned",
    "View",
]
# Module-level functions and the exception classes (checked for presence only).
_MODULE_NAMES = [
    "format_version",
    "ValiseError",
    "CorruptionError",
    "BusyError",
    "UnsupportedError",
    "ValidationError",
    "NotCalibratedError",
    "SchemaMismatchError",
]


def _public(names: object) -> set[str]:
    return {n for n in dir(names) if not n.startswith("_")}


def _stub_class_members(tree: ast.Module) -> dict[str, set[str]]:
    """Map each class in the .pyi to the set of its public member names."""
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        members: set[str] = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not item.name.startswith("_"):
                    members.add(item.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(
                item.target, ast.Name
            ):
                if not item.target.id.startswith("_"):
                    members.add(item.target.id)
        out[node.name] = members
    return out


def _stub_module_names(tree: ast.Module) -> set[str]:
    """Top-level function and class names declared in the .pyi."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def main() -> int:
    source = _PYI.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PYI))
    stub_classes = _stub_class_members(tree)
    stub_module = _stub_module_names(tree)

    errors: list[str] = []

    # Module-level functions + exception classes must be present in the stub.
    for name in _MODULE_NAMES:
        if not hasattr(native, name):
            errors.append(f"native module is missing expected symbol {name!r}")
        if name not in stub_module:
            errors.append(f"_native.pyi is missing module symbol {name!r}")

    # Each pyclass: runtime public names must equal the stub's declared names.
    for cls_name in _CLASSES:
        runtime_cls = getattr(native, cls_name, None)
        if runtime_cls is None:
            errors.append(f"native module is missing class {cls_name!r}")
            continue
        if cls_name not in stub_classes:
            errors.append(f"_native.pyi is missing class {cls_name!r}")
            continue
        runtime_members = _public(runtime_cls)
        stub_members = stub_classes[cls_name]
        missing = runtime_members - stub_members
        extra = stub_members - runtime_members
        if missing:
            errors.append(
                f"{cls_name}: native has public members not in stub: "
                f"{sorted(missing)}"
            )
        if extra:
            errors.append(
                f"{cls_name}: stub declares members not on native class: "
                f"{sorted(extra)}"
            )

    if errors:
        print("native-stub drift detected:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_native_stub: OK (native surface matches _native.pyi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
