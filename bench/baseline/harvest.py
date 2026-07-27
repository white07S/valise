#!/usr/bin/env python3
"""Harvest criterion estimates from target/criterion into a flat JSON
snapshot. Writes to stdout — redirect into a per-arch file under
bench/baseline/.

Usage:
    cargo bench --features bench
    python3 bench/baseline/harvest.py > bench/baseline/neon_m_series.json
"""
import datetime
import json
import pathlib
import platform
import subprocess
import sys


def cpu_brand() -> str:
    if platform.system() == "Darwin":
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        return r.stdout.strip()
    if platform.system() == "Linux":
        try:
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return ""


def main() -> int:
    root = pathlib.Path("target/criterion")
    if not root.exists():
        print("target/criterion missing — run `cargo bench --features bench` first.",
              file=sys.stderr)
        return 1

    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    out = {
        "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
        "arch": platform.machine(),
        "os": platform.system(),
        "hostname": platform.node(),
        "cpu_brand": cpu_brand(),
        "rustc": rustc,
        "kernels": {},
    }

    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir() or group_dir.name in ("report",):
            continue
        group = {}
        for variant_dir in sorted(group_dir.iterdir()):
            if not variant_dir.is_dir() or variant_dir.name in ("report",):
                continue
            sub = {}
            sizes = sorted(
                [d for d in variant_dir.iterdir() if d.is_dir() and d.name not in ("report",)],
                key=lambda p: int(p.name) if p.name.isdigit() else 1_000_000,
            )
            for size_dir in sizes:
                est = size_dir / "new" / "estimates.json"
                if not est.exists():
                    continue
                data = json.loads(est.read_text())
                mean = data.get("mean", {}).get("point_estimate")
                median = data.get("median", {}).get("point_estimate")
                if mean is None:
                    continue
                sub[size_dir.name] = {
                    "mean_ns": round(mean, 3),
                    "median_ns": round(median, 3) if median else None,
                }
            if sub:
                group[variant_dir.name] = sub
        if group:
            out["kernels"][group_dir.name] = group

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
