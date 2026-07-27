#!/usr/bin/env python3
"""QEMU guest power-cut campaign for Valise (FAST paper durability evidence).

Per iteration:
  1. fresh 2 GiB raw data disk;
  2. COMMIT boot: the guest (Alpine linux-virt kernel + minimal initramfs,
     see make_initramfs.sh) mkfs+mounts the disk, and the static worker
     streams put+commit, acking `ACK <seq> <gen> <sha256/16>` on the
     serial console strictly AFTER commit() returns;
  3. at a seeded delay past the first ack, SIGKILL the qemu process —
     the guest loses its page cache (power cut at the virtual device
     boundary; writes completed by qemu to the host file survive, like
     a powered disk behind a dead host OS);
  4. VERIFY boot: same disk, the worker reopens the store read-only
     (scan-back recovery permitted), byte-verifies every frame against
     the same seeded derivation, reports, powers off;
  5. classify: every commit acked to the HOST (complete ack lines
     received before the kill — the host-observable ack set) must be
     present and byte-correct; no wrong data may ever be served.

Repeated across ext4 data=ordered, ext4 data=writeback, xfs, btrfs.
Configs run in parallel (one qemu each); iterations are sequential
within a config. All randomness is seeded (SplitMix64); the kill delay
value is deterministic, only its wall-clock landing point is real-time.

Usage: campaign.py [--iters 50] [--seed HEX] [--accel hvf]
                   [--configs ext4-ordered,ext4-writeback,xfs,btrfs]
                   [--out bench/results/paper/vmcrash_campaign.json]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GUEST = os.path.join(HERE, "guest")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

MASK64 = (1 << 64) - 1

def splitmix64(x):
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64

def derive_payload(seed, seq):
    """Mirror of worker/src/main.rs::derive_payload — byte-identical."""
    x = splitmix64((seed ^ (seq * 0x9E3779B97F4A7C15)) & MASK64)
    length = 512 + (x % 3584)
    buf = bytearray()
    while len(buf) < length:
        x = splitmix64(x)
        buf += x.to_bytes(8, "little")
    return bytes(buf[:length])

def expected_hash(seed, seq):
    return hashlib.sha256(derive_payload(seed, seq)).hexdigest()[:16]

CONFIGS = {
    "ext4-ordered":   {"fs": "ext4",  "mntopts": "data=ordered"},
    "ext4-writeback": {"fs": "ext4",  "mntopts": "data=writeback"},
    "xfs":            {"fs": "xfs",   "mntopts": ""},
    "btrfs":          {"fs": "btrfs", "mntopts": ""},
}

ACK_RE = re.compile(r"^ACK (\d+) (\d+) ([0-9a-f]{16})$")
GEN_RE = re.compile(r"^GEN (\d+)$")
PRESENT_RE = re.compile(r"^PRESENT (\d+) ([0-9a-f]{16}|-) (ok|BAD.*|ERR:.*)$")
DONE_RE = re.compile(
    r"^VERIFYDONE ok=(true|false) max_seq=(-?\d+) gen=(\d+) unopenable=(true|false)$"
)

def qemu_cmd(args, disk, fs, mntopts, mode, fresh, seed):
    append = (
        f"console=ttyAMA0 rdinit=/init loglevel=4"
        f" valisemode={mode} valisefs={fs} valisedev=/dev/vda"
        f" valisefresh={1 if fresh else 0} valiseseed={seed:#x}"
    )
    if mntopts:
        append += f" valisemnt={mntopts}"
    return [
        args.qemu, "-accel", args.accel, "-M", "virt", "-cpu", args.cpu,
        "-m", "1024",
        "-kernel", os.path.join(GUEST, "vmlinuz-virt"),
        "-initrd", os.path.join(GUEST, "initramfs.cpio.gz"),
        "-append", append,
        "-drive", f"file={disk},format=raw,if=none,id=data,cache={args.cache}",
        "-device", "virtio-blk-pci,drive=data",
        "-serial", "stdio", "-monitor", "none", "-display", "none",
        "-no-reboot",
    ]

class ConsoleReader:
    """Line-reader thread over qemu stdout; complete lines only."""

    def __init__(self, stream):
        self.lines = []
        self.lock = threading.Lock()
        self.first_ack = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(stream,), daemon=True)
        self.thread.start()

    def _run(self, stream):
        for raw in stream:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            with self.lock:
                self.lines.append(line)
            if not self.first_ack.is_set() and ACK_RE.match(line):
                self.first_ack.set()

    def snapshot(self):
        with self.lock:
            return list(self.lines)

def run_commit_boot(args, disk, seed, delay_ms, log):
    """Boot in commit mode, SIGKILL at delay_ms past the first ack.

    Returns (acks, error): acks = {seq: (gen, hash)} from COMPLETE ack
    lines the host had received; the reader thread is joined after the
    kill, so this is exactly the pre-kill host-observable ack set.
    """
    cmd = qemu_cmd(args, disk, log["fs"], log["mntopts"], "commit", True, seed)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    reader = ConsoleReader(proc.stdout)
    t0 = time.monotonic()
    if not reader.first_ack.wait(timeout=args.boot_timeout):
        proc.kill()
        proc.wait()
        reader.thread.join(timeout=10)
        tail = "; ".join(reader.snapshot()[-8:])
        return None, f"no ack within {args.boot_timeout}s (console tail: {tail})"
    log["first_ack_s"] = round(time.monotonic() - t0, 2)
    time.sleep(delay_ms / 1000.0)
    proc.kill()  # SIGKILL: guest power cut
    proc.wait()
    reader.thread.join(timeout=10)
    acks = {}
    for line in reader.snapshot():
        m = ACK_RE.match(line)
        if m:
            acks[int(m.group(1))] = (int(m.group(2)), m.group(3))
    if not acks:
        return None, "ack seen by reader but none parsed (harness bug)"
    return acks, None

def run_verify_boot(args, disk, seed, log):
    """Boot in verify mode; guest powers itself off. Returns parsed report."""
    cmd = qemu_cmd(args, disk, log["fs"], log["mntopts"], "verify", False, seed)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    reader = ConsoleReader(proc.stdout)
    try:
        proc.wait(timeout=args.verify_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    reader.thread.join(timeout=10)
    lines = reader.snapshot()
    report = {"gen": None, "present": {}, "done": None, "unopenable": False}
    for line in lines:
        m = GEN_RE.match(line)
        if m:
            report["gen"] = int(m.group(1))
            continue
        m = PRESENT_RE.match(line)
        if m:
            seq, digest, status = int(m.group(1)), m.group(2), m.group(3)
            report["present"][seq] = (digest, status)
            continue
        m = DONE_RE.match(line)
        if m:
            report["done"] = {
                "ok": m.group(1) == "true",
                "max_seq": int(m.group(2)),
                "gen": int(m.group(3)),
            }
            report["unopenable"] = m.group(4) == "true"
    if report["done"] is None:
        report["console_tail"] = lines[-8:]
    return report

def classify(seed, acks, report):
    """One iteration's verdict from the ack set + verify report."""
    out = {
        "acked_max": max(acks), "acked_count": len(acks),
        "lost": [], "wrong_data": False, "unopenable": False,
        "verify_incomplete": False, "detail": [],
    }
    if report["done"] is None:
        out["verify_incomplete"] = True
        out["detail"].append(
            f"no VERIFYDONE (console tail: {report.get('console_tail')})")
        return out
    if report["unopenable"]:
        out["unopenable"] = True
        return out
    done = report["done"]
    present = report["present"]
    out["present_max"] = done["max_seq"]
    out["served_gen"] = done["gen"]
    # Chain shape: seq s was committed as generation s+2 (create()'s
    # empty snapshot is generation 1). A served snapshot of generation g
    # must therefore expose exactly g-1 frames, seq 0..g-2.
    if done["gen"] != done["max_seq"] + 2:
        out["wrong_data"] = True
        out["detail"].append(
            f"self-inconsistent snapshot: gen {done['gen']} with "
            f"max_seq {done['max_seq']} (want gen == max_seq + 2)")
    # Every host-acked commit must be present and byte-correct.
    for seq in sorted(acks):
        gen, ack_hash = acks[seq]
        want = expected_hash(seed, seq)
        if ack_hash != want:
            out["wrong_data"] = True
            out["detail"].append(
                f"seq {seq}: ack hash {ack_hash} != host derivation {want}")
        got = present.get(seq)
        if got is None:
            out["lost"].append(seq)
            out["detail"].append(f"seq {seq}: acked (gen {gen}) but ABSENT")
        elif got[1] != "ok":
            if got[1].startswith("ERR"):
                out["lost"].append(seq)
            else:
                out["wrong_data"] = True
            out["detail"].append(f"seq {seq}: acked (gen {gen}) but {got[1]}")
        elif got[0] != want:
            out["wrong_data"] = True
            out["detail"].append(
                f"seq {seq}: served hash {got[0]} != expected {want}")
    # Frames past the acked max (commit finished, ack line not received
    # before the kill) must ALSO be byte-correct if served.
    for seq, (digest, status) in sorted(present.items()):
        if seq in acks:
            continue
        if status != "ok" or digest != expected_hash(seed, seq):
            out["wrong_data"] = True
            out["detail"].append(f"unacked seq {seq}: served but {status}")
    out["unacked_tail"] = max(0, done["max_seq"] - out["acked_max"])
    return out

def run_config(args, name, results, errors):
    cfg = CONFIGS[name]
    cfg_seed = splitmix64(args.seed ^ int.from_bytes(name.encode(), "little") & MASK64)
    workdir = tempfile.mkdtemp(prefix=f"vmcrash-{name}-", dir=args.tmpdir)
    disk = os.path.join(workdir, "data.img")
    iters = []
    try:
        for i in range(args.iters):
            it_seed = splitmix64((cfg_seed + i) & MASK64)
            payload_seed = splitmix64(it_seed ^ 0xACC0FFEE)
            delay_ms = 2000 + splitmix64(it_seed ^ 0xDE1A) % 8000
            log = {"iter": i, "fs": cfg["fs"], "mntopts": cfg["mntopts"],
                   "payload_seed": f"{payload_seed:#018x}", "delay_ms": delay_ms}
            subprocess.run(
                [args.qemu_img, "create", "-f", "raw", disk, "2G"],
                check=True, capture_output=True)
            t0 = time.monotonic()
            acks, err = run_commit_boot(args, disk, payload_seed, delay_ms, log)
            if err:
                log["error"] = f"commit boot: {err}"
                iters.append(log)
                errors.append(f"{name}#{i}: {log['error']}")
                continue
            report = run_verify_boot(args, disk, payload_seed, log)
            verdict = classify(payload_seed, acks, report)
            log.update(verdict)
            log["wall_s"] = round(time.monotonic() - t0, 2)
            iters.append(log)
            flag = ("LOST" if verdict["lost"] else
                    "WRONG" if verdict["wrong_data"] else
                    "UNOPENABLE" if verdict["unopenable"] else
                    "INCOMPLETE" if verdict["verify_incomplete"] else "ok")
            print(f"[{name} {i + 1}/{args.iters}] delay={delay_ms}ms "
                  f"acked={verdict['acked_count']} "
                  f"tail={verdict.get('unacked_tail', '?')} {flag} "
                  f"({log['wall_s']}s)", file=sys.stderr)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    ok = sum(1 for l in iters if not l.get("lost") and not l.get("wrong_data")
             and not l.get("unopenable") and not l.get("verify_incomplete")
             and "error" not in l)
    results[name] = {
        "config": {
            "fs": cfg["fs"], "mntopts": cfg["mntopts"] or "(defaults)",
            "cache_mode": args.cache, "accel": args.accel,
        },
        "totals": {
            "iterations": len(iters),
            "all_acked_present": ok,
            "lost_acked": sum(1 for l in iters if l.get("lost")),
            "wrong_data": sum(1 for l in iters if l.get("wrong_data")),
            "unopenable": sum(1 for l in iters if l.get("unopenable")),
            "verify_incomplete": sum(1 for l in iters if l.get("verify_incomplete")),
            "harness_errors": sum(1 for l in iters if "error" in l),
            "acked_commits_total": sum(l.get("acked_count", 0) for l in iters),
            "unacked_tail_total": sum(l.get("unacked_tail", 0) for l in iters),
        },
        "iterations": iters,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=lambda s: int(s, 16), default=0x5EED2026FA57C0DE)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--accel", default="hvf")
    ap.add_argument("--cpu", default="host")
    ap.add_argument("--cache", default="writethrough")
    ap.add_argument("--qemu", default="/opt/homebrew/bin/qemu-system-aarch64")
    ap.add_argument("--qemu-img", default="/opt/homebrew/bin/qemu-img")
    ap.add_argument("--boot-timeout", type=float, default=90.0)
    ap.add_argument("--verify-timeout", type=float, default=180.0)
    ap.add_argument("--tmpdir", default=tempfile.gettempdir())
    ap.add_argument("--out", default=os.path.join(
        REPO, "bench", "results", "paper", "vmcrash_campaign.json"))
    args = ap.parse_args()

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    for n in names:
        if n not in CONFIGS:
            sys.exit(f"unknown config {n!r} (have: {', '.join(CONFIGS)})")

    t0 = time.monotonic()
    results, errors = {}, []
    threads = [threading.Thread(target=run_config, args=(args, n, results, errors))
               for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    kver_dir = os.path.join(GUEST, "rootfs", "lib", "modules")
    kver = os.listdir(kver_dir)[0] if os.path.isdir(kver_dir) else "unknown"
    qemu_ver = subprocess.run([args.qemu, "--version"], capture_output=True,
                              text=True).stdout.splitlines()[0]
    doc = {
        "campaign": "vmcrash: qemu guest power-cut (SIGKILL) durability",
        "host": {"qemu": qemu_ver, "accel": args.accel,
                 "machine": "virt", "cpu": args.cpu,
                 "data_disk": "raw 2G, virtio-blk-pci, cache=" + args.cache},
        "guest": {"kernel": f"Alpine linux-virt {kver} (aarch64)",
                  "userland": "minimal initramfs: busybox + mkfs tools + "
                              "static musl worker as init (bench/vmcrash/)"},
        "protocol": {
            "ack_definition": "complete `ACK seq gen hash` lines received by "
                              "the host before SIGKILL (written by the worker "
                              "strictly after commit() returned)",
            "kill": "SIGKILL of the qemu process at a seeded 2-10 s delay "
                    "past the first ack (guest page cache lost; writes "
                    "completed by qemu to the host file survive)",
            "seed": f"{args.seed:#018x}",
        },
        "results": {n: results[n] for n in names if n in results},
        "harness_errors": errors,
        "campaign_wall_seconds": round(time.monotonic() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)
    for n in names:
        if n in results:
            print(f"{n}: {json.dumps(results[n]['totals'])}", file=sys.stderr)

if __name__ == "__main__":
    main()
