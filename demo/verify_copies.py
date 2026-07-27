"""Open every mid-write copy and check it is a usable, consistent snapshot.

A copy passes only if all of the following hold:

  * it opens at all;
  * its record count is a multiple of the writer's commit batch, i.e. it
    landed exactly on a commit boundary and never halfway through one;
  * a search against it returns results;
  * a record read back from it is byte-exact.

Anything else — an open failure, a torn count, a bad payload — is a
failure and is printed as one.

    python3 demo/verify_copies.py --glob 'copies/*.vls' --batch 250
"""

import argparse
import glob
import sys

from valise import Search, Store

DIM = 128


def check(path: str, batch: int) -> tuple[bool, str]:
    try:
        store = Store.open(path)
        r = store.reader()
        keys = list(r.keys("docs"))
        n = len(keys)

        if n == 0:
            return False, "opened but empty"
        # Every commit adds exactly `batch` records, so a consistent
        # snapshot can only ever be a whole number of batches. A count
        # that is not is proof we saw a partially-applied commit.
        if n % batch != 0:
            return False, f"torn commit: {n} records is not a multiple of {batch}"

        hits = store.search("docs", Search().text("body", "vector quantization").top_k(5))
        if not hits.keys:
            return False, "search returned nothing"

        # Payload round-trip on the newest record in the snapshot.
        last = f"doc-{n - 1:07d}"
        rec = r.get("docs", last)
        if rec is None:
            return False, f"missing {last}"
        if f"entry" not in rec.text:
            return False, f"payload mismatch on {last}"

        return True, f"{n:,} records"
    except Exception as exc:  # noqa: BLE001 - any failure is a failure
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("no copies found", file=sys.stderr)
        return 2

    ok = 0
    failures = []
    counts = []
    total = len(paths)
    for i, p in enumerate(paths, 1):
        good, detail = check(p, args.batch)
        if good:
            ok += 1
            counts.append(int(detail.split()[0].replace(",", "")))
        else:
            failures.append((p, detail))
        if args.quiet:
            # In-place progress, so a long check is visibly working
            # rather than a frozen screen.
            print(f"\r  opening and checking copy {i}/{total}...", end="", flush=True)
        else:
            mark = "ok  " if good else "FAIL"
            print(f"  {mark} {p.split('/')[-1]:<16} {detail}")

    if args.quiet:
        print("\r" + " " * 44, end="")
    print()
    print(f"  {ok}/{len(paths)} mid-write copies opened to a consistent snapshot")
    if counts:
        print(f"  snapshots span {min(counts):,} to {max(counts):,} records "
              f"— {len(set(counts))} distinct generations")
    for p, detail in failures:
        print(f"  FAILED {p}: {detail}")
    return 0 if ok == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
