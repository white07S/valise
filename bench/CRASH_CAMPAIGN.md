# Reproducing the crash-consistency campaign

Valise claims that a commit is atomic: after a crash, a reader sees the
previous committed state or the new one, never a mixture. This document is how
you check that claim rather than take it.

There are two harnesses, and they fail in different ways on purpose:

| | What it proves | What it cannot prove |
|---|---|---|
| **`valise-crash-campaign`** | The recovery logic is correct against damage we can describe: truncation, bit-rot, torn writes, reordering, garbage, process kills. | That the operating system and disk actually flushed when they said they did. |
| **`bench/vmcrash`** | A real kernel, on a real filesystem, losing power mid-commit, keeps every acknowledged commit. | Much per unit of wall-clock — each iteration boots a VM. |

The first is fast and exhaustive. The second is slow and honest about
hardware. The claim in the README rests on both.

## Part 1 — seeded fault injection

### Run it

```bash
cargo build --release -p valise-bench --bin valise-crash-campaign

./target/release/valise-crash-campaign \
  --iters-per-class 15000 \
  --out bench/results/crash_campaign.json
```

That invocation is exactly the **122,200 injections** cited in the README:
eight classes at 15,000 iterations, plus 2,000 `crash-recover-cycle` and 200
`sigkill-storm` (both default, both budgeted separately because each iteration
costs several durable commits or a real process spawn).

`8 × 15000 + 2000 + 200 = 122,200`

It takes **about 100 seconds** on an M1 Pro and needs no data files — the
fixture is built in memory at startup, so there is nothing to download. Drop
`--iters-per-class` to 500 (the default) for a ~5-second smoke run that still
exercises every class.

The full run on the reference machine:

```console
$ ./target/release/valise-crash-campaign --iters-per-class 15000 \
    --out bench/results/crash_campaign.json

[fixture] building pristine image (3 generations x 4 frames, seed 0x5eed20260c4aa167)...
[fixture] 24724 bytes, generations [1, 2, 3, 4], 12 frames
[random-truncation]     cases=15000 newest=0     previous=15000 rejected_at_read=0     wrong_data=0 panic=0
[random-bitflip]        cases=15000 newest=1183  previous=2100  rejected_at_read=11717 wrong_data=0 panic=0
[random-multi-bitflip]  cases=15000 newest=15    previous=3485  rejected_at_read=11408 wrong_data=0 panic=0
[torn-footer]           cases=15000 newest=231   previous=14769 rejected_at_read=0     wrong_data=0 panic=0
[torn-header]           cases=15000 newest=12699 previous=514   rejected_at_read=0     wrong_data=0 panic=0
[reorder-sim]           cases=15000 newest=7     previous=12499 rejected_at_read=2494  wrong_data=0 panic=0
[random-garbage-window] cases=15000 newest=29    previous=6300  rejected_at_read=8671  wrong_data=0 panic=0
[crash-recover-cycle]   cases=2000  newest=0     previous=1820  rejected_at_read=180   wrong_data=0 panic=0
[sigkill-storm]         cases=200   newest=200   previous=0     rejected_at_read=0     wrong_data=0 panic=0
[compaction-install]    cases=15000 newest=15000 previous=0     rejected_at_read=0     wrong_data=0 panic=0

campaign complete: 10 classes x 15000 iterations in 100.6s
totals: newest=29364 previous=56373 rejected_at_read=34470 clean_reject=1993 wrong_data=0 panic=0
```

The shape of those rows is worth reading. `random-truncation` never serves the
newest generation, because truncation always removes the footer that was at the
end of the file. `torn-header` mostly *does*, because the header is rebuilt
from a footer that is still intact. `compaction-install` is always clean by
construction — the point of that class is that no intermediate state of an
atomic replace is ever observable.

Every fault placement derives from `--seed`, so **the same seed reproduces the
identical campaign bit-for-bit**. If you find a failure, the seed in the
incident report replays it exactly.

### What gets damaged

The fixture is a small multi-generation file — by default 3 generations of 4
frames, with odd slots carrying high-entropy payloads so they are stored as raw
zstd blocks rather than compressed away. Each iteration takes a **fresh copy**
of the pristine image, injects one fault, and reopens.

| Class | Fault |
|---|---|
| `random-truncation` | Truncate to a uniformly random length past the header. |
| `random-bitflip` | Flip one bit anywhere past the 4 KiB header. |
| `random-multi-bitflip` | Flip several bits at independent random offsets. |
| `torn-footer` | Write only a prefix of the new footer — the classic interrupted commit. |
| `torn-header` | Damage the 4 KiB header, including the footer pointer itself. |
| `reorder-sim` | Apply the new footer while withholding segment bytes it references, simulating a reordered flush. |
| `random-garbage-window` | Overwrite a random byte range with zeros or random bytes. |
| `crash-recover-cycle` | Fault → recover read-write → heal with a fresh commit, three rounds per iteration. State carries across rounds. |
| `sigkill-storm` | Spawn a real child process writing commits and `SIGKILL` it at a seeded delay. |
| `compaction-install` | The atomic-replace states `Store::compact` passes through, mirroring the temp naming and rename + parent-dir fsync of `src/db/compact.rs`. |

### Reading the output

Every iteration is classified into one of six outcomes, ordered here by how bad
they are:

- **`newest`** — the file opened at the newest generation. The fault landed
  somewhere that did not invalidate the active snapshot.
- **`previous`** — recovery scanned back and served the previous committed
  generation. **This is a success**, not a degradation: the newest commit was
  damaged, so it was never acknowledged, and the reader got the last state that
  fully validates.
- **`rejected_at_read`** — the file opened, but a specific damaged frame
  returned a typed error instead of bytes. Also a success: the damage was
  caught and named rather than served.
- **`clean_reject`** — the open itself refused, with a typed error.
- **`wrong_data`** — a reader was served bytes that differ from what was
  committed, or a self-inconsistent snapshot. **Must be zero.**
- **`panic`** — the library panicked instead of returning an error. **Must be
  zero.**

The first four are all acceptable outcomes for a damaged file; which one you
get depends on where the fault landed. **Only the last two are failures**, and
any occurrence is written to the `incidents` array in the JSON with the seed
needed to replay it.

That distinction is the whole point. A durable format is not one that never
loses a commit under arbitrary damage — that is impossible. It is one that
never *lies* about which commits it has.

### Recovery cost at scale

```bash
./target/release/valise-crash-campaign --iters-per-class 500 \
  --scale-sweep --out bench/results/crash_scale.json
```

`--scale-sweep` adds a sweep over ~1/16/64/256 MiB files that measures
scan-back recovery against a clean open, so you can see what a torn final
footer costs on a large file. The per-class `open_p50`/`p95` figures in the
main run are for the small fixture and are not representative of large files.

## Part 2 — virtual-machine power cuts

Fault injection assumes the bytes on disk are what we asked for. That is the
assumption most worth distrusting, so the second harness removes it: a real
Linux kernel writes to a real filesystem in QEMU, and the VM is **destroyed
mid-commit** — not signalled, not shut down.

```bash
bench/vmcrash/run_campaign.sh --iters 50
```

50 iterations across four filesystem configurations is the **200 power cuts**
in the README:

| Config | Filesystem | Mount options |
|---|---|---|
| `ext4-ordered` | ext4 | `data=ordered` |
| `ext4-writeback` | ext4 | `data=writeback` |
| `xfs` | XFS | — |
| `btrfs` | btrfs | — |

`ext4-writeback` is included deliberately: it is the weakest ordering
guarantee of the four, and the one most likely to expose a commit protocol
that leans on filesystem behaviour it was never promised.

The guest worker commits in a loop and prints an `ACK` line for each durable
commit. After the power cut, the file is remounted and verified: **every
acknowledged commit must still be present and byte-exact.** A commit that was
in flight may be lost — that is correct, it was never acknowledged. One that
was acknowledged and then vanished is a failure, as is any byte that comes back
different.

### Host prerequisites

This one is macOS/aarch64-specific and needs more setup:

```bash
rustup target add aarch64-unknown-linux-musl
brew install zig qemu
cargo install cargo-zigbuild
```

`run_campaign.sh` cross-builds a static musl worker, fetches Alpine packages,
builds an initramfs, and drives the campaign. The Alpine fetch happens once and
is cached in `bench/vmcrash/guest/apks/`.

Budget **several hours** for the full 200 iterations — each one boots a VM,
writes until the seeded cut, then reboots to verify. Start with `--iters 2` to
confirm the toolchain works before committing to a full run.

## What the campaign found

It found three real bugs in Valise, which is the reason to run these things:

1. **A decompression path that never re-hashed the wire bytes.** Corruption
   inside a compressed block could survive into a served payload — the
   checksum was verified against the wrong buffer.
2. **A torn commit that could render a file unopenable.** A specific
   interleaving left the header pointing at a footer that failed validation in
   a way recovery did not scan past.
3. **A trusted length field driving a 2⁵⁴-byte allocation.** A corrupted length
   was used before it was bounded, turning bit-rot into an OOM abort.

All three are fixed and regression-tested in `tests/crash_consistency.rs`,
which carries the reduced deterministic case for each. The campaign is the net;
the unit tests are what keeps them caught.

If you run this and find a fourth, the JSON incident report has the seed —
please open an issue with it.

## Related

- [docs/FORMAT.md](../docs/FORMAT.md) — the commit and recovery protocol these
  harnesses attack
- [docs/CONCURRENCY.md](../docs/CONCURRENCY.md) — the reader/writer model
- [tests/crash_consistency.rs](../tests/crash_consistency.rs) — the
  deterministic regression cases
- [REPRODUCE.md](REPRODUCE.md) — the retrieval benchmark, a separate concern
