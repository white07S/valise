#!/bin/bash
# Render .github/copy-under-write.gif from demo/copy-under-write.tape.
#
#   demo/record.sh              # record against the published wheel
#   VALISE_DEMO_LOCAL=1 demo/record.sh   # ... against the working tree
#
# Stages the demo scripts, a venv holding the `valise` wheel, and the
# `valise` CLI at /tmp/valise-demo, then hands that to vhs. The path is
# fixed rather than a mktemp so the tape can be read, and run, on its
# own.
#
# The recording runs the real thing: the writer really commits for the
# whole take, and the copies are really taken while it does.
#
# Needs: vhs (brew install vhs), python3, cargo.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STAGE=/tmp/valise-demo

echo "== staging $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin"
cp "$HERE/writer.py" "$HERE/verify_copies.py" "$HERE/copy_storm.sh" "$STAGE/"
chmod +x "$STAGE/copy_storm.sh"

echo "== building the valise CLI"
( cd "$ROOT" && cargo build --release --bin valise >/dev/null )
cp "$ROOT/target/release/valise" "$STAGE/bin/valise"

echo "== venv"
python3 -m venv "$STAGE/.venv"
"$STAGE/.venv/bin/pip" install -q --upgrade pip
if [[ -n "${VALISE_DEMO_LOCAL:-}" ]]; then
  "$STAGE/.venv/bin/pip" install -q maturin
  ( cd "$ROOT/bindings/valise-py" && "$STAGE/.venv/bin/maturin" develop -q )
else
  "$STAGE/.venv/bin/pip" install -q valise
fi

echo "== recording"
cd "$ROOT"
vhs "$HERE/copy-under-write.tape"

echo "== done"
ls -lh "$ROOT/.github/copy-under-write.gif"
