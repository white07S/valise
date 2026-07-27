#!/bin/bash
# Take N copies of a capsule while a writer is actively committing to it.
#
#   demo/copy_storm.sh corpus.vls copies 50
#
# Plain `cp`, no coordination with the writer, no quiescing, no lock. The
# point is that this is the naive thing everybody already does, and that
# it is safe here.
set -euo pipefail

SRC="${1:?usage: copy_storm.sh <src.vls> <dest-dir> [count]}"
DEST="${2:?usage: copy_storm.sh <src.vls> <dest-dir> [count]}"
COUNT="${3:-50}"

mkdir -p "$DEST"
rm -f "$DEST"/*.vls

for i in $(seq 1 "$COUNT"); do
  cp "$SRC" "$DEST/copy-$(printf '%02d' "$i").vls"
  printf "\r  copied %d/%d while the writer ran" "$i" "$COUNT"
  sleep 0.04
done
printf "\n"
