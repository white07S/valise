#!/bin/bash
# End-to-end driver for the QEMU guest power-cut campaign (macOS host).
#
#   bench/vmcrash/run_campaign.sh [campaign.py args...]
#
# Steps: cross-build the static guest worker (aarch64-unknown-linux-musl
# via cargo-zigbuild), fetch the Alpine guest artifacts if missing,
# build the initramfs, then run the campaign. Results land in
# bench/results/paper/vmcrash_campaign.json unless --out overrides.
#
# Host prerequisites: rustup target aarch64-unknown-linux-musl, zig +
# cargo-zigbuild (brew install zig; cargo install cargo-zigbuild),
# qemu-system-aarch64 + qemu-img (brew install qemu), python3.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== worker (static musl, aarch64)"
( cd "$HERE/worker" && cargo zigbuild --release --target aarch64-unknown-linux-musl )

if ! ls "$HERE"/guest/apks/*.apk >/dev/null 2>&1; then
  echo "== fetching Alpine v3.22 aarch64 apks"
  ( cd "$HERE/guest" &&
    curl -sL -o APKINDEX.tar.gz \
      https://dl-cdn.alpinelinux.org/alpine/v3.22/main/aarch64/APKINDEX.tar.gz &&
    tar -xzf APKINDEX.tar.gz APKINDEX &&
    python3 fetch_apks.py APKINDEX \
      https://dl-cdn.alpinelinux.org/alpine/v3.22/main/aarch64 apks \
      linux-virt musl busybox e2fsprogs xfsprogs btrfs-progs )
fi

echo "== initramfs + kernel"
"$HERE/make_initramfs.sh"

echo "== campaign"
exec python3 "$HERE/campaign.py" "$@"
