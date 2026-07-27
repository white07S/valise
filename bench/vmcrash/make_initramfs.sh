#!/bin/bash
# Build the guest initramfs + extract the guest kernel for the vmcrash
# campaign, from pre-downloaded Alpine v3.22 aarch64 apks (see
# guest/fetch_apks.py) plus the cross-built static worker binary.
#
# Outputs (under bench/vmcrash/guest/):
#   vmlinuz-virt      - Alpine linux-virt kernel image (from the apk)
#   initramfs.cpio.gz - /init (busybox sh script) + busybox + mkfs tools
#                       + trimmed kernel module tree + /worker (static)
#
# Everything runs on macOS: bsdtar extracts the apks, bsdcpio packs newc.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GUEST="$HERE/guest"
ROOT="$GUEST/rootfs"
WORKER="$HERE/worker/target/aarch64-unknown-linux-musl/release/valise-vmcrash-worker"

[ -x "$WORKER" ] || { echo "worker binary missing: $WORKER" >&2; exit 1; }
ls "$GUEST"/apks/*.apk >/dev/null 2>&1 || { echo "no apks in $GUEST/apks" >&2; exit 1; }

rm -rf "$ROOT"
mkdir -p "$ROOT"

# 1. Extract every apk into the rootfs. apks are gzip tars; ignore the
#    apk metadata entries and extraction warnings about them.
for apk in "$GUEST"/apks/*.apk; do
  tar -xzf "$apk" -C "$ROOT" \
    --exclude '.SIGN.*' --exclude '.PKGINFO' --exclude '.pre-*' \
    --exclude '.post-*' --exclude '.trigger' 2>/dev/null || true
done

# 2. Pull the kernel image out and trim the module tree to what the
#    campaign needs: filesystems + their lib/crypto deps + virtio.
KVER="$(ls "$ROOT/lib/modules" | head -1)"
echo "kernel modules version: $KVER"
cp "$ROOT"/boot/vmlinuz-virt "$GUEST/vmlinuz-virt"
MODS="$ROOT/lib/modules/$KVER"
KEEP="$GUEST/mods-keep"
rm -rf "$KEEP"
mkdir -p "$KEEP"
for sub in \
  kernel/fs/ext4 kernel/fs/jbd2 kernel/fs/mbcache.ko.gz \
  kernel/fs/xfs kernel/fs/btrfs kernel/fs/netfs kernel/fs/nls \
  kernel/lib kernel/crypto kernel/arch \
  kernel/drivers/virtio kernel/drivers/block \
  kernel/drivers/char/hw_random ; do
  src="$MODS/$sub"
  [ -e "$src" ] || continue
  mkdir -p "$KEEP/$(dirname "$sub")"
  cp -R "$src" "$KEEP/$sub"
done
for meta in modules.builtin modules.builtin.modinfo modules.order; do
  cp "$MODS/$meta" "$KEEP/$meta" 2>/dev/null || true
done
rm -rf "$MODS"
mkdir -p "$MODS"
cp -R "$KEEP"/. "$MODS/"
rm -rf "$KEEP" "$ROOT/boot"

# Alpine ships modules as .ko.gz; the kernel's module loader (via
# busybox modprobe, which does not decompress) rejects them with
# "Invalid ELF header magic". Decompress them all at build time.
find "$MODS" -name '*.ko.gz' -exec gunzip {} +

# 3. Install the worker + /init.
install -m 0755 "$WORKER" "$ROOT/worker"
cat > "$ROOT/init" <<'INIT'
#!/bin/sh
# vmcrash guest init: mount pseudo-fs, load fs modules, mkfs/mount the
# data disk per kernel cmdline, then exec the static worker as PID 1.
/bin/busybox mkdir -p /proc /sys /dev /data /tmp
/bin/busybox mount -t proc proc /proc
/bin/busybox mount -t sysfs sys /sys
/bin/busybox mount -t devtmpfs dev /dev 2>/dev/null
/bin/busybox --install -s /bin 2>/dev/null
export PATH=/bin:/sbin:/usr/bin:/usr/sbin

FS= DEV=/dev/vdb MNT= FRESH=0
for tok in $(cat /proc/cmdline); do
  case "$tok" in
    valisefs=*)    FS="${tok#valisefs=}" ;;
    valisedev=*)   DEV="${tok#valisedev=}" ;;
    valisemnt=*)   MNT="${tok#valisemnt=}" ;;
    valisefresh=*) FRESH="${tok#valisefresh=}" ;;
  esac
done

depmod 2>/dev/null
for m in virtio virtio_ring virtio_pci virtio_mmio virtio_blk crc32c \
         crc32c_generic libcrc32c jbd2 mbcache "$FS"; do
  modprobe "$m" 2>/dev/null
done

if [ ! -b "$DEV" ]; then
  echo "INITFAIL no block device $DEV"
  busybox poweroff -f
fi

if [ "$FRESH" = "1" ]; then
  case "$FS" in
    ext4)  mkfs.ext4 -q -F "$DEV" ;;
    xfs)   mkfs.xfs -q -f "$DEV" ;;
    btrfs) mkfs.btrfs -q -f "$DEV" >/dev/null ;;
    *)     false ;;
  esac || { echo "INITFAIL mkfs $FS"; busybox poweroff -f; }
fi

if [ -n "$MNT" ]; then
  mount -t "$FS" -o "$MNT" "$DEV" /data
else
  mount -t "$FS" "$DEV" /data
fi
if [ "$?" != "0" ]; then
  echo "INITFAIL mount $FS mnt=$MNT"
  busybox poweroff -f
fi

echo "BOOTOK fs=$FS mnt=$MNT fresh=$FRESH"
exec /worker
INIT
chmod 0755 "$ROOT/init"

# 4. Pack (newc cpio, gzip).
( cd "$ROOT" && find . | cpio -o -H newc 2>/dev/null ) | gzip -6 > "$GUEST/initramfs.cpio.gz"
echo "kernel:    $(ls -lh "$GUEST/vmlinuz-virt" | awk '{print $5}')"
echo "initramfs: $(ls -lh "$GUEST/initramfs.cpio.gz" | awk '{print $5}')"
