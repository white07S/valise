#!/usr/bin/env bash
#
# run_remote.sh — cross-machine leg of the Valise portability benchmark.
#
# Ships the built artifacts to a Linux x86-64 host (`lab`, reachable over
# passwordless ssh), runs the cold-start `probe` on the transferred bytes,
# and pulls the probe JSON back. The whole point is to demonstrate the
# portability difference: Valise moves as ONE file; the composed deployment
# must be tar-gzipped, transferred, and unpacked into a directory of
# sidecars before it can answer a query. The scp wall time is captured for
# both so the "bytes + packaging" axes are measured end to end.
#
# The top-10 id lists in the two probe JSONs (local vs remote) are meant to
# be diffed: Valise's results are byte-identical across macOS and Linux;
# whether the composed deployment reproduces is exactly what the experiment
# reports.
#
# Usage:
#   bench/portability/run_remote.sh <scenario> <local-out-dir>
#
#   <scenario>       airgapped-manual | agent-corpus
#   <local-out-dir>  the --out-dir a prior `build` wrote (holds valise/,
#                    composed/, composed.tar.gz, build.json)
#
# Env overrides:
#   REMOTE       ssh host alias                (default: lab)
#   REMOTE_REPO  repo path on the remote       (default: ~/valise_file_crash)
#   REMOTE_WORK  scratch dir on the remote     (default: ~/valise_portability)
#   DATASET_DIR  local cohere dataset dir (holds queries.f32)
#                                              (default: bench/datasets/cohere-medium-1m-f32)
#
# It is intentionally simple and readable — meant to be run by hand.

set -euo pipefail

SCENARIO="${1:?usage: run_remote.sh <scenario> <local-out-dir>}"
OUT_DIR="${2:?usage: run_remote.sh <scenario> <local-out-dir>}"

REMOTE="${REMOTE:-lab}"
REMOTE_REPO="${REMOTE_REPO:-~/valise_file_crash}"
REMOTE_WORK="${REMOTE_WORK:-~/valise_portability}"
DATASET_DIR="${DATASET_DIR:-bench/datasets/cohere-medium-1m-f32}"

QUERIES_LOCAL="bench/portability/queries.${SCENARIO}.json"
BIN="target/release/valise-portability-bench"

VALISE_LOCAL="${OUT_DIR}/valise/corpus.vls"
TAR_LOCAL="${OUT_DIR}/composed.tar.gz"
QUERIES_F32_LOCAL="${DATASET_DIR}/queries.f32"

for f in "$VALISE_LOCAL" "$TAR_LOCAL" "$QUERIES_LOCAL" "$QUERIES_F32_LOCAL"; do
  [ -e "$f" ] || { echo "missing required input: $f" >&2; exit 1; }
done

REMOTE_SCEN="${REMOTE_WORK}/${SCENARIO}"

echo "==== Valise portability: remote leg (${REMOTE}, scenario=${SCENARIO}) ===="
echo "-- remote workdir: ${REMOTE_SCEN}"
ssh "$REMOTE" "mkdir -p ${REMOTE_SCEN}/valise ${REMOTE_SCEN}/dataset ${REMOTE_SCEN}/out"

# --- (a) transfer, with wall-clock capture ------------------------------------
echo
echo "-- scp Valise single file ($(du -h "$VALISE_LOCAL" | cut -f1)) --"
time scp -q "$VALISE_LOCAL" "${REMOTE}:${REMOTE_SCEN}/valise/corpus.vls"

echo
echo "-- scp composed tarball ($(du -h "$TAR_LOCAL" | cut -f1)) --"
time scp -q "$TAR_LOCAL" "${REMOTE}:${REMOTE_SCEN}/composed.tar.gz"

echo
echo "-- unpack composed tarball on ${REMOTE} --"
time ssh "$REMOTE" "cd ${REMOTE_SCEN} && tar -xzf composed.tar.gz"

echo
echo "-- scp probe inputs (queries.json + queries.f32) --"
scp -q "$QUERIES_LOCAL" "${REMOTE}:${REMOTE_SCEN}/queries.json"
scp -q "$QUERIES_F32_LOCAL" "${REMOTE}:${REMOTE_SCEN}/dataset/queries.f32"

# --- (b) build the bench bin (if needed) + run both probes on the remote ------
echo
echo "-- build valise-portability-bench on ${REMOTE} (if needed) --"
ssh "$REMOTE" "export PATH=\"\$HOME/.cargo/bin:\$PATH\"; cd ${REMOTE_REPO} && cargo build --release -p valise-bench --bin valise-portability-bench"

REMOTE_BIN="${REMOTE_REPO}/${BIN}"

echo
echo "-- probe Valise (remote) --"
ssh "$REMOTE" "${REMOTE_BIN} probe \
  --deployment valise \
  --path ${REMOTE_SCEN}/valise/corpus.vls \
  --queries ${REMOTE_SCEN}/queries.json \
  --dataset-dir ${REMOTE_SCEN}/dataset \
  --out ${REMOTE_SCEN}/out/probe-valise.json"

echo
echo "-- probe composed (remote) --"
ssh "$REMOTE" "${REMOTE_BIN} probe \
  --deployment composed \
  --path ${REMOTE_SCEN}/composed \
  --queries ${REMOTE_SCEN}/queries.json \
  --dataset-dir ${REMOTE_SCEN}/dataset \
  --out ${REMOTE_SCEN}/out/probe-composed.json"

# --- (c) pull the JSON back + combined summary --------------------------------
mkdir -p "${OUT_DIR}/remote"
scp -q "${REMOTE}:${REMOTE_SCEN}/out/probe-valise.json" "${OUT_DIR}/remote/probe-valise.json"
scp -q "${REMOTE}:${REMOTE_SCEN}/out/probe-composed.json" "${OUT_DIR}/remote/probe-composed.json"

echo
echo "==== remote probe JSON (scenario=${SCENARIO}) ===="
echo "--- Valise (remote) ---"
cat "${OUT_DIR}/remote/probe-valise.json"
echo
echo "--- composed (remote) ---"
cat "${OUT_DIR}/remote/probe-composed.json"
echo
echo "==== reproducibility check (diff top-10 id lists local vs remote) ===="
echo "Compare against the LOCAL probe JSONs, e.g.:"
echo "  diff <(jq -S '{t:.top10_text_ids,v:.top10_vector_ids,h:.top10_hybrid_ids}' ${OUT_DIR}/probe-valise.json) \\"
echo "       <(jq -S '{t:.top10_text_ids,v:.top10_vector_ids,h:.top10_hybrid_ids}' ${OUT_DIR}/remote/probe-valise.json)"
echo "Valise should diff clean (byte-identical across macOS/Linux)."
