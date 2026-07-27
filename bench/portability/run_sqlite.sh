#!/usr/bin/env bash
#
# run_sqlite.sh — SQLite single-file hybrid baseline for the portability
# experiment (FTS5 + sqlite-vec vec0 + payloads in ONE .db file).
#
# Companion to bench/src/bin/valise-portability-bench.rs / run_remote.sh:
# same scenarios, same probe metrics, same race protocol, third
# deployment "sqlite". Uses the bench python venv (has sqlite-vec).
#
# Usage:
#   bench/portability/run_sqlite.sh local            # build+probe+race, both scenarios
#   bench/portability/run_sqlite.sh remote <scenario> <local-sqlite-dir>
#
# The remote leg ships ONE file (corpus.db — that is the point), installs
# sqlite-vec into the existing ~/diskann-venv on the remote, and runs the
# same cold probe there. It assumes run_remote.sh has already staged
# queries.f32 at ${REMOTE_WORK}/<scenario>/dataset/queries.f32.
#
# Env overrides:
#   REMOTE       ssh host alias        (default: lab)
#   REMOTE_WORK  remote scratch dir    (default: ~/valise_portability)
#   REMOTE_VENV  remote python venv    (default: ~/diskann-venv)
#   OUT_ROOT     local output root     (default: /tmp/valise-port)
#   COPIES       race copies per mode  (default: 50)

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

PY=bench/python/.venv/bin/python
SCRIPT=bench/python/portability_sqlite.py
OUT_ROOT="${OUT_ROOT:-/tmp/valise-port}"
COPIES="${COPIES:-50}"
RESULTS=bench/results/paper

cmd_local() {
  for SCEN in agent-corpus airgapped-manual; do
    SQL_DIR="${OUT_ROOT}/${SCEN}/sqlite"

    echo "==== [${SCEN}] build ===="
    "$PY" "$SCRIPT" build --scenario "$SCEN" --out "$SQL_DIR"
    cp "${SQL_DIR}/build.json" "${RESULTS}/portability_sqlite_build_${SCEN}.json"

    echo "==== [${SCEN}] cold probe ===="
    "$PY" "$SCRIPT" probe \
      --path "${SQL_DIR}/corpus.db" \
      --queries "bench/portability/queries.${SCEN}.json" \
      --dataset-dir bench/datasets/cohere-medium-1m-f32 \
      --out "${OUT_ROOT}/probe_local_${SCEN}_sqlite.json"
    cp "${OUT_ROOT}/probe_local_${SCEN}_sqlite.json" \
       "${RESULTS}/portability_sqlite_probe_local_${SCEN}.json"

    for MODE in delete wal; do
      echo "==== [${SCEN}] race mode=${MODE} copies=${COPIES} ===="
      "$PY" "$SCRIPT" race \
        --scenario "$SCEN" \
        --db "${SQL_DIR}/corpus.db" \
        --copies "$COPIES" \
        --mode "$MODE" \
        --out "${OUT_ROOT}/race_${SCEN}_sqlite_${MODE}.json"
      cp "${OUT_ROOT}/race_${SCEN}_sqlite_${MODE}.json" \
         "${RESULTS}/portability_sqlite_race_${SCEN}_${MODE}.json"
    done
  done
}

cmd_remote() {
  SCEN="${1:?usage: run_sqlite.sh remote <scenario> <local-sqlite-dir>}"
  SQL_DIR="${2:?usage: run_sqlite.sh remote <scenario> <local-sqlite-dir>}"
  REMOTE="${REMOTE:-lab}"
  REMOTE_WORK="${REMOTE_WORK:-~/valise_portability}"
  REMOTE_VENV="${REMOTE_VENV:-~/diskann-venv}"
  REMOTE_SCEN="${REMOTE_WORK}/${SCEN}"

  echo "==== sqlite portability: remote leg (${REMOTE}, scenario=${SCEN}) ===="
  ssh "$REMOTE" "mkdir -p ${REMOTE_SCEN}/sqlite ${REMOTE_SCEN}/out"

  # sqlite-vec into the existing venv (idempotent one-liner).
  ssh "$REMOTE" "${REMOTE_VENV}/bin/pip install --quiet sqlite-vec"

  echo "-- scp the ONE file ($(du -h "${SQL_DIR}/corpus.db" | cut -f1)) --"
  time scp -q "${SQL_DIR}/corpus.db" "${REMOTE}:${REMOTE_SCEN}/sqlite/corpus.db"

  scp -q "$SCRIPT" "${REMOTE}:${REMOTE_WORK}/portability_sqlite.py"
  scp -q "bench/portability/queries.${SCEN}.json" "${REMOTE}:${REMOTE_SCEN}/queries.json"
  # queries.f32 is staged by run_remote.sh at ${REMOTE_SCEN}/dataset/.

  echo "-- cold probe (remote) --"
  ssh "$REMOTE" "${REMOTE_VENV}/bin/python ${REMOTE_WORK}/portability_sqlite.py probe \
    --path ${REMOTE_SCEN}/sqlite/corpus.db \
    --queries ${REMOTE_SCEN}/queries.json \
    --dataset-dir ${REMOTE_SCEN}/dataset \
    --out ${REMOTE_SCEN}/out/probe-sqlite.json"

  mkdir -p "${SQL_DIR}/remote"
  scp -q "${REMOTE}:${REMOTE_SCEN}/out/probe-sqlite.json" "${SQL_DIR}/remote/probe-sqlite.json"
  echo "--- sqlite (remote) ---"
  cat "${SQL_DIR}/remote/probe-sqlite.json"
  echo
  echo "Diff the top-10 id lists against the local probe to check"
  echo "cross-platform reproducibility, e.g.:"
  echo "  diff <(jq -S '{t:.top10_text_ids,v:.top10_vector_ids,h:.top10_hybrid_ids}' ${OUT_ROOT}/probe_local_${SCEN}_sqlite.json) \\"
  echo "       <(jq -S '{t:.top10_text_ids,v:.top10_vector_ids,h:.top10_hybrid_ids}' ${SQL_DIR}/remote/probe-sqlite.json)"
}

case "${1:?usage: run_sqlite.sh {local|remote <scenario> <local-sqlite-dir>}}" in
  local)  cmd_local ;;
  remote) shift; cmd_remote "$@" ;;
  *) echo "unknown subcommand: $1" >&2; exit 1 ;;
esac
