// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Integration tests for the DB layer P4 maintenance: compaction (GC of
//! tombstoned bytes via fresh-file rewrite + atomic swap), stats, and
//! compaction after delete / forget_before.

use valise::db::{
    Calibrate, CompactOptions, Key, Partition, Record, Schema, Search, Space, Store, Text, Vector,
};

const DIM: usize = 64;
const DAY: i64 = 86_400;

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

fn vec_n(seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    for slot in v.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *slot = ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    }
    let norm: f32 = v.iter().map(|f| f * f).sum::<f32>().sqrt().max(1e-6);
    for slot in v.iter_mut() {
        *slot /= norm;
    }
    v
}

fn sample() -> Vec<Vec<f32>> {
    (0..128).map(vec_n).collect()
}

fn hybrid_store(path: &std::path::Path) -> (Store, Space, Space) {
    let store = Store::create(path).unwrap();
    let en = store.define_space("english", Text::english()).unwrap();
    let v = store
        .define_space(
            "dense",
            Vector::dim(DIM as u32).calibrate(Calibrate::now(sample())),
        )
        .unwrap();
    (store, en, v)
}

#[test]
fn compact_is_noop_when_nothing_tombstoned() {
    let path = tmpfile("noop.vls");
    let (store, en, _v) = hybrid_store(&path);
    store
        .collection("docs", Schema::new().text_with("body", Text::space(&en)))
        .unwrap();
    {
        let mut w = store.writer();
        w.put("docs", "a", Record::new().text("body", "hello"))
            .unwrap();
        w.commit().unwrap();
    }
    let report = store.compact(CompactOptions::default()).unwrap();
    assert!(!report.compacted, "no tombstones → no-op");
    assert!(store.get("docs", "a").unwrap().is_some());
}

#[test]
fn compact_reclaims_bytes_and_preserves_survivors() {
    let path = tmpfile("reclaim.vls");
    let (store, en, v) = hybrid_store(&path);
    store
        .collection(
            "kb",
            Schema::new()
                .text_with("body", Text::space(&en))
                .vector("dense", Vector::space(&v)),
        )
        .unwrap();

    {
        let mut w = store.writer();
        for i in 0..200u64 {
            w.put(
                "kb",
                i,
                Record::new()
                    .text(
                        "body",
                        "quantum retrieval vector database note for compaction",
                    )
                    .vector("dense", &vec_n(i)),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }
    // Delete the odd keys.
    {
        let mut w = store.writer();
        for i in (1..200u64).step_by(2) {
            assert!(w.delete("kb", i).unwrap());
        }
        w.commit().unwrap();
    }

    let before = store.stats();
    assert_eq!(before.frames_active, 100);
    assert!((before.tombstone_ratio() - 0.5).abs() < 0.01);
    assert!(before.needs_compaction(0.3));

    let report = store.compact(CompactOptions::default()).unwrap();
    assert!(report.compacted);
    assert_eq!(report.frames_before, 200);
    assert_eq!(report.frames_after, 100);
    assert_eq!(report.vectors_before, 200);
    assert_eq!(report.vectors_after, 100);
    assert!(
        report.bytes_after < report.bytes_before,
        "bytes reclaimed: {report:?}"
    );

    let after = store.stats();
    assert_eq!(after.frames_total, 100);
    assert_eq!(after.frames_active, 100);
    assert_eq!(after.tombstone_ratio(), 0.0);

    // Survivors (even keys) still present, with text + vector intact; deleted
    // (odd) gone. Uses Store ops which reload the post-compact published state.
    let got = store.get("kb", 4u64).unwrap().unwrap();
    assert_eq!(
        got.text.as_deref(),
        Some("quantum retrieval vector database note for compaction")
    );
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].1.len(), DIM);
    assert!(store.get("kb", 3u64).unwrap().is_none());

    // Both channels work against the swapped-in file. (Use vector-only for the
    // nearest-neighbor assertion: all survivors share identical text, so a
    // hybrid query's tied BM25 channel would swamp the vector signal under RRF.)
    let q = vec_n(50);
    let vhits = store
        .search("kb", Search::new().vector("dense", &q).top_k(5))
        .unwrap();
    assert_eq!(
        vhits[0].key,
        Key::from(50u64),
        "nearest survivor returned post-compact"
    );

    let thits = store
        .search(
            "kb",
            Search::new().text("body", "quantum retrieval").top_k(10),
        )
        .unwrap();
    assert_eq!(thits.len(), 10);
    assert!(
        thits
            .iter()
            .all(|h| matches!(&h.key, Key::U64(n) if n % 2 == 0)),
        "only even (survivor) keys returned: {:?}",
        thits.iter().map(|h| &h.key).collect::<Vec<_>>()
    );
}

#[test]
fn compact_preserves_parent_tree() {
    let path = tmpfile("tree.vls");
    let (store, en, _v) = hybrid_store(&path);
    store
        .collection("docs", Schema::new().text_with("body", Text::space(&en)))
        .unwrap();

    {
        let mut w = store.writer();
        w.put("docs", "parent", Record::new().text("body", "doc"))
            .unwrap();
        w.put(
            "docs",
            "child",
            Record::new()
                .text("body", "chunk")
                .child_of(Key::from("parent")),
        )
        .unwrap();
        w.put("docs", "trash", Record::new().text("body", "trash"))
            .unwrap();
        w.commit().unwrap();
        w.delete("docs", "trash").unwrap();
        w.commit().unwrap();
    }

    let report = store.compact(CompactOptions::default()).unwrap();
    assert!(report.compacted);
    // Parent + child survive (and the child's parent link is remapped, not
    // orphaned, since parents stream before children).
    assert!(store.get("docs", "parent").unwrap().is_some());
    assert!(store.get("docs", "child").unwrap().is_some());
    assert!(store.get("docs", "trash").unwrap().is_none());
}

#[test]
fn compact_after_forget_before() {
    let path = tmpfile("forget_compact.vls");
    let (store, en, _v) = hybrid_store(&path);
    let mem = store
        .partitioned(
            "mem",
            Schema::new().text_with("body", Text::space(&en)),
            Partition::ByDay,
        )
        .unwrap();
    {
        let mut w = store.writer();
        for d in 0..4i64 {
            w.put_into(
                &mem,
                format!("k{d}"),
                Record::new().text("body", "common").at(d * DAY + 10),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }
    mem.forget_before(2 * DAY).unwrap(); // soft-tombstone days 0,1
    assert!(store.stats().needs_compaction(0.3));

    let report = store.compact(CompactOptions::default()).unwrap();
    assert!(report.compacted);
    assert_eq!(report.frames_after, 2);

    let hits = store
        .search_view(&mem.all(), Search::new().text("body", "common").top_k(10))
        .unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn compact_recalibrate_path() {
    let path = tmpfile("recal.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "c",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample())),
            ),
        )
        .unwrap();
    {
        let mut w = store.writer();
        for i in 0..60u64 {
            w.put("c", i, Record::new().vector("dense", &vec_n(i)))
                .unwrap();
        }
        w.commit().unwrap();
        for i in (1..60u64).step_by(3) {
            w.delete("c", i).unwrap();
        }
        w.commit().unwrap();
    }
    let report = store.compact(CompactOptions { recalibrate: true }).unwrap();
    assert!(report.compacted);
    // A survivor still reconstructs and is its own nearest neighbor.
    let q = vec_n(6);
    let hits = store
        .search("c", Search::new().vector("dense", &q).top_k(3))
        .unwrap();
    assert_eq!(hits[0].key, Key::from(6u64));
}

#[test]
fn compact_survives_reopen() {
    let path = tmpfile("reopen.vls");
    {
        let (store, en, v) = hybrid_store(&path);
        store
            .collection(
                "kb",
                Schema::new()
                    .text_with("body", Text::space(&en))
                    .vector("dense", Vector::space(&v)),
            )
            .unwrap();
        {
            let mut w = store.writer();
            for i in 0..40u64 {
                w.put(
                    "kb",
                    i,
                    Record::new()
                        .text("body", "persisted note")
                        .vector("dense", &vec_n(i)),
                )
                .unwrap();
            }
            w.commit().unwrap();
            for i in (0..40u64).step_by(2) {
                w.delete("kb", i).unwrap();
            }
            w.commit().unwrap();
            // Compacting while the writer is still held must fail fast, not
            // deadlock on the non-reentrant writer lock.
            assert!(matches!(
                store.compact(CompactOptions::default()),
                Err(valise::Error::Busy(_))
            ));
        } // writer dropped
        let report = store.compact(CompactOptions::default()).unwrap();
        assert!(report.compacted);
        assert_eq!(report.frames_after, 20);
    }

    // Reopen the compacted file: survivors persisted, identity rebuilt.
    let store = Store::open(&path).unwrap();
    let en = store.space("english").unwrap();
    let v = store.space("dense").unwrap();
    store
        .collection(
            "kb",
            Schema::new()
                .text_with("body", Text::space(&en))
                .vector("dense", Vector::space(&v)),
        )
        .unwrap();
    assert!(store.get("kb", 1u64).unwrap().is_some()); // odd survived
    assert!(store.get("kb", 0u64).unwrap().is_none()); // even deleted
    assert_eq!(store.stats().frames_total, 20);
}
