// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Integration tests for persisted collection schemas (SIMPLE_API_SPEC §4):
//! reopen needs no re-declaration. Covers the reopen-just-works contract
//! (text + vector, deferred-then-calibrated and still-deferred UPQ),
//! stage-only durability, watermark created_at ordering, additive/divergent
//! redeclare semantics against the persisted doc, the compact-after-reopen
//! text-indexing fix, old-file upgrade via one re-declare, partition
//! collections, and unknown-JSON-key tolerance of the on-disk doc.

use valise::db::{Calibrate, Codec, Key, Partition, Record, Schema, Search, Store, Text, Vector};
use valise::{CodecFamily, Error, OpenMode, PutFrame, ValiseFile};

const DIM: usize = 64;
const DAY: i64 = 86_400;

/// Leading uri byte of a schema doc frame (pinned: not a `Key` tag).
const SCHEMA_URI_TAG: u8 = 0xF5;
/// The reserved schema-doc collection name (pinned).
const SCHEMA_COLLECTION: &str = "~valise.schema";

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

/// Engine-level view of `coll`'s active schema doc frames:
/// `(frame_id, payload)` pairs. Opens the file read-only; callers must have
/// dropped any `Store` first (advisory lock).
fn doc_frames(valise: &ValiseFile, coll: &str) -> Vec<(valise::FrameId, Vec<u8>)> {
    let Some(cid) = valise
        .collections()
        .iter()
        .find(|c| c.name == SCHEMA_COLLECTION)
        .map(|c| c.collection_id)
    else {
        return Vec::new();
    };
    let mut want = vec![SCHEMA_URI_TAG];
    want.extend_from_slice(coll.as_bytes());
    let frame_ids: Vec<valise::FrameId> = valise
        .active_frames()
        .filter(|f| f.collection_id == cid)
        .map(|f| f.frame_id)
        .collect();
    let mut out = Vec::new();
    for fid in frame_ids {
        if valise.read_frame_uri(fid).unwrap().as_deref() == Some(want.as_slice()) {
            out.push((fid, valise.read_payload(fid).unwrap()));
        }
    }
    out
}

#[test]
fn reopen_just_works_without_redeclare() {
    let path = tmpfile("reopen.vls");
    {
        let store = Store::create(&path).unwrap();
        // Hybrid collection with a deferred (auto-calibrating) QAM field.
        store
            .collection(
                "kb",
                Schema::new().text("body").vector(
                    "dense",
                    Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(8)),
                ),
            )
            .unwrap();
        // Vector-only collection whose UPQ field stays DEFERRED (no vectors
        // are ever ingested before the reopen).
        store
            .collection(
                "pending",
                Schema::new().vector(
                    "u",
                    Vector::dim(DIM as u32)
                        .codec(Codec::upq())
                        .calibrate(Calibrate::auto_sample(4)),
                ),
            )
            .unwrap();
        let mut w = store.writer();
        for i in 0..16u64 {
            w.put(
                "kb",
                i,
                Record::new()
                    .text("body", "persisted hybrid record about vector quantization")
                    .vector("dense", &vec_n(i)),
            )
            .unwrap();
        }
        w.commit().unwrap();

        // The hidden doc store never leaks into user-facing listings/stats.
        assert!(
            store.collections().iter().all(|c| !c.name.starts_with('~')),
            "collections() must exclude ~valise.schema: {:?}",
            store.collections()
        );
        let stats = store.stats();
        assert_eq!(stats.frames_total, 16, "doc frames must not be counted");
        assert_eq!(stats.frames_active, 16);
    }

    // Reopen: NO re-declaration of anything.
    let store = Store::open(&path).unwrap();

    // get: text + vector materialize per the restored schema.
    let got = store.get("kb", 3u64).unwrap().expect("key survives");
    assert_eq!(
        got.text.as_deref(),
        Some("persisted hybrid record about vector quantization")
    );
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].0, "dense");
    assert_eq!(got.vectors[0].1.len(), DIM);

    // Text search works.
    let thits = store
        .search("kb", Search::new().text("body", "quantization").top_k(5))
        .unwrap();
    assert!(!thits.is_empty(), "text channel restored from doc");

    // Vector search works (calibrated space recovered from the catalog).
    let vhits = store
        .search("kb", Search::new().vector("dense", &vec_n(7)).top_k(3))
        .unwrap();
    assert_eq!(vhits[0].key, Key::from(7u64));

    // The still-deferred UPQ field is restored as deferred: searching it is
    // NotCalibrated (naming the auto space), not "no schema".
    let err = store
        .search("pending", Search::new().vector("u", &vec_n(0)).top_k(3))
        .unwrap_err();
    match &err {
        Error::NotCalibrated { space } => assert_eq!(space, "~auto/pending/u"),
        other => panic!("expected NotCalibrated, got {other}"),
    }

    // …and its CODEC CHOICE survived: the first vector commit calibrates UPQ.
    {
        let mut w = store.writer();
        for i in 0..8u64 {
            w.put("pending", i, Record::new().vector("u", &vec_n(100 + i)))
                .unwrap();
        }
        w.commit().unwrap();
    }
    let families: Vec<CodecFamily> = store
        .raw()
        .reader()
        .codecs()
        .iter()
        .map(|c| c.family)
        .collect();
    assert!(
        families.contains(&CodecFamily::Upq),
        "deferred UPQ choice must survive reopen, got {families:?}"
    );
    let hits = store
        .search("pending", Search::new().vector("u", &vec_n(103)).top_k(3))
        .unwrap();
    assert_eq!(hits[0].key, Key::from(3u64));

    // The reserved collection is invisible to keyed reads.
    assert!(store.get(SCHEMA_COLLECTION, "kb").is_err());
}

#[test]
fn declare_durability_is_stage_only() {
    let path = tmpfile("stage_only.vls");
    {
        let store = Store::create(&path).unwrap();
        // Engine create persists the (empty) file; the declare itself is a
        // pending mutation with no commit behind it.
        store.collection("kb", Schema::new().text("body")).unwrap();
        // Dropped WITHOUT any commit.
    }
    {
        let store = Store::open(&path).unwrap();
        assert!(
            store.collections().is_empty(),
            "an uncommitted declare must leave nothing"
        );
        assert!(
            store
                .search("kb", Search::new().text("body", "x").top_k(1))
                .is_err()
        );

        // Declare again; ANY writer commit persists it (group-committed,
        // like create_collection itself).
        store.collection("kb", Schema::new().text("body")).unwrap();
        let mut w = store.writer();
        w.commit().unwrap();
    }
    let store = Store::open(&path).unwrap();
    let names: Vec<String> = store.collections().into_iter().map(|c| c.name).collect();
    assert_eq!(names, vec!["kb".to_string()]);
    // Shape restored: searching is well-formed (empty, but not "no schema").
    let hits = store
        .search("kb", Search::new().text("body", "anything").top_k(3))
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn watermark_created_at_orders_doc_after_existing_frames() {
    let path = tmpfile("watermark.vls");
    {
        let store = Store::create(&path).unwrap();
        store.collection("a", Schema::new().text("body")).unwrap();
        {
            let mut w = store.writer();
            w.put("a", "x", Record::new().text("body", "one").at(1000))
                .unwrap();
            w.put("a", "y", Record::new().text("body", "two").at(1500))
                .unwrap();
            w.commit().unwrap();
        } // drop the writer: the lock is not reentrant

        // Declare AFTER frames exist: the doc frame rides at the watermark
        // (1500) and must not break the time index's global ordering.
        store.collection("b", Schema::new().text("body")).unwrap();

        // Puts at created_at ≥ watermark still succeed (the doc added no
        // new ingest constraint), and the commit carrying the doc is clean.
        let mut w = store.writer();
        w.put("a", "z", Record::new().text("body", "three").at(1500))
            .unwrap();
        w.put("b", "k", Record::new().text("body", "four").at(2000))
            .unwrap();
        w.commit().unwrap();
    }
    let store = Store::open(&path).unwrap();
    assert_eq!(
        store.get("b", "k").unwrap().unwrap().created_at,
        2000,
        "schema restored and data intact"
    );
    assert_eq!(store.get("a", "z").unwrap().unwrap().created_at, 1500);
}

#[test]
fn additive_redeclare_rewrites_doc() {
    let path = tmpfile("additive.vls");
    {
        let store = Store::create(&path).unwrap();
        store.collection("kb", Schema::new().text("body")).unwrap();
        let mut w = store.writer();
        w.put("kb", "a", Record::new().text("body", "text only era"))
            .unwrap();
        w.commit().unwrap();
    }
    {
        let store = Store::open(&path).unwrap();
        // Additive: append a vector field. The persisted doc is rewritten
        // (tombstone old + put new).
        store
            .collection(
                "kb",
                Schema::new().text("body").vector(
                    "dense",
                    Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(4)),
                ),
            )
            .unwrap();
        let mut w = store.writer();
        w.put(
            "kb",
            "b",
            Record::new()
                .text("body", "hybrid era")
                .vector("dense", &vec_n(1)),
        )
        .unwrap();
        w.commit().unwrap();
    }
    // Engine-level: exactly ONE active doc frame for kb (old one tombstoned).
    {
        let valise = ValiseFile::open_read_only(&path).unwrap();
        assert_eq!(
            doc_frames(&valise, "kb").len(),
            1,
            "old doc must be replaced"
        );
    }
    // Reopen with no declare: the grown schema is what's restored.
    let store = Store::open(&path).unwrap();
    let got = store.get("kb", "b").unwrap().unwrap();
    assert_eq!(got.vectors.len(), 1);
    let hits = store
        .search("kb", Search::new().vector("dense", &vec_n(1)).top_k(1))
        .unwrap();
    assert_eq!(hits[0].key, Key::from("b"));
}

#[test]
fn divergent_redeclare_errors_with_schema_mismatch() {
    let path = tmpfile("divergent.vls");
    {
        let store = Store::create(&path).unwrap();
        store
            .collection(
                "kb",
                Schema::new()
                    .text("body")
                    .vector("dense", Vector::dim(DIM as u32)),
            )
            .unwrap();
        // Renamed field (in-process shape check).
        let e = store
            .collection(
                "kb",
                Schema::new()
                    .text("title")
                    .vector("dense", Vector::dim(DIM as u32)),
            )
            .unwrap_err();
        assert!(
            matches!(&e, Error::SchemaMismatch { collection, .. } if collection == "kb"),
            "{e}"
        );
        let mut w = store.writer();
        w.commit().unwrap();
    }
    // Codec change on an existing field has the SAME (name, kind, space)
    // signature — only the persisted doc can catch it.
    let store = Store::open(&path).unwrap();
    let e = store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::dim(DIM as u32).codec(Codec::upq())),
        )
        .unwrap_err();
    let Error::SchemaMismatch { collection, detail } = &e else {
        panic!("expected SchemaMismatch, got {e}");
    };
    assert_eq!(collection, "kb");
    assert!(detail.contains("'dense'"), "field-level detail: {detail}");
    // The failed declaration had no side effects: the original schema still
    // re-declares as a no-op and the store still works.
    store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::dim(DIM as u32)),
        )
        .unwrap();
}

#[test]
fn compact_round_trip_preserves_schemas_and_text_search() {
    let path = tmpfile("compact_rt.vls");
    {
        let store = Store::create(&path).unwrap();
        store
            .collection(
                "kb",
                Schema::new().text("body").vector(
                    "dense",
                    Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(16)),
                ),
            )
            .unwrap();
        let mut w = store.writer();
        for i in 0..30u64 {
            w.put(
                "kb",
                i,
                Record::new()
                    .text("body", "compaction era retrieval note")
                    .vector("dense", &vec_n(i)),
            )
            .unwrap();
        }
        w.commit().unwrap();
        for i in 0..10u64 {
            w.delete("kb", i).unwrap();
        }
        w.commit().unwrap();
    }

    // Reopen WITHOUT re-declaring, then compact. Before schemas were
    // persisted this silently dropped text indexing (coll_text came from
    // in-process shapes, which a reopen didn't have).
    let store = Store::open(&path).unwrap();
    let report = store
        .compact(valise::db::CompactOptions::default())
        .unwrap();
    assert!(report.compacted);
    assert_eq!(report.frames_after, 20, "report excludes doc frames");

    let thits = store
        .search(
            "kb",
            Search::new().text("body", "compaction retrieval").top_k(5),
        )
        .unwrap();
    assert!(
        !thits.is_empty(),
        "text search must survive compact-after-reopen (latent-bug fix)"
    );
    let vhits = store
        .search("kb", Search::new().vector("dense", &vec_n(15)).top_k(3))
        .unwrap();
    assert_eq!(vhits[0].key, Key::from(15u64));

    // And the compacted file still reopens schema-complete.
    drop(store);
    let store = Store::open(&path).unwrap();
    assert!(store.get("kb", 15u64).unwrap().is_some());
    assert!(store.get("kb", 3u64).unwrap().is_none());
    let thits = store
        .search("kb", Search::new().text("body", "compaction").top_k(5))
        .unwrap();
    assert!(!thits.is_empty());
}

#[test]
fn partition_collections_persist_their_schema_docs() {
    let path = tmpfile("partition.vls");
    {
        let store = Store::create(&path).unwrap();
        let mem = store
            .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
            .unwrap();
        let mut w = store.writer();
        w.put_into(
            &mem,
            "k0",
            Record::new().text("body", "day zero note").at(10),
        )
        .unwrap();
        w.put_into(
            &mem,
            "k1",
            Record::new().text("body", "day one note").at(DAY + 10),
        )
        .unwrap();
        w.commit().unwrap();
    }
    // Reopen: the lazily created partition collections restore their shapes
    // from their own docs — readable with no partitioned()/collection() call.
    let store = Store::open(&path).unwrap();
    let names: Vec<String> = store.collections().into_iter().map(|c| c.name).collect();
    assert!(names.contains(&"mem:1970-01-01".to_string()), "{names:?}");
    assert!(names.contains(&"mem:1970-01-02".to_string()), "{names:?}");
    let got = store.get("mem:1970-01-01", "k0").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("day zero note"));
    let hits = store
        .search(
            "mem:1970-01-02",
            Search::new().text("body", "note").top_k(3),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, Key::from("k1"));
}

#[test]
fn old_file_upgrade_one_redeclare_persists() {
    let path = tmpfile("upgrade.vls");
    let schema = || Schema::new().text("body");
    {
        let store = Store::create(&path).unwrap();
        store.collection("kb", schema()).unwrap();
        let mut w = store.writer();
        w.put("kb", "a", Record::new().text("body", "legacy record"))
            .unwrap();
        w.commit().unwrap();
    }
    // Simulate a pre-schema-persistence file: strip the doc frame at the
    // engine level.
    {
        let mut valise = ValiseFile::open(&path, OpenMode::ReadWrite).unwrap();
        let docs = doc_frames(&valise, "kb");
        assert_eq!(docs.len(), 1);
        valise.delete_frame(docs[0].0).unwrap();
        valise.commit().unwrap();
    }
    {
        // Doc-less collection: graceful shape-less degrade, not a broken open.
        let store = Store::open(&path).unwrap();
        let names: Vec<String> = store.collections().into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["kb".to_string()]);
        let e = store
            .search("kb", Search::new().text("body", "legacy").top_k(3))
            .unwrap_err();
        assert!(e.to_string().contains("has no schema"), "{e}");

        // ONE re-declare persists the doc…
        store.collection("kb", schema()).unwrap();
        assert!(
            store
                .search("kb", Search::new().text("body", "legacy").top_k(3))
                .is_ok()
        );
        let mut w = store.writer();
        w.commit().unwrap();
    }
    // …and every later open just works.
    let store = Store::open(&path).unwrap();
    let hits = store
        .search("kb", Search::new().text("body", "legacy").top_k(3))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, Key::from("a"));
}

#[test]
fn unknown_json_keys_in_doc_are_tolerated() {
    let path = tmpfile("forward_compat.vls");
    let schema = || {
        Schema::new().text("body").vector(
            "dense",
            Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(4)),
        )
    };
    {
        let store = Store::create(&path).unwrap();
        store.collection("kb", schema()).unwrap();
        let mut w = store.writer();
        w.put(
            "kb",
            "a",
            Record::new()
                .text("body", "future proof record")
                .vector("dense", &vec_n(9)),
        )
        .unwrap();
        w.commit().unwrap();
    }
    // Rewrite the doc with unknown keys at every level (as a newer version
    // of the layer might), engine-level.
    {
        let mut valise = ValiseFile::open(&path, OpenMode::ReadWrite).unwrap();
        let docs = doc_frames(&valise, "kb");
        assert_eq!(docs.len(), 1);
        let json = String::from_utf8(docs[0].1.clone()).unwrap();
        let json = json.replacen(
            r#"{"valise_schema":1,"#,
            r#"{"valise_schema":1,"future_top":true,"#,
            1,
        );
        let json = json.replacen(
            r#"{"name":"body","#,
            r#"{"name":"body","future_field":[1,2],"#,
            1,
        );
        let json = json.replacen(
            r#"{"lang":"english"}"#,
            r#"{"lang":"english","future_lang_opt":2}"#,
            1,
        );
        assert!(json.contains("future_top") && json.contains("future_field"));
        assert!(json.contains("future_lang_opt"));

        let cid = valise
            .collections()
            .iter()
            .find(|c| c.name == SCHEMA_COLLECTION)
            .unwrap()
            .collection_id;
        valise.delete_frame(docs[0].0).unwrap();
        let watermark = valise
            .active_frames()
            .map(|f| f.created_at)
            .max()
            .unwrap_or(0)
            .max(0);
        let mut uri = vec![SCHEMA_URI_TAG];
        uri.extend_from_slice(b"kb");
        let mut pf = PutFrame::document(cid, json.as_bytes()).with_uri(&uri);
        pf.created_at = Some(watermark);
        valise.put_frame(pf).unwrap();
        valise.commit().unwrap();
    }
    // The decorated doc still restores the schema…
    let store = Store::open(&path).unwrap();
    let got = store.get("kb", "a").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("future proof record"));
    let hits = store
        .search("kb", Search::new().text("body", "future").top_k(3))
        .unwrap();
    assert_eq!(hits.len(), 1);
    // …and the unknown keys are invisible to the redeclare comparison: an
    // identical declaration is still a no-op, not a SchemaMismatch.
    store.collection("kb", schema()).unwrap();
}

/// A divergent re-declaration must error even BEFORE any commit: staged doc
/// frames are not readable until commit, so this is caught by the in-process
/// doc comparison — and with zero side effects (no space redefinition).
#[test]
fn divergent_redeclare_errors_before_first_commit() {
    let path = tmpfile("precommit_divergent.vls");
    let store = Store::create(&path).unwrap();
    store.collection("kb", Schema::new().text("body")).unwrap();
    // Text analyzer change — no commit has happened yet.
    let e = store
        .collection("kb", Schema::new().text_with("body", Text::raw()))
        .unwrap_err();
    assert!(
        matches!(&e, Error::SchemaMismatch { collection, .. } if collection == "kb"),
        "expected SchemaMismatch, got {e}"
    );
    // Vector dim change on a fresh collection, also pre-commit, must raise
    // SchemaMismatch (not a space-redefinition error from a partial bind).
    store
        .collection("v", Schema::new().vector("dense", Vector::dim(64)))
        .unwrap();
    let e = store
        .collection("v", Schema::new().vector("dense", Vector::dim(128)))
        .unwrap_err();
    assert!(
        matches!(&e, Error::SchemaMismatch { collection, .. } if collection == "v"),
        "expected SchemaMismatch, got {e}"
    );
    // Identical re-declare remains a no-op.
    store.collection("kb", Schema::new().text("body")).unwrap();
}
