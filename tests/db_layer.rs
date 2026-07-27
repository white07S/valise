// Prints results for the reader; stdout is this target's output channel.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! Integration tests for the DB layer (P1: schema + write).
//!
//! Covers: schema declaration (inline auto-spaces and shared spaces, eager
//! and deferred calibration), shapes (text-only / vector-only / hybrid /
//! multi-vector), put/get round trips, update (upsert) remapping, delete,
//! persistence across reopen (identity rebuilt from in-file keys), deferred
//! calibration triggered at commit, reserved-name rejection, and re-declare
//! (no-op / additive / SchemaMismatch) semantics.

use valise::Error;
use valise::db::{Calibrate, Codec, Key, Record, Schema, Store, Text, Vector};

const DIM: usize = 64;

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

/// Deterministic unit-ish vector, distinct per seed.
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

fn sample(n: u64) -> Vec<Vec<f32>> {
    (0..n).map(vec_n).collect()
}

/// Vector-put throughput at realistic dim=768 (the dim where the finiteness
/// validation actually costs something). Run with:
///   cargo test --test db_layer --release -- --ignored --nocapture bench_vector_put_768
#[test]
#[ignore]
fn bench_vector_put_768() {
    const D: usize = 768;
    fn v768(seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; D];
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        for s in v.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        }
        v
    }
    let path = tmpfile("bench768.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "c",
            Schema::new().vector(
                "dense",
                Vector::dim(D as u32).calibrate(Calibrate::now((0..256).map(v768).collect())),
            ),
        )
        .unwrap();

    const N: u64 = 100_000;
    let embs: Vec<Vec<f32>> = (0..N).map(v768).collect();
    let t0 = std::time::Instant::now();
    {
        let mut w = store.writer();
        for i in 0..N {
            w.put("c", i, Record::new().vector("dense", &embs[i as usize]))
                .unwrap();
        }
        w.commit().unwrap();
    }
    let el = t0.elapsed();
    eprintln!(
        "db-layer vector put (dim {D}): {N} in {el:?} ({:.0}/s, {:.0} ns/put incl. finiteness check)",
        N as f64 / el.as_secs_f64(),
        el.as_secs_f64() * 1e9 / N as f64,
    );
}

/// Write-path micro-benchmark. Run with:
///   cargo test --test db_layer --release -- --ignored --nocapture bench_write_path
#[test]
#[ignore]
fn bench_write_path() {
    let path = tmpfile("bench.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "kb",
            Schema::new().text("body").vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(256))),
            ),
        )
        .unwrap();

    const N: u64 = 20_000;
    let embs: Vec<Vec<f32>> = (0..N).map(vec_n).collect();
    let t0 = std::time::Instant::now();
    {
        let mut w = store.writer();
        for i in 0..N {
            w.put(
                "kb",
                i,
                Record::new()
                    .text("body", "hybrid record for the write-path benchmark")
                    .vector("dense", &embs[i as usize]),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }
    let put_elapsed = t0.elapsed();

    // Update half the keys (exercises the upsert tombstone+remap path).
    let t1 = std::time::Instant::now();
    {
        let mut w = store.writer();
        for i in 0..(N / 2) {
            w.put(
                "kb",
                i,
                Record::new()
                    .text("body", "updated record")
                    .vector("dense", &embs[i as usize]),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }
    let upd_elapsed = t1.elapsed();

    // get throughput.
    let t2 = std::time::Instant::now();
    for i in 0..N {
        let _ = store.get("kb", i).unwrap();
    }
    let get_elapsed = t2.elapsed();

    eprintln!(
        "put {N} hybrid: {put_elapsed:?} ({:.0}/s) | update {} : {upd_elapsed:?} | get {N}: {get_elapsed:?} ({:.0}/s)",
        N as f64 / put_elapsed.as_secs_f64(),
        N / 2,
        N as f64 / get_elapsed.as_secs_f64(),
    );
}

#[test]
fn text_only_put_get_delete() {
    let path = tmpfile("text_only.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    {
        let mut w = store.writer();
        w.put(
            "docs",
            "doc-1",
            Record::new().text("body", "vector quantization explained"),
        )
        .unwrap();
        w.commit().unwrap();
    }

    let got = store.get("docs", "doc-1").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("vector quantization explained"));
    assert!(got.vectors.is_empty());

    // Delete tombstones + unmaps.
    {
        let mut w = store.writer();
        assert!(w.delete("docs", "doc-1").unwrap());
        assert!(!w.delete("docs", "missing").unwrap());
        w.commit().unwrap();
    }
    assert!(store.get("docs", "doc-1").unwrap().is_none());
}

#[test]
fn vector_only_eager_sample_calibration() {
    let path = tmpfile("vec_only.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "mem",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
            ),
        )
        .unwrap();

    {
        let mut w = store.writer();
        for i in 0..16u64 {
            let emb = vec_n(1000 + i);
            w.put("mem", i, Record::new().vector("dense", &emb))
                .unwrap();
        }
        w.commit().unwrap();
    }

    let got = store.get("mem", 3u64).unwrap().unwrap();
    assert!(got.text.is_none());
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].0, "dense");
    assert_eq!(got.vectors[0].1.len(), DIM);
    assert!(got.vectors[0].1.iter().all(|f| f.is_finite()));
}

#[test]
fn hybrid_deferred_calibration() {
    let path = tmpfile("hybrid.vls");
    let store = Store::create(&path).unwrap();
    // Deferred (auto) calibration: nothing is registered until the first
    // commit calibrates it from the first 8 staged vectors.
    store
        .collection(
            "kb",
            Schema::new().text("body").vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(8)),
            ),
        )
        .unwrap();

    {
        let mut w = store.writer();
        for i in 0..12u64 {
            let emb = vec_n(i);
            w.put(
                "kb",
                format!("k{i}"),
                Record::new()
                    .text("body", "quantization and retrieval")
                    .vector("dense", &emb),
            )
            .unwrap();
        }
        // Calibration happens here, from the first 8 staged vectors.
        w.commit().unwrap();
    }

    let got = store.get("kb", "k5").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("quantization and retrieval"));
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].1.len(), DIM);
}

#[test]
fn multimodal_two_vector_fields() {
    let path = tmpfile("multimodal.vls");
    let store = Store::create(&path).unwrap();
    // Two inline vector fields auto-define two distinct private spaces.
    store
        .collection(
            "assets",
            Schema::new()
                .vector(
                    "text",
                    Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
                )
                .vector(
                    "image",
                    Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
                ),
        )
        .unwrap();

    let a = vec_n(7);
    let b = vec_n(99);
    {
        let mut w = store.writer();
        w.put(
            "assets",
            "a1",
            Record::new().vector("text", &a).vector("image", &b),
        )
        .unwrap();
        w.commit().unwrap();
    }

    let got = store.get("assets", "a1").unwrap().unwrap();
    assert_eq!(got.vectors.len(), 2);
    let fields: Vec<&str> = got.vectors.iter().map(|(f, _)| f.as_str()).collect();
    assert!(fields.contains(&"text"));
    assert!(fields.contains(&"image"));
}

#[test]
fn update_remaps_key() {
    let path = tmpfile("update.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    {
        let mut w = store.writer();
        w.put("docs", "d", Record::new().text("body", "first version"))
            .unwrap();
        w.commit().unwrap();
    }
    {
        let mut w = store.writer();
        w.put("docs", "d", Record::new().text("body", "second version"))
            .unwrap();
        w.commit().unwrap();
    }

    let got = store.get("docs", "d").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("second version"));
}

#[test]
fn put_auto_generates_key() {
    let path = tmpfile("auto.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let key = {
        let mut w = store.writer();
        let k = w
            .put_auto("docs", Record::new().text("body", "auto"))
            .unwrap();
        w.commit().unwrap();
        k
    };
    let got = store.get("docs", &key).unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("auto"));
}

#[test]
fn persists_across_reopen() {
    let path = tmpfile("reopen.vls");
    {
        let store = Store::create(&path).unwrap();
        // Shared spaces: defined once, bound by reference in the schema.
        let en = store.define_space("english", Text::english()).unwrap();
        let v = store
            .define_space(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
            )
            .unwrap();
        store
            .collection(
                "kb",
                Schema::new()
                    .text_with("body", Text::space(&en))
                    .vector("dense", Vector::space(&v)),
            )
            .unwrap();
        let mut w = store.writer();
        let emb = vec_n(42);
        w.put(
            "kb",
            "survivor",
            Record::new()
                .text("body", "persisted record")
                .vector("dense", &emb),
        )
        .unwrap();
        w.commit().unwrap();
    }

    // Reopen: identity index is rebuilt from the in-file keys (uri_ref),
    // spaces come back by name, and the collection's schema is restored
    // from its persisted doc — no re-declaration needed.
    let store = Store::open(&path).unwrap();
    assert!(store.space("english").unwrap().is_text());
    assert!(store.space("dense").unwrap().is_vector());

    let got = store.get("kb", "survivor").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("persisted record"));
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].1.len(), DIM);

    // Update after reopen still works (identity resolved the prior frame).
    {
        let mut w = store.writer();
        w.put(
            "kb",
            "survivor",
            Record::new().text("body", "updated after reopen"),
        )
        .unwrap();
        w.commit().unwrap();
    }
    let got = store.get("kb", "survivor").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("updated after reopen"));
    // The vector was dropped by the text-only update (its field was omitted).
    assert!(got.vectors.is_empty());
}

#[test]
fn schema_rejects_two_text_fields() {
    let path = tmpfile("two_text.vls");
    let store = Store::create(&path).unwrap();
    let err = store
        .collection("bad", Schema::new().text("a").text("b"))
        .unwrap_err();
    assert!(err.to_string().contains("at most one text field"), "{err}");
}

#[test]
fn put_rejects_dim_mismatch() {
    let path = tmpfile("dim.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "c",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
            ),
        )
        .unwrap();

    let mut w = store.writer();
    let wrong = vec![0.0f32; DIM + 1];
    let err = w
        .put("c", "x", Record::new().vector("dense", &wrong))
        .unwrap_err();
    assert!(err.to_string().contains("dim"), "{err}");
}

#[test]
fn update_vector_record_across_commits() {
    // Exercises the committed-flag path: after the first commit the prior
    // version's vector must be tombstonable, so the second put must not be
    // rejected with "vectors not yet committed".
    let path = tmpfile("update_vec.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "c",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
            ),
        )
        .unwrap();

    let a = vec_n(1);
    let b = vec_n(2);
    {
        let mut w = store.writer();
        w.put("c", "k", Record::new().vector("dense", &a)).unwrap();
        w.commit().unwrap();
    }
    {
        let mut w = store.writer();
        // Must succeed: prior vector is committed and thus tombstonable.
        w.put("c", "k", Record::new().vector("dense", &b)).unwrap();
        w.commit().unwrap();
    }
    let got = store.get("c", "k").unwrap().unwrap();
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].1.len(), DIM);
}

/// A vector-only schema whose codec calibrates at the first commit.
fn deferred_vector_schema() -> Schema {
    Schema::new().vector(
        "dense",
        Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(8)),
    )
}

#[test]
fn deferred_space_delete_same_batch_does_not_poison_commit() {
    // Pre-calibration (deferred) space: a same-batch delete tombstones the
    // owner frame while its vector is still staged. flush_deferred must not
    // re-put onto the tombstoned frame.
    let path = tmpfile("deferred_del.vls");
    let store = Store::create(&path).unwrap();
    store.collection("c", deferred_vector_schema()).unwrap();

    let mut w = store.writer();
    w.put("c", "a", Record::new().vector("dense", &vec_n(1)))
        .unwrap();
    assert!(w.delete("c", "a").unwrap());
    w.commit().expect("commit must not fail");
    drop(w);
    assert!(store.get("c", "a").unwrap().is_none());
}

#[test]
fn deferred_space_upsert_same_batch_does_not_poison_commit() {
    let path = tmpfile("deferred_up.vls");
    let store = Store::create(&path).unwrap();
    store.collection("c", deferred_vector_schema()).unwrap();

    let a = vec_n(1);
    let b = vec_n(2);
    {
        let mut w = store.writer();
        w.put("c", "k", Record::new().vector("dense", &a)).unwrap();
        // Re-put in the same pre-calibration batch — the first staged vector is
        // pinned to a frame this put tombstones.
        w.put("c", "k", Record::new().vector("dense", &b)).unwrap();
        w.commit().expect("commit must not fail");
    }
    // Exactly one active vector survives under "k".
    let got = store.get("c", "k").unwrap().unwrap();
    assert_eq!(got.vectors.len(), 1);
    assert_eq!(got.vectors[0].1.len(), DIM);
}

#[test]
fn put_rejects_non_finite_vector() {
    let path = tmpfile("nonfinite.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "c",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample(64))),
            ),
        )
        .unwrap();

    let mut w = store.writer();
    // NaN and Inf are both rejected immediately at the offending put — naming
    // the field — not deferred to a batch-poisoning commit failure.
    let mut nan_vec = vec_n(1);
    nan_vec[7] = f32::NAN;
    let e = w
        .put("c", "bad-nan", Record::new().vector("dense", &nan_vec))
        .unwrap_err();
    assert!(
        e.to_string().contains("non-finite") && e.to_string().contains("dense"),
        "{e}"
    );

    let mut inf_vec = vec_n(2);
    inf_vec[0] = f32::INFINITY;
    let e = w
        .put("c", "bad-inf", Record::new().vector("dense", &inf_vec))
        .unwrap_err();
    assert!(e.to_string().contains("non-finite"), "{e}");

    // A clean vector still commits, and neither bad key was recorded.
    w.put("c", "good", Record::new().vector("dense", &vec_n(3)))
        .unwrap();
    w.commit().unwrap();
    assert!(store.get("c", "good").unwrap().is_some());
    assert!(store.get("c", "bad-nan").unwrap().is_none());
    assert!(store.get("c", "bad-inf").unwrap().is_none());
}

#[test]
fn redefine_space_with_different_spec_errors() {
    let path = tmpfile("redefine.vls");
    let store = Store::create(&path).unwrap();
    let spec = |metric| {
        Vector::dim(DIM as u32)
            .metric(metric)
            .calibrate(Calibrate::auto_sample(8))
    };
    store
        .define_space("v", spec(valise::db::Metric::Cosine))
        .unwrap();
    // Identical redefine is idempotent.
    store
        .define_space("v", spec(valise::db::Metric::Cosine))
        .unwrap();
    // Divergent metric is a caller bug, not a silent no-op.
    let e = store
        .define_space("v", spec(valise::db::Metric::Dot))
        .unwrap_err();
    assert!(
        e.to_string().contains("metric") || e.to_string().contains("different"),
        "{e}"
    );
}

#[test]
fn vector_space_rejects_non_multiple_of_64_dim() {
    let path = tmpfile("baddim.vls");
    let store = Store::create(&path).unwrap();
    // Shared tier.
    let err = store.define_space("emb", Vector::dim(100)).unwrap_err();
    assert!(err.to_string().contains("multiple of 64"), "{err}");
    // Inline tier hits the same check.
    let err = store
        .collection("c", Schema::new().vector("dense", Vector::dim(100)))
        .unwrap_err();
    assert!(err.to_string().contains("multiple of 64"), "{err}");
}

#[test]
fn child_of_builds_tree() {
    let path = tmpfile("tree.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let mut w = store.writer();
    w.put("docs", "parent", Record::new().text("body", "doc"))
        .unwrap();
    w.commit().unwrap();
    // Child references the committed parent.
    w.put(
        "docs",
        "child",
        Record::new()
            .text("body", "chunk")
            .child_of(Key::from("parent")),
    )
    .unwrap();
    w.commit().unwrap();

    assert!(store.get("docs", "child").unwrap().is_some());
}

#[test]
fn reserved_tilde_names_rejected() {
    let path = tmpfile("tilde.vls");
    let store = Store::create(&path).unwrap();

    let e = store
        .collection("~valise.schema", Schema::new().text("body"))
        .unwrap_err();
    assert!(e.to_string().contains("reserved"), "{e}");

    let e = store
        .define_space("~auto/x/y", Text::english())
        .unwrap_err();
    assert!(e.to_string().contains("reserved"), "{e}");

    let e = match store.partitioned(
        "~mem",
        Schema::new().text("body"),
        valise::db::Partition::ByDay,
    ) {
        Err(e) => e,
        Ok(_) => panic!("partitioned must reject a reserved base name"),
    };
    assert!(e.to_string().contains("reserved"), "{e}");
}

#[test]
fn auto_spaces_are_listed_and_flagged() {
    let path = tmpfile("spaces.vls");
    let store = Store::create(&path).unwrap();
    store.define_space("shared-en", Text::english()).unwrap();
    store
        .collection(
            "notes",
            Schema::new()
                .text("body")
                .vector("dense", Vector::dim(DIM as u32)),
        )
        .unwrap();

    let spaces = store.spaces();
    let names: Vec<(&str, bool)> = spaces.iter().map(|s| (s.name.as_str(), s.auto)).collect();
    assert!(names.contains(&("~auto/notes/body", true)), "{names:?}");
    assert!(names.contains(&("~auto/notes/dense", true)), "{names:?}");
    assert!(names.contains(&("shared-en", false)), "{names:?}");
    let dense = spaces
        .iter()
        .find(|s| s.name == "~auto/notes/dense")
        .unwrap();
    assert_eq!(
        dense.kind,
        valise::db::SpaceKind::Vector { dim: DIM as u32 }
    );
}

#[test]
fn redeclare_identical_noop_additive_ok_divergent_errors() {
    let path = tmpfile("redeclare.vls");
    let store = Store::create(&path).unwrap();
    let schema = || Schema::new().text("body");
    store.collection("kb", schema()).unwrap();

    // Identical re-declare is a no-op.
    store.collection("kb", schema()).unwrap();

    // Additive re-declare (appending a field) is accepted.
    store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::dim(DIM as u32)),
        )
        .unwrap();

    // Divergent re-declare (renamed field) is a SchemaMismatch, not a silent
    // overwrite.
    let e = store
        .collection("kb", Schema::new().text("title"))
        .unwrap_err();
    assert!(
        matches!(&e, Error::SchemaMismatch { collection, .. } if collection == "kb"),
        "{e}"
    );

    // Dropping fields is also a mismatch.
    let e = store.collection("kb", schema()).unwrap_err();
    assert!(matches!(e, Error::SchemaMismatch { .. }), "{e}");
}

#[test]
fn shared_space_binding_rejects_codec_and_calibrate() {
    let path = tmpfile("shared_codec.vls");
    let store = Store::create(&path).unwrap();
    let v = store.define_space("emb", Vector::dim(DIM as u32)).unwrap();

    // codec()/calibrate()/metric() on a shared binding error at declaration.
    let e = store
        .collection(
            "c",
            Schema::new().vector("dense", Vector::space(&v).codec(Codec::upq())),
        )
        .unwrap_err();
    assert!(e.to_string().contains("codec"), "{e}");

    let e = store
        .collection(
            "c",
            Schema::new().vector(
                "dense",
                Vector::space(&v).calibrate(Calibrate::auto_sample(4)),
            ),
        )
        .unwrap_err();
    assert!(e.to_string().contains("calibrate"), "{e}");

    // define_space rejects the same misuse.
    let e = store
        .define_space("emb2", Vector::space(&v).codec(Codec::upq()))
        .unwrap_err();
    assert!(e.to_string().contains("codec"), "{e}");

    // A clean shared binding still works.
    store
        .collection("c", Schema::new().vector("dense", Vector::space(&v)))
        .unwrap();
}

/// The declared codec family must be what actually lands in the engine
/// catalog — for both the eager (`Calibrate::now`) and the deferred
/// (first-commit) calibration paths.
#[test]
fn codec_choice_reaches_engine_catalog() {
    use valise::CodecFamily;

    // Eager: UPQ fitted at declaration time.
    let path = tmpfile("codec_eager_upq.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "kb",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32)
                    .codec(Codec::upq())
                    .calibrate(Calibrate::now(sample(64))),
            ),
        )
        .unwrap();
    let families: Vec<CodecFamily> = store
        .raw()
        .reader()
        .codecs()
        .iter()
        .map(|c| c.family)
        .collect();
    assert_eq!(families, vec![CodecFamily::Upq], "eager UPQ choice lost");

    // Deferred: UPQ fitted by the writer's first-commit flush.
    let path = tmpfile("codec_deferred_upq.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "kb",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32)
                    .codec(Codec::upq())
                    .calibrate(Calibrate::auto_sample(8)),
            ),
        )
        .unwrap();
    {
        let mut w = store.writer();
        for i in 0..16u64 {
            let emb = vec_n(2000 + i);
            w.put("kb", i, Record::new().vector("dense", &emb)).unwrap();
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
    assert_eq!(families, vec![CodecFamily::Upq], "deferred UPQ choice lost");
    // And the UPQ-backed field still searches.
    let hits = store
        .search(
            "kb",
            valise::db::Search::new()
                .vector("dense", &vec_n(2003))
                .top_k(3),
        )
        .unwrap();
    assert!(!hits.is_empty());

    // Non-default QAM bit budget survives the deferred path too.
    let path = tmpfile("codec_deferred_qam88.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection(
            "kb",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32)
                    .codec(Codec::qam_bits(8, 8))
                    .calibrate(Calibrate::auto_sample(8)),
            ),
        )
        .unwrap();
    {
        let mut w = store.writer();
        for i in 0..16u64 {
            let emb = vec_n(3000 + i);
            w.put("kb", i, Record::new().vector("dense", &emb)).unwrap();
        }
        w.commit().unwrap();
    }
    let qam = store.raw().reader().codecs();
    assert_eq!(qam.len(), 1);
    assert_eq!(qam[0].family, CodecFamily::QamLloydMax);
}
