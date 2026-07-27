// Prints results for the reader; stdout is this target's output channel.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! Integration tests for the DB layer P2 read path: text/vector/hybrid search,
//! fusion (summing RRF + weighted), recency (range/halflife/rrf-channel),
//! collection isolation over a shared text space, rerank modes, get_many,
//! `SearchResult` ergonomics, and the field-validation error surface.

use valise::Error;
use valise::db::{
    Calibrate, Fusion, Hit, Key, Recency, Record, Rerank, Schema, Search, Space, Store, Text,
    Vector,
};

const DIM: usize = 64;

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

/// A shared, eagerly calibrated vector space.
fn vec_space(store: &Store, name: &str) -> Space {
    store
        .define_space(
            name,
            Vector::dim(DIM as u32).calibrate(Calibrate::now(sample())),
        )
        .unwrap()
}

/// Read-path micro-benchmark. Run with:
///   cargo test --test db_search --release -- --ignored --nocapture bench_search
#[test]
#[ignore]
fn bench_search() {
    let path = tmpfile("bench_search.vls");
    let store = Store::create(&path).unwrap();
    let v = vec_space(&store, "dense");
    store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::space(&v)),
        )
        .unwrap();

    const N: u64 = 20_000;
    {
        let mut w = store.writer();
        for i in 0..N {
            w.put(
                "kb",
                i,
                Record::new()
                    .text(
                        "body",
                        "quantum retrieval vector database benchmark document",
                    )
                    .vector("dense", &vec_n(i)),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }

    let reader = store.reader();
    const Q: usize = 1_000;

    let bench = |label: &str, f: &dyn Fn(usize) -> Search| {
        let t = std::time::Instant::now();
        let mut total = 0usize;
        for i in 0..Q {
            total += reader.search("kb", f(i)).unwrap().len();
        }
        let el = t.elapsed();
        eprintln!(
            "{label}: {Q} queries in {el:?} ({:.0}/s, {:.1}us/q), avg_hits={:.1}",
            Q as f64 / el.as_secs_f64(),
            el.as_secs_f64() * 1e6 / Q as f64,
            total as f64 / Q as f64,
        );
    };

    bench("text-only ", &|_| {
        Search::new().text("body", "quantum retrieval").top_k(10)
    });
    bench("vector-only", &|i| {
        Search::new().vector("dense", &vec_n(i as u64)).top_k(10)
    });
    bench("hybrid-rrf ", &|i| {
        Search::new()
            .text("body", "quantum retrieval")
            .vector("dense", &vec_n(i as u64))
            .fuse(Fusion::rrf(60))
            .top_k(10)
    });
    bench("hybrid+recency", &|i| {
        Search::new()
            .text("body", "quantum retrieval")
            .vector("dense", &vec_n(i as u64))
            .recency(Recency::half_life(30.0))
            .now(0)
            .top_k(10)
    });
}

#[test]
fn text_search_ranks_match_first() {
    let path = tmpfile("text_search.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let mut w = store.writer();
    w.put(
        "docs",
        "q",
        Record::new().text("body", "quantum entanglement physics"),
    )
    .unwrap();
    w.put(
        "docs",
        "c",
        Record::new().text("body", "chocolate cake recipe"),
    )
    .unwrap();
    w.put(
        "docs",
        "g",
        Record::new().text("body", "gardening tips for spring"),
    )
    .unwrap();
    w.commit().unwrap();

    let hits = store
        .search("docs", Search::new().text("body", "quantum").top_k(5))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].key, Key::from("q"));
}

#[test]
fn vector_search_ranks_nearest_first() {
    let path = tmpfile("vec_search.vls");
    let store = Store::create(&path).unwrap();
    let v = vec_space(&store, "dense");
    store
        .collection("mem", Schema::new().vector("dense", Vector::space(&v)))
        .unwrap();

    let mut w = store.writer();
    for i in 0..20u64 {
        w.put("mem", i, Record::new().vector("dense", &vec_n(500 + i)))
            .unwrap();
    }
    w.commit().unwrap();

    // Query equals item 507's vector → it must come back first.
    let q = vec_n(507);
    let hits = store
        .search("mem", Search::new().vector("dense", &q).top_k(3))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].key, Key::from(7u64));
}

#[test]
fn delete_then_search_returns_survivors() {
    // Regression: deleting a text-indexed doc + commit must not break text
    // search for the survivors. (Bug: the BM25 vote+rerank path dropped
    // negative-IDF survivors once tombstones shrank the active corpus below a
    // term's df.)
    let path = tmpfile("delete_search.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    {
        let mut w = store.writer();
        w.put("docs", "a", Record::new().text("body", "common alpha"))
            .unwrap();
        w.put("docs", "b", Record::new().text("body", "common beta"))
            .unwrap();
        w.put("docs", "c", Record::new().text("body", "common gamma"))
            .unwrap();
        w.commit().unwrap();
    }
    assert_eq!(
        store
            .search("docs", Search::new().text("body", "common").top_k(10))
            .unwrap()
            .len(),
        3
    );
    {
        let mut w = store.writer();
        assert!(w.delete("docs", "a").unwrap());
        w.commit().unwrap();
    }
    let hits = store
        .search("docs", Search::new().text("body", "common").top_k(10))
        .unwrap();
    assert_eq!(
        hits.len(),
        2,
        "survivors must still be searchable after a delete"
    );
    let keys: Vec<&Key> = hits.iter().map(|h| &h.key).collect();
    assert!(keys.contains(&&Key::from("b")) && keys.contains(&&Key::from("c")));
}

#[test]
fn collection_isolation_over_shared_text_space() {
    // Two collections share the "english" text space. A text search (global in
    // the engine) must be filtered to the queried collection.
    let path = tmpfile("isolation.vls");
    let store = Store::create(&path).unwrap();
    let en = store.define_space("english", Text::english()).unwrap();
    store
        .collection("a", Schema::new().text_with("body", Text::space(&en)))
        .unwrap();
    store
        .collection("b", Schema::new().text_with("body", Text::space(&en)))
        .unwrap();

    let mut w = store.writer();
    w.put("a", "a1", Record::new().text("body", "alpha shared term"))
        .unwrap();
    w.put("b", "b1", Record::new().text("body", "alpha shared term"))
        .unwrap();
    w.commit().unwrap();

    let hits = store
        .search("a", Search::new().text("body", "alpha").top_k(10))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, Key::from("a1"));
    assert_eq!(hits[0].collection, "a");
}

#[test]
fn hybrid_rrf_sums_channels_not_averages() {
    // Discriminating test: X is in BOTH channels; Y is in the vector channel
    // only and ranks #0 there. Summing RRF ranks X first (it accrues both
    // channels); averaging RRF would rank Y first. X must win.
    let path = tmpfile("rrf_sum.vls");
    let store = Store::create(&path).unwrap();
    let v = vec_space(&store, "dense");
    store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::space(&v)),
        )
        .unwrap();

    let q = vec_n(900);
    let mut x_vec = q.clone();
    x_vec[0] += 0.05; // X: close to the query but not identical → vector rank ~1
    let y_vec = q.clone(); // Y: identical to the query → vector rank 0

    let mut w = store.writer();
    // X carries the text term AND a near vector.
    w.put(
        "kb",
        "X",
        Record::new()
            .text("body", "quantum")
            .vector("dense", &x_vec),
    )
    .unwrap();
    // Y is the nearest vector but has no matching text term.
    w.put(
        "kb",
        "Y",
        Record::new().text("body", "banana").vector("dense", &y_vec),
    )
    .unwrap();
    for i in 0..20u64 {
        w.put(
            "kb",
            format!("f{i}"),
            Record::new()
                .text("body", "filler")
                .vector("dense", &vec_n(i)),
        )
        .unwrap();
    }
    w.commit().unwrap();

    let hits = store
        .search(
            "kb",
            Search::new()
                .text("body", "quantum")
                .vector("dense", &q)
                .fuse(Fusion::rrf(60))
                .top_k(5),
        )
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(
        hits[0].key,
        Key::from("X"),
        "both-channel hit must beat single-channel hit (sum, not average)"
    );
}

#[test]
fn weighted_fusion_runs() {
    let path = tmpfile("weighted.vls");
    let store = Store::create(&path).unwrap();
    let v = vec_space(&store, "dense");
    store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::space(&v)),
        )
        .unwrap();

    let q = vec_n(33);
    let mut w = store.writer();
    w.put(
        "kb",
        "hit",
        Record::new()
            .text("body", "relevant quantum doc")
            .vector("dense", &q),
    )
    .unwrap();
    for i in 0..15u64 {
        w.put(
            "kb",
            format!("f{i}"),
            Record::new()
                .text("body", "other")
                .vector("dense", &vec_n(i)),
        )
        .unwrap();
    }
    w.commit().unwrap();

    let hits = store
        .search(
            "kb",
            Search::new()
                .text("body", "quantum")
                .vector("dense", &q)
                .fuse(Fusion::weighted(0.5, 0.5))
                .top_k(5),
        )
        .unwrap();
    assert_eq!(hits[0].key, Key::from("hit"));
}

#[test]
fn recency_range_hard_filters() {
    let path = tmpfile("range.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let mut w = store.writer();
    w.put(
        "docs",
        "old",
        Record::new().text("body", "common term").at(100),
    )
    .unwrap();
    w.put(
        "docs",
        "mid",
        Record::new().text("body", "common term").at(200),
    )
    .unwrap();
    w.put(
        "docs",
        "new",
        Record::new().text("body", "common term").at(300),
    )
    .unwrap();
    w.commit().unwrap();

    let hits = store
        .search(
            "docs",
            Search::new()
                .text("body", "common")
                .recency(Recency::range(150, 250))
                .top_k(10),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, Key::from("mid"));
}

#[test]
fn recency_halflife_favors_newer() {
    let path = tmpfile("halflife.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let day = 86_400i64;
    let mut w = store.writer();
    // Identical text → equal BM25; recency decay breaks the tie.
    w.put(
        "docs",
        "old",
        Record::new().text("body", "memory note").at(0),
    )
    .unwrap();
    w.put(
        "docs",
        "new",
        Record::new().text("body", "memory note").at(100 * day),
    )
    .unwrap();
    w.commit().unwrap();

    let hits = store
        .search(
            "docs",
            Search::new()
                .text("body", "memory")
                .recency(Recency::half_life(30.0))
                .now(100 * day)
                .top_k(5),
        )
        .unwrap();
    assert_eq!(hits[0].key, Key::from("new"));
}

#[test]
fn recency_rrf_channel_favors_newer() {
    let path = tmpfile("rrfchan.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let day = 86_400i64;
    let mut w = store.writer();
    w.put(
        "docs",
        "old",
        Record::new().text("body", "memory note").at(0),
    )
    .unwrap();
    w.put(
        "docs",
        "new",
        Record::new().text("body", "memory note").at(100 * day),
    )
    .unwrap();
    w.commit().unwrap();

    let hits = store
        .search(
            "docs",
            Search::new()
                .text("body", "memory")
                .fuse(Fusion::rrf(60))
                .recency(Recency::rrf_channel(30.0, 2.0))
                .now(100 * day)
                .top_k(5),
        )
        .unwrap();
    assert_eq!(hits[0].key, Key::from("new"));
}

#[test]
fn get_many_batches() {
    let path = tmpfile("getmany.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let mut w = store.writer();
    w.put("docs", "a", Record::new().text("body", "one"))
        .unwrap();
    w.put("docs", "b", Record::new().text("body", "two"))
        .unwrap();
    w.commit().unwrap();

    let reader = store.reader();
    let got = reader
        .get_many(
            "docs",
            &[Key::from("a"), Key::from("missing"), Key::from("b")],
        )
        .unwrap();
    assert_eq!(got.len(), 3);
    assert_eq!(got[0].as_ref().unwrap().text.as_deref(), Some("one"));
    assert!(got[1].is_none());
    assert_eq!(got[2].as_ref().unwrap().text.as_deref(), Some("two"));
}

#[test]
fn search_error_surface() {
    let path = tmpfile("errs.vls");
    let store = Store::create(&path).unwrap();
    // Deferred (auto-calibrating) vector field: not calibrated until a commit.
    store
        .collection(
            "kb",
            Schema::new().text("body").vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::auto_sample(8)),
            ),
        )
        .unwrap();

    // Unknown field.
    let e = store
        .search("kb", Search::new().text("nope", "x"))
        .unwrap_err();
    assert!(e.to_string().contains("unknown field"), "{e}");

    // Text scorer aimed at a vector field.
    let e = store
        .search("kb", Search::new().text("dense", "x"))
        .unwrap_err();
    assert!(e.to_string().contains("vector field"), "{e}");

    // Vector search against an uncalibrated (deferred) auto space: the typed
    // NotCalibrated error names the auto space and guides to a commit.
    let q = vec_n(1);
    let e = store
        .search("kb", Search::new().vector_with("dense", &q, Rerank::Fast))
        .unwrap_err();
    assert!(
        matches!(&e, Error::NotCalibrated { space } if space == "~auto/kb/dense"),
        "{e}"
    );
    assert!(e.to_string().contains("not calibrated"), "{e}");

    // No channels.
    let e = store.search("kb", Search::new()).unwrap_err();
    assert!(e.to_string().contains("no channels"), "{e}");
}

#[test]
fn rerank_fast_and_accurate_both_work() {
    let path = tmpfile("rerank.vls");
    let store = Store::create(&path).unwrap();
    let v = vec_space(&store, "dense");
    store
        .collection("mem", Schema::new().vector("dense", Vector::space(&v)))
        .unwrap();

    let mut w = store.writer();
    for i in 0..20u64 {
        w.put("mem", i, Record::new().vector("dense", &vec_n(700 + i)))
            .unwrap();
    }
    w.commit().unwrap();

    let q = vec_n(705);
    for rerank in [Rerank::Fast, Rerank::Accurate] {
        let hits = store
            .search(
                "mem",
                Search::new().vector_with("dense", &q, rerank).top_k(3),
            )
            .unwrap();
        assert_eq!(hits[0].key, Key::from(5u64), "{rerank:?}");
    }
}

#[test]
fn search_result_derefs_and_iterates() {
    let path = tmpfile("result.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    let mut w = store.writer();
    for i in 0..3u64 {
        w.put("docs", i, Record::new().text("body", "common note"))
            .unwrap();
    }
    w.commit().unwrap();

    let result = store
        .search("docs", Search::new().text("body", "common").top_k(10))
        .unwrap();

    // Deref to [Hit]: len / index / iter / slice coercion.
    assert_eq!(result.len(), 3);
    assert!(result[0].score >= result[2].score);
    fn takes_slice(hits: &[Hit]) -> usize {
        hits.len()
    }
    assert_eq!(takes_slice(&result), 3);

    // Borrowed iteration.
    let borrowed: Vec<&Key> = (&result).into_iter().map(|h| &h.key).collect();
    assert_eq!(borrowed.len(), 3);

    // The hits field is public, and owned iteration consumes the result.
    assert_eq!(result.hits.len(), 3);
    let owned: Vec<Hit> = result.into_iter().collect();
    assert_eq!(owned.len(), 3);
}
