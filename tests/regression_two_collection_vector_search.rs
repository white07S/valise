// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Regression: a vector search must survive a commit that touched only a
//! *different* collection.
//!
//! Found by dogfooding — a two-collection layout (one holding an embedded
//! corpus, one holding mutable per-record state) segfaulted the Python
//! binding on the first vector query issued after a state-only commit.

use valise::db::{Calibrate, Key, Record, Schema, Search, Space, Store, Vector};

const DIM: usize = 64;

fn vec_n(seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
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

fn vec_space(store: &Store, name: &str) -> Space {
    store
        .define_space(
            name,
            Vector::dim(DIM as u32).calibrate(Calibrate::now((0..128).map(vec_n).collect())),
        )
        .unwrap()
}

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

#[test]
fn vector_search_survives_commit_to_another_collection() {
    let path = tmpfile("two_coll.vls");
    let store = Store::create(&path).unwrap();
    let space = vec_space(&store, "dense");

    // `threads` carries the vectors; `state` deliberately has no vector
    // space at all — it is the mutable side-table.
    store
        .collection(
            "threads",
            Schema::new()
                .text("body")
                .vector("dense", Vector::space(&space)),
        )
        .unwrap();
    store
        .collection("state", Schema::new().text("meta"))
        .unwrap();

    let q = vec_n(7);
    // Each writer gets its own scope: a `WriteConnection` holds the
    // single-writer slot until it drops, so shadowing one with another
    // in the same scope self-deadlocks.
    {
        let mut w = store.writer();
        w.put(
            "threads",
            "a",
            Record::new()
                .text("body", "hnsw memory usage")
                .vector("dense", &q),
        )
        .unwrap();
        w.put("state", "a", Record::new().text("meta", "status=new"))
            .unwrap();
        w.commit().unwrap();
    }

    // A vector query before the state-only commit works.
    let before = store
        .search(
            "threads",
            Search::new()
                .text("body", "hnsw memory")
                .vector("dense", &q)
                .top_k(5),
        )
        .unwrap();
    assert_eq!(
        before.len(),
        1,
        "setup: the vector query must find the record"
    );

    // Commit that touches ONLY the collection without a vector space.
    {
        let mut w = store.writer();
        w.put("state", "a", Record::new().text("meta", "status=done"))
            .unwrap();
        w.commit().unwrap();
    }

    // The same query again. This is what crashed.
    let after = store
        .search(
            "threads",
            Search::new()
                .text("body", "hnsw memory")
                .vector("dense", &q)
                .top_k(5),
        )
        .unwrap();
    assert_eq!(
        after.len(),
        1,
        "the vector query must still work after a commit to another collection"
    );
    assert_eq!(after[0].key, Key::from("a"), "and return the right record");
}

/// The same hazard without a second collection: any commit that touches
/// no vector still remaps, so a text-only update to a collection that
/// *has* a vector space must not strand the pointer cache either.
#[test]
fn vector_search_survives_a_text_only_commit() {
    let path = tmpfile("text_only.vls");
    let store = Store::create(&path).unwrap();
    let space = vec_space(&store, "dense");
    store
        .collection(
            "kb",
            Schema::new()
                .text("body")
                .vector("dense", Vector::space(&space)),
        )
        .unwrap();

    let q = vec_n(11);
    {
        let mut w = store.writer();
        w.put(
            "kb",
            "withvec",
            Record::new()
                .text("body", "quantization budget")
                .vector("dense", &q),
        )
        .unwrap();
        w.commit().unwrap();
    }

    // A record with text but no vector: the commit dirties no vector,
    // no codec and no embedding space, so `vectors_changed` is false.
    {
        let mut w = store.writer();
        w.put(
            "kb",
            "textonly",
            Record::new().text("body", "quantization notes"),
        )
        .unwrap();
        w.commit().unwrap();
    }

    let hits = store
        .search("kb", Search::new().vector("dense", &q).top_k(5))
        .unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the vector query must survive a commit that added no vector"
    );
}

/// Repeated vectorless commits must stay safe: the cache is invalidated
/// on each one and lazily rebuilt by the search in between.
#[test]
fn vector_search_survives_repeated_vectorless_commits() {
    let path = tmpfile("repeat.vls");
    let store = Store::create(&path).unwrap();
    let space = vec_space(&store, "dense");
    store
        .collection(
            "threads",
            Schema::new()
                .text("body")
                .vector("dense", Vector::space(&space)),
        )
        .unwrap();
    store
        .collection("state", Schema::new().text("meta"))
        .unwrap();

    let q = vec_n(3);
    {
        let mut w = store.writer();
        w.put(
            "threads",
            "a",
            Record::new()
                .text("body", "sketch candidates")
                .vector("dense", &q),
        )
        .unwrap();
        w.commit().unwrap();
    }

    for round in 0..8u32 {
        {
            let mut w = store.writer();
            w.put(
                "state",
                "a",
                Record::new().text("meta", &format!("round={round}")),
            )
            .unwrap();
            w.commit().unwrap();
        }
        let hits = store
            .search("threads", Search::new().vector("dense", &q).top_k(5))
            .unwrap();
        assert_eq!(hits.len(), 1, "round {round}: vector query must survive");
    }
}
