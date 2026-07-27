// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Prints results for the reader; stdout is this target's output channel.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! Quickstart: the application-facing `Store` API end to end.
//!
//! Opens a single-file store, declares a text + vector collection in one
//! call, ingests a handful of records, and runs text, vector, and hybrid
//! searches. Runs in milliseconds with no external data:
//!
//!     cargo run --example quickstart

use valise::prelude::*;

/// Toy embedding: deterministic pseudo-random unit vector seeded by the
/// text. A real application would call an embedding model instead —
/// Valise stores vectors; it does not produce them.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut state = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
    });
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("valise_quickstart");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("kb.vls");
    let _ = std::fs::remove_file(&path);

    // 1. Open the store — one file, no sidecars; created if missing.
    let store = Store::open(&path)?;

    // 2. One call declares the collection: an English BM25 text field plus a
    //    cosine vector field whose codec auto-calibrates from the first
    //    committed vectors. dim must be a multiple of 64.
    let dim = 64;
    store.collection(
        "notes",
        Schema::new().text("body").vector(
            "dense",
            Vector::dim(dim as u32).calibrate(Calibrate::auto_sample(1024)),
        ),
    )?;

    // 3. Ingest + commit (commit is the durability point and triggers the
    //    first calibration).
    let notes = [
        (
            "rust-io",
            "Rust file IO with explicit fsync durability barriers",
        ),
        (
            "polar-q",
            "Polar quantization stores vectors as amplitude and phase",
        ),
        ("bm25", "BM25 ranks documents by lexical term statistics"),
        (
            "capsule",
            "A capsule file packs payloads, text index, and vectors",
        ),
    ];
    let mut w = store.writer();
    for (key, body) in notes {
        w.put(
            "notes",
            key,
            Record::new()
                .text("body", body)
                .vector("dense", &embed(body, dim)),
        )?;
    }
    w.commit()?;
    drop(w); // release the single-writer lock (and its borrow of `store`)

    // 4. Search: lexical, vector, and fused hybrid (defaults: BM25,
    //    accurate rerank, RRF fusion).
    let text_hits = store.search(
        "notes",
        Search::new().text("body", "vector quantization").top_k(3),
    )?;
    println!("text:   {:?}", keys(&text_hits));

    let q = embed("how are vectors stored", dim);
    let vec_hits = store.search("notes", Search::new().vector("dense", &q).top_k(3))?;
    println!("vector: {:?}", keys(&vec_hits));

    let hybrid_hits = store.search(
        "notes",
        Search::new()
            .text("body", "vector quantization")
            .vector("dense", &q)
            .fuse(Fusion::rrf(60))
            .top_k(3),
    )?;
    println!("hybrid: {:?}", keys(&hybrid_hits));

    // 5. Fetch a record back by key.
    let stored = store
        .get("notes", "polar-q")?
        .expect("polar-q was committed");
    println!("get:    polar-q -> {:?}", stored.text);

    // 6. Reopen: the schema is persisted in-file, so a second process (or a
    //    later run) needs no re-declaration — reads and searches just work.
    drop(store);
    let store = Store::open(&path)?;
    let again = store
        .get("notes", "polar-q")?
        .expect("schema restored from file");
    println!("reopen: polar-q -> {:?}", again.text);

    Ok(())
}

fn keys(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(|h| format!("{:?}", h.key)).collect()
}
