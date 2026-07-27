// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Integration tests for `Store::writer_owned` (the owned, lifetime-free
//! single-writer handle that the FFI layer builds on).
//!
//! Proves: (1) an `OwnedWriter` can `put` + `commit` and the write is visible
//! through `get`; (2) the owned writer holds the same single-writer lock — a
//! second `writer_owned()` blocks while the first is alive and proceeds once it
//! is dropped.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use valise::db::{Record, Schema, Store};

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

#[test]
fn owned_writer_put_commit_roundtrip() {
    let path = tmpfile("owned_roundtrip.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    {
        let mut w = store.writer_owned();
        w.put(
            "docs",
            "doc-1",
            Record::new().text("body", "owned writer round trip"),
        )
        .unwrap();
        w.commit().unwrap();
    }

    let got = store.get("docs", "doc-1").unwrap().unwrap();
    assert_eq!(got.text.as_deref(), Some("owned writer round trip"));
}

#[test]
fn owned_writer_holds_single_writer_lock() {
    let path = tmpfile("owned_lock.vls");
    let store = Arc::new(Store::create(&path).unwrap());
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    // Hold the owned writer on this thread.
    let mut held = store.writer_owned();

    // A second thread tries to acquire another owned writer; it must block until
    // we drop `held`.
    let acquired = Arc::new(AtomicBool::new(false));
    let store2 = Arc::clone(&store);
    let acquired2 = Arc::clone(&acquired);
    let t = std::thread::spawn(move || {
        let mut w = store2.writer_owned();
        acquired2.store(true, Ordering::SeqCst);
        w.put("docs", "doc-2", Record::new().text("body", "second writer"))
            .unwrap();
        w.commit().unwrap();
    });

    // Give the thread time to (fail to) acquire. It must still be blocked.
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !acquired.load(Ordering::SeqCst),
        "second writer_owned() must block while the first is alive"
    );

    // Do real work on the held writer, then release the lock.
    held.put("docs", "doc-1", Record::new().text("body", "first writer"))
        .unwrap();
    held.commit().unwrap();
    drop(held);

    // Now the second thread can proceed.
    t.join().unwrap();
    assert!(acquired.load(Ordering::SeqCst));

    assert!(store.get("docs", "doc-1").unwrap().is_some());
    assert!(store.get("docs", "doc-2").unwrap().is_some());
}
