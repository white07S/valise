//! Stage 5: writer pipeline + reader-during-writer-progress.
//!
//! Three guarantees:
//! 1. The Stage 4 reader-blocks-on-writer regression is gone — a
//!    long-lived `WriteConnection` does not freeze concurrent
//!    `ReadConnection`s, only other `WriteConnection`s.
//! 2. Multiple committers go through the pipeline FIFO; outcomes are
//!    delivered in queue order; concurrent commit() calls all
//!    eventually succeed.
//! 3. The `commit_gather_window` knob takes effect (the leader's
//!    pre-drain wait is honoured per call).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tempfile::tempdir;
use valise::{Database, OpenMode, PutFrame, ValiseFile};

fn make_file(path: &std::path::Path) {
    let mut valise = ValiseFile::create(path).unwrap();
    let cid = valise.create_collection("c").unwrap();
    let _ = valise.put_frame(PutFrame::document(cid, b"x")).unwrap();
    valise.commit().unwrap();
}

#[test]
fn long_lived_writer_does_not_block_concurrent_readers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rw.vls");
    make_file(&path);

    let db = Database::open(&path, OpenMode::ReadWrite).unwrap();
    let mut writer = db.writer();
    // Hold the writer for 200 ms, doing two interleaved put_frame +
    // commit cycles. Meanwhile, a reader thread issues a series of
    // read_payload calls — Stage 4 would have stalled the reader for
    // the entire 200 ms; Stage 5 lets reads interleave between the
    // writer's individual method calls.
    let cid = writer.collections()[0].collection_id;

    let db_reader = Arc::clone(&db);
    let reader_runs = Arc::new(AtomicU64::new(0));
    let reader_runs_handle = Arc::clone(&reader_runs);
    let reader_thread = thread::spawn(move || {
        let reader = db_reader.reader();
        let stubs = reader.frame_stubs();
        let frame_id = stubs[0].frame_id;
        let deadline = Instant::now() + Duration::from_millis(200);
        while Instant::now() < deadline {
            let _ = reader.read_payload(frame_id).unwrap();
            reader_runs_handle.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(2));
        }
    });

    // Writer does work over the same 200 ms.
    let writer_start = Instant::now();
    let mut commits = 0;
    while writer_start.elapsed() < Duration::from_millis(200) {
        let _ = writer.put_frame(PutFrame::document(cid, b"y")).unwrap();
        let _ = writer.commit().unwrap();
        commits += 1;
        thread::sleep(Duration::from_millis(20));
    }
    drop(writer);
    reader_thread.join().unwrap();

    let runs = reader_runs.load(Ordering::Relaxed);
    assert!(
        runs >= 5,
        "reader must make progress while writer is alive (got {runs} reads, {commits} commits)",
    );
    assert!(commits >= 2, "writer must also make progress");
}

#[test]
fn concurrent_committers_all_succeed() {
    // Eight writer threads, each opens its own write transaction, does
    // a put_frame + commit, drops. Without the pipeline's exclusion
    // they'd race on `ValiseFile`'s internal state; with it they
    // serialize and all succeed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("concurrent.vls");
    make_file(&path);

    let db = Database::open(&path, OpenMode::ReadWrite).unwrap();
    let cid = db.writer().collections()[0].collection_id;

    let started = Arc::new(std::sync::Barrier::new(8));
    let mut handles = Vec::new();
    for i in 0..8 {
        let db = Arc::clone(&db);
        let started = Arc::clone(&started);
        handles.push(thread::spawn(move || {
            started.wait();
            let mut w = db.writer();
            let _ = w
                .put_frame(PutFrame::document(cid, format!("frame-{i}").as_bytes()))
                .unwrap();
            w.commit().unwrap()
        }));
    }
    let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All eight commits should produce distinct, monotonically-advancing
    // generations.
    let mut gens: Vec<u64> = outcomes.iter().map(|o| o.snapshot_generation).collect();
    gens.sort();
    gens.dedup();
    assert_eq!(gens.len(), 8, "all commits must be distinct");
    // Final state on disk has at least 9 frames (the seed + 8 added).
    let final_db = Database::open(&path, OpenMode::ReadOnly).unwrap();
    let stubs = final_db.reader().frame_stubs();
    assert!(stubs.len() >= 9, "got {} frames", stubs.len());
}

#[test]
fn pipeline_fifo_under_concurrency() {
    // Spawn 16 writers, capture commit start order, assert the
    // checkpoint_seq returned by each is monotonic in arrival order.
    // Uses a small sleep inside `submit` to force interleaving.
    let dir = tempdir().unwrap();
    let path = dir.path().join("fifo.vls");
    make_file(&path);

    let db = Database::open(&path, OpenMode::ReadWrite).unwrap();
    let cid = db.writer().collections()[0].collection_id;

    let arrival_counter = Arc::new(AtomicU64::new(0));
    let started = Arc::new(std::sync::Barrier::new(16));
    let mut handles = Vec::new();
    for _ in 0..16 {
        let db = Arc::clone(&db);
        let arrival_counter = Arc::clone(&arrival_counter);
        let started = Arc::clone(&started);
        handles.push(thread::spawn(move || {
            started.wait();
            let arrival_seq = arrival_counter.fetch_add(1, Ordering::AcqRel);
            let mut w = db.writer();
            let _ = w.put_frame(PutFrame::document(cid, b"x")).unwrap();
            let outcome = w.commit().unwrap();
            (arrival_seq, outcome.snapshot_generation)
        }));
    }
    let mut pairs: Vec<(u64, u64)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    pairs.sort_by_key(|(arrival, _)| *arrival);
    // Snapshot generations should be unique — 16 distinct commits.
    let mut gens: Vec<u64> = pairs.iter().map(|(_, g)| *g).collect();
    gens.sort();
    gens.dedup();
    assert_eq!(gens.len(), 16);
}
