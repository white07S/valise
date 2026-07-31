// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Compaction renames a fresh file over the store path, unlinking the old
//! inode. A writer opened before the swap still holds an fd on that inode;
//! committing through it used to fsync into the unlinked file and return
//! `Ok`, losing the write. The commit path now checks the fd's link count
//! and returns `Err(Busy)` instead.

use valise::db::{CompactOptions, Record, Schema, Store};

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

#[test]
fn stale_inode_after_concurrent_compaction_is_loud_err_not_silent_ok() {
    let path = tmpfile("fix4.vls");

    // Seed a store with a real tombstone so compaction actually rewrites
    // the file (put k0, delete k0, put k1 as a surviving key).
    {
        let s = Store::create(&path).unwrap();
        s.collection("c", Schema::new().text("body")).unwrap();
        {
            let mut w = s.writer();
            w.put("c", "k0", Record::new().text("body", "zero"))
                .unwrap();
            w.commit().unwrap();
        }
        {
            let mut w = s.writer();
            w.delete("c", "k0").unwrap();
            w.commit().unwrap();
        }
        {
            let mut w = s.writer();
            w.put("c", "k1", Record::new().text("body", "one")).unwrap();
            w.commit().unwrap();
        }
    } // drop the seeding handle so the file is quiescent

    // Two independent handles onto the same path. `store_w` will hold an fd
    // on the inode that `store_c`'s compaction is about to unlink.
    let store_w = Store::open(&path).unwrap();
    let store_c = Store::open(&path).unwrap();

    // Compaction rewrites a fresh inode and renames it over the path,
    // unlinking the inode that `store_w` still has open.
    let report = store_c.compact(CompactOptions::default()).unwrap();
    assert!(report.compacted, "tombstone present → compaction must run");

    // Commit through the stale handle. Its fd points at the now-unlinked
    // (nlink == 0) old inode. The guard must convert what would have been a
    // silent lost write into a retryable error.
    let mut w = store_w.writer();
    w.put("c", "survivor", Record::new().text("body", "must-survive"))
        .unwrap();
    let res = w.commit();

    assert!(
        res.is_err(),
        "commit into an inode unlinked by concurrent compaction must fail \
         loudly (Err), not silently swallow the write and return Ok"
    );
}
