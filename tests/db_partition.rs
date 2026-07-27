// Prints results for the reader; stdout is this target's output channel.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! Integration tests for the DB layer P3 scale-out: partition routing,
//! views + global-fusion search_view, soft forget_before, and the Bulk loader.

use std::sync::Arc;

use valise::db::{
    Calibrate, Fusion, Key, Partition, Recency, Record, Rerank, Schema, Search, Store, Vector,
    Window,
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

/// search_view fan-out benchmark over many day-partitions.
///   cargo test --test db_partition --release -- --ignored --nocapture bench_view
#[test]
#[ignore]
fn bench_view() {
    let path = tmpfile("bench_view.vls");
    let store = Store::create(&path).unwrap();
    let mem = store
        .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
        .unwrap();

    const DAYS: i64 = 180;
    const PER_DAY: u64 = 100;
    {
        let mut w = store.writer();
        let mut id = 0u64;
        for d in 0..DAYS {
            for _ in 0..PER_DAY {
                w.put_into(
                    &mem,
                    Key::from(id),
                    Record::new()
                        .text("body", "quantum retrieval benchmark note")
                        .at(d * DAY + 10),
                )
                .unwrap();
                id += 1;
            }
        }
        w.commit().unwrap();
    }

    let view = mem.all();
    let q = 1000;
    let t = std::time::Instant::now();
    let mut total = 0;
    for _ in 0..q {
        total += store
            .search_view(
                &view,
                Search::new().text("body", "quantum retrieval").top_k(10),
            )
            .unwrap()
            .len();
    }
    let el = t.elapsed();
    eprintln!(
        "search_view over {DAYS} partitions ({} docs): {q} queries in {el:?} ({:.1}us/q), avg_hits={:.1}",
        DAYS as u64 * PER_DAY,
        el.as_secs_f64() * 1e6 / q as f64,
        total as f64 / q as f64,
    );
}

#[test]
fn by_day_routing_creates_partitions() {
    let path = tmpfile("byday.vls");
    let store = Store::create(&path).unwrap();
    let mem = store
        .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
        .unwrap();

    {
        let mut w = store.writer();
        // day 0 (1970-01-01), day 1 (1970-01-02), day 1 again
        w.put_into(&mem, "a", Record::new().text("body", "alpha").at(0))
            .unwrap();
        w.put_into(&mem, "b", Record::new().text("body", "beta").at(DAY))
            .unwrap();
        w.put_into(&mem, "c", Record::new().text("body", "gamma").at(DAY + 60))
            .unwrap();
        w.commit().unwrap();
    }

    let names: Vec<String> = store.collections().into_iter().map(|c| c.name).collect();
    assert!(names.contains(&"mem:1970-01-01".to_string()), "{names:?}");
    assert!(names.contains(&"mem:1970-01-02".to_string()), "{names:?}");

    // Records land in their routed physical collections.
    assert!(store.get("mem:1970-01-01", "a").unwrap().is_some());
    assert!(store.get("mem:1970-01-02", "b").unwrap().is_some());
    assert!(store.get("mem:1970-01-02", "c").unwrap().is_some());
    // 'b' is not in day 1970-01-01.
    assert!(store.get("mem:1970-01-01", "b").unwrap().is_none());
}

#[test]
fn search_view_window_and_global_ranking() {
    let path = tmpfile("view.vls");
    let store = Store::create(&path).unwrap();
    let mem = store
        .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
        .unwrap();

    {
        let mut w = store.writer();
        for d in 0..5i64 {
            w.put_into(
                &mem,
                format!("k{d}"),
                Record::new()
                    .text("body", "memory note common")
                    .at(d * DAY + 10),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }

    // Window [day1, day3] → 3 partitions, 3 hits.
    let view = mem.view(Window::Range {
        from: DAY,
        to: 3 * DAY + DAY - 1,
    });
    assert_eq!(view.collections().len(), 3);
    let hits = store
        .search_view(&view, Search::new().text("body", "memory").top_k(10))
        .unwrap();
    let keys: Vec<String> = hits
        .iter()
        .map(|h| match &h.key {
            Key::Str(s) => s.clone(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(hits.len(), 3, "{keys:?}");
    assert!(keys.contains(&"k1".to_string()));
    assert!(keys.contains(&"k2".to_string()));
    assert!(keys.contains(&"k3".to_string()));
    // Hit.collection disambiguates the partition.
    assert!(hits.iter().any(|h| h.collection == "mem:1970-01-02"));

    // all() sees every partition.
    assert_eq!(mem.all().collections().len(), 5);
}

#[test]
fn view_as_of_last_days_is_deterministic() {
    let path = tmpfile("lastdays.vls");
    let store = Store::create(&path).unwrap();
    let mem = store
        .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
        .unwrap();
    {
        let mut w = store.writer();
        for d in 0..6i64 {
            w.put_into(
                &mem,
                format!("k{d}"),
                Record::new().text("body", "x").at(d * DAY + 10),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }
    // "now" mid-day-5: LastDays(2) ⇒ window [day3+.., day5+..] ⇒ days 3,4,5.
    let now = 5 * DAY + 50_000;
    let view = mem.view_as_of(Window::LastDays(2), now);
    let mut names: Vec<&str> = view.collections().iter().map(|s| s.as_str()).collect();
    names.sort();
    assert_eq!(
        names,
        ["mem:1970-01-04", "mem:1970-01-05", "mem:1970-01-06"]
    );
}

#[test]
fn search_view_vector_across_partitions() {
    let path = tmpfile("view_vec.vls");
    let store = Store::create(&path).unwrap();
    // Inline vector spec: the auto space is defined once for the base, shared
    // by every partition.
    let mem = store
        .partitioned(
            "mem",
            Schema::new().vector(
                "dense",
                Vector::dim(DIM as u32).calibrate(Calibrate::now(sample())),
            ),
            Partition::ByDay,
        )
        .unwrap();

    {
        let mut w = store.writer();
        for d in 0..6i64 {
            w.put_into(
                &mem,
                format!("k{d}"),
                Record::new()
                    .vector("dense", &vec_n(d as u64))
                    .at(d * DAY + 10),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }

    // Query == k4's vector; over the full view it should rank first globally.
    let q = vec_n(4);
    let hits = store
        .search_view(&mem.all(), Search::new().vector("dense", &q).top_k(3))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].key, Key::from("k4"));
    assert_eq!(hits[0].collection, "mem:1970-01-05");
}

#[test]
fn forget_before_soft_tombstones_old_partitions() {
    let path = tmpfile("forget.vls");
    let store = Store::create(&path).unwrap();
    let mem = store
        .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
        .unwrap();

    {
        let mut w = store.writer();
        for d in 0..4i64 {
            w.put_into(
                &mem,
                format!("k{d}"),
                Record::new().text("body", "common note").at(d * DAY + 10),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }

    // Forget everything whose period ends ≤ 2 days: day0 [0,DAY) and day1 [DAY,2DAY).
    let forgotten = mem.forget_before(2 * DAY).unwrap();
    assert_eq!(forgotten, 2);

    // The forgotten records no longer surface or resolve.
    assert!(store.get("mem:1970-01-01", "k0").unwrap().is_none());
    assert!(store.get("mem:1970-01-02", "k1").unwrap().is_none());
    assert!(store.get("mem:1970-01-03", "k2").unwrap().is_some());

    let hits = store
        .search_view(&mem.all(), Search::new().text("body", "common").top_k(10))
        .unwrap();
    assert_eq!(hits.len(), 2, "only days 2,3 remain");
    let keys: Vec<&Key> = hits.iter().map(|h| &h.key).collect();
    assert!(keys.contains(&&Key::from("k2")));
    assert!(keys.contains(&&Key::from("k3")));

    // Idempotent-ish: forgetting again forgets nothing new.
    assert_eq!(mem.forget_before(2 * DAY).unwrap(), 0);
}

#[test]
fn search_view_narrow_window_fills_top_k() {
    // A view over one partition out of many must return a full top_k even when
    // its docs rank globally *below* the old fixed over-fetch (100). Make the
    // target partition's docs strictly less relevant (tf=1) than every other
    // partition's (tf=2), so globally they sit at ranks ~870–900 — past 100,
    // reachable only via the scope-aware over-fetch widening.
    let path = tmpfile("narrow.vls");
    let store = Store::create(&path).unwrap();
    let mem = store
        .partitioned("mem", Schema::new().text("body"), Partition::ByDay)
        .unwrap();

    {
        let mut w = store.writer();
        // Day 0 (target): 30 docs with the query term once (lower BM25).
        for i in 0..30u64 {
            w.put_into(
                &mem,
                format!("t{i}"),
                Record::new().text("body", "common").at(10),
            )
            .unwrap();
        }
        // Days 1..29: 30 docs each with the term twice (higher BM25) → 870 docs
        // that all out-rank the target globally.
        for d in 1..29i64 {
            for i in 0..30u64 {
                w.put_into(
                    &mem,
                    format!("d{d}_{i}"),
                    Record::new().text("body", "common common").at(d * DAY + 10),
                )
                .unwrap();
            }
        }
        w.commit().unwrap();
    }

    let view = mem.view(Window::Partitions(vec!["1970-01-01".to_string()]));
    assert_eq!(view.collections(), &["mem:1970-01-01".to_string()]);
    let hits = store
        .search_view(&view, Search::new().text("body", "common").top_k(10))
        .unwrap();
    assert_eq!(hits.len(), 10, "narrow view must fill top_k");
    assert!(hits.iter().all(|h| h.collection == "mem:1970-01-01"));
}

#[test]
fn forget_before_rejects_custom_partition() {
    let path = tmpfile("forget_custom.vls");
    let store = Store::create(&path).unwrap();
    let router: Arc<dyn Fn(&Record) -> String + Send + Sync> = Arc::new(|_| "bucket".to_string());
    let p = store
        .partitioned("t", Schema::new().text("body"), Partition::Custom(router))
        .unwrap();
    let err = p.forget_before(i64::MAX).unwrap_err();
    assert!(err.to_string().contains("time partition"), "{err}");
}

#[test]
fn custom_and_by_month_routing() {
    let path = tmpfile("custom.vls");
    let store = Store::create(&path).unwrap();

    // Custom router by a captured prefix.
    let router: Arc<dyn Fn(&Record) -> String + Send + Sync> = Arc::new(|r| {
        if r.created_at().unwrap_or(0) < DAY {
            "early".into()
        } else {
            "late".into()
        }
    });
    let custom = store
        .partitioned("c", Schema::new().text("body"), Partition::Custom(router))
        .unwrap();
    // ByMonth router.
    let by_month = store
        .partitioned("m", Schema::new().text("body"), Partition::ByMonth)
        .unwrap();

    {
        // The engine time index requires created_at non-decreasing in commit
        // order, so ingest oldest-first.
        let mut w = store.writer();
        w.put_into(&custom, "e", Record::new().text("body", "x").at(0))
            .unwrap();
        w.put_into(&by_month, "jan", Record::new().text("body", "x").at(0))
            .unwrap();
        w.put_into(&custom, "l", Record::new().text("body", "x").at(5 * DAY))
            .unwrap();
        w.commit().unwrap();
    }

    let names: Vec<String> = store.collections().into_iter().map(|c| c.name).collect();
    assert!(names.contains(&"c:early".to_string()), "{names:?}");
    assert!(names.contains(&"c:late".to_string()), "{names:?}");
    assert!(names.contains(&"m:1970-01".to_string()), "{names:?}");
}

#[test]
fn bulk_loads_and_group_commits() {
    let path = tmpfile("bulk.vls");
    let store = Store::create(&path).unwrap();
    store
        .collection("docs", Schema::new().text("body"))
        .unwrap();

    {
        let mut w = store.writer();
        let mut bulk = w.bulk();
        for i in 0..200u64 {
            bulk.put("docs", i, Record::new().text("body", "bulk doc"))
                .unwrap();
        }
        bulk.flush().unwrap();
    }

    assert!(store.get("docs", 0u64).unwrap().is_some());
    assert!(store.get("docs", 199u64).unwrap().is_some());
}

#[test]
fn bulk_repeated_key_is_size_independent() {
    // A key re-put within the same window forces a flush so the update is
    // valid, rather than tripping the uncommitted-vector rule.
    let path = tmpfile("bulk_dup.vls");
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

    let a = vec_n(1);
    let b = vec_n(2);
    {
        let mut w = store.writer();
        let mut bulk = w.bulk();
        bulk.put("c", "k", Record::new().vector("dense", &a))
            .unwrap();
        // Same key again, with a vector → must succeed (auto-flush between).
        bulk.put("c", "k", Record::new().vector("dense", &b))
            .unwrap();
        bulk.flush().unwrap();
    }

    let got = store.get("c", "k").unwrap().unwrap();
    assert_eq!(got.vectors.len(), 1);

    // Hybrid hybrid+RrfChannel fusion-with-weighted mismatch is rejected.
    let q = vec_n(2);
    let err = store
        .search(
            "c",
            Search::new()
                .vector_with("dense", &q, Rerank::Fast)
                .fuse(Fusion::weighted(1.0, 1.0))
                .recency(Recency::rrf_channel(1.0, 1.0)),
        )
        .unwrap_err();
    assert!(err.to_string().contains("RrfChannel requires"), "{err}");
}
