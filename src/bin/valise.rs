// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `valise` — inspect and query a capsule file from the shell.
//!
//! Stdout is the CLI's output channel, so the crate-wide `print_stdout`
//! lint is relaxed here only.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::{path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use valise::prelude::*;

#[derive(Parser)]
#[command(
    name = "valise",
    version,
    about = "Inspect and query a Valise capsule file",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarize a capsule: size, collections, spaces, record counts.
    Info {
        /// Path to the `.vls` file.
        file: PathBuf,
        /// Break the file size down by segment type — where the bytes went.
        #[arg(long)]
        segments: bool,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },
    /// Run a text search against one collection.
    Search {
        /// Path to the `.vls` file.
        file: PathBuf,
        /// Collection to search.
        collection: String,
        /// Query string.
        query: String,
        /// Text field to match against.
        #[arg(long, default_value = "body")]
        field: String,
        /// Maximum number of hits.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },
    /// Fetch a single record by key.
    Get {
        /// Path to the `.vls` file.
        file: PathBuf,
        /// Collection holding the record.
        collection: String,
        /// Record key.
        key: String,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },
    /// Stream every record to stdout as JSON Lines.
    Export {
        /// Path to the `.vls` file.
        file: PathBuf,
        /// Restrict the export to one collection.
        #[arg(long)]
        collection: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("valise: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Info {
            file,
            segments,
            json,
        } => info(&file, segments, json),
        Command::Search {
            file,
            collection,
            query,
            field,
            top_k,
            json,
        } => search(&file, &collection, &query, &field, top_k, json),
        Command::Get {
            file,
            collection,
            key,
            json,
        } => get(&file, &collection, &key, json),
        Command::Export { file, collection } => export(&file, collection.as_deref()),
    }
}

/// `Store::open` is open-or-create; the CLI only ever reads, so refuse a
/// missing path rather than silently creating an empty capsule.
fn open_existing(path: &PathBuf) -> Result<Store> {
    if !path.exists() {
        return Err(Error::Other(format!("no such file: {}", path.display())));
    }
    Store::open(path)
}

fn info(path: &PathBuf, segments: bool, json: bool) -> Result<()> {
    let store = open_existing(path)?;
    let stats = store.stats();
    let collections = store.collections();
    let spaces = store.spaces();
    // Live bytes only — tombstoned segments are excluded, so after deletes
    // this sums to less than the file on disk until `compact` reclaims it.
    let breakdown = store.raw().reader().storage_breakdown();

    if json {
        let payload = serde_json::json!({
            "file": path.display().to_string(),
            "file_bytes": stats.file_bytes,
            "frames_total": stats.frames_total,
            "frames_active": stats.frames_active,
            "vectors_total": stats.vectors_total,
            "vectors_active": stats.vectors_active,
            "tombstone_ratio": stats.tombstone_ratio(),
            "collections": collections.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
            "spaces": spaces.iter().map(|s| {
                serde_json::json!({ "name": s.name, "kind": format!("{:?}", s.kind), "auto": s.auto })
            }).collect::<Vec<_>>(),
            "segments": breakdown.iter().map(|(t, count, bytes)| {
                serde_json::json!({ "type": format!("{t:?}"), "count": count, "bytes": bytes })
            }).collect::<Vec<_>>(),
        });
        println!("{}", json_pretty(&payload)?);
        return Ok(());
    }

    println!("{}", path.display());
    println!("  size            {}", human_bytes(stats.file_bytes));
    println!(
        "  records         {} active / {} total",
        stats.frames_active, stats.frames_total
    );
    println!(
        "  vectors         {} active / {} total",
        stats.vectors_active, stats.vectors_total
    );
    println!(
        "  tombstones      {:.1}%{}",
        stats.tombstone_ratio() * 100.0,
        if stats.needs_compaction(0.3) {
            "  (compaction recommended)"
        } else {
            ""
        }
    );

    if collections.is_empty() {
        println!("  collections     (none)");
    } else {
        println!("  collections     {}", collections.len());
        for c in &collections {
            println!("    - {}", c.name);
        }
    }
    if !spaces.is_empty() {
        println!("  spaces          {}", spaces.len());
        for s in &spaces {
            let auto = if s.auto { "  [auto]" } else { "" };
            println!("    - {} ({:?}){auto}", s.name, s.kind);
        }
    }

    if segments {
        let live: u64 = breakdown.iter().map(|(_, _, b)| b).sum();
        println!("\n  where the bytes are (live segments only):");
        println!(
            "    {:<22} {:>6}  {:>12}  {:>7}",
            "segment", "count", "bytes", "share"
        );
        for (kind, count, bytes) in &breakdown {
            let pct = if live == 0 {
                0.0
            } else {
                *bytes as f64 * 100.0 / live as f64
            };
            println!(
                "    {:<22} {count:>6}  {:>12}  {pct:>6.1}%",
                format!("{kind:?}"),
                human_bytes(*bytes)
            );
        }
        println!(
            "    {:<22} {:>6}  {:>12}",
            "TOTAL (live)",
            "",
            human_bytes(live)
        );
        // The gap is tombstoned segments plus the header and TOC footer.
        let overhead = stats.file_bytes.saturating_sub(live);
        println!(
            "    {:<22} {:>6}  {:>12}   header, footer, and any reclaimable gap",
            "file on disk",
            "",
            human_bytes(stats.file_bytes)
        );
        if overhead > 0 {
            println!(
                "    {:<22} {:>6}  {:>12}",
                "  of which not live",
                "",
                human_bytes(overhead)
            );
        }
    }
    Ok(())
}

fn search(
    path: &PathBuf,
    collection: &str,
    query: &str,
    field: &str,
    top_k: usize,
    json: bool,
) -> Result<()> {
    let store = open_existing(path)?;
    let hits = store.search(collection, Search::new().text(field, query).top_k(top_k))?;

    if json {
        let rows: Vec<_> = hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "key": key_string(&h.key),
                    "score": h.score,
                    "collection": h.collection,
                })
            })
            .collect();
        println!("{}", json_pretty(&rows)?);
        return Ok(());
    }

    if hits.is_empty() {
        println!("no matches");
        return Ok(());
    }
    for (i, h) in hits.iter().enumerate() {
        println!("{:>3}. {:<32} {:.4}", i + 1, key_string(&h.key), h.score);
    }
    Ok(())
}

fn get(path: &PathBuf, collection: &str, key: &str, json: bool) -> Result<()> {
    let store = open_existing(path)?;
    let Some(rec) = store.get(collection, key)? else {
        return Err(Error::Other(format!(
            "no record {key:?} in collection {collection:?}"
        )));
    };

    if json {
        println!("{}", json_pretty(&record_json(&rec))?);
        return Ok(());
    }

    println!("key         {}", key_string(&rec.key));
    println!("collection  {}", rec.collection);
    println!("created_at  {}", rec.created_at);
    if let Some(text) = &rec.text {
        println!("text        {text}");
    }
    for (name, v) in &rec.vectors {
        println!("vector      {name} (dim {})", v.len());
    }
    Ok(())
}

fn export(path: &PathBuf, only: Option<&str>) -> Result<()> {
    let store = open_existing(path)?;
    let reader = store.reader();

    let collections: Vec<CollectionInfo> = match only {
        Some(name) => store
            .collections()
            .into_iter()
            .filter(|c| c.name == name)
            .collect(),
        None => store.collections(),
    };
    if collections.is_empty() {
        return Err(Error::Other(match only {
            Some(name) => format!("no such collection: {name}"),
            None => "capsule has no collections".to_string(),
        }));
    }

    // Batch the fetch so each chunk takes the schema/identity locks once
    // rather than per record.
    const CHUNK: usize = 512;
    for c in &collections {
        let keys = reader.keys(&c.name)?;
        for chunk in keys.chunks(CHUNK) {
            for rec in reader.get_many(&c.name, chunk)?.into_iter().flatten() {
                println!("{}", json_line(&record_json(&rec))?);
            }
        }
    }
    Ok(())
}

/// `Error::Serde` carries a message rather than a `serde_json::Error`, so
/// serialization failures are mapped here instead of via `?`.
fn json_pretty<T: serde::Serialize>(v: &T) -> Result<String> {
    serde_json::to_string_pretty(v).map_err(|e| Error::Serde(e.to_string()))
}

fn json_line<T: serde::Serialize>(v: &T) -> Result<String> {
    serde_json::to_string(v).map_err(|e| Error::Serde(e.to_string()))
}

fn record_json(rec: &Stored) -> serde_json::Value {
    serde_json::json!({
        "key": key_string(&rec.key),
        "collection": rec.collection,
        "created_at": rec.created_at,
        "text": rec.text,
        "vectors": rec.vectors.iter()
            .map(|(n, v)| serde_json::json!({ "field": n, "dim": v.len() }))
            .collect::<Vec<_>>(),
    })
}

fn key_string(key: &Key) -> String {
    match key {
        Key::Str(s) => s.clone(),
        Key::U64(n) => n.to_string(),
        Key::Bytes(b) => hex::encode(b),
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}
