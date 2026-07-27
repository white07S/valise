//! The read handle: `get` / `get_many` / `search`.
//!
//! `Reader` is a cheap `Clone + Send` handle. It is **uniformly live** —
//! every read reflects the last committed state (last-committed-wins). A single
//! `search` holds one `db.inner` read guard for its whole duration, so all of
//! its channels, the collection-membership filter, the time filter, and the
//! key resolution observe one consistent committed snapshot (a writer cannot
//! commit mid-search). Fusion is computed layer-side and writes nothing.

use std::collections::HashSet;
use std::sync::Arc;

use arc_swap::ArcSwap;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::db::Published;
use crate::db::collection::{Field, Shape};
use crate::db::partition::View;
use crate::db::query::{Fusion, Hit, Recency, Rerank, Search, SearchResult};
use crate::db::record::{Key, Stored};
use crate::db::space::{SchemaRegistry, VecState};
use crate::error::{Error, Result};
use crate::file::{
    QueryAlgorithm, ReadVectorResult, Reconstruct, TextQuery, TimeQuery, ValiseFile,
    VectorSearchQuery,
};
use crate::format::{CollectionId, EmbeddingSpaceId, FrameId, TextSpaceId};

/// Cheap, cloneable read handle. Reloads the published `{db, schema, identity}`
/// triple once per operation, so a compaction swap is observed atomically (no
/// torn db/identity) and the handle is always last-committed-wins.
#[derive(Clone)]
pub struct Reader {
    published: Arc<ArcSwap<Published>>,
}

impl Reader {
    pub(crate) fn new(published: Arc<ArcSwap<Published>>) -> Self {
        Self { published }
    }

    /// Fetch a record by key, materializing its text and vectors per the
    /// collection's schema. Returns `None` if the key is unknown.
    pub fn get(&self, coll: &str, key: impl Into<Key>) -> Result<Option<Stored>> {
        let key: Key = key.into();
        let p = self.published.load_full();
        let resolved = {
            let schema = p.schema.lock();
            resolve_fields(&schema, coll)?
        };
        let entry = {
            let identity = p.identity.lock();
            identity.get(resolved.collection_id, &key.encode()).cloned()
        };
        let Some(entry) = entry else {
            return Ok(None);
        };
        let valise = p.db.inner.read();
        Ok(Some(materialize(&valise, key, coll, &entry, &resolved)?))
    }

    /// Every committed key in a collection, in unspecified order.
    ///
    /// The building block for a full scan: pair it with [`Reader::get_many`]
    /// to walk a capsule end to end (that is what `valise export` does).
    /// Uncommitted records are excluded. Cost is O(records in `coll`), and
    /// the returned `Vec` holds one owned `Key` per record — chunk it for
    /// very large collections.
    pub fn keys(&self, coll: &str) -> Result<Vec<Key>> {
        let p = self.published.load_full();
        let resolved = {
            let schema = p.schema.lock();
            resolve_fields(&schema, coll)?
        };
        let encoded = {
            let identity = p.identity.lock();
            identity.committed_keys(resolved.collection_id)
        };
        Ok(encoded.iter().filter_map(|b| Key::decode(b)).collect())
    }

    /// Batched [`Reader::get`]. Resolves the shape and takes each lock once, so
    /// it avoids the N+1 a per-key loop would incur. Result order matches
    /// `keys`; a missing key yields `None`.
    pub fn get_many(&self, coll: &str, keys: &[Key]) -> Result<Vec<Option<Stored>>> {
        let p = self.published.load_full();
        let resolved = {
            let schema = p.schema.lock();
            resolve_fields(&schema, coll)?
        };
        let entries: Vec<Option<crate::db::identity::Entry>> = {
            let identity = p.identity.lock();
            keys.iter()
                .map(|k| identity.get(resolved.collection_id, &k.encode()).cloned())
                .collect()
        };
        let valise = p.db.inner.read();
        let mut out = Vec::with_capacity(keys.len());
        for (key, entry) in keys.iter().zip(entries) {
            match entry {
                Some(e) => out.push(Some(materialize(
                    &valise,
                    key.clone(),
                    coll,
                    &e,
                    &resolved,
                )?)),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    /// Run a multi-channel search over a single collection.
    pub fn search(&self, coll: &str, q: Search) -> Result<SearchResult> {
        if q.top_k == 0 {
            return Ok(SearchResult::new(Vec::new()));
        }
        let p = self.published.load_full();
        let (scope, plan) = {
            let schema = p.schema.lock();
            let rec = schema
                .collections
                .get(coll)
                .ok_or_else(|| Error::Format(format!("search: unknown collection '{coll}'")))?;
            let shape = rec.shape.as_ref().ok_or_else(|| {
                Error::Format(format!(
                    "search: collection '{coll}' has no schema in this process; call collection(name, schema)"
                ))
            })?;
            let plan = resolve_plan(&schema, shape, coll, &q)?;
            (vec![(rec.collection_id, coll.to_owned())], plan)
        };
        Ok(SearchResult::new(self.execute(&p, &scope, plan, &q)?))
    }

    /// Search across a partitioned [`View`] as ONE multi-collection query — a
    /// global `query_text` (filtered to the view's collections) plus a
    /// `vector_search` scoped to the view's collection set, then ONE global
    /// fusion. This keeps RRF ranks globally comparable (no per-partition
    /// fan-out, so small partitions can't dominate) and is atomic under one
    /// read guard.
    pub fn search_view(&self, view: &View, q: Search) -> Result<SearchResult> {
        if q.top_k == 0 {
            return Ok(SearchResult::new(Vec::new()));
        }
        let p = self.published.load_full();
        let (scope, plan) = {
            let schema = p.schema.lock();
            let mut scope = Vec::with_capacity(view.collections.len());
            for name in &view.collections {
                // Skip view members that don't exist as engine collections.
                if let Some(rec) = schema.collections.get(name) {
                    scope.push((rec.collection_id, name.clone()));
                }
            }
            let plan = resolve_plan(&schema, &view.shape, "view", &q)?;
            (scope, plan)
        };
        if scope.is_empty() {
            return Ok(SearchResult::new(Vec::new()));
        }
        Ok(SearchResult::new(self.execute(&p, &scope, plan, &q)?))
    }

    /// Shared executor: build each channel scoped to `scope`'s collections,
    /// fuse globally, apply recency, resolve the top-k keys. Holds one read
    /// guard for the whole search.
    fn execute(
        &self,
        p: &Published,
        scope: &[(CollectionId, String)],
        plan: ResolvedSearch,
        q: &Search,
    ) -> Result<Vec<Hit>> {
        let valise = p.db.inner.read();
        // Text search is global (no engine-side collection filter), so when the
        // search is scoped to a strict subset of collections we post-filter the
        // candidate set — most global candidates fall outside a narrow view and
        // get dropped. Over-fetch much wider (capped) so the post-scope-filter
        // set still clears top_k. (Vector pre-filters via `collection_filter`,
        // so it needs no scope-aware widening.) Residual limitation: a view
        // that is a tiny fraction of a very large corpus can still under-fill
        // at the cap — the real fix is a push-down collection filter in
        // `query_text` (engine follow-up). Layer-internal `~` collections
        // (the schema-doc store) hold no searchable content and don't count.
        let user_collections = valise
            .collections()
            .iter()
            .filter(|c| !c.name.starts_with('~'))
            .count();
        let scoped = scope.len() < user_collections;
        let over_fetch_text = if scoped {
            (q.top_k * 64).clamp(1_000, 50_000)
        } else {
            (q.top_k * 4).max(100)
        };
        let over_fetch_vector = if matches!(q.recency, Some(Recency::Range { .. })) {
            (q.top_k * 8).max(100)
        } else {
            (q.top_k * 2).max(50)
        };

        // frame → owning collection name (borrowed from `scope`), for
        // `Hit.collection`. Owned only at the final ≤top_k push.
        let mut frame_coll: FxHashMap<FrameId, &str> = FxHashMap::default();
        let mut channels: Vec<Channel> = Vec::new();

        if let Some(t) = plan.text {
            let hits = valise.query_text(TextQuery {
                text_space_id: t.space_id,
                query: t.query,
                algorithm: t.algorithm,
                top_k: Some(over_fetch_text),
                channel_k: Some(over_fetch_text),
            })?;
            // Text is global; keep only frames in a scope collection (binary
            // search the sorted per-collection member slices), recording which.
            let members: Vec<(&str, &[FrameId])> = scope
                .iter()
                .map(|(cid, name)| (name.as_str(), valise.collection_member_frames(*cid)))
                .collect();
            let mut ranked = Vec::with_capacity(hits.len());
            let mut scores = FxHashMap::default();
            for h in hits {
                if let Some((name, _)) = members
                    .iter()
                    .find(|(_, slice)| slice.binary_search(&h.frame_id).is_ok())
                {
                    frame_coll.entry(h.frame_id).or_insert(*name);
                    ranked.push(h.frame_id);
                    scores.insert(h.frame_id, h.score);
                }
            }
            channels.push(Channel {
                kind: ChannelKind::Text,
                ranked,
                scores,
            });
        }

        if !plan.vectors.is_empty() {
            let filter: HashSet<CollectionId> = scope.iter().map(|(c, _)| *c).collect();
            let cid_name: FxHashMap<CollectionId, &str> =
                scope.iter().map(|(c, n)| (*c, n.as_str())).collect();
            for v in plan.vectors {
                let mut vsq = match v.rerank {
                    Rerank::Accurate => {
                        VectorSearchQuery::accurate(v.space_id, v.query, over_fetch_vector)
                    }
                    Rerank::Fast => VectorSearchQuery::fast(v.space_id, v.query, over_fetch_vector),
                };
                vsq.collection_filter = Some(filter.clone());
                let hits = valise.vector_search(vsq)?;
                let mut ranked = Vec::with_capacity(hits.len());
                let mut scores = FxHashMap::default();
                for h in hits {
                    if let Some(name) = cid_name.get(&h.collection_id) {
                        frame_coll.entry(h.frame_id).or_insert(*name);
                    }
                    ranked.push(h.frame_id);
                    // distance (smaller = better) → similarity (higher = better)
                    scores.insert(h.frame_id, -h.score);
                }
                channels.push(Channel {
                    kind: ChannelKind::Vector,
                    ranked,
                    scores,
                });
            }
        }

        // Hard Range filter: prune candidates outside the window (union the
        // per-collection time ranges across the scope).
        if let Some(Recency::Range { from, to }) = q.recency {
            let mut allowed: FxHashSet<FrameId> = FxHashSet::default();
            for (cid, _) in scope {
                allowed.extend(valise.time_range_query(TimeQuery {
                    from,
                    to,
                    collection_id: Some(*cid),
                }));
            }
            for ch in &mut channels {
                ch.ranked.retain(|f| allowed.contains(f));
                ch.scores.retain(|f, _| allowed.contains(f));
            }
        }

        // Fuse globally.
        let mut fused: FxHashMap<FrameId, f32> = FxHashMap::default();
        match q.fusion {
            Fusion::Rrf { k } => {
                let k = k as f32;
                for ch in &channels {
                    for (pos, f) in ch.ranked.iter().enumerate() {
                        // Canonical RRF: SUM reciprocal ranks (missing = 0).
                        *fused.entry(*f).or_insert(0.0) += 1.0 / (k + pos as f32 + 1.0);
                    }
                }
                if let Some(Recency::RrfChannel { weight, .. }) = q.recency {
                    let universe: Vec<FrameId> = fused.keys().copied().collect();
                    let newest_first = sort_by_created_desc(&valise, &universe)?;
                    for (pos, f) in newest_first.into_iter().enumerate() {
                        *fused.entry(f).or_insert(0.0) += weight / (k + pos as f32 + 1.0);
                    }
                }
            }
            Fusion::Weighted { text, vector } => {
                if matches!(q.recency, Some(Recency::RrfChannel { .. })) {
                    return Err(Error::Unsupported(
                        "Recency::RrfChannel requires Fusion::Rrf".into(),
                    ));
                }
                for ch in &channels {
                    let w = match ch.kind {
                        ChannelKind::Text => text,
                        ChannelKind::Vector => vector,
                    };
                    // Min-max normalize this channel inline (no intermediate map).
                    let (lo, hi) = ch
                        .scores
                        .values()
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                            (lo.min(v), hi.max(v))
                        });
                    let range = hi - lo;
                    for (f, &v) in &ch.scores {
                        // Degenerate (single/tied channel): neutral 0.5, not
                        // 1.0 — 1.0 would over-reward a tied channel.
                        let n = if range > 0.0 { (v - lo) / range } else { 0.5 };
                        *fused.entry(*f).or_insert(0.0) += w * n;
                    }
                }
            }
        }

        // Soft HalfLife multiplier.
        if let Some(Recency::HalfLife { days }) = q.recency {
            let now = q.now_override.unwrap_or_else(now_secs);
            let hl = (days * 86_400.0).max(1.0);
            for (f, score) in fused.iter_mut() {
                let ca = valise.frame_created_at(*f).unwrap_or(now);
                let age = (now - ca).max(0) as f64;
                *score *= 0.5f64.powf(age / hl) as f32;
            }
        }

        // Rank and resolve the top-k keys (dropping frames this layer can't
        // map back to a key — e.g. foreign uris).
        let mut ranked: Vec<(FrameId, f32)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut hits = Vec::with_capacity(q.top_k.min(ranked.len()));
        for (frame_id, score) in ranked {
            if hits.len() >= q.top_k {
                break;
            }
            if let Some(uri) = valise.read_frame_uri(frame_id)?
                && let Some(key) = Key::decode(&uri)
            {
                let collection = frame_coll
                    .get(&frame_id)
                    .copied()
                    .or_else(|| scope.first().map(|(_, n)| n.as_str()))
                    .unwrap_or_default()
                    .to_owned();
                hits.push(Hit {
                    key,
                    score,
                    collection,
                });
            }
        }
        Ok(hits)
    }
}

#[derive(Clone, Copy)]
enum ChannelKind {
    Text,
    Vector,
}

struct Channel {
    kind: ChannelKind,
    /// Frame ids best-first.
    ranked: Vec<FrameId>,
    /// Higher = better, aligned with `ranked` (used by weighted fusion).
    scores: FxHashMap<FrameId, f32>,
}

struct ResolvedFields {
    collection_id: CollectionId,
    has_text: bool,
    /// `(embedding_space_id, field_name)` for each vector field of the shape.
    vec_fields: Vec<(EmbeddingSpaceId, String)>,
}

struct ResolvedText {
    space_id: TextSpaceId,
    query: String,
    algorithm: QueryAlgorithm,
}

struct ResolvedVector {
    space_id: EmbeddingSpaceId,
    query: Vec<f32>,
    rerank: Rerank,
}

struct ResolvedSearch {
    text: Option<ResolvedText>,
    vectors: Vec<ResolvedVector>,
}

fn resolve_fields(schema: &SchemaRegistry, coll: &str) -> Result<ResolvedFields> {
    let rec = schema
        .collections
        .get(coll)
        .ok_or_else(|| Error::Format(format!("unknown collection '{coll}'")))?;
    let shape = rec.shape.as_ref().ok_or_else(|| {
        Error::Format(format!(
            "collection '{coll}' has no schema in this process; call collection(name, schema)"
        ))
    })?;
    let mut vec_fields = Vec::new();
    for (fname, space) in shape.vector_fields() {
        if let Some(sid) = schema.vector_space_id(space.name()) {
            vec_fields.push((sid, fname.to_owned()));
        }
    }
    Ok(ResolvedFields {
        collection_id: rec.collection_id,
        has_text: shape.text_field().is_some(),
        vec_fields,
    })
}

/// Validate the search against a shape and resolve channels to engine ids.
/// `label` names the target (a collection or `"view"`) for error messages.
fn resolve_plan(
    schema: &SchemaRegistry,
    shape: &Shape,
    label: &str,
    q: &Search,
) -> Result<ResolvedSearch> {
    if q.text.is_none() && q.vectors.is_empty() {
        return Err(Error::Format(
            "search: no channels (set text and/or vector)".into(),
        ));
    }

    let text = match &q.text {
        Some(t) => match shape.field(&t.field) {
            Some(Field::Text(space)) => Some(ResolvedText {
                space_id: schema.text_space_id(space.name()).ok_or_else(|| {
                    Error::Format(format!(
                        "search: text space '{}' not registered",
                        space.name()
                    ))
                })?,
                query: t.query.clone(),
                algorithm: t.scorer.to_algorithm(),
            }),
            Some(Field::Vector(_)) => {
                return Err(Error::Format(format!(
                    "search: field '{}' is a vector field, not text",
                    t.field
                )));
            }
            None => {
                return Err(Error::Format(format!(
                    "search: unknown field '{}' in '{label}'",
                    t.field
                )));
            }
        },
        None => None,
    };

    let mut vectors = Vec::with_capacity(q.vectors.len());
    for v in &q.vectors {
        match shape.field(&v.field) {
            Some(Field::Vector(space)) => {
                let entry = schema.vector_spaces.get(space.name()).ok_or_else(|| {
                    Error::Format(format!(
                        "search: vector space '{}' not registered",
                        space.name()
                    ))
                })?;
                let space_id = match entry.state {
                    VecState::Ready { space_id } => space_id,
                    VecState::Deferred { .. } => {
                        return Err(Error::NotCalibrated {
                            space: space.name().to_owned(),
                        });
                    }
                };
                if entry.dim as usize != v.query.len() {
                    return Err(Error::Format(format!(
                        "search: field '{}' expects dim {}, got {}",
                        v.field,
                        entry.dim,
                        v.query.len()
                    )));
                }
                vectors.push(ResolvedVector {
                    space_id,
                    query: v.query.clone(),
                    rerank: v.rerank,
                });
            }
            Some(Field::Text(_)) => {
                return Err(Error::Format(format!(
                    "search: field '{}' is a text field, not vector",
                    v.field
                )));
            }
            None => {
                return Err(Error::Format(format!(
                    "search: unknown field '{}' in '{label}'",
                    v.field
                )));
            }
        }
    }

    Ok(ResolvedSearch { text, vectors })
}

fn materialize(
    valise: &ValiseFile,
    key: Key,
    coll: &str,
    entry: &crate::db::identity::Entry,
    resolved: &ResolvedFields,
) -> Result<Stored> {
    let frame = valise.frame_full(entry.frame_id)?;
    let text = if resolved.has_text {
        let bytes = valise.read_payload(entry.frame_id)?;
        if bytes.is_empty() {
            None
        } else {
            Some(String::from_utf8(bytes).map_err(|e| Error::Format(e.to_string()))?)
        }
    } else {
        None
    };
    let mut vectors = Vec::with_capacity(entry.vectors.len());
    for (space_id, vid) in &entry.vectors {
        if let Some((_, fname)) = resolved.vec_fields.iter().find(|(sid, _)| sid == space_id) {
            // We requested F32Vector, so a non-F32 result is an engine
            // invariant violation, not a caller error — surface it as Integrity
            // rather than panicking (`expect_f32`).
            let values = match valise.read_vector(*vid, Reconstruct::F32Vector)? {
                ReadVectorResult::F32Vector(v) => v,
                other => {
                    return Err(Error::Integrity(format!(
                        "get: expected F32Vector for {vid:?}, got {other:?}"
                    )));
                }
            };
            vectors.push((fname.clone(), values));
        }
    }
    Ok(Stored {
        key,
        collection: coll.to_owned(),
        created_at: frame.created_at,
        text,
        vectors,
    })
}

fn sort_by_created_desc(valise: &ValiseFile, frames: &[FrameId]) -> Result<Vec<FrameId>> {
    let mut paired: Vec<(FrameId, i64)> = frames
        .iter()
        .map(|&f| Ok((f, valise.frame_created_at(f)?)))
        .collect::<Result<_>>()?;
    paired.sort_by_key(|p| std::cmp::Reverse(p.1));
    Ok(paired.into_iter().map(|(f, _)| f).collect())
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
