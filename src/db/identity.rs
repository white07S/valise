//! Stable record identity: external [`Key`](crate::db::Key) → current internal
//! ids. This is what lets the layer synthesize `update` / `delete-by-key` over
//! an append-only engine (DB-layer plan §9.D-1).
//!
//! The map is persisted *indirectly*: each keyed frame stores its encoded key
//! in `FrameDesc.uri_ref` (hook #1), and [`IdentityIndex::rebuild`] reconstructs
//! the map at open by scanning active frames + vectors. No sidecar.

use rustc_hash::FxHashMap;

use crate::db::record::Key;
use crate::error::Result;
use crate::file::ValiseFile;
use crate::format::{CollectionId, EmbeddingSpaceId, FrameId, VectorId};

/// The internal ids a key currently resolves to.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) frame_id: FrameId,
    /// `(embedding_space_id, vector_id)` for each vector the record owns. The
    /// space id (not a field name) is stored because field names are not
    /// persisted; `get` maps space → field via the collection's shape.
    pub(crate) vectors: Vec<(EmbeddingSpaceId, VectorId)>,
    /// `false` while the record is buffered in an uncommitted batch. Vectors of
    /// uncommitted records cannot be tombstoned (engine rejects deleting an
    /// uncommitted vector), which gates same-batch re-puts/deletes.
    pub(crate) committed: bool,
}

/// Identity index, partitioned by collection so the per-key lookup keys on a
/// bare `&[u8]` (no owned-key allocation per `get`/`remove`/`push_vector`). Uses
/// `FxHashMap` (keys are opaque bytes; no DoS surface) for cheaper hashing than
/// SipHash on the hot path. The reverse `frame → key` map is rebuilt transiently
/// inside [`Self::rebuild`] only — it is never needed in the steady state, so it
/// is not maintained on `insert`/`remove`.
#[derive(Default)]
pub(crate) struct IdentityIndex {
    map: FxHashMap<CollectionId, FxHashMap<Vec<u8>, Entry>>,
    /// Keys inserted as uncommitted since the last commit. `mark_all_committed`
    /// drains this instead of scanning every entry — turning per-commit cost
    /// from O(total records) into O(batch size). Stale entries (a key removed
    /// before commit) are harmless: marking a missing key is a no-op.
    uncommitted: Vec<(CollectionId, Vec<u8>)>,
}

impl IdentityIndex {
    pub fn get(&self, collection: CollectionId, key_bytes: &[u8]) -> Option<&Entry> {
        self.map.get(&collection)?.get(key_bytes)
    }

    /// Every committed key in a collection, as encoded bytes. Unordered —
    /// the backing map has no meaningful iteration order. Uncommitted
    /// entries are skipped so a scan never surfaces records that a crash
    /// would erase.
    pub fn committed_keys(&self, collection: CollectionId) -> Vec<Vec<u8>> {
        self.map
            .get(&collection)
            .map(|inner| {
                inner
                    .iter()
                    .filter(|(_, e)| e.committed)
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Insert or replace the entry for `(collection, key)`.
    pub fn insert(&mut self, collection: CollectionId, key_bytes: Vec<u8>, entry: Entry) {
        if !entry.committed {
            self.uncommitted.push((collection, key_bytes.clone()));
        }
        self.map
            .entry(collection)
            .or_default()
            .insert(key_bytes, entry);
    }

    /// Remove the entry for `(collection, key)`, returning it if present.
    pub fn remove(&mut self, collection: CollectionId, key_bytes: &[u8]) -> Option<Entry> {
        self.map.get_mut(&collection)?.remove(key_bytes)
    }

    /// Drop every entry for a collection (whole-partition forget). Also prunes
    /// the uncommitted dirty set of that collection's keys so a later commit
    /// doesn't try to mark vanished entries.
    pub fn remove_collection(&mut self, collection: CollectionId) {
        self.map.remove(&collection);
        self.uncommitted.retain(|(c, _)| *c != collection);
    }

    /// Append a freshly flushed vector id to an existing entry (used when a
    /// deferred-calibration vector finally lands at commit).
    pub fn push_vector(
        &mut self,
        collection: CollectionId,
        key_bytes: &[u8],
        space_id: EmbeddingSpaceId,
        vector_id: VectorId,
    ) {
        if let Some(inner) = self.map.get_mut(&collection)
            && let Some(entry) = inner.get_mut(key_bytes)
        {
            entry.vectors.push((space_id, vector_id));
        }
    }

    /// Mark this batch's uncommitted entries committed. Called after a
    /// successful engine commit. O(batch size), not O(total records).
    pub fn mark_all_committed(&mut self) {
        for (collection, key_bytes) in self.uncommitted.drain(..) {
            if let Some(inner) = self.map.get_mut(&collection)
                && let Some(entry) = inner.get_mut(&key_bytes)
            {
                entry.committed = true;
            }
        }
    }

    /// Rebuild the index from a freshly opened file: every active frame that
    /// carries a layer-written key becomes an entry, and active vectors are
    /// attached to their owner frame's entry. All entries are committed.
    pub fn rebuild(&mut self, valise: &ValiseFile) -> Result<()> {
        self.map.clear();
        self.uncommitted.clear();

        // Transient frame → (collection, key) reverse map, used only to attach
        // vectors to their owner frame below.
        let mut by_frame: FxHashMap<FrameId, (CollectionId, Vec<u8>)> = FxHashMap::default();

        // Frames first. Collect frame ids up front so the immutable
        // `active_frames` borrow is released before the per-frame
        // `read_frame_uri` calls.
        let frame_ids: Vec<(FrameId, CollectionId)> = valise
            .active_frames()
            .map(|f| (f.frame_id, f.collection_id))
            .collect();
        for (frame_id, collection_id) in frame_ids {
            let Some(uri) = valise.read_frame_uri(frame_id)? else {
                continue;
            };
            if Key::decode(&uri).is_none() {
                // A frame with a uri this layer didn't write — ignore it.
                continue;
            }
            by_frame.insert(frame_id, (collection_id, uri.clone()));
            self.map.entry(collection_id).or_default().insert(
                uri,
                Entry {
                    frame_id,
                    vectors: Vec::new(),
                    committed: true,
                },
            );
        }

        // Vectors: attach each active vector to its owner frame's entry.
        let space_ids: Vec<EmbeddingSpaceId> = valise
            .embedding_spaces()
            .iter()
            .map(|e| e.embedding_space_id)
            .collect();
        for space_id in space_ids {
            let owned: Vec<(FrameId, VectorId)> = valise
                .active_vectors(space_id)
                .map(|v| (v.owner_frame_id, v.vector_id))
                .collect();
            for (owner_frame, vector_id) in owned {
                if let Some((coll, key)) = by_frame.get(&owner_frame)
                    && let Some(inner) = self.map.get_mut(coll)
                    && let Some(entry) = inner.get_mut(key)
                {
                    entry.vectors.push((space_id, vector_id));
                }
            }
        }
        Ok(())
    }
}
