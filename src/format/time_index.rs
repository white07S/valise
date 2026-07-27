//! Time index records, spec §17.
//!
//! Per-entry record only. The segment payload that aggregates these
//! (`TimeIndexSegment`) and its wire codec live in `time_index_segment.rs`.

use crate::format::{CollectionId, FrameId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TimeIndexEntry {
    pub collection_id: CollectionId,
    pub frame_id: FrameId,
    pub created_at: i64,
    pub updated_at: i64,
}
