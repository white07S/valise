// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
