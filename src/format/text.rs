// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Text retrieval segment records, spec §12.

use crate::format::{CollectionId, FrameId, SegmentId, TermId, TextSpaceId};

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TermDictionaryEntry {
    pub term_id: TermId,
    pub term_bytes: Vec<u8>,
    pub collection_freq: u64,
    pub doc_freq: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PostingList {
    pub term_id: TermId,
    pub df: u32,
    pub postings: Vec<Posting>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Posting {
    pub frame_id: FrameId,
    pub collection_id: CollectionId,
    pub field_mask: u32,
    pub term_freq: u32,
    pub positions_ref: Option<PositionsRef>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PositionsRef {
    pub segment_id: SegmentId,
    pub ordinal: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct DocStatsEntry {
    pub frame_id: FrameId,
    pub collection_id: CollectionId,
    pub text_space_id: TextSpaceId,
    pub field_lengths: Vec<u32>,
    pub total_terms: u32,
    pub unique_terms: u32,
}
