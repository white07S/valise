//! Payload references and payload segment helpers, spec §11.

use crate::format::SegmentId;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PayloadRef {
    pub segment_id: SegmentId,
    pub bytes_offset: u64,
    pub bytes_length: u64,
}
