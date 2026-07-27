// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Payload references and payload segment helpers, spec §11.

use crate::format::SegmentId;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PayloadRef {
    pub segment_id: SegmentId,
    pub bytes_offset: u64,
    pub bytes_length: u64,
}
