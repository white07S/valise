// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("format error: {0}")]
    Format(String),

    #[error("serialization error: {0}")]
    Serde(String),

    #[error("integrity check failed: {0}")]
    Integrity(String),

    #[error("signature verification failed")]
    Signature,

    #[error("compression error: {0}")]
    Compression(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// A vector field/space was searched before its codec was calibrated.
    /// Calibration happens at the first commit that contains vectors for the
    /// space (`Calibrate::auto`) or eagerly at declaration (`Calibrate::now`).
    #[error(
        "vector space '{space}' is not calibrated yet — commit a batch containing vectors first"
    )]
    NotCalibrated { space: String },

    /// A collection was re-declared with a schema that diverges from the one
    /// already bound to it. `detail` carries the field-level difference.
    /// Identical re-declares are no-ops and additive re-declares (appending
    /// new fields) are accepted; anything else is this error.
    #[error("schema mismatch for collection '{collection}': {detail}")]
    SchemaMismatch { collection: String, detail: String },

    /// A bounded resource (reader slot, writer slot, advisory lock) is
    /// held by another party past the caller's deadline. Transient by
    /// definition — the caller may retry.
    #[error("resource busy: {0}")]
    Busy(String),

    #[error("other: {0}")]
    Other(String),
}
