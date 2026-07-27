// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Result, error::Error};

pub(super) fn current_unix_timestamp() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| Error::Other(format!("system clock before Unix epoch: {err}")))?;
    i64::try_from(duration.as_secs()).map_err(|_| Error::Other("timestamp overflow".into()))
}
