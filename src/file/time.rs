use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Result, error::Error};

pub(super) fn current_unix_timestamp() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| Error::Other(format!("system clock before Unix epoch: {err}")))?;
    i64::try_from(duration.as_secs()).map_err(|_| Error::Other("timestamp overflow".into()))
}
