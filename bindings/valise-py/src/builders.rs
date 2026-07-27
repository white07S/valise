// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Builder pyclasses: [`PyRecord`], [`PySearch`]. (The schema builder lives
//! in `crate::schema` next to its lowering helpers.)

mod record;
mod search;

pub(crate) use record::PyRecord;
pub(crate) use search::PySearch;
