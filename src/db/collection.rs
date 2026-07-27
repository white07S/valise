// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The resolved collection shape (named fields bound to spaces) and the
//! collection handle.
//!
//! [`Shape`] is the *resolved*, internal form of a declared
//! [`Schema`](crate::db::Schema): every field is bound to a concrete
//! file-global [`Space`] (a private `~auto/...` one for inline specs, or a
//! shared one). Text-only, vector-only, hybrid, multi-vector, and multimodal
//! collections are all just different field sets (DB-layer plan §4). A
//! [`Collection`] binds a shape to a physical engine collection id.

use crate::db::space::Space;
use crate::error::{Error, Result};
use crate::format::CollectionId;

/// One named field of a [`Shape`], bound to a file-global space.
#[derive(Clone, Debug)]
pub(crate) enum Field {
    Text(Space),
    Vector(Space),
}

/// The resolved set of named fields a collection's records may fill. Built
/// by [`crate::db::space::SchemaRegistry::bind_schema`].
#[derive(Clone, Debug, Default)]
pub(crate) struct Shape {
    pub(crate) fields: Vec<(String, Field)>,
}

impl Shape {
    /// The single text field's `(name, space)`, if any.
    pub(crate) fn text_field(&self) -> Option<(&str, &Space)> {
        self.fields.iter().find_map(|(name, f)| match f {
            Field::Text(s) => Some((name.as_str(), s)),
            Field::Vector(_) => None,
        })
    }

    /// Iterate `(name, space)` for every vector field, in declared order.
    pub(crate) fn vector_fields(&self) -> impl Iterator<Item = (&str, &Space)> {
        self.fields.iter().filter_map(|(name, f)| match f {
            Field::Vector(s) => Some((name.as_str(), s)),
            Field::Text(_) => None,
        })
    }

    /// Resolve a field by name, returning its `Field` (kind + space).
    pub(crate) fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, f)| f)
    }

    /// Enforce the v1 shape constraints:
    /// - field names are unique,
    /// - text fields are bound to text spaces and vector fields to vector
    ///   spaces,
    /// - at most one text field (engine indexes one text space per frame),
    /// - no two vector fields share the same vector space (so a stored
    ///   vector maps back to its field unambiguously at read time).
    ///
    /// Declared schemas are also pre-validated before any auto space is
    /// defined (`validate_schema` in `space.rs`); this is the post-binding
    /// safety net.
    pub(crate) fn validate(&self) -> Result<()> {
        let mut text_fields = 0usize;
        let mut seen_names: Vec<&str> = Vec::with_capacity(self.fields.len());
        let mut seen_vec_spaces: Vec<&str> = Vec::new();
        for (name, field) in &self.fields {
            if seen_names.contains(&name.as_str()) {
                return Err(Error::Format(format!(
                    "schema: duplicate field name '{name}'"
                )));
            }
            seen_names.push(name);
            match field {
                Field::Text(space) => {
                    if !space.is_text() {
                        return Err(Error::Format(format!(
                            "schema: field '{name}' is declared text but bound to vector space '{}'",
                            space.name()
                        )));
                    }
                    text_fields += 1;
                }
                Field::Vector(space) => {
                    if !space.is_vector() {
                        return Err(Error::Format(format!(
                            "schema: field '{name}' is declared vector but bound to text space '{}'",
                            space.name()
                        )));
                    }
                    if seen_vec_spaces.contains(&space.name()) {
                        return Err(Error::Format(format!(
                            "schema: vector fields must use distinct spaces; '{}' is reused",
                            space.name()
                        )));
                    }
                    seen_vec_spaces.push(space.name());
                }
            }
        }
        if text_fields > 1 {
            return Err(Error::Unsupported(
                "schema: at most one text field per collection in v1 (use the doc→chunk pattern for multiple text fields)".into(),
            ));
        }
        Ok(())
    }
}

/// A handle to a physical collection: its engine id and name. The bound
/// shape lives in the store's schema registry (keyed by name), so the handle
/// stays cheap and never goes stale across re-declares.
#[derive(Clone, Debug)]
pub struct Collection {
    pub(crate) id: CollectionId,
    pub(crate) name: String,
}

impl Collection {
    #[must_use]
    pub fn id(&self) -> CollectionId {
        self.id
    }
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Lightweight collection summary returned by [`crate::db::Store::collections`].
#[derive(Clone, Debug)]
pub struct CollectionInfo {
    pub name: String,
    pub id: CollectionId,
}
