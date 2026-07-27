//! Record identity, values, and the application-facing record builder.
//!
//! A [`Record`] is what an application hands to [`crate::db::Writer::put`]:
//! a set of named field values (text / vector), an optional stable [`Key`],
//! an optional timestamp, and an optional parent (for the doc→chunk frame
//! tree). The DB layer lowers a `Record` onto the engine's
//! `put_frame` / `index_frame_text` / `put_vector` primitives (see
//! `writer.rs`).
//!
//! Tags / metadata filtering are a P2 concern (DB-layer plan §5.7, hook
//! #7) and are intentionally absent here — accepting tags we cannot yet
//! persist would be lossy.

/// Application-supplied stable identity for a record.
///
/// The key is persisted in-file via `FrameDesc.uri_ref` (DB-layer hook
/// #1), so `put`/`delete`/`get` by key survive a process restart with no
/// sidecar. Each variant encodes to a self-describing byte string via
/// [`Key::encode`] so that two variants can never collide in the identity
/// index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Str(String),
    U64(u64),
    Bytes(Vec<u8>),
}

impl Key {
    const TAG_STR: u8 = 1;
    const TAG_U64: u8 = 2;
    const TAG_BYTES: u8 = 3;

    /// Encode to the self-describing byte string persisted as the frame's
    /// `uri`. The leading tag byte disambiguates the variant on decode.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Key::Str(s) => {
                let mut out = Vec::with_capacity(1 + s.len());
                out.push(Self::TAG_STR);
                out.extend_from_slice(s.as_bytes());
                out
            }
            Key::U64(n) => {
                let mut out = Vec::with_capacity(1 + 8);
                out.push(Self::TAG_U64);
                out.extend_from_slice(&n.to_le_bytes());
                out
            }
            Key::Bytes(b) => {
                let mut out = Vec::with_capacity(1 + b.len());
                out.push(Self::TAG_BYTES);
                out.extend_from_slice(b);
                out
            }
        }
    }

    /// Decode a key previously produced by [`Key::encode`]. Returns
    /// `None` for an empty or unrecognized-tag byte string (a frame whose
    /// `uri` was not written by this layer).
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Key> {
        let (&tag, rest) = bytes.split_first()?;
        match tag {
            Self::TAG_STR => String::from_utf8(rest.to_vec()).ok().map(Key::Str),
            Self::TAG_U64 => {
                let arr: [u8; 8] = rest.try_into().ok()?;
                Some(Key::U64(u64::from_le_bytes(arr)))
            }
            Self::TAG_BYTES => Some(Key::Bytes(rest.to_vec())),
            _ => None,
        }
    }
}

impl From<&str> for Key {
    fn from(s: &str) -> Self {
        Key::Str(s.to_owned())
    }
}
impl From<String> for Key {
    fn from(s: String) -> Self {
        Key::Str(s)
    }
}
impl From<u64> for Key {
    fn from(n: u64) -> Self {
        Key::U64(n)
    }
}
/// Lets `impl Into<Key>` parameters accept an already-held `&Key` (clones).
impl From<&Key> for Key {
    fn from(k: &Key) -> Self {
        k.clone()
    }
}

/// A single field value inside a [`Record`]. Borrows from the caller for
/// the duration of the `put` — text and vectors are copied into the
/// engine's own buffers synchronously.
#[derive(Clone, Debug)]
pub enum Value<'a> {
    Text(&'a str),
    Vector(&'a [f32]),
}

/// Application-facing record. Built fluently, then handed to
/// [`crate::db::Writer::put`].
///
/// A record lowers to exactly one engine *frame* (the owner) plus one
/// `put_vector` per vector field. The frame's payload is the single text
/// field's bytes (or empty for a vector-only record).
#[derive(Clone, Debug, Default)]
pub struct Record<'a> {
    /// Field names are borrowed (usually `'static` literals) to avoid a
    /// per-field allocation on the `put` hot path.
    pub(crate) fields: Vec<(&'a str, Value<'a>)>,
    pub(crate) created_at: Option<i64>,
    pub(crate) parent: Option<Key>,
}

impl<'a> Record<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            created_at: None,
            parent: None,
        }
    }

    /// Bind text to a named field. The field must be a `Text` field of
    /// the target collection's shape.
    #[must_use]
    pub fn text(mut self, field: &'a str, s: &'a str) -> Self {
        self.fields.push((field, Value::Text(s)));
        self
    }

    /// Bind a vector to a named field. The field must be a `Vector` field
    /// of the target collection's shape and match the bound space's dim.
    #[must_use]
    pub fn vector(mut self, field: &'a str, v: &'a [f32]) -> Self {
        self.fields.push((field, Value::Vector(v)));
        self
    }

    /// Set the record's `created_at` (unix seconds). Defaults to the
    /// engine's ingest-time clock when unset.
    #[must_use]
    pub fn at(mut self, unix_secs: i64) -> Self {
        self.created_at = Some(unix_secs);
        self
    }

    /// Mark this record as a child of `parent` (same collection), building
    /// the engine's doc→chunk frame tree. The parent must already exist
    /// (committed or earlier in the same batch).
    #[must_use]
    pub fn child_of(mut self, parent: Key) -> Self {
        self.parent = Some(parent);
        self
    }

    /// The record's `created_at` (unix seconds), if set. Exposed so a
    /// [`crate::db::Partition::Custom`] router can route on the record's time.
    #[must_use]
    pub fn created_at(&self) -> Option<i64> {
        self.created_at
    }

    /// The single text field's value, if any.
    pub(crate) fn text_value(&self) -> Option<&'a str> {
        self.fields.iter().find_map(|(_, v)| match v {
            Value::Text(s) => Some(*s),
            Value::Vector(_) => None,
        })
    }

    /// Iterate `(field_name, vector)` pairs in declared order.
    pub(crate) fn vector_values(&self) -> impl Iterator<Item = (&'a str, &'a [f32])> {
        self.fields.iter().filter_map(|(name, v)| match v {
            Value::Vector(values) => Some((*name, *values)),
            Value::Text(_) => None,
        })
    }
}

/// A materialized record read back via [`crate::db::Store::get`].
#[derive(Clone, Debug)]
pub struct Stored {
    pub key: Key,
    pub collection: String,
    pub created_at: i64,
    /// The text field's content, if the shape has a (non-empty) text field.
    pub text: Option<String>,
    /// Reconstructed vectors, one per populated vector field, in shape order.
    pub vectors: Vec<(String, Vec<f32>)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_roundtrip_all_variants() {
        for k in [
            Key::Str("hello/world".into()),
            Key::Str(String::new()),
            Key::U64(0),
            Key::U64(u64::MAX),
            Key::Bytes(vec![]),
            Key::Bytes(vec![0, 1, 2, 255]),
        ] {
            let encoded = k.encode();
            assert_eq!(Key::decode(&encoded), Some(k));
        }
    }

    #[test]
    fn key_variants_never_collide() {
        // A u64 whose LE bytes spell ASCII must not decode as the Str form.
        let u = Key::U64(0x6f6c6c6568).encode();
        let s = Key::Str("hello".into()).encode();
        assert_ne!(u, s);
    }

    #[test]
    fn key_decode_rejects_garbage() {
        assert_eq!(Key::decode(&[]), None);
        assert_eq!(Key::decode(&[99, 1, 2]), None);
        // U64 with wrong length.
        assert_eq!(Key::decode(&[Key::U64(0).encode()[0], 1, 2, 3]), None);
    }

    #[test]
    fn record_builder_collects_fields() {
        let v = [1.0f32, 2.0];
        let r = Record::new()
            .text("body", "hi")
            .vector("dense", &v)
            .at(42)
            .child_of(Key::U64(7));
        assert_eq!(r.text_value(), Some("hi"));
        assert_eq!(r.created_at, Some(42));
        assert_eq!(r.parent, Some(Key::U64(7)));
        let vecs: Vec<_> = r.vector_values().collect();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].0, "dense");
    }
}
