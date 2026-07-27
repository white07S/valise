//! Footer TOC encoding, spec §19 and §19.1.

use crate::{
    error::{Error, Result},
    format::{
        Checksum, CollectionId, TextSpaceId, bincode_config, blake3_checksum,
        create_contract::CreateContractV1, registry::IdAllocator, segment::SegmentRef,
    },
};

pub const TOC_MAGIC: [u8; 4] = *b"VLTC";
pub const TOC_VERSION: u16 = 1;
pub const TOC_HEADER_LEN: usize = 88;

#[derive(Clone, Debug, PartialEq)]
pub struct TocFooter {
    pub body: TocFooterBody,
    pub checksum: Checksum,
}

impl TocFooter {
    /// Build an empty footer carrying the given file-level contract.
    /// Every Valise file MUST carry a contract — there is no legacy
    /// shape after the v1 consolidation.
    pub fn empty(snapshot_generation: u64, create_contract: CreateContractV1) -> Result<Self> {
        TocFooterCodec::encode_body(TocFooterBody {
            snapshot_generation,
            create_contract,
            ..TocFooterBody::default()
        })
    }
}

/// Footer body. Single canonical shape — `bincode-serde` round-trips
/// it directly. Field order is wire-stable; new fields append at the
/// end of the scalar block (before the variable-length `Vec` fields)
/// so future additions stay dense.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TocFooterBody {
    pub snapshot_generation: u64,
    pub id_allocator: IdAllocator,
    pub segment_catalog_root: Option<SegmentRef>,
    pub collection_catalog_root: Option<SegmentRef>,
    pub frame_catalog_root: Option<SegmentRef>,
    pub text_space_catalog_root: Option<SegmentRef>,
    pub analyzer_catalog_root: Option<SegmentRef>,
    pub field_schema_catalog_root: Option<SegmentRef>,
    pub retrieval_profile_catalog_root: Option<SegmentRef>,
    pub embedding_space_catalog_root: Option<SegmentRef>,
    pub codec_catalog_root: Option<SegmentRef>,
    pub vector_catalog_root: Option<SegmentRef>,
    pub fusion_profile_catalog_root: Option<SegmentRef>,
    pub text_index_roots: Vec<TextIndexRoot>,
    pub collection_filter_roots: Vec<CollectionFilterRoot>,
    pub time_index_roots: Vec<TimeIndexRoot>,
    /// File-level create-time contract (plan §4). Required. The 32-byte
    /// BLAKE3 of the encoded contract is stamped into the file header
    /// at byte 832 — `open()` recomputes the digest and rejects the
    /// file on mismatch.
    pub create_contract: CreateContractV1,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TextIndexRoot {
    pub text_space_id: TextSpaceId,
    pub term_dictionary_root: Option<SegmentRef>,
    pub postings_root: Option<SegmentRef>,
    pub positions_root: Option<SegmentRef>,
    pub doc_stats_root: Option<SegmentRef>,
    pub token_set_roots: Vec<SegmentRef>,
    pub minhash_signature_roots: Vec<SegmentRef>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CollectionFilterRoot {
    pub collection_id: CollectionId,
    pub frame_filter_root: Option<SegmentRef>,
    pub vector_filter_root: Option<SegmentRef>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TimeIndexRoot {
    pub collection_id: Option<CollectionId>,
    pub root: SegmentRef,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TocCatalogRoots {
    pub collection_catalog_root: Option<SegmentRef>,
    pub frame_catalog_root: Option<SegmentRef>,
    pub text_space_catalog_root: Option<SegmentRef>,
    pub analyzer_catalog_root: Option<SegmentRef>,
    pub field_schema_catalog_root: Option<SegmentRef>,
    pub retrieval_profile_catalog_root: Option<SegmentRef>,
    pub embedding_space_catalog_root: Option<SegmentRef>,
    pub codec_catalog_root: Option<SegmentRef>,
    pub vector_catalog_root: Option<SegmentRef>,
    pub fusion_profile_catalog_root: Option<SegmentRef>,
}

impl TocFooterBody {
    #[must_use]
    pub fn catalog_roots(&self) -> TocCatalogRoots {
        TocCatalogRoots {
            collection_catalog_root: self.collection_catalog_root,
            frame_catalog_root: self.frame_catalog_root,
            text_space_catalog_root: self.text_space_catalog_root,
            analyzer_catalog_root: self.analyzer_catalog_root,
            field_schema_catalog_root: self.field_schema_catalog_root,
            retrieval_profile_catalog_root: self.retrieval_profile_catalog_root,
            embedding_space_catalog_root: self.embedding_space_catalog_root,
            codec_catalog_root: self.codec_catalog_root,
            vector_catalog_root: self.vector_catalog_root,
            fusion_profile_catalog_root: self.fusion_profile_catalog_root,
        }
    }
}

pub struct TocFooterCodec;

impl TocFooterCodec {
    /// Encode a body into a fully self-validating `TocFooter`.
    ///
    /// The on-disk shape carries TWO embedded checksums (see `encode`):
    ///   - `body_checksum` at offset 24..56 — BLAKE3 over the body bytes.
    ///   - `footer_checksum` at offset 56..88 — BLAKE3 over the
    ///     header fields (magic | version | header_len | snapshot_gen |
    ///     body_len | body_checksum) AND the body bytes.
    ///
    /// Because both are present on disk, `decode` alone catches any
    /// single-byte corruption in EITHER the body or the header
    /// fields, without needing to consult `crate::format::header::Header::toc_checksum`.
    /// The header copy is a redundant belt-and-suspenders cross-check
    /// that will be removed in the v3 format.
    pub fn encode_body(body: TocFooterBody) -> Result<TocFooter> {
        let encoded_body = encode_body_bytes(&body)?;
        let checksum = footer_checksum(body.snapshot_generation, &encoded_body);
        Ok(TocFooter { body, checksum })
    }

    pub fn encode(footer: &TocFooter) -> Result<Vec<u8>> {
        let body = encode_body_bytes(&footer.body)?;
        let body_checksum = blake3_checksum(&body);
        let checksum = footer_checksum(footer.body.snapshot_generation, &body);
        if checksum != footer.checksum {
            return Err(Error::Integrity("TOC footer checksum is stale".into()));
        }

        let mut out = vec![0u8; TOC_HEADER_LEN + body.len()];
        out[0..4].copy_from_slice(&TOC_MAGIC);
        put_u16(&mut out, 4, TOC_VERSION);
        put_u16(&mut out, 6, TOC_HEADER_LEN as u16);
        put_u64(&mut out, 8, footer.body.snapshot_generation);
        put_u64(&mut out, 16, body.len() as u64);
        out[24..56].copy_from_slice(&body_checksum);
        out[56..88].copy_from_slice(&checksum);
        out[TOC_HEADER_LEN..].copy_from_slice(&body);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<(TocFooter, usize)> {
        if bytes.len() < TOC_HEADER_LEN {
            return Err(Error::Format("TOC footer truncated".into()));
        }
        if bytes[0..4] != TOC_MAGIC {
            return Err(Error::Format("TOC footer magic mismatch".into()));
        }
        let version = get_u16(bytes, 4)?;
        if version != TOC_VERSION {
            return Err(Error::Format(format!("unsupported TOC version: {version}")));
        }
        let header_len = get_u16(bytes, 6)? as usize;
        if header_len != TOC_HEADER_LEN {
            return Err(Error::Format(format!(
                "invalid TOC header length: expected {TOC_HEADER_LEN}, got {header_len}"
            )));
        }
        let snapshot_generation = get_u64(bytes, 8)?;
        let body_len = get_u64(bytes, 16)? as usize;
        let total_len = TOC_HEADER_LEN
            .checked_add(body_len)
            .ok_or_else(|| Error::Format("TOC footer length overflow".into()))?;
        if bytes.len() < total_len {
            return Err(Error::Format("TOC footer body truncated".into()));
        }

        let body = &bytes[TOC_HEADER_LEN..total_len];
        let expected_body_checksum: Checksum = bytes[24..56]
            .try_into()
            .map_err(|_| Error::Format("TOC body checksum truncated".into()))?;
        let actual_body_checksum = blake3_checksum(body);
        if expected_body_checksum != actual_body_checksum {
            return Err(Error::Integrity("TOC body checksum mismatch".into()));
        }

        let expected_footer_checksum: Checksum = bytes[56..88]
            .try_into()
            .map_err(|_| Error::Format("TOC footer checksum truncated".into()))?;
        let actual_footer_checksum = footer_checksum(snapshot_generation, body);
        if expected_footer_checksum != actual_footer_checksum {
            return Err(Error::Integrity("TOC footer checksum mismatch".into()));
        }

        let (body, consumed): (TocFooterBody, usize) =
            bincode::serde::decode_from_slice(body, bincode_config())
                .map_err(|err| Error::Serde(err.to_string()))?;
        if consumed != body_len {
            return Err(Error::Format("TOC body has trailing bytes".into()));
        }
        if body.snapshot_generation != snapshot_generation {
            return Err(Error::Integrity("TOC snapshot generation mismatch".into()));
        }

        Ok((
            TocFooter {
                body,
                checksum: expected_footer_checksum,
            },
            total_len,
        ))
    }

    pub fn encoded_len_from_header(bytes: &[u8; TOC_HEADER_LEN]) -> Result<usize> {
        if bytes[0..4] != TOC_MAGIC {
            return Err(Error::Format("TOC footer magic mismatch".into()));
        }
        let version = get_u16(bytes, 4)?;
        if version != TOC_VERSION {
            return Err(Error::Format(format!("unsupported TOC version: {version}")));
        }
        let header_len = get_u16(bytes, 6)? as usize;
        if header_len != TOC_HEADER_LEN {
            return Err(Error::Format(format!(
                "invalid TOC header length: expected {TOC_HEADER_LEN}, got {header_len}"
            )));
        }
        let body_len = get_u64(bytes, 16)? as usize;
        TOC_HEADER_LEN
            .checked_add(body_len)
            .ok_or_else(|| Error::Format("TOC footer length overflow".into()))
    }
}

fn encode_body_bytes(body: &TocFooterBody) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(body, bincode_config())
        .map_err(|err| Error::Serde(err.to_string()))
}

fn footer_checksum(snapshot_generation: u64, body: &[u8]) -> Checksum {
    let mut prefix = [0u8; 56];
    prefix[0..4].copy_from_slice(&TOC_MAGIC);
    put_u16(&mut prefix, 4, TOC_VERSION);
    put_u16(&mut prefix, 6, TOC_HEADER_LEN as u16);
    put_u64(&mut prefix, 8, snapshot_generation);
    put_u64(&mut prefix, 16, body.len() as u64);
    prefix[24..56].copy_from_slice(&blake3_checksum(body));

    let mut hasher = blake3::Hasher::new();
    hasher.update(&prefix);
    hasher.update(body);
    *hasher.finalize().as_bytes()
}

fn put_u16(buf: &mut [u8], offset: usize, value: u16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .map_err(|_| Error::Format("TOC u16 field truncated".into()))?,
    ))
}

fn get_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| Error::Format("TOC u64 field truncated".into()))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_footer(generation: u64) -> TocFooter {
        TocFooter::empty(generation, CreateContractV1::default()).expect("make footer")
    }

    #[test]
    fn toc_footer_round_trip() {
        let footer = sample_footer(7);
        let bytes = TocFooterCodec::encode(&footer).expect("encode footer");
        let (decoded, consumed) = TocFooterCodec::decode(&bytes).expect("decode footer");
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded, footer);
        assert_eq!(decoded.body.create_contract, CreateContractV1::default());
    }

    #[test]
    fn rejects_stale_checksum() {
        let mut footer = sample_footer(7);
        footer.body.snapshot_generation = 8;
        let err = TocFooterCodec::encode(&footer).expect_err("stale checksum should fail");
        assert!(err.to_string().contains("stale"));
    }

    #[test]
    fn rejects_corrupt_body() {
        let footer = sample_footer(7);
        let mut bytes = TocFooterCodec::encode(&footer).expect("encode footer");
        let last = bytes.last_mut().expect("footer has body");
        *last ^= 0xFF;
        let err = TocFooterCodec::decode(&bytes).expect_err("corrupt body should fail");
        assert!(err.to_string().contains("checksum"));
    }

    // The remainder of this module locks down the invariant that
    // `TocFooterCodec::decode` is fully self-validating — it catches
    // every single-byte corruption class in the on-disk footer
    // *without* consulting any header checksum. This self-validation is
    // what allowed `Header.toc_checksum` to be deleted in the v2 format
    // (it is now reserved-zero; see `format/header.rs`).
    //
    // For a flip at offset `i` we expect:
    //   - magic (0..4) / version (4..6) → typed format error
    //   - snapshot_generation, body_checksum, footer_checksum, body
    //     bytes → checksum error from inside `decode`
    //   - body_len → either checksum error (length valid but slice
    //     wrong) or truncation error (length past EOF)

    /// Flip a single byte at `offset` in the encoded footer and
    /// assert that decode rejects it. Returns the error so the
    /// caller can pattern-match on the specific message.
    fn decode_after_flip(offset: usize) -> crate::error::Error {
        let footer = sample_footer(7);
        let mut bytes = TocFooterCodec::encode(&footer).expect("encode footer");
        bytes[offset] ^= 0xFF;
        TocFooterCodec::decode(&bytes).expect_err("flip should fail")
    }

    #[test]
    fn detects_magic_flip() {
        let err = decode_after_flip(0);
        assert!(err.to_string().contains("magic"), "got: {err}");
    }

    #[test]
    fn detects_version_flip() {
        let err = decode_after_flip(4);
        assert!(
            err.to_string().contains("version") || err.to_string().contains("header length"),
            "got: {err}"
        );
    }

    #[test]
    fn detects_snapshot_generation_flip() {
        // snapshot_generation lives at bytes 8..16. Flipping it
        // changes the value `decode` reads, which then feeds
        // `footer_checksum(snap, body)` and produces a hash that
        // disagrees with the stored footer_checksum at 56..88.
        let err = decode_after_flip(8);
        assert!(err.to_string().contains("checksum"), "got: {err}");
    }

    #[test]
    fn detects_body_checksum_field_flip() {
        // bytes 24..56 hold the precomputed body checksum. Flipping
        // a byte makes `expected_body_checksum != blake3(body)`.
        let err = decode_after_flip(24);
        assert!(err.to_string().contains("checksum"), "got: {err}");
    }

    #[test]
    fn detects_footer_checksum_field_flip() {
        // bytes 56..88 hold the precomputed footer checksum. Flipping
        // a byte makes `expected_footer_checksum != recomputed`.
        let err = decode_after_flip(56);
        assert!(err.to_string().contains("checksum"), "got: {err}");
    }

    #[test]
    fn detects_body_byte_flip() {
        // First body byte sits at TOC_HEADER_LEN. Flipping any body
        // byte invalidates `blake3(body)` against the stored
        // body_checksum at 24..56.
        let err = decode_after_flip(TOC_HEADER_LEN);
        assert!(err.to_string().contains("checksum"), "got: {err}");
    }

    #[test]
    fn detects_body_len_flip_upward() {
        // body_len lives at bytes 16..24 (u64 LE). Flip the high
        // byte to push declared length past EOF — caught by the
        // truncation check before any hashing happens.
        let footer = sample_footer(7);
        let mut bytes = TocFooterCodec::encode(&footer).expect("encode footer");
        bytes[23] ^= 0x80; // turn on a high bit to make body_len huge
        let err = TocFooterCodec::decode(&bytes).expect_err("flip should fail");
        assert!(err.to_string().contains("truncated"), "got: {err}");
    }

    #[test]
    fn detects_body_len_flip_downward() {
        // Cut declared body_len in half — `decode` slices a shorter
        // body and either fails the body_checksum check (wrong slice
        // hashes differently) or the bincode `consumed != body_len`
        // assertion. Either way: the footer is rejected.
        let footer = sample_footer(7);
        let mut bytes = TocFooterCodec::encode(&footer).expect("encode footer");
        let original_body_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let truncated = original_body_len.saturating_sub(1);
        bytes[16..24].copy_from_slice(&truncated.to_le_bytes());
        let err = TocFooterCodec::decode(&bytes).expect_err("flip should fail");
        let s = err.to_string();
        assert!(
            s.contains("checksum") || s.contains("trailing") || s.contains("truncated"),
            "got: {s}"
        );
    }

    #[test]
    fn full_self_validation_does_not_consult_external_state() {
        // The behavioral assertion: `TocFooterCodec::decode` takes
        // ONLY the footer bytes. There is no `&Header`, no `&File`,
        // no shared state — so by construction it cannot consult
        // `Header.toc_checksum`. A successful decode therefore proves
        // self-validation purely from the on-disk footer.
        let footer = sample_footer(42);
        let bytes = TocFooterCodec::encode(&footer).expect("encode footer");
        let (decoded, consumed) = TocFooterCodec::decode(&bytes).expect("decode footer");
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded, footer);
        assert_eq!(decoded.body.snapshot_generation, 42);
    }
}
