// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Versioned catalog segment encoding.
//!
//! Fixed headers are still hand-coded. Catalog records remain serde/bincode for
//! v0.1-alpha, but each record is length-prefixed and wrapped with table kind
//! plus schema version so future readers can dispatch per table/version instead
//! of relying on a raw `Vec<T>` blob.

use crate::{
    error::{Error, Result},
    format::bincode_config,
};

pub const CATALOG_MAGIC: [u8; 4] = *b"VLCC";
pub const CATALOG_RECORD_VERSION: u16 = 1;
pub const CATALOG_HEADER_LEN: usize = 12;
pub const CATALOG_RECORD_HEADER_LEN: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[repr(u16)]
pub enum CatalogTableKind {
    Segment = 0,
    Collection = 1,
    Frame = 2,
    TextSpace = 3,
    Analyzer = 4,
    FieldSchema = 5,
    RetrievalProfile = 6,
    EmbeddingSpace = 7,
    Codec = 8,
    Vector = 9,
    IdAllocator = 10,
    FusionProfile = 12,
}

impl TryFrom<u16> for CatalogTableKind {
    type Error = Error;

    fn try_from(value: u16) -> Result<Self> {
        match value {
            0 => Ok(Self::Segment),
            1 => Ok(Self::Collection),
            2 => Ok(Self::Frame),
            3 => Ok(Self::TextSpace),
            4 => Ok(Self::Analyzer),
            5 => Ok(Self::FieldSchema),
            6 => Ok(Self::RetrievalProfile),
            7 => Ok(Self::EmbeddingSpace),
            8 => Ok(Self::Codec),
            9 => Ok(Self::Vector),
            10 => Ok(Self::IdAllocator),
            12 => Ok(Self::FusionProfile),
            _ => Err(Error::Format(format!(
                "unknown catalog table kind: {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CatalogRecordHeader {
    pub version: u16,
    pub table: CatalogTableKind,
    pub payload_len: u32,
}

impl CatalogRecordHeader {
    pub fn new(table: CatalogTableKind, payload_len: usize) -> Result<Self> {
        let payload_len = u32::try_from(payload_len)
            .map_err(|_| Error::Format("catalog record too large".into()))?;
        Ok(Self {
            version: CATALOG_RECORD_VERSION,
            table,
            payload_len,
        })
    }

    #[must_use]
    pub fn encode(&self) -> [u8; CATALOG_RECORD_HEADER_LEN] {
        let mut out = [0u8; CATALOG_RECORD_HEADER_LEN];
        out[0..2].copy_from_slice(&self.version.to_le_bytes());
        out[2..4].copy_from_slice(&(self.table as u16).to_le_bytes());
        out[4..8].copy_from_slice(&self.payload_len.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8; CATALOG_RECORD_HEADER_LEN]) -> Result<Self> {
        let version = read_u16(bytes, 0)?;
        if version != CATALOG_RECORD_VERSION {
            return Err(Error::Format(format!(
                "unsupported catalog record version: {version}"
            )));
        }
        Ok(Self {
            version,
            table: CatalogTableKind::try_from(read_u16(bytes, 2)?)?,
            payload_len: read_u32(bytes, 4)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogRecordEnvelope<'a> {
    pub header: CatalogRecordHeader,
    pub payload: &'a [u8],
}

pub fn encode_catalog_record<T: serde::Serialize>(
    table: CatalogTableKind,
    record: &T,
) -> Result<Vec<u8>> {
    let payload = bincode::serde::encode_to_vec(record, bincode_config())
        .map_err(|err| Error::Serde(err.to_string()))?;
    let header = CatalogRecordHeader::new(table, payload.len())?;
    let mut out = Vec::with_capacity(CATALOG_RECORD_HEADER_LEN + payload.len());
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_catalog_record<T: serde::de::DeserializeOwned>(
    expected_table: CatalogTableKind,
    bytes: &[u8],
) -> Result<(T, usize)> {
    let (envelope, consumed) = decode_catalog_record_envelope(bytes)?;
    if envelope.header.table != expected_table {
        return Err(Error::Format(format!(
            "catalog record table mismatch: expected {:?}, got {:?}",
            expected_table, envelope.header.table
        )));
    }
    let (record, record_consumed): (T, usize) =
        bincode::serde::decode_from_slice(envelope.payload, bincode_config())
            .map_err(|err| Error::Serde(err.to_string()))?;
    if record_consumed != envelope.payload.len() {
        return Err(Error::Format("catalog record has trailing bytes".into()));
    }
    Ok((record, consumed))
}

pub fn decode_catalog_record_envelope(bytes: &[u8]) -> Result<(CatalogRecordEnvelope<'_>, usize)> {
    if bytes.len() < CATALOG_RECORD_HEADER_LEN {
        return Err(Error::Format("catalog record header truncated".into()));
    }
    let header_bytes: [u8; CATALOG_RECORD_HEADER_LEN] = bytes[..CATALOG_RECORD_HEADER_LEN]
        .try_into()
        .map_err(|_| Error::Format("catalog record header truncated".into()))?;
    let header = CatalogRecordHeader::decode(&header_bytes)?;
    let payload_len = header.payload_len as usize;
    let end = CATALOG_RECORD_HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| Error::Format("catalog record length overflow".into()))?;
    let payload = bytes
        .get(CATALOG_RECORD_HEADER_LEN..end)
        .ok_or_else(|| Error::Format("catalog record truncated".into()))?;
    Ok((CatalogRecordEnvelope { header, payload }, end))
}

pub fn encode_catalog_records<T: serde::Serialize>(
    table: CatalogTableKind,
    records: &[T],
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&CATALOG_MAGIC);
    out.extend_from_slice(&(table as u16).to_le_bytes());
    out.extend_from_slice(&CATALOG_RECORD_VERSION.to_le_bytes());
    let record_count = u32::try_from(records.len())
        .map_err(|_| Error::Format("too many catalog records".into()))?;
    out.extend_from_slice(&record_count.to_le_bytes());

    for record in records {
        out.extend_from_slice(&encode_catalog_record(table, record)?);
    }
    Ok(out)
}

pub fn decode_catalog_records<T: serde::de::DeserializeOwned>(
    expected_table: CatalogTableKind,
    bytes: &[u8],
) -> Result<Vec<T>> {
    if bytes.len() < CATALOG_HEADER_LEN {
        return Err(Error::Format("catalog segment truncated".into()));
    }
    if bytes[0..4] != CATALOG_MAGIC {
        return Err(Error::Format("catalog magic mismatch".into()));
    }
    let table = CatalogTableKind::try_from(read_u16(bytes, 4)?)?;
    if table != expected_table {
        return Err(Error::Format(format!(
            "catalog table mismatch: expected {expected_table:?}, got {table:?}"
        )));
    }
    let version = read_u16(bytes, 6)?;
    if version != CATALOG_RECORD_VERSION {
        return Err(Error::Format(format!(
            "unsupported catalog record version: {version}"
        )));
    }
    let count = read_u32(bytes, 8)? as usize;
    let mut offset = CATALOG_HEADER_LEN;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let (record, consumed) = decode_catalog_record(expected_table, &bytes[offset..])?;
        records.push(record);
        offset += consumed;
    }
    if offset != bytes.len() {
        return Err(Error::Format("catalog segment has trailing bytes".into()));
    }
    Ok(records)
}

pub fn encode_catalog_delta<T: serde::Serialize>(
    table: CatalogTableKind,
    previous_root: Option<crate::format::segment::SegmentRef>,
    records: &[T],
) -> Result<Vec<u8>> {
    let mut out = encode_catalog_records(table, records)?;
    let previous = bincode::serde::encode_to_vec(previous_root, bincode_config())
        .map_err(|err| Error::Serde(err.to_string()))?;
    let previous_len = u32::try_from(previous.len())
        .map_err(|_| Error::Format("catalog previous root too large".into()))?;
    out.extend_from_slice(&previous_len.to_le_bytes());
    out.extend_from_slice(&previous);
    Ok(out)
}

pub struct CatalogDelta<T> {
    pub previous_root: Option<crate::format::segment::SegmentRef>,
    pub records: Vec<T>,
}

pub fn decode_catalog_delta<T: serde::de::DeserializeOwned>(
    expected_table: CatalogTableKind,
    bytes: &[u8],
) -> Result<CatalogDelta<T>> {
    let (records, offset) = decode_catalog_records_prefix(expected_table, bytes)?;
    let previous_root = decode_previous_root_suffix(bytes, offset)?;
    Ok(CatalogDelta {
        previous_root,
        records,
    })
}

pub fn decode_catalog_previous_root(
    expected_table: CatalogTableKind,
    bytes: &[u8],
) -> Result<Option<crate::format::segment::SegmentRef>> {
    let offset = catalog_records_end_offset(expected_table, bytes)?;
    decode_previous_root_suffix(bytes, offset)
}

fn catalog_records_end_offset(expected_table: CatalogTableKind, bytes: &[u8]) -> Result<usize> {
    if bytes.len() < CATALOG_HEADER_LEN {
        return Err(Error::Format("catalog segment truncated".into()));
    }
    if bytes[0..4] != CATALOG_MAGIC {
        return Err(Error::Format("catalog magic mismatch".into()));
    }
    let table = CatalogTableKind::try_from(read_u16(bytes, 4)?)?;
    if table != expected_table {
        return Err(Error::Format(format!(
            "catalog table mismatch: expected {expected_table:?}, got {table:?}"
        )));
    }
    let count = read_u32(bytes, 8)? as usize;
    let mut offset = CATALOG_HEADER_LEN;
    for _ in 0..count {
        let (_, consumed) = decode_catalog_record_envelope(&bytes[offset..])?;
        offset += consumed;
    }
    Ok(offset)
}

fn decode_catalog_records_prefix<T: serde::de::DeserializeOwned>(
    expected_table: CatalogTableKind,
    bytes: &[u8],
) -> Result<(Vec<T>, usize)> {
    if bytes.len() < CATALOG_HEADER_LEN {
        return Err(Error::Format("catalog segment truncated".into()));
    }
    if bytes[0..4] != CATALOG_MAGIC {
        return Err(Error::Format("catalog magic mismatch".into()));
    }
    let table = CatalogTableKind::try_from(read_u16(bytes, 4)?)?;
    if table != expected_table {
        return Err(Error::Format(format!(
            "catalog table mismatch: expected {expected_table:?}, got {table:?}"
        )));
    }
    let version = read_u16(bytes, 6)?;
    if version != CATALOG_RECORD_VERSION {
        return Err(Error::Format(format!(
            "unsupported catalog record version: {version}"
        )));
    }
    let count = read_u32(bytes, 8)? as usize;
    let mut offset = CATALOG_HEADER_LEN;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let (record, consumed) = decode_catalog_record(expected_table, &bytes[offset..])?;
        records.push(record);
        offset += consumed;
    }
    Ok((records, offset))
}

fn decode_previous_root_suffix(
    bytes: &[u8],
    offset: usize,
) -> Result<Option<crate::format::segment::SegmentRef>> {
    if offset == bytes.len() {
        return Ok(None);
    }
    let previous_len = read_u32(bytes, offset)? as usize;
    let offset = offset + 4;
    let end = offset
        .checked_add(previous_len)
        .ok_or_else(|| Error::Format("catalog previous root length overflow".into()))?;
    let previous_bytes = bytes
        .get(offset..end)
        .ok_or_else(|| Error::Format("catalog previous root truncated".into()))?;
    let (previous_root, consumed): (Option<crate::format::segment::SegmentRef>, usize) =
        bincode::serde::decode_from_slice(previous_bytes, bincode_config())
            .map_err(|err| Error::Serde(err.to_string()))?;
    if consumed != previous_bytes.len() {
        return Err(Error::Format(
            "catalog previous root has trailing bytes".into(),
        ));
    }
    if end != bytes.len() {
        return Err(Error::Format("catalog segment has trailing bytes".into()));
    }
    Ok(previous_root)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| Error::Format("catalog u16 field truncated".into()))?
            .try_into()
            .map_err(|_| Error::Format("catalog u16 field truncated".into()))?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| Error::Format("catalog u32 field truncated".into()))?
            .try_into()
            .map_err(|_| Error::Format("catalog u32 field truncated".into()))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_records_roundtrip() {
        let records = vec![1_u64, 2, 3];
        let encoded = encode_catalog_records(CatalogTableKind::Frame, &records).expect("encode");
        let decoded: Vec<u64> =
            decode_catalog_records(CatalogTableKind::Frame, &encoded).expect("decode");

        assert_eq!(decoded, records);
    }

    #[test]
    fn catalog_records_reject_wrong_table() {
        let encoded = encode_catalog_records(CatalogTableKind::Frame, &[1_u64]).expect("encode");
        let err = decode_catalog_records::<u64>(CatalogTableKind::Collection, &encoded)
            .expect_err("wrong table should fail");

        assert!(err.to_string().contains("catalog table mismatch"));
    }

    #[test]
    fn catalog_record_envelope_exposes_table_and_length() {
        let encoded = encode_catalog_record(CatalogTableKind::Frame, &42_u64).expect("encode");
        let (envelope, consumed) =
            decode_catalog_record_envelope(&encoded).expect("decode envelope");

        assert_eq!(envelope.header.version, CATALOG_RECORD_VERSION);
        assert_eq!(envelope.header.table, CatalogTableKind::Frame);
        assert_eq!(envelope.header.payload_len as usize, envelope.payload.len());
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn catalog_records_reject_record_table_mismatch() {
        let mut encoded =
            encode_catalog_records(CatalogTableKind::Frame, &[1_u64]).expect("encode");
        let record_table_offset = CATALOG_HEADER_LEN + 2;
        encoded[record_table_offset..record_table_offset + 2]
            .copy_from_slice(&(CatalogTableKind::Collection as u16).to_le_bytes());

        let err = decode_catalog_records::<u64>(CatalogTableKind::Frame, &encoded)
            .expect_err("record table mismatch should fail");

        assert!(err.to_string().contains("record table mismatch"));
    }

    #[test]
    fn catalog_records_reject_truncated_record() {
        let mut encoded =
            encode_catalog_records(CatalogTableKind::Frame, &[1_u64]).expect("encode");
        encoded.pop();

        let err = decode_catalog_records::<u64>(CatalogTableKind::Frame, &encoded)
            .expect_err("truncated record should fail");

        assert!(err.to_string().contains("truncated"));
    }
}
