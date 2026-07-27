// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Fixed 4 KB file header, spec §7 (v2 layout).
//!
//! v2 (current) — Phase 4 of the WAL elimination plan. The embedded WAL
//! is gone; recovery reads the header → seeks to `footer_offset` →
//! validates the TOC footer's embedded self-checksum. The 32 bytes that
//! used to hold the four WAL u64s (40..72) and the 32 bytes that used
//! to hold `toc_checksum` (80..112) are now reserved-zero.

use std::io::{Read, Seek, SeekFrom, Write};

use uuid::Uuid;

use crate::{
    concurrency::coordination,
    error::{Error, Result},
    format::{
        CREATE_CONTRACT_DIGEST_LEN, CREATE_CONTRACT_DIGEST_OFFSET, Checksum,
        FEATURE_COORDINATION_REGION, FEATURE_CREATE_CONTRACT, FORMAT_MAJOR, FORMAT_MINOR,
        HEADER_SIZE,
    },
};

pub const MAGIC: [u8; 4] = *b"VLS\0";

const MAGIC_OFFSET: usize = 0;
const FORMAT_MAJOR_OFFSET: usize = 4;
const FORMAT_MINOR_OFFSET: usize = 6;
const HEADER_SIZE_OFFSET: usize = 8;
const FILE_UUID_OFFSET: usize = 12;
const FLAGS_OFFSET: usize = 28;
const FOOTER_OFFSET_OFFSET: usize = 32;
// Bytes 40..72 reserved-zero (former wal_offset / wal_size /
// wal_checkpoint_seq / wal_head_seq, deleted in v2).
const SNAPSHOT_GENERATION_OFFSET: usize = 72;
// Bytes 80..112 reserved-zero (former toc_checksum, deleted in v2 —
// the TOC footer carries its own checksum).
const FEATURE_BITMAP_OFFSET: usize = 112;
#[cfg(test)]
const RESERVED_OFFSET: usize = 120;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Header {
    pub format_major: u16,
    pub format_minor: u16,
    pub file_uuid: Uuid,
    pub flags: u32,
    pub footer_offset: u64,
    pub snapshot_generation: u64,
    pub feature_bitmap: u64,
    /// 32-byte BLAKE3 digest of `TocFooterBody.create_contract` (plan §4.3).
    /// All-zero when [`FEATURE_CREATE_CONTRACT`] is clear in
    /// `feature_bitmap`. Stamped once at file creation; commit-time
    /// header rewrites (`write_logical_prefix`) don't touch the bytes
    /// that carry it, so a later binary that re-opens an old file under
    /// the new schema sees a valid digest.
    pub create_contract_digest: Checksum,
}

impl Header {
    #[must_use]
    pub fn new(file_uuid: Uuid) -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            file_uuid,
            flags: 0,
            footer_offset: 0,
            snapshot_generation: 0,
            feature_bitmap: FEATURE_COORDINATION_REGION | FEATURE_CREATE_CONTRACT,
            create_contract_digest: [0; 32],
        }
    }

    #[must_use]
    pub fn new_default(file_uuid: Uuid) -> Self {
        Self::new(file_uuid)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_major != FORMAT_MAJOR {
            return Err(Error::Format(format!(
                "unsupported format major version: {} (binary expects {FORMAT_MAJOR}). \
                 v1 (pre-WAL-elimination) files are not readable; see MIGRATION.md.",
                self.format_major
            )));
        }
        // Each FORMAT_MINOR bump has been a catalog/TOC shape change that
        // bincode cannot decode against the older layout (v2.2 = ANN
        // profile burial, v2.3 = vote-profile burial). We reject older
        // minors at open rather than letting bincode fail mid-decode of
        // the TOC footer body.
        if self.format_minor != FORMAT_MINOR {
            return Err(Error::Format(format!(
                "unsupported format minor version: {} (binary expects {FORMAT_MINOR}). \
                 Earlier v2.x files are not readable by a v2.3 binary; see MIGRATION.md.",
                self.format_minor
            )));
        }
        if self.footer_offset != 0 && self.footer_offset < HEADER_SIZE as u64 {
            return Err(Error::Format(
                "footer offset points inside header region".into(),
            ));
        }
        Ok(())
    }
}

pub struct HeaderCodec;

impl HeaderCodec {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Header> {
        let mut buf = [0u8; HEADER_SIZE];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut buf)?;
        Self::decode(&buf)
    }

    /// Write the **entire** 4 KB header — including a fresh
    /// `stamp_initial` of the coordination region. Use this only at file
    /// creation, never at commit-update time. Subsequent header rewrites
    /// must go through [`Self::write_logical_prefix`] so they preserve
    /// the live atomic state in the coord region.
    pub fn write_initial<W: Write + Seek>(writer: &mut W, header: &Header) -> Result<()> {
        let buf = Self::encode(header)?;
        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&buf)?;
        Ok(())
    }

    /// Write only the logical-header prefix (bytes `0..120`), preserving
    /// the coord region and the trailing reserved area on disk. This is
    /// the path used by every header update after creation — `commit()`
    /// — so that mutations made by Stage 3b to `coord_published_*` and
    /// reader/writer slot bytes are not silently clobbered when the
    /// logical header rotates.
    pub fn write_logical_prefix<W: Write + Seek>(writer: &mut W, header: &Header) -> Result<()> {
        let buf = Self::encode(header)?;
        let logical_end = FEATURE_BITMAP_OFFSET + 8; // 120
        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&buf[..logical_end])?;
        Ok(())
    }

    /// Back-compat shim. Delegates to `write_logical_prefix` so existing
    /// call sites preserve the coord region without churn.
    pub fn write<W: Write + Seek>(writer: &mut W, header: &Header) -> Result<()> {
        Self::write_logical_prefix(writer, header)
    }

    pub fn encode(header: &Header) -> Result<[u8; HEADER_SIZE]> {
        header.validate()?;

        let mut buf = [0u8; HEADER_SIZE];
        buf[MAGIC_OFFSET..MAGIC_OFFSET + 4].copy_from_slice(&MAGIC);
        put_u16(&mut buf, FORMAT_MAJOR_OFFSET, header.format_major);
        put_u16(&mut buf, FORMAT_MINOR_OFFSET, header.format_minor);
        put_u32(&mut buf, HEADER_SIZE_OFFSET, HEADER_SIZE as u32);
        buf[FILE_UUID_OFFSET..FILE_UUID_OFFSET + 16].copy_from_slice(header.file_uuid.as_bytes());
        put_u32(&mut buf, FLAGS_OFFSET, header.flags);
        put_u64(&mut buf, FOOTER_OFFSET_OFFSET, header.footer_offset);
        // bytes 40..72 reserved-zero (deleted WAL fields)
        put_u64(
            &mut buf,
            SNAPSHOT_GENERATION_OFFSET,
            header.snapshot_generation,
        );
        // bytes 80..112 reserved-zero (deleted toc_checksum)
        put_u64(&mut buf, FEATURE_BITMAP_OFFSET, header.feature_bitmap);
        if header.feature_bitmap & FEATURE_COORDINATION_REGION != 0 {
            coordination::stamp_initial(&mut buf);
        }
        if header.feature_bitmap & FEATURE_CREATE_CONTRACT != 0 {
            buf[CREATE_CONTRACT_DIGEST_OFFSET
                ..CREATE_CONTRACT_DIGEST_OFFSET + CREATE_CONTRACT_DIGEST_LEN]
                .copy_from_slice(&header.create_contract_digest);
        }
        Ok(buf)
    }

    pub fn decode(bytes: &[u8; HEADER_SIZE]) -> Result<Header> {
        let magic = bytes
            .get(MAGIC_OFFSET..MAGIC_OFFSET + 4)
            .ok_or_else(|| Error::Format("header magic truncated".into()))?;
        if magic != MAGIC {
            return Err(Error::Format("header magic mismatch".into()));
        }

        let header_size = get_u32(bytes, HEADER_SIZE_OFFSET)?;
        if header_size != HEADER_SIZE as u32 {
            return Err(Error::Format(format!(
                "invalid header size: expected {HEADER_SIZE}, got {header_size}"
            )));
        }

        let file_uuid_bytes: [u8; 16] = bytes[FILE_UUID_OFFSET..FILE_UUID_OFFSET + 16]
            .try_into()
            .map_err(|_| Error::Format("file UUID truncated".into()))?;

        let feature_bitmap = get_u64(bytes, FEATURE_BITMAP_OFFSET)?;
        let create_contract_digest: Checksum = if feature_bitmap & FEATURE_CREATE_CONTRACT != 0 {
            bytes[CREATE_CONTRACT_DIGEST_OFFSET
                ..CREATE_CONTRACT_DIGEST_OFFSET + CREATE_CONTRACT_DIGEST_LEN]
                .try_into()
                .map_err(|_| Error::Format("create-contract digest truncated".into()))?
        } else {
            [0u8; 32]
        };

        let header = Header {
            format_major: get_u16(bytes, FORMAT_MAJOR_OFFSET)?,
            format_minor: get_u16(bytes, FORMAT_MINOR_OFFSET)?,
            file_uuid: Uuid::from_bytes(file_uuid_bytes),
            flags: get_u32(bytes, FLAGS_OFFSET)?,
            footer_offset: get_u64(bytes, FOOTER_OFFSET_OFFSET)?,
            snapshot_generation: get_u64(bytes, SNAPSHOT_GENERATION_OFFSET)?,
            feature_bitmap,
            create_contract_digest,
        };
        header.validate()?;
        Ok(header)
    }
}

fn put_u16(buf: &mut [u8], offset: usize, value: u16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let raw = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| Error::Format("header u16 field truncated".into()))?;
    Ok(u16::from_le_bytes(raw.try_into().map_err(|_| {
        Error::Format("header u16 field truncated".into())
    })?))
}

fn get_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::Format("header u32 field truncated".into()))?;
    Ok(u32::from_le_bytes(raw.try_into().map_err(|_| {
        Error::Format("header u32 field truncated".into())
    })?))
}

fn get_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::Format("header u64 field truncated".into()))?;
    Ok(u64::from_le_bytes(raw.try_into().map_err(|_| {
        Error::Format("header u64 field truncated".into())
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrips() {
        let mut header = Header::new_default(Uuid::from_u128(0xA11CE));
        header.footer_offset = HEADER_SIZE as u64;
        header.snapshot_generation = 7;

        let encoded = HeaderCodec::encode(&header).expect("encode header");
        let decoded = HeaderCodec::decode(&encoded).expect("decode header");

        assert_eq!(decoded, header);
    }

    #[test]
    fn rejects_bad_magic() {
        let header = Header::new_default(Uuid::from_u128(1));
        let mut encoded = HeaderCodec::encode(&header).expect("encode header");
        encoded[0] = b'M';

        let err = HeaderCodec::decode(&encoded).expect_err("bad magic should fail");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_mismatched_minor() {
        let header = Header::new_default(Uuid::from_u128(1));
        let mut encoded = HeaderCodec::encode(&header).expect("encode header");
        encoded[FORMAT_MINOR_OFFSET..FORMAT_MINOR_OFFSET + 2]
            .copy_from_slice(&(FORMAT_MINOR + 1).to_le_bytes());

        let err =
            HeaderCodec::decode(&encoded).expect_err("v2.x binary must reject foreign minors");
        assert!(err.to_string().contains("minor version"));
    }

    #[test]
    fn accepts_reserved_bytes_set() {
        let header = Header::new_default(Uuid::from_u128(1));
        let mut encoded = HeaderCodec::encode(&header).expect("encode header");
        encoded[RESERVED_OFFSET] = 1;

        let decoded = HeaderCodec::decode(&encoded).expect("reserved-byte set must not fail");
        assert_eq!(decoded.format_minor, FORMAT_MINOR);
    }

    #[test]
    fn rejects_footer_inside_header_region() {
        let mut header = Header::new_default(Uuid::from_u128(1));
        header.footer_offset = 100; // inside header (0..4096)

        let err = HeaderCodec::encode(&header).expect_err("invalid footer offset should fail");
        assert!(err.to_string().contains("footer offset"));
    }

    #[test]
    fn rejects_v1_format() {
        let mut header = Header::new_default(Uuid::from_u128(1));
        header.format_major = 1;
        let err = HeaderCodec::encode(&header).expect_err("v1 should be rejected");
        assert!(err.to_string().contains("unsupported"));
    }
}
