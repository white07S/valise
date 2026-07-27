// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use crate::{
    error::{Error, Result},
    format::toc::{TOC_MAGIC, TocFooter, TocFooterCodec},
};

pub(super) fn read_toc_footer(file: &mut File, offset: u64) -> Result<TocFooter> {
    if offset == 0 {
        return Err(Error::Format(
            "header does not point to a TOC footer".into(),
        ));
    }
    let file_len = file.metadata()?.len();
    if offset >= file_len {
        return Err(Error::Format("TOC footer offset is past EOF".into()));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut header = [0u8; crate::format::toc::TOC_HEADER_LEN];
    file.read_exact(&mut header)?;
    let footer_len = TocFooterCodec::encoded_len_from_header(&header)?;
    if offset + footer_len as u64 > file_len {
        return Err(Error::Format("TOC footer body is past EOF".into()));
    }
    let mut bytes = vec![0u8; footer_len];
    bytes[..crate::format::toc::TOC_HEADER_LEN].copy_from_slice(&header);
    file.read_exact(&mut bytes[crate::format::toc::TOC_HEADER_LEN..])?;
    let (footer, _) = TocFooterCodec::decode(&bytes)?;
    Ok(footer)
}

pub(super) fn write_toc_footer_at_end(file: &mut File, footer: &TocFooter) -> Result<()> {
    let bytes = TocFooterCodec::encode(footer)?;
    file.seek(SeekFrom::End(0))?;
    file.write_all(&bytes)?;
    Ok(())
}

/// A fully self-validating TOC footer discovered by [`scan_for_footer_candidates`],
/// together with the byte offset it decodes from.
pub(super) struct FooterCandidate {
    pub(super) offset: u64,
    pub(super) footer: TocFooter,
}

/// Recovery-path scan: stream the file in bounded chunks hunting for the
/// `VLTC` magic and return every offset whose bytes decode as a fully
/// self-validating TOC footer (embedded body + footer BLAKE3, version and
/// length bounds — see [`TocFooterCodec::decode`]).
///
/// Decoy `VLTC` bytes inside payload data fail the double-checksum
/// validation and are skipped without aborting the scan; forging a footer
/// that passes both BLAKE3 digests is effectively impossible by accident.
/// This runs only when the header-referenced footer is unusable or
/// suspect (see `lifecycle::resolve_footer_state`), so the linear read is
/// acceptable; chunking keeps peak memory bounded for multi-GB files.
pub(super) fn scan_for_footer_candidates(file: &mut File) -> Result<Vec<FooterCandidate>> {
    /// Streaming chunk size. Large enough to amortize syscalls, small
    /// enough that the recovery path never holds a file-sized buffer.
    const SCAN_CHUNK: usize = 8 * 1024 * 1024;
    let magic_len = TOC_MAGIC.len();
    let file_len = file.metadata()?.len();
    // Footers never live inside the fixed 4 KB header region
    // (`Header::validate` rejects `footer_offset < HEADER_SIZE`), so the
    // scan starts right past it.
    let scan_start = crate::format::HEADER_SIZE as u64;
    let mut candidates = Vec::new();
    if file_len <= scan_start {
        return Ok(candidates);
    }

    let mut buf: Vec<u8> = Vec::new();
    // Trailing `magic_len - 1` bytes of the previous chunk, so a magic
    // spanning a chunk boundary is still seen once (and only once).
    let mut carry: Vec<u8> = Vec::new();
    let mut next_read = scan_start;
    while next_read < file_len {
        let want = SCAN_CHUNK.min((file_len - next_read) as usize);
        let base = next_read - carry.len() as u64;
        buf.clear();
        buf.extend_from_slice(&carry);
        let carry_len = carry.len();
        buf.resize(carry_len + want, 0);
        file.seek(SeekFrom::Start(next_read))?;
        file.read_exact(&mut buf[carry_len..])?;
        next_read += want as u64;

        let mut i = 0usize;
        while i + magic_len <= buf.len() {
            let Some(rel) = buf[i..].iter().position(|&b| b == TOC_MAGIC[0]) else {
                break;
            };
            let at = i + rel;
            if at + magic_len > buf.len() {
                break;
            }
            if buf[at..at + magic_len] == TOC_MAGIC {
                let offset = base + at as u64;
                // Full bounded decode + double-BLAKE3 validation via the
                // regular footer reader. A failed candidate (decoy bytes,
                // torn footer, bad checksum, body past EOF) is skipped;
                // the scan keeps going.
                if let Ok(footer) = read_toc_footer(file, offset) {
                    candidates.push(FooterCandidate { offset, footer });
                }
            }
            i = at + 1;
        }

        if next_read < file_len {
            let keep = buf.len().min(magic_len - 1);
            carry = buf[buf.len() - keep..].to_vec();
        }
    }
    Ok(candidates)
}
