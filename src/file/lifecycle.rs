//! `ValiseFile` lifecycle: create / open / durability.

use super::*;

impl ValiseFile {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::create_with_options(path, CreateOptions::default())
    }

    pub fn create_with_options(path: impl AsRef<Path>, options: CreateOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        // Phase 1: lower the user-facing options into the persisted
        // `CreateContractV1` and validate before any bytes hit the disk.
        // Validation errors here surface BEFORE `create_new(true)`
        // creates an empty file at the chosen path.
        let create_contract = options.build_contract()?;
        let contract_digest = create_contract.digest()?;

        let mut file = FsOpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        // Stage 3b: new files always carry the coordination region, so
        // cross-process writer exclusion is provided by the writer-slot
        // byte-range lock (`F_SETLK` on macOS, `F_OFD_SETLK` on Linux).
        // The legacy whole-file `flock` is intentionally NOT taken here:
        // on macOS it would share kernel lock state with `F_SETLK` and
        // self-conflict on subsequent commits.

        // File creation is always strictly durable — `Durability::Buffered`
        // applies only to subsequent write traffic (put_frame / put_vector
        // / register_*). The file itself, its initial empty TOC, and the
        // parent directory entry must survive a power loss the moment
        // `create_with_options` returns Ok.
        let init_durability = Durability::FullSync;
        let mut header = Header::new(Uuid::new_v4());
        // Phase 1: stamp the contract digest into the header before the
        // first write. The digest lives at byte 832 (post-coord
        // reserved area) so subsequent `write_logical_prefix` rewrites
        // never touch it; it gets baked once at create time.
        header.create_contract_digest = contract_digest;
        file.set_len(crate::format::HEADER_SIZE as u64)?;
        // First write at create stamps the full 4 KB header including the
        // canonical empty coordination region (Stage 3a) and the contract digest.
        HeaderCodec::write_initial(&mut file, &header)?;
        sync_file(&file, init_durability)?;

        let footer = TocFooter::empty(1, create_contract.clone())?;
        let footer_offset = file.metadata()?.len();
        write_toc_footer_at_end(&mut file, &footer)?;
        sync_file(&file, init_durability)?;

        header.footer_offset = footer_offset;
        header.snapshot_generation = footer.body.snapshot_generation;
        // Subsequent header rewrites must preserve the coord region's
        // mutated atomics — `write` aliases `write_logical_prefix`.
        // The create-contract digest at byte 832 also survives this
        // because `write_logical_prefix` only rewrites bytes 0..120.
        HeaderCodec::write(&mut file, &header)?;
        sync_file(&file, init_durability)?;
        sync_parent_dir(&path, init_durability)?;

        let file_mmap = remap_file(&file)?.map(Arc::new);
        let verified_payload_segments = Arc::new(RwLock::new(HashSet::new()));
        let initial_snapshot = build_snapshot(
            &header,
            file_mmap.clone(),
            &CatalogSnapshot::default(),
            &HashMap::new(),
            &SegmentRegistry {
                catalog: Vec::new(),
                by_id: HashMap::new(),
                loaded: true,
            },
            &HashMap::new(),
            verified_payload_segments.clone(),
        );
        Ok(Self {
            path,
            file: Mutex::new(file),
            file_mmap,
            published_snapshot: ArcSwap::new(Arc::new(initial_snapshot)),
            commit_fsync_barrier: RwLock::new(None),
            mode: OpenMode::ReadWrite,
            durability: options.durability,
            header,
            footer,
            create_contract,
            catalog: CatalogSnapshot::default(),
            frame_locators: HashMap::new(),
            vector_base: RwLock::new(VectorBasePtrs::default()),
            id_allocator: IdAllocator::default(),
            segment_registry: RwLock::new(SegmentRegistry {
                catalog: Vec::new(),
                by_id: HashMap::new(),
                loaded: true,
            }),
            dirty_segment_ids: HashSet::new(),
            dirty_collection_ids: HashSet::new(),
            dirty_frame_ids: HashSet::new(),
            dirty_embedding_space_ids: HashSet::new(),
            dirty_codec_ids: HashSet::new(),
            dirty_vector_ids: HashSet::new(),
            dirty_text_space_ids: HashSet::new(),
            dirty_analyzer_ids: HashSet::new(),
            dirty_field_schema_ids: HashSet::new(),
            dirty_retrieval_profile_ids: HashSet::new(),
            dirty_fusion_profile_ids: HashSet::new(),
            codec_cache: HashMap::new(),
            pending_vector_batches: HashMap::new(),
            pending_payload_buf: Vec::new(),
            pending_payload_frames: Vec::new(),
            pending_payload_by_frame: HashMap::new(),
            pending_uri_buf: Vec::new(),
            pending_uris: Vec::new(),
            ingest_profile: IngestProfile::default(),
            ingest_profile_enabled: std::env::var_os("VALISE_INGEST_PROFILE").is_some(),
            pending_text_indexes: HashMap::new(),
            text_space_states: HashMap::new(),
            tombstoned_frame_count: 0,
            analyzer_cache: HashMap::new(),
            vector_by_id_cache: HashMap::new(),
            collection_filter_cache: filter_io::CollectionFilterCache::new(),
            time_index_cache: time_index_io::TimeIndexCache::new(),
            verified_payload_segments,
            dirty: false,
        })
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path, OpenMode::ReadOnly)
    }

    pub fn open_read_write(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path, OpenMode::ReadWrite)
    }

    pub fn open(path: impl AsRef<Path>, mode: OpenMode) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut options = FsOpenOptions::new();
        options.read(true);
        if mode == OpenMode::ReadWrite {
            options.write(true);
        }
        let mut file = options.open(&path)?;
        // Every Valise file (post-consolidation) carries the coordination
        // region — cross-process arbitration runs through the in-region
        // byte locks. The legacy whole-file `flock` fallback is gone.

        let prof_top = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
        let mark = std::time::Instant::now();
        let mut header = HeaderCodec::read(&mut file)?;
        if prof_top {
            crate::prof_eprintln!(
                "    [open] header_read:        {:.3}s",
                mark.elapsed().as_secs_f64()
            );
        }
        // Required: every file MUST advertise both the coord region
        // and the create contract. `Header::decode` already enforced
        // the magic / version / size / checksum invariants; we only
        // re-check the feature bits here so a hand-crafted header
        // can't slip through.
        if header.feature_bitmap & crate::format::FEATURE_COORDINATION_REGION == 0 {
            return Err(Error::Format(
                "open: file lacks FEATURE_COORDINATION_REGION (legacy files are not supported)"
                    .into(),
            ));
        }
        if header.feature_bitmap & crate::format::FEATURE_CREATE_CONTRACT == 0 {
            return Err(Error::Format(
                "open: file lacks FEATURE_CREATE_CONTRACT (legacy files are not supported)".into(),
            ));
        }

        // v2 has no WAL: open is read-header → read-TOC → load roots →
        // done. No replay, no invalidation. When the header-referenced
        // footer is unusable (torn commit, reordered writeback, media
        // corruption) `resolve_footer_state` falls back to scan-back
        // recovery and re-anchors on the most recent durable snapshot.
        let (footer_offset, footer, state) = resolve_footer_state(&mut file, &header)?;

        // Scan-back (or the generation cross-check) may have re-anchored
        // the open on a footer other than the one the on-disk header
        // referenced. Fix the in-memory header so the served snapshot is
        // self-consistent — the reported generation is always the
        // recovered footer's own. The on-disk header is NOT rewritten
        // here (the open may be read-only); a ReadWrite recovery heals
        // it at the next commit's header rewrite.
        header.footer_offset = footer_offset;
        header.snapshot_generation = footer.body.snapshot_generation;

        let create_contract = footer.body.create_contract.clone();
        let OpenState {
            catalog,
            frame_locators,
            id_allocator,
            codec_cache,
            analyzer_cache,
            text_space_states,
            collection_filter_cache,
            time_index_cache,
            vector_by_id_cache,
            segment_registry,
        } = state;

        let prof = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
        let mark = std::time::Instant::now();
        let file_mmap = remap_file(&file)?.map(Arc::new);
        if prof {
            crate::prof_eprintln!(
                "    [open] mmap_remap:         {:.3}s",
                mark.elapsed().as_secs_f64()
            );
        }
        let verified_payload_segments = Arc::new(RwLock::new(HashSet::new()));
        let initial_snapshot = build_snapshot(
            &header,
            file_mmap.clone(),
            &catalog,
            &frame_locators,
            &segment_registry,
            &vector_by_id_cache,
            verified_payload_segments.clone(),
        );
        let tombstoned_frame_count: u64 = catalog
            .frame_stubs
            .iter()
            .filter(|s| s.status == FrameStatus::Tombstoned)
            .count() as u64;
        Ok(Self {
            path,
            file: Mutex::new(file),
            file_mmap,
            published_snapshot: ArcSwap::new(Arc::new(initial_snapshot)),
            commit_fsync_barrier: RwLock::new(None),
            mode,
            durability: Durability::default(),
            header,
            footer,
            create_contract,
            catalog,
            frame_locators,
            vector_base: RwLock::new(VectorBasePtrs::default()),
            id_allocator,
            segment_registry: RwLock::new(segment_registry),
            dirty_segment_ids: HashSet::new(),
            dirty_collection_ids: HashSet::new(),
            dirty_frame_ids: HashSet::new(),
            dirty_embedding_space_ids: HashSet::new(),
            dirty_codec_ids: HashSet::new(),
            dirty_vector_ids: HashSet::new(),
            dirty_text_space_ids: HashSet::new(),
            dirty_analyzer_ids: HashSet::new(),
            dirty_field_schema_ids: HashSet::new(),
            dirty_retrieval_profile_ids: HashSet::new(),
            dirty_fusion_profile_ids: HashSet::new(),
            codec_cache,
            pending_vector_batches: HashMap::new(),
            pending_payload_buf: Vec::new(),
            pending_payload_frames: Vec::new(),
            pending_payload_by_frame: HashMap::new(),
            pending_uri_buf: Vec::new(),
            pending_uris: Vec::new(),
            ingest_profile: IngestProfile::default(),
            ingest_profile_enabled: std::env::var_os("VALISE_INGEST_PROFILE").is_some(),
            pending_text_indexes: HashMap::new(),
            text_space_states,
            tombstoned_frame_count,
            analyzer_cache,
            vector_by_id_cache,
            collection_filter_cache,
            time_index_cache,
            verified_payload_segments,
            dirty: false,
        })
    }
}

/// Everything loadable from a TOC footer at open time: the catalog
/// snapshot plus every derived cache and the (eagerly loaded) segment
/// registry. Bundled so the scan-back recovery path can treat "this
/// footer's snapshot fully loads and validates" as a single operation.
struct OpenState {
    catalog: CatalogSnapshot,
    frame_locators: HashMap<FrameId, catalog_io::FrameLocator>,
    id_allocator: IdAllocator,
    codec_cache: HashMap<CodecId, Box<dyn VectorCodec>>,
    analyzer_cache: HashMap<TextSpaceId, Analyzer>,
    text_space_states: HashMap<TextSpaceId, TextSpaceState>,
    collection_filter_cache: filter_io::CollectionFilterCache,
    time_index_cache: time_index_io::TimeIndexCache,
    vector_by_id_cache: HashMap<VectorId, VectorDesc>,
    segment_registry: SegmentRegistry,
}

/// Resolve the TOC footer that anchors this open, together with the full
/// snapshot state loaded from it. Returns the footer's byte offset so the
/// caller can fix up the in-memory header after a recovery.
///
/// Fast path (every clean open): the header-referenced footer decodes,
/// matches the create-contract anchor AND the header's generation label,
/// and every root reachable from it loads — identical to the classic v2
/// open sequence.
///
/// Recovery path (scan-back): when the referenced footer is missing,
/// torn, checksum-invalid, past EOF, generation-inconsistent with the
/// header (torn header rewrite), or any of its snapshot roots fail to
/// load, scan the whole file for candidate `NXTC` footers and recover
/// the highest-generation fully-valid snapshot. The previous commit's
/// footer is guaranteed durable — that commit's own fsync barrier
/// covered it — so any file that ever committed successfully has at
/// least one recoverable snapshot.
fn resolve_footer_state(file: &mut File, header: &Header) -> Result<(u64, TocFooter, OpenState)> {
    let primary = read_toc_footer(file, header.footer_offset).and_then(|footer| {
        check_footer_contract(&footer, header)?;
        Ok(footer)
    });
    let suspect_reason = match primary {
        Ok(footer) => {
            if footer.body.snapshot_generation == header.snapshot_generation {
                match load_open_state(file, &footer) {
                    Ok(state) => return Ok((header.footer_offset, footer, state)),
                    Err(err) => {
                        format!("snapshot roots under the referenced footer failed to load: {err}")
                    }
                }
            } else {
                // Torn-header cross-check: the referenced footer is
                // valid but carries a different generation than the
                // header's label — half of the 120-byte logical header
                // rewrite landed without the other. The header can no
                // longer be trusted as the snapshot anchor; the scan
                // re-derives it from the footers alone (and may well
                // re-select the referenced footer).
                format!(
                    "header/footer generation mismatch (header {}, footer {}) — torn header \
                     rewrite suspected",
                    header.snapshot_generation, footer.body.snapshot_generation
                )
            }
        }
        Err(err) => format!("referenced footer failed validation: {err}"),
    };
    recover_via_scan(file, header, &suspect_reason)
}

/// Scan-back recovery: enumerate every fully-valid footer in the file,
/// then anchor the open on the highest-generation candidate whose entire
/// snapshot state loads. Candidates whose roots fail integrity are
/// rejected and the next-best generation is tried.
fn recover_via_scan(
    file: &mut File,
    header: &Header,
    cause: &str,
) -> Result<(u64, TocFooter, OpenState)> {
    let file_len = file.metadata()?.len();
    tracing::warn!(
        target: "valise::recovery",
        cause,
        header_footer_offset = header.footer_offset,
        header_generation = header.snapshot_generation,
        file_len,
        "open: header-referenced TOC footer unusable or suspect; starting scan-back recovery"
    );
    let scanned = scan_for_footer_candidates(file)?;
    // Primary lineage check: candidates must carry THIS file's
    // create-time contract, anchored by the 32-byte digest at header
    // byte 832 (stamped once at create, never touched by the 0..120
    // logical-prefix rewrite). This pins out footers copied in from a
    // foreign file.
    //
    // Fallback (footer consensus): the digest region itself sits on the
    // same device as everything else — if it is corrupted, the header
    // copy matches NO valid footer and the anchor is implausible. In
    // that case the contract is re-derived from the footers themselves:
    // eligible candidates are those whose embedded contract digest
    // matches the consensus of the independently checksum-valid footers
    // in the file (2+ agreeing footers, or the single footer of a
    // fresh file). The header anchor stays authoritative whenever it
    // matches at least one valid candidate.
    let (mut candidates, unanchored): (Vec<FooterCandidate>, Vec<FooterCandidate>) = scanned
        .into_iter()
        .partition(|c| check_footer_contract(&c.footer, header).is_ok());
    if candidates.is_empty() && !unanchored.is_empty() {
        tracing::warn!(
            target: "valise::recovery",
            valid_footers = unanchored.len(),
            "open: header create-contract digest matches NO checksum-valid footer — header \
             digest region untrusted; falling back to footer-consensus contract validation"
        );
        candidates = footer_consensus_candidates(unanchored);
    }
    // Highest snapshot_generation wins; on a (theoretically impossible)
    // generation tie prefer the later-appended copy.
    candidates.sort_by(|a, b| {
        (b.footer.body.snapshot_generation, b.offset)
            .cmp(&(a.footer.body.snapshot_generation, a.offset))
    });
    let valid_candidates = candidates.len();
    for candidate in candidates {
        match load_open_state(file, &candidate.footer) {
            Ok(state) => {
                tracing::warn!(
                    target: "valise::recovery",
                    recovered_generation = candidate.footer.body.snapshot_generation,
                    recovered_footer_offset = candidate.offset,
                    scan_distance_bytes = file_len.saturating_sub(candidate.offset),
                    bytes_scanned = file_len.saturating_sub(crate::format::HEADER_SIZE as u64),
                    valid_candidates,
                    "open: scan-back recovered the most recent durable snapshot"
                );
                return Ok((candidate.offset, candidate.footer, state));
            }
            Err(err) => {
                tracing::warn!(
                    target: "valise::recovery",
                    candidate_generation = candidate.footer.body.snapshot_generation,
                    candidate_offset = candidate.offset,
                    error = %err,
                    "open: scan-back candidate rejected (snapshot roots failed to load); \
                     trying the next generation"
                );
            }
        }
    }
    Err(Error::Integrity(format!(
        "open: {cause}; scan-back found no recoverable snapshot"
    )))
}

/// The 32-byte BLAKE3 digest stamped in the header at byte 832 MUST match
/// the digest of the contract in the footer body — that's the
/// immutability anchor for the file-level invariants (and, for scan-back,
/// the proof a candidate footer belongs to this file's lineage).
fn check_footer_contract(footer: &TocFooter, header: &Header) -> Result<()> {
    if footer.body.create_contract.digest()? != header.create_contract_digest {
        return Err(Error::Integrity("create-contract digest mismatch".into()));
    }
    Ok(())
}

/// Footer-consensus contract validation, used only when the header's
/// byte-832 digest copy matches no checksum-valid footer (i.e. the
/// header region itself is corrupt). Every candidate here already passed
/// the double-BLAKE3 footer validation, so its embedded contract is
/// authentic *for that footer*; consensus establishes which contract is
/// this file's lineage:
///
/// - A single valid footer (fresh file) is its own consensus.
/// - Otherwise the eligible digest is the one shared by the MOST
///   candidates and by at least two of them. Ties are broken toward the
///   group containing the lowest-offset footer — the create-time footer
///   is the oldest bytes in the file, written before any
///   attacker-influenced payload could have been appended.
/// - Two-plus footers that all disagree yield no consensus: an empty
///   result, so the open fails closed.
///
/// Candidates whose embedded contract fails to re-encode for digesting
/// are dropped (they cannot vote or be validated).
fn footer_consensus_candidates(candidates: Vec<FooterCandidate>) -> Vec<FooterCandidate> {
    let mut digestible: Vec<(crate::format::Checksum, FooterCandidate)> = candidates
        .into_iter()
        .filter_map(|c| {
            let digest = c.footer.body.create_contract.digest().ok()?;
            Some((digest, c))
        })
        .collect();
    if digestible.len() == 1 {
        let (digest, candidate) = digestible.remove(0);
        tracing::warn!(
            target: "valise::recovery",
            consensus_digest = %hex::encode(digest),
            "open: single valid footer is its own contract consensus (fresh-file case)"
        );
        return vec![candidate];
    }
    // (votes, lowest offset seen) per digest.
    let mut groups: HashMap<crate::format::Checksum, (usize, u64)> = HashMap::new();
    for (digest, candidate) in &digestible {
        let entry = groups.entry(*digest).or_insert((0, u64::MAX));
        entry.0 += 1;
        entry.1 = entry.1.min(candidate.offset);
    }
    // Most votes wins; ties go to the group holding the oldest footer.
    let consensus = groups
        .iter()
        .max_by(|a, b| (a.1.0, std::cmp::Reverse(a.1.1)).cmp(&(b.1.0, std::cmp::Reverse(b.1.1))))
        .map(|(digest, &(votes, _))| (*digest, votes));
    match consensus {
        Some((digest, votes)) if votes >= 2 => {
            tracing::warn!(
                target: "valise::recovery",
                consensus_digest = %hex::encode(digest),
                agreeing_footers = votes,
                "open: contract consensus established from agreeing valid footers; header \
                 digest region bypassed"
            );
            digestible
                .into_iter()
                .filter_map(|(d, c)| (d == digest).then_some(c))
                .collect()
        }
        _ => {
            tracing::warn!(
                target: "valise::recovery",
                valid_footers = digestible.len(),
                "open: multiple valid footers disagree on the contract with no 2+ consensus; \
                 failing closed"
            );
            Vec::new()
        }
    }
}

/// Load every open-time root reachable from `footer`: the catalog
/// snapshot, codec + analyzer caches, text-space states, collection
/// filters, time indexes, and the segment registry. Any integrity
/// failure in any root makes the footer unusable as an open anchor —
/// the caller either fails the open or rejects the scan-back candidate.
fn load_open_state(file: &mut File, footer: &TocFooter) -> Result<OpenState> {
    let prof = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
    let mark = std::time::Instant::now();
    let (catalog, frame_locators) = read_catalog_snapshot(file, &footer.body)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] catalog_snapshot:   {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mut id_allocator = footer.body.id_allocator.clone();
    observe_catalog_ids(&catalog, &mut id_allocator);
    let mark = std::time::Instant::now();
    let codec_cache = build_codec_cache(file, &catalog.codecs)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] codec_cache:        {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    let analyzer_cache = build_analyzer_cache(&catalog)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] analyzer_cache:     {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    let text_space_states =
        rebuild_all_text_space_states(file, &footer.body, &mut id_allocator, &catalog.text_spaces)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] text_space_states:  {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    let collection_filter_cache = filter_io::load_collection_filter_cache(file, &footer.body)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] collection_filter:  {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    let time_index_cache = time_index_io::load_time_index_cache(file, &footer.body)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] time_index:         {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    let vector_by_id_cache = rebuild_vector_by_id_cache(&catalog);
    if prof {
        crate::prof_eprintln!(
            "    [open] vector_by_id_cache: {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    // Stage 6 fix: the published snapshot must carry a *loaded* segment
    // registry, otherwise readers driving directly off the snapshot
    // (`Snapshot::frame_full` / `Snapshot::read_payload`, bypassing
    // `Database::inner.read()`) can't resolve segment ids. Loading it
    // here also folds the registry chain into the "footer fully
    // validates" contract used by scan-back candidate selection.
    let cat = read_segment_catalog(file, &footer.body)?;
    let by_id = build_segment_map(&cat);
    let segment_registry = SegmentRegistry {
        catalog: cat,
        by_id,
        loaded: true,
    };
    Ok(OpenState {
        catalog,
        frame_locators,
        id_allocator,
        codec_cache,
        analyzer_cache,
        text_space_states,
        collection_filter_cache,
        time_index_cache,
        vector_by_id_cache,
        segment_registry,
    })
}
