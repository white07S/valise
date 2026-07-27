//! `ValiseFile` schema-object registration.

use super::*;

impl ValiseFile {
    /// Register an embedding space.
    ///
    /// Validation:
    /// - `dimension` ≤ `CreateContractV1.max_dim`.
    /// - `dtype` ∈ `CreateContractV1.allowed_dtypes`.
    /// - non-f8: `primary_codec_id` MUST be `Some` and resolve to a
    ///   `CodecFamily::QamLloydMax` codec.
    /// - f8: `primary_codec_id` MUST be `None` (raw bytes; no codec
    ///   encoding at ingest).
    /// - `secondary_codec_id` MUST be `None` (unused since v2.3).
    pub fn register_embedding_space(
        &mut self,
        spec: EmbeddingSpaceSpec,
    ) -> Result<EmbeddingSpaceId> {
        self.ensure_write()?;

        // ---- Contract gates (plan §4.4 + §7.3) -----------------------
        if spec.dimension > self.create_contract.max_dim {
            return Err(Error::Unsupported(format!(
                "register_embedding_space: dimension {} exceeds contract.max_dim {}",
                spec.dimension, self.create_contract.max_dim
            )));
        }
        let allowed = DtypeSet::from_bits_truncate(self.create_contract.allowed_dtypes);
        if !allowed.contains(spec.dtype.as_set_bit()) {
            return Err(Error::Unsupported(format!(
                "register_embedding_space: dtype {:?} not in contract.allowed_dtypes",
                spec.dtype
            )));
        }

        // ---- Dtype × codec slot rules (plan §7.3 matrix) -------------
        if spec.dtype.is_f8() && spec.primary_codec_id.is_some() {
            return Err(Error::Format(
                "register_embedding_space: f8 spaces must not register a primary codec".into(),
            ));
        }
        if !spec.dtype.is_f8() && spec.primary_codec_id.is_none() {
            return Err(Error::Format(
                "register_embedding_space: non-f8 spaces must register a primary codec".into(),
            ));
        }

        // ---- Codec family checks ------------------------------------
        if let Some(primary_id) = spec.primary_codec_id {
            let codec_desc = self
                .catalog
                .codecs
                .iter()
                .find(|c| c.codec_id == primary_id)
                .ok_or_else(|| {
                    Error::Format(format!(
                        "register_embedding_space: unknown primary_codec_id {}",
                        primary_id.0
                    ))
                })?;
            // Supported primary codec families: QamLloydMax and (since
            // v2.4) Upq.
            match codec_desc.family {
                CodecFamily::QamLloydMax | CodecFamily::Upq => {}
                CodecFamily::_LegacyInt4PcaZero => {
                    return Err(Error::Unsupported(
                        "register_embedding_space: Int4PcaZero codec family is no longer \
                         supported (removed in v2.2). Re-register the space with a QAM codec."
                            .into(),
                    ));
                }
            }
            // Reject dim mismatch with the codec's actual dim. Pulled
            // from the cache rather than re-decoding params.
            let codec = self.codec_cache.get(&primary_id).ok_or_else(|| {
                Error::Integrity(format!(
                    "register_embedding_space: codec {} not in cache",
                    primary_id.0
                ))
            })?;
            debug_assert_eq!(
                codec.family(),
                codec_desc.family,
                "BUG: cached codec family diverges from its catalog descriptor"
            );
            if codec.dimension() != spec.dimension {
                return Err(Error::Format(format!(
                    "register_embedding_space: primary codec dim {} != spec dim {}",
                    codec.dimension(),
                    spec.dimension
                )));
            }
        }
        // `secondary_codec_id` was only ever consumed by the CSR vote-index
        // build, which was removed in v2.3 (vote → sign-sketch; see
        // docs/VECTOR_SEARCH.md). Nothing reads it now, so accepting a value
        // would silently do nothing. Reject it rather than persist a no-op.
        // The persisted field stays (always `None`) to avoid a format break.
        if spec.secondary_codec_id.is_some() {
            return Err(Error::Unsupported(
                "register_embedding_space: secondary_codec_id is unused since the vote-index \
                 removal (v2.3); pass None"
                    .into(),
            ));
        }

        let embedding_space_id = self.id_allocator.allocate_embedding_space_id()?;
        let desc = EmbeddingSpaceDesc {
            embedding_space_id,
            provider: spec.provider,
            model: spec.model,
            dimension: spec.dimension,
            metric: spec.metric,
            normalized: spec.normalized,
            dtype: spec.dtype,
            primary_codec_id: spec.primary_codec_id,
            // Always None: rejected above (unused since the v2.3 vote-index
            // removal). Field retained for format stability.
            secondary_codec_id: spec.secondary_codec_id,
            // Reserved flag word; no live consumer since vote/ANN burial.
            flags: 0,
        };
        upsert_embedding_space(&mut self.catalog, desc);
        self.dirty_embedding_space_ids.insert(embedding_space_id);
        self.dirty = true;
        Ok(embedding_space_id)
    }

    /// Register an analyzer for use by text spaces (spec §10.4). The
    /// `analyzer_id` field on `desc` is overwritten with a freshly allocated
    /// id; supply any placeholder value (e.g. `AnalyzerId(0)`).
    pub fn register_analyzer(&mut self, desc: AnalyzerDesc) -> Result<AnalyzerId> {
        self.ensure_write()?;
        self.ensure_text_enabled("register_analyzer")?;
        let analyzer_id = self.id_allocator.allocate_analyzer_id()?;
        let desc = AnalyzerDesc {
            analyzer_id,
            ..desc
        };
        upsert_analyzer(&mut self.catalog, desc);
        self.dirty_analyzer_ids.insert(analyzer_id);
        self.dirty = true;
        Ok(analyzer_id)
    }

    /// Register a field schema for use by text spaces (spec §10.5). The
    /// `field_schema_id` on `desc` is overwritten with a freshly allocated id.
    pub fn register_field_schema(&mut self, desc: FieldSchemaDesc) -> Result<FieldSchemaId> {
        self.ensure_write()?;
        self.ensure_text_enabled("register_field_schema")?;
        let field_schema_id = self.id_allocator.allocate_field_schema_id()?;
        let desc = FieldSchemaDesc {
            field_schema_id,
            ..desc
        };
        upsert_field_schema(&mut self.catalog, desc);
        self.dirty_field_schema_ids.insert(field_schema_id);
        self.dirty = true;
        Ok(field_schema_id)
    }

    /// Register a retrieval profile (BM25, TF-IDF, Jaccard, …) per spec §10.6.
    /// The `profile_id` on `desc` is overwritten with a freshly allocated id.
    pub fn register_retrieval_profile(
        &mut self,
        desc: RetrievalProfileDesc,
    ) -> Result<RetrievalProfileId> {
        self.ensure_write()?;
        self.ensure_text_enabled("register_retrieval_profile")?;
        let profile_id = self.id_allocator.allocate_retrieval_profile_id()?;
        let desc = RetrievalProfileDesc { profile_id, ..desc };
        upsert_retrieval_profile(&mut self.catalog, desc);
        self.dirty_retrieval_profile_ids.insert(profile_id);
        self.dirty = true;
        Ok(profile_id)
    }

    /// Register a fusion profile per spec §18.3. The `fusion_profile_id` on
    /// `desc` is overwritten with a freshly allocated id. Weights and
    /// normalization are validated before the catalog upsert.
    pub fn register_fusion_profile(&mut self, desc: FusionProfileDesc) -> Result<FusionProfileId> {
        self.ensure_write()?;
        desc.validate()?;
        let fusion_profile_id = self.id_allocator.allocate_fusion_profile_id()?;
        let desc = FusionProfileDesc {
            fusion_profile_id,
            ..desc
        };
        upsert_fusion_profile(&mut self.catalog, desc);
        self.dirty_fusion_profile_ids.insert(fusion_profile_id);
        self.dirty = true;
        Ok(fusion_profile_id)
    }

    /// Register a text space (spec §10.3). Validates that the referenced
    /// `analyzer_id`, `field_schema_id`, and `default_profile_id` are already
    /// registered before the catalog upsert. The `text_space_id` on
    /// `desc` is overwritten with a freshly allocated id.
    pub fn register_text_space(&mut self, desc: TextSpaceDesc) -> Result<TextSpaceId> {
        self.ensure_write()?;
        self.ensure_text_enabled("register_text_space")?;
        if !self
            .catalog
            .analyzers
            .iter()
            .any(|a| a.analyzer_id == desc.analyzer_id)
        {
            return Err(Error::Format(format!(
                "register_text_space: unknown analyzer_id {}",
                desc.analyzer_id.0
            )));
        }
        if !self
            .catalog
            .field_schemas
            .iter()
            .any(|f| f.field_schema_id == desc.field_schema_id)
        {
            return Err(Error::Format(format!(
                "register_text_space: unknown field_schema_id {}",
                desc.field_schema_id.0
            )));
        }
        if !self
            .catalog
            .retrieval_profiles
            .iter()
            .any(|p| p.profile_id == desc.default_profile_id)
        {
            return Err(Error::Format(format!(
                "register_text_space: unknown default_profile_id {}",
                desc.default_profile_id.0
            )));
        }
        let text_space_id = self.id_allocator.allocate_text_space_id()?;
        let desc = TextSpaceDesc {
            text_space_id,
            ..desc
        };
        // Build the analyzer now so we can cache it; if construction fails
        // (e.g., v1-reserved tokenizer), we surface the error before
        // the catalog upsert.
        let analyzer_desc = self
            .catalog
            .analyzers
            .iter()
            .find(|a| a.analyzer_id == desc.analyzer_id)
            .expect("BUG: analyzer existence checked above");
        let analyzer = Analyzer::from_desc(analyzer_desc)?;

        upsert_text_space(&mut self.catalog, desc);
        self.dirty_text_space_ids.insert(text_space_id);
        self.analyzer_cache.insert(text_space_id, analyzer);
        self.text_space_states
            .insert(text_space_id, TextSpaceState::default());
        self.dirty = true;
        Ok(text_space_id)
    }
}
