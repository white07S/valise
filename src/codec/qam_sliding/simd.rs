//! Per-arch SIMD kernels for the qam_sliding integer raw-dot path.
//!
//! Layout mirrors `codec/qam_lloyd_max/simd/` (the reference shape — see the
//! "Per-arch SIMD" rule in `SKILL.md`): each architecture's kernel lives in a sibling
//! file named purely by arch (`aarch64.rs`, `x86_64.rs`); the `simd/`
//! directory already namespaces them, so the kernel *functions* keep
//! their descriptive `raw_dot_*` names while the *files* do not repeat it.
//!
//! The `#[cfg(target_arch …)]` runtime dispatch lives in the parent
//! `qam_sliding.rs` (`QamSlidingEngine::raw_dot_int`), next to the scalar
//! fallback it falls through to.

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;
#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;
