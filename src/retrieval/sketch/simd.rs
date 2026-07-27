//! Per-arch SIMD hamming kernels for the sign-sketch candidate scan.
//!
//! Layout mirrors `codec/qam_lloyd_max/simd/` (the reference shape — see the
//! "Per-arch SIMD" rule in `SKILL.md`): each architecture's kernel lives in a sibling
//! file named purely by arch (`aarch64.rs`, `x86_64.rs`); the `simd/`
//! directory already namespaces them, so the kernel *functions* keep
//! their descriptive `hamming_*` names while the *files* do not repeat it.
//!
//! The `#[cfg(target_arch …)]` dispatch (`hamming`, `scan_candidates`)
//! and the bench re-exposure live in the parent `sketch.rs`. The x86_64
//! parity tests live in `simd/x86_64_tests.rs` but are declared as a
//! child of `sketch` via `#[path]` so they keep reaching `hamming_scalar`.

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;
#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

// x86_64 kernel parity tests live beside the kernel they exercise. Declared
// here (not in `sketch.rs`) so the file path matches the module nesting
// without a `#[path]` override; the tests reach `hamming_scalar` via
// `super::super` (the parent `sketch` module).
#[cfg(all(test, target_arch = "x86_64"))]
mod x86_64_tests;
