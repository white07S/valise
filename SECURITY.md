# Security Policy

## Supported versions

Valise is pre-1.0. Security fixes land on the latest released version only.
Until 1.0, the format and API may change between minor versions; see
[MIGRATION.md](MIGRATION.md).

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/white07S/valise/security/advisories/new)
for this repository. That opens a private advisory visible only to the
maintainers.

Please include:

- A description of the issue and why you believe it is a security problem.
- Steps to reproduce, ideally a minimal capsule file or code sample.
- The version, platform, and filesystem involved.
- Any suggested fix, if you have one.

You can expect an acknowledgement within a week. Because this is a
small project, please treat that as a realistic estimate rather than a
guaranteed service level.

## What is in scope

Valise is an embedded library: it runs in your process, with your
privileges, on files you point it at. The interesting boundary is
therefore **untrusted capsule files**. In scope:

- Memory-safety failures (out-of-bounds reads/writes, use-after-free)
  triggered by a malformed or hostile `.vls` file.
- Panics or unbounded allocation reachable from parsing a capsule, where a
  clean `Err` is the correct behavior.
- Reads that escape the bounds declared in a segment, TOC, or header.
- Silent corruption: returning bytes that were never committed, or
  surviving a checksum mismatch without an error.
- Path traversal or unexpected file creation outside the capsule path.

## What is out of scope

- **Encryption and signing.** Valise does not encrypt or sign capsules. A
  `.vls` file is readable by anyone who can read the file. Use filesystem
  or disk encryption if you need confidentiality. `Error::Signature` exists
  in the error enum as a reserved variant and is not currently produced.
- **Hard delete.** `delete` writes a tombstone and `compact` reclaims the
  bytes, but neither is a compliance-grade erase primitive. Assume deleted
  payload bytes may remain recoverable from the file until compaction, and
  possibly from the underlying storage afterward.
- **Denial of service from your own data.** Feeding a very large corpus and
  running out of memory or disk is a capacity-planning problem.
- Vulnerabilities in dependencies — report those upstream, though we
  appreciate a heads-up so the version can be bumped.

## Hardening notes

- Treat capsule files from third parties the way you would treat any other
  untrusted input format. There is no sandbox between a capsule and your
  process.
- Segment payloads carry BLAKE3 checksums and the TOC is self-checksummed,
  so accidental corruption is detected. These are integrity checks, not
  authentication: an attacker who can rewrite the file can rewrite the
  checksums with it.
