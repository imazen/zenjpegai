# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->

### Added
- Repository skeleton.
- `bitio`, `container`: MSB-first bit IO, Exp-Golomb, marker/region/thread container parsing and writing.
- `weights`: reader for the upstream PyTorch `.pth` checkpoints (no Python needed; the pickle interpreter is data-only).
- `mans`: me-tANS entropy coder (tables, decoder, encoder), bit-exact against the reference C++ extension.
