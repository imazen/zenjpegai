//! Adversarial 8-bit RGB sources for the f16 model-weight study (task `f16w`).
//!
//! Pure pixel generation, no dependencies: this file is `#[path]`-included by
//! `gen_synthetic_inputs.rs` (writes the PNGs under the reference inputs tree) and by
//! `tests/f16_weights.rs` (encodes the same pixels in memory), so the streams tested are
//! exactly the files shipped.
//!
//! The patterns push the networks toward the places where weight rounding matters most:
//! near-zero latents (flat fields and ±1 LSB noise, where activations sit at subnormal-f16
//! scale), saturated transform extremes (pure primaries), and the largest spatial gradients
//! the source format can express (impulses, maximum-contrast edges, 1 px checkerboards).

/// A generated source: interleaved RGB8.
pub struct SyntheticImage {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` bytes, RGB interleaved.
    pub rgb: Vec<u8>,
}

/// Flat-field levels the study covers (8-bit).
const GREYS: [u8; 7] = [0, 1, 2, 127, 128, 254, 255];

fn flat(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut out = vec![0; w * h * 3];
    for p in out.as_chunks_mut::<3>().0.iter_mut() {
        *p = rgb;
    }
    out
}

/// xorshift64* — deterministic ±1 LSB dither, no `rand` dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32
    }
}

/// Flat field of `v` with per-pixel noise in `{-1, 0, +1}`, clamped to 0..=255.
fn grey_noise(w: usize, h: usize, v: u8) -> Vec<u8> {
    let mut out = flat(w, h, [v, v, v]);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ u64::from(v));
    for p in out.as_chunks_mut::<3>().0.iter_mut() {
        // One dither per pixel, applied to all three channels: this stays a grey source.
        let d = (rng.next() % 3) as i32 - 1;
        let s = (v as i32 + d).clamp(0, 255) as u8;
        *p = [s, s, s];
    }
    out
}

/// Black field with one white pixel dead centre: the largest impulse the format expresses.
fn impulse(w: usize, h: usize) -> Vec<u8> {
    let mut out = flat(w, h, [0, 0, 0]);
    let c = (h / 2) * w + w / 2;
    out[3 * c..3 * c + 3].copy_from_slice(&[255, 255, 255]);
    out
}

/// Maximum-contrast vertical step: left half black, right half white.
fn step_edge(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 0 } else { 255 };
            out.extend_from_slice(&[v, v, v]);
        }
    }
    out
}

/// Black/white checkerboard of `cell`-pixel squares.
fn checker(w: usize, h: usize, cell: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let v = if (x / cell + y / cell).is_multiple_of(2) {
                0
            } else {
                255
            };
            out.extend_from_slice(&[v, v, v]);
        }
    }
    out
}

/// Ramp 0..255 along x (repeated down the picture).
fn ramp_h(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let v = (x * 255 / (w - 1).max(1)) as u8;
            out.extend_from_slice(&[v, v, v]);
        }
    }
    out
}

/// Ramp 0..255 along y.
fn ramp_v(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let v = (y * 255 / (h - 1).max(1)) as u8;
        for _x in 0..w {
            out.extend_from_slice(&[v, v, v]);
        }
    }
    out
}

/// The named 256x256 sources (25 images).
pub fn images_256() -> Vec<SyntheticImage> {
    const W: usize = 256;
    const H: usize = 256;
    let mut out = Vec::new();
    let mut push = |name: &'static str, rgb: Vec<u8>| {
        out.push(SyntheticImage {
            name,
            width: W,
            height: H,
            rgb,
        })
    };
    for &v in &GREYS {
        push(
            match v {
                0 => "grey000",
                1 => "grey001",
                2 => "grey002",
                127 => "grey127",
                128 => "grey128",
                254 => "grey254",
                _ => "grey255",
            },
            flat(W, H, [v, v, v]),
        );
    }
    for &v in &GREYS {
        push(
            match v {
                0 => "noise000",
                1 => "noise001",
                2 => "noise002",
                127 => "noise127",
                128 => "noise128",
                254 => "noise254",
                _ => "noise255",
            },
            grey_noise(W, H, v),
        );
    }
    push("impulse", impulse(W, H));
    push("step_edge", step_edge(W, H));
    for (name, rgb) in [
        ("red", [255, 0, 0]),
        ("green", [0, 255, 0]),
        ("blue", [0, 0, 255]),
        ("cyan", [0, 255, 255]),
        ("magenta", [255, 0, 255]),
        ("yellow", [255, 255, 0]),
    ] {
        push(name, flat(W, H, rgb));
    }
    push("checker1", checker(W, H, 1));
    push("checker2", checker(W, H, 2));
    push("ramp_h", ramp_h(W, H));
    push("ramp_v", ramp_v(W, H));
    out
}

/// One 1024x1024 mosaic combining every pattern in an 8x8 grid of 128x128 cells; cells past
/// the 25 named ones are flat mid-grey (another near-flat case).
pub fn mosaic_1024() -> SyntheticImage {
    const CELL: usize = 128;
    const GRID: usize = 8;
    let (w, h) = (GRID * CELL, GRID * CELL);
    let mut out = flat(w, h, [128, 128, 128]);
    let cells: Vec<Vec<u8>> = images_256()
        .iter()
        .map(|i| {
            // Regenerate each pattern at cell size.
            match i.name {
                n if n.starts_with("grey") => {
                    let v = n[4..].parse::<u8>().unwrap();
                    flat(CELL, CELL, [v, v, v])
                }
                n if n.starts_with("noise") => {
                    let v = n[5..].parse::<u8>().unwrap();
                    grey_noise(CELL, CELL, v)
                }
                "impulse" => impulse(CELL, CELL),
                "step_edge" => step_edge(CELL, CELL),
                "red" => flat(CELL, CELL, [255, 0, 0]),
                "green" => flat(CELL, CELL, [0, 255, 0]),
                "blue" => flat(CELL, CELL, [0, 0, 255]),
                "cyan" => flat(CELL, CELL, [0, 255, 255]),
                "magenta" => flat(CELL, CELL, [255, 0, 255]),
                "yellow" => flat(CELL, CELL, [255, 255, 0]),
                "checker1" => checker(CELL, CELL, 1),
                "checker2" => checker(CELL, CELL, 2),
                "ramp_h" => ramp_h(CELL, CELL),
                "ramp_v" => ramp_v(CELL, CELL),
                _ => flat(CELL, CELL, [128, 128, 128]),
            }
        })
        .collect();
    for (i, cell) in cells.iter().enumerate() {
        let (cx, cy) = (i % GRID * CELL, i / GRID * CELL);
        for y in 0..CELL {
            let dst = ((cy + y) * w + cx) * 3;
            out[dst..dst + CELL * 3].copy_from_slice(&cell[y * CELL * 3..(y + 1) * CELL * 3]);
        }
    }
    SyntheticImage {
        name: "mosaic1024",
        width: w,
        height: h,
        rgb: out,
    }
}

/// Every source the study encodes: the 256x256 set, then the mosaic.
pub fn all() -> Vec<SyntheticImage> {
    let mut out = images_256();
    out.push(mosaic_1024());
    out
}
