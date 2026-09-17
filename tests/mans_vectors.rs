//! me-tANS parity against the reference software's C++ extension.
//!
//! `tests/vectors/mans/*.bin` are produced by `scripts/ref_vectors/gen_mans_vectors.py`, which
//! drives the reference `ANSEncoder` directly. For every vector we require:
//!   1. decoding the reference payload yields the original symbols, and
//!   2. encoding the symbols yields the reference payload byte for byte (thread sizes included).

use zenjpegai::mans::{AnsTables, MAX_Z};

struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        a
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
}

enum Call {
    Residual {
        sigma: Vec<u8>,
        mask: Vec<bool>,
        values: Vec<i16>,
    },
    Z {
        cdfs: Vec<[u8; MAX_Z]>,
        size: usize,
        symbols: Vec<u8>,
    },
}

struct Vector {
    calls: Vec<Call>,
    thread_sizes: Vec<usize>,
    payload: Vec<u8>,
}

fn load(name: &str) -> Vector {
    let path = format!("{}/tests/vectors/mans/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut c = Cursor(&bytes);
    assert_eq!(c.take(4), b"ZJMV");
    assert_eq!(c.u32(), 1, "version");
    let kind = c.u32();
    let num_threads = c.u32() as usize;
    let num_calls = c.u32() as usize;
    let mut calls = Vec::new();
    for _ in 0..num_calls {
        if kind == 0 {
            let len = c.u32() as usize;
            let sigma = c.take(len).to_vec();
            let mask = c.take(len).iter().map(|&b| b != 0).collect();
            let values = c
                .take(len * 2)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b))
                .collect();
            calls.push(Call::Residual {
                sigma,
                mask,
                values,
            });
        } else {
            let channels = c.u32() as usize;
            let size = c.u32() as usize;
            let cdfs = c.take(channels * MAX_Z).as_chunks::<MAX_Z>().0.to_vec();
            let symbols = c.take(channels * size).to_vec();
            calls.push(Call::Z {
                cdfs,
                size,
                symbols,
            });
        }
    }
    assert_eq!(c.u32() as usize, num_threads);
    let thread_sizes: Vec<usize> = (0..num_threads).map(|_| c.u32() as usize).collect();
    let payload_len = c.u32() as usize;
    let payload = c.take(payload_len).to_vec();
    assert!(c.0.is_empty(), "trailing bytes in {name}");
    assert_eq!(thread_sizes.iter().sum::<usize>(), payload.len());
    Vector {
        calls,
        thread_sizes,
        payload,
    }
}

fn split<'a>(payload: &'a [u8], sizes: &[usize]) -> Vec<&'a [u8]> {
    let mut rest = payload;
    sizes
        .iter()
        .map(|&s| {
            let (a, b) = rest.split_at(s);
            rest = b;
            a
        })
        .collect()
}

fn check(name: &str) {
    let v = load(name);
    let tables = AnsTables::new();

    // 1. decode the reference payload
    let threads = split(&v.payload, &v.thread_sizes);
    let mut dec = tables.decoder(&threads).unwrap();
    for (i, call) in v.calls.iter().enumerate() {
        match call {
            Call::Residual {
                sigma,
                mask,
                values,
            } => {
                let mut out = vec![0i16; values.len()];
                dec.decode_residual(sigma, mask, &mut out).unwrap();
                for k in 0..values.len() {
                    if mask[k] {
                        assert_eq!(out[k], values[k], "{name}: call {i} symbol {k}");
                    } else {
                        assert_eq!(out[k], 0, "{name}: call {i} skipped symbol {k} was written");
                    }
                }
            }
            Call::Z {
                cdfs,
                size,
                symbols,
            } => {
                let mut out = vec![0u8; symbols.len()];
                dec.decode_z(cdfs, *size, &mut out).unwrap();
                assert_eq!(&out, symbols, "{name}: call {i}");
            }
        }
    }

    // 2. encode and compare bytes; calls go in reverse of decode order
    let mut enc = tables.encoder(v.thread_sizes.len()).unwrap();
    for call in v.calls.iter().rev() {
        match call {
            Call::Residual {
                sigma,
                mask,
                values,
            } => enc.encode_residual(sigma, mask, values).unwrap(),
            Call::Z {
                cdfs,
                size,
                symbols,
            } => enc.encode_z(cdfs, *size, symbols).unwrap(),
        }
    }
    let ours = enc.finish();
    let our_sizes: Vec<usize> = ours.iter().map(|t| t.len()).collect();
    assert_eq!(our_sizes, v.thread_sizes, "{name}: thread sizes");
    assert_eq!(ours.concat(), v.payload, "{name}: payload bytes");
}

macro_rules! vector_tests {
    ($($fn_name:ident => $file:literal),* $(,)?) => {
        $( #[test] fn $fn_name() { check($file); } )*
    };
}

vector_tests! {
    residual_1_thread => "residual_t1.bin",
    residual_2_threads => "residual_t2.bin",
    residual_4_threads => "residual_t4.bin",
    residual_8_threads => "residual_t8.bin",
    residual_16_threads => "residual_t16.bin",
    z_1_thread => "z_t1.bin",
    z_2_threads => "z_t2.bin",
    z_4_threads => "z_t4.bin",
    z_8_threads => "z_t8.bin",
    z_16_threads => "z_t16.bin",
}
