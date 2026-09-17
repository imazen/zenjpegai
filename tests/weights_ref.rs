//! Checkpoint loader against the upstream `.pth` files. Expected hashes are FNV-1a 64 of the
//! tensors' little-endian bytes as loaded by `torch.load` (torch 1.10.2) at upstream b9e573f.
#![cfg(feature = "reference-tests")]

mod common;
use common::{fnv1a64, read_model};
use zenjpegai::weights::{Checkpoint, DType};

#[test]
fn common_int_y_012() {
    let file = read_model("VM_common_int/Y_0.012.pth");
    let ck = Checkpoint::parse(&file).unwrap();
    assert_eq!(ck.tensor_names().count(), 72);
    assert_eq!(ck.int("epoch"), Some(3));

    let t = ck.i32("hyper_entropy.freqs_int").unwrap();
    assert_eq!(t.shape, [160, 63]);
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0x0c56_aa1d_d7f9_fc54
    );

    let t = ck.i8("hyper_scale_decoder.conv1.weight").unwrap();
    assert_eq!(t.shape, [160, 160, 1, 1]);
    assert_eq!(
        fnv1a64(t.data.iter().map(|&v| v as u8)),
        0xdd71_96ed_9560_e22e
    );

    let t = ck.i32("hyper_scale_decoder.pointwise.bias").unwrap();
    assert_eq!(t.shape, [2560]);
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0xf0b8_47ac_b3a8_16d9
    );

    let t = ck
        .i8("hyper_scale_decoder.depthwise.per_channel_shifts")
        .unwrap();
    assert_eq!(t.shape, [1, 160, 1, 1]);
    assert_eq!(
        fnv1a64(t.data.iter().map(|&v| v as u8)),
        0x2915_3e63_3496_dd7d
    );

    let t = ck.f32("hyper_decoder.conv4.weight").unwrap();
    assert_eq!(t.shape, [640, 160, 3, 3]);
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0x6f50_c331_d7ac_ce3f
    );

    let t = ck.bool("hyper_scale_decoder.conv1.is_quantized").unwrap();
    assert_eq!(
        (t.shape.as_slice(), t.data.as_slice()),
        (&[1usize][..], &[true][..])
    );

    let t = ck.i64("entropy._quantized_cdf").unwrap();
    assert_eq!(t.shape, [35, 1223]);
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0x89fb_0acc_2853_b6a6
    );

    assert_eq!(ck.info("vr_vec.c").unwrap().dtype, DType::F32);
    assert!(
        ck.f32("hyper_entropy.freqs_int").is_err(),
        "dtype mismatch must be an error"
    );
    assert!(ck.f32("no.such.tensor").is_err());
}

/// This file also carries a pickled optimizer state (nested dicts, lists, floats); it must parse
/// and the model tensors must be unaffected.
#[test]
fn common_int_y_075_with_optimizer_state() {
    let file = read_model("VM_common_int/Y_0.075.pth");
    let ck = Checkpoint::parse(&file).unwrap();
    assert_eq!(ck.tensor_names().count(), 72);
    assert_eq!(ck.int("epoch"), Some(2));
    let t = ck.i32("hyper_entropy.freqs_int").unwrap();
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0xd94f_4487_af05_ba15
    );
    let t = ck.f32("context.MCM.3.conv.0.weight").unwrap();
    assert_eq!(t.shape, [160, 480, 1, 1]);
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0xba9c_8342_585d_d04a
    );
}

#[test]
fn bop_decoder_uv_012() {
    let file = read_model("VM_bop/decoder_UV_0.012.pth");
    let ck = Checkpoint::parse(&file).unwrap();
    assert_eq!(ck.tensor_names().count(), 11);
    let t = ck.f32("conv2_t.weight").unwrap();
    assert_eq!(t.shape, [144, 64, 4, 4]);
    assert_eq!(
        fnv1a64(t.data.iter().flat_map(|v| v.to_le_bytes())),
        0x8cc5_6973_19ca_f450
    );
}

/// Every checkpoint the pipeline references must at least parse.
#[test]
fn every_upstream_checkpoint_parses() {
    let models = common::ref_root().join("models");
    let mut n = 0;
    for dir in [
        "VM_common_int",
        "VM_bop",
        "VM_hop",
        "VM_sop",
        "eICCI_bophop_2d020448_20240229",
    ] {
        for entry in std::fs::read_dir(models.join(dir)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "pth") {
                let file = std::fs::read(&path).unwrap();
                let ck =
                    Checkpoint::parse(&file).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                assert!(ck.tensor_names().count() > 0, "{}", path.display());
                n += 1;
            }
        }
    }
    assert_eq!(
        n,
        8 + 16 + 16 + 8 + 20,
        "unexpected number of upstream checkpoints"
    );
}
