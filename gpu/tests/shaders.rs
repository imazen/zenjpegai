//! Every kernel generator's emitted WGSL must parse — the first thing a real
//! device would reject. This runs anywhere (no adapter, no models): `naga`'s
//! WGSL front-end is the same parser wgpu uses, so a shader that fails here
//! would fail `create_shader_module` on hardware.
use zenjpegai_gpu::kernels::*;

fn parses(src: &str, name: &str) {
    naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
        panic!("{name}: WGSL did not parse: {e:?}\n--- source ---\n{src}");
    });
}

#[test]
fn conv_shaders_parse() {
    for w16 in [false, true] {
        for (ob, px, kh, stride, tile_ich) in [
            (1u32, 1u32, 3u32, 1u32, 0u32),
            (2, 2, 3, 1, 0),
            (4, 1, 1, 1, 0),
            (2, 1, 3, 2, 0),
            // conv_tiled asserts px == 1 && icg4.is_multiple_of(tile_ich).
            (4, 1, 3, 1, 4),
            (1, 1, 3, 1, 4),
        ] {
            for act in [Act::None, Act::Relu] {
                for res in [None, Some(Res::Add), Some(Res::Gate)] {
                    for pre_relu6 in [false, true] {
                        let v = ConvVariant {
                            kh,
                            kw: kh,
                            stride,
                            ob,
                            wg: WG,
                            px,
                            act,
                            pre_relu6,
                            res,
                            icg4: 4,
                            ocg4: 4,
                            tile_ich,
                            w16,
                        };
                        if tile_ich > 0 {
                            parses(&conv_tiled(&v, tile_ich), &format!("conv_tiled {v:?}"));
                        } else {
                            parses(&conv(&v), &format!("conv {v:?}"));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn conv_transpose_parses() {
    for w16 in [false, true] {
        for ob in [1u32, 2, 4] {
            parses(&conv_transpose(4, 2, ob, 16, w16), "conv_transpose");
        }
    }
}

#[test]
fn pointwise_and_pad_parse() {
    parses(&depthwise3x3(4), "depthwise3x3");
    parses(&copy_channels(), "copy_channels");
    parses(&pad(), "pad");
    parses(&elu_gate(), "elu_gate");
    parses(&pixel_shuffle(2), "pixel_shuffle");
    parses(&emit(2), "emit");
    parses(&layer_norm(), "layer_norm");
    parses(&gram_chunks(), "gram_chunks");
    parses(&gram_reduce(), "gram_reduce");
    parses(&attention_softmax(), "attention_softmax");
    parses(&attention_apply(), "attention_apply");
    parses(&emit_output(), "emit_output");
    parses(&yuv_to_rgba(), "yuv_to_rgba");
    for op in [
        Pointwise::Relu,
        Pointwise::Add,
        Pointwise::Gate,
        Pointwise::SigmoidMulAdd,
    ] {
        parses(&pointwise(op), &format!("pointwise {op:?}"));
    }
}
