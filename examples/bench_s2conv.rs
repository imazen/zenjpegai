use zenjpegai::nn::Conv2d;
use zenjpegai::nn::fast::{BTensor, Engine, PackedConv};
use zenjpegai::tensor::Tensor;
fn main() {
    let (in_ch, out_ch, h, w) = (128usize, 128usize, 512usize, 352usize);
    let conv = Conv2d::new(
        in_ch,
        out_ch,
        (3, 3),
        2,
        (1, 1),
        1,
        vec![0.01; out_ch * in_ch * 9],
        Some(vec![0.0; out_ch]),
    )
    .unwrap();
    let x = Tensor::from_vec(in_ch, h, w, vec![0.5; in_ch * h * w]).unwrap();
    for threads in [false, true] {
        let eng = Engine::with(zenjpegai::nn::fast::Tier::detect(), threads);
        let v = eng.tier.block();
        let packed = PackedConv::new(&conv, v, [0; 4]).unwrap();
        let bx = BTensor::from_planar(&x, v).unwrap();
        let iters = if threads { 800 } else { 200 };
        let t0 = std::time::Instant::now();
        let mut acc = 0.0;
        for _ in 0..iters {
            acc += packed.forward(&eng, &bx).unwrap().data[123];
        }
        eprintln!(
            "threads={threads}: {:.2} ms/conv (acc {acc})",
            t0.elapsed().as_secs_f64() * 1e3 / iters as f64
        );
    }
}
