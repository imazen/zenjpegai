//! Writes a black RGB PNG with white rectangles: a region-of-interest mask for the reference
//! encoder's quality-map tool (`qp_map_type = 3`, `ROI_map_in_file`).
//!
//! `cargo run --features cli --example make_roi_mask -- W H OUT.png X,Y,W,H [X,Y,W,H ...]`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (w, h): (usize, usize) = (args[0].parse().unwrap(), args[1].parse().unwrap());
    let mut px = vec![rgb::Rgb { r: 0u8, g: 0, b: 0 }; w * h];
    for rect in &args[3..] {
        let v: Vec<usize> = rect.split(',').map(|s| s.parse().unwrap()).collect();
        for y in v[1]..(v[1] + v[3]).min(h) {
            for x in v[0]..(v[0] + v[2]).min(w) {
                px[y * w + x] = rgb::Rgb {
                    r: 255,
                    g: 255,
                    b: 255,
                };
            }
        }
    }
    let png = zenpng::encode_rgb8(
        imgref::ImgRef::new(&px, w, h),
        None,
        &zenpng::EncodeConfig::default(),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .unwrap();
    std::fs::write(&args[2], png).unwrap();
}
