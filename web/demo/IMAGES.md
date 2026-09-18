# Demo images — provenance

8 images from the [imazen-26 corpus](https://github.com/imazen/imazen-26)
(`~/work/zen/imazen-26`, PNG-v3 SDR derivatives — no EXIF/GPS), chosen for content diversity
(photo, landscape, food, artwork reproduction, born-digital document, bilevel scan, synthetic
line art, AI-generated flat graphic). Downscaled to ~1 MP (Lanczos, `zenresize`) with
`web/scripts/prep-demo-images` (zenpng decode/encode + zenresize; Imazen tooling only, per
`CLAUDE.md`). Encoded at two rates (target 0.25 and 0.75 bpp) with the **reference** JPEG AI
encoder (`jpeg-ai-reference-software`, `python -m src.reco.coders.encoder --set_target_bpp {25,75}
--cfg cfg/tools_off.json cfg/profiles/base.json -target_device cpu`; every stream picked its own
`model_id` automatically — 0..3 all appear across the 16 streams).

None of these bytes are in git: streams (`.jai`), the two model bundle parts each stream's
`model_id` needs (`m<id>_common.zjb` + `m<id>_bop.zjb`, common part shared across every stream
that reuses a model), and the reference-decoder PNGs used as the pixel-parity oracle in the
Playwright suite are published as assets of the `demo-assets-v1` GitHub release (prerelease).
`manifest.json` in that release ties it together: per-image slug, source category/license,
and per-rate bytes/width/height/`modelId`.

| slug | source (imazen-26) | category | license |
| --- | --- | --- | --- |
| `car` | `1000-lilith-photos-general/1000_..._4032x3024.jpg` | everyday photo | PD-own (lilith) |
| `mountain` | `1400-lilith-nature/1400_..._4608x3456.jpg` | landscape | PD-own (lilith) |
| `dumplings` | `1600-lilith-food/1600_..._4032x3024.jpg` | food | PD-own (lilith) |
| `artwork` | `3000-art-institute-of-chicago-photos/3000_aic_for-sunday-s-dinner_111377_1269x2250.jpg` | open-access artwork reproduction | CC0 (Art Institute of Chicago) |
| `brochure` | `5000-national-park-service-brochures/color/5000_nps_bibe-2023-wild-and-scenic-rivers_color_p01_4961x7016.png` | born-digital document page | PD-USGov (NPS) |
| `patent-scan` | `6000-lilith-scans-public-patents/lynn_conway_us5046022_1bitoriginal/6000_..._p001_2320x3408.png` | bilevel scan (rescanned colour render) | PD (USPTO) |
| `line-plot` | `7000-lilith-plots/aliased-line-patterns/7000_plots_line-00056-s6bcec02a_1024x1024.png` | synthetic hard-edge line art | PD-own (lilith, generated) |
| `clipart` | `9000-lilith-ai-clipart/9000_gen_clipart_avocado-half_1024x1536.png` | AI-generated flat/transparent graphic | PD-own (lilith, AI-generated) |

Full per-category license text and attribution: imazen-26 `README.md`
("Licensing status"). Model weights inside every `.zjb`: upstream JPEG AI reference software,
BSD-licensed — notice text embedded in each bundle (`PackedBundle::notice()`) and shown on the
demo page and in `upstream-notices/LICENSE`.

Regenerate (models dir default `$ZENJPEGAI_REF/models`, override with `--models`):

```
web/scripts/prep-demo-images/target/release/prep-demo-images ~/tmp/zjw/demo-1mp ~/tmp/zjw/demo-src/*.png
cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate && export PYTHONPATH=.
for f in ~/tmp/zjw/demo-1mp/*.png; do for bpp in 25 75; do
  python -m src.reco.coders.encoder "$f" out.bits --set_target_bpp $bpp \
    --cfg cfg/tools_off.json cfg/profiles/base.json -target_device cpu
done; done
target/release/zenjpegai pack-models --models $ZENJPEGAI_REF/models --model <0..3> --only common --out m<id>_common.zjb
target/release/zenjpegai pack-models --models $ZENJPEGAI_REF/models --model <0..3> --op bop --only synthesis --out m<id>_bop.zjb
target/release/zenjpegai decode out.bits out.native.png --models m<id>_bop.zjb   # after merging common+bop in one PackedBundle, or use a full --only-less pack
gh release create demo-assets-v1 --prerelease --title "Demo assets v1" --notes "..." \
  <slug>_bpp{25,75}.jai <slug>_bpp{25,75}.native.png m{0,1,2,3}_{common,bop}.zjb manifest.json
```
