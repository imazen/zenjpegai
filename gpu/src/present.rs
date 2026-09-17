//! Getting a decoded picture onto a surface (or back to the CPU as RGBA) without touching the
//! float planes on the CPU.

use crate::context::GpuContext;
use crate::error::{GpuError, Result};

const BLIT_WGSL: &str = "
@group(0) @binding(0) var src: texture_2d<f32>;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
  // One triangle covering the target.
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return V(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y));
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> {
  let size = vec2<f32>(textureDimensions(src));
  let at = vec2<i32>(clamp(v.uv * size, vec2<f32>(0.0), size - vec2<f32>(1.0)));
  return textureLoad(src, at, 0);
}
";

/// Draws an `rgba8unorm` picture texture ([`crate::GpuDecoded::to_rgba_texture`]) onto a render
/// target of a given format, e.g. the current texture of a canvas surface. Texels are fetched,
/// not filtered: size the canvas to the picture (CSS scales it) for a 1:1 copy.
pub struct Blitter {
    pipeline: wgpu::RenderPipeline,
}

impl Blitter {
    /// `target` is the surface's format (`surface.get_capabilities(adapter).formats[0]`, in the
    /// browser usually `Bgra8Unorm` or `Rgba8Unorm`).
    pub fn new(ctx: &GpuContext, target: wgpu::TextureFormat) -> Self {
        let dev = ctx.device();
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zenjpegai blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("zenjpegai blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(target.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline }
    }

    /// Record and submit the draw of `picture` onto `target`.
    pub fn blit(&self, ctx: &GpuContext, picture: &wgpu::Texture, target: &wgpu::TextureView) {
        let dev = ctx.device();
        let view = picture.create_view(&Default::default());
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zenjpegai blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue().submit([enc.finish()]);
    }
}

/// Read an `rgba8unorm` texture (with `COPY_SRC` usage) back as tightly packed RGBA bytes, e.g.
/// for `ImageData` / `createImageBitmap` when the page has no WebGPU canvas.
pub async fn read_rgba8(ctx: &GpuContext, tex: &wgpu::Texture) -> Result<Vec<u8>> {
    let (w, h) = (tex.width() as usize, tex.height() as usize);
    let stride = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
    let bytes = (stride * h) as u64;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("zenjpegai rgba readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: None,
            },
        },
        tex.size(),
    );
    ctx.queue().submit([enc.finish()]);
    ctx.map_read(&staging, bytes).await?;
    let view = staging
        .slice(0..bytes)
        .get_mapped_range()
        .map_err(|e| GpuError::Device(e.to_string()))?;
    let mut out = Vec::with_capacity(w * h * 4);
    for row in 0..h {
        out.extend_from_slice(&view[row * stride..][..w * 4]);
    }
    Ok(out)
}
