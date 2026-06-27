use video_pipeline::{Frame, FrameCtx, Pixel};

pub trait Mixer {
    fn mix(&mut self, frames: Vec<Frame>, ctx: FrameCtx) -> Frame;
}

impl<F: FnMut(Vec<Frame>, FrameCtx) -> Frame> Mixer for F {
    fn mix(&mut self, frames: Vec<Frame>, ctx: FrameCtx) -> Frame {
        self(frames, ctx)
    }
}

pub struct PerPixelMix {
    f: Box<dyn FnMut(&[Pixel], FrameCtx) -> Pixel>,
}

impl PerPixelMix {
    pub fn new(f: impl FnMut(&[Pixel], FrameCtx) -> Pixel + 'static) -> Self {
        Self { f: Box::new(f) }
    }

    pub fn new_blend(ratio: &[f32]) -> Self {
        let ratio = ratio.to_vec();
        Self::new(move |pixels, _ctx| {
            let mut result = [0f32; 4];
            for (i, pixel) in pixels.iter().enumerate() {
                let [r, g, b, a] = pixel.to_array();
                result[0] += r as f32 * ratio[i];
                result[1] += g as f32 * ratio[i];
                result[2] += b as f32 * ratio[i];
                result[3] += a as f32 * ratio[i];
            }
            Pixel::new(
                result[0] as u8,
                result[1] as u8,
                result[2] as u8,
                result[3] as u8,
            )
        })
    }
}

impl Mixer for PerPixelMix {
    fn mix(&mut self, frames: Vec<Frame>, ctx: FrameCtx) -> Frame {
        let min_width = frames.iter().map(|f| f.width()).min().unwrap_or(0);
        let min_height = frames.iter().map(|f| f.height()).min().unwrap_or(0);

        let mut canvas = Frame::black(min_height, min_height, ctx.pts);

        for y in 0..min_height {
            for x in 0..min_width {
                let pxs = frames.iter().map(|f| f.get_pixel(x, y)).collect::<Vec<_>>();
                let result = (self.f)(&pxs, ctx);
                canvas.set_pixel(x, y, result);
            }
        }

        canvas
    }
}
