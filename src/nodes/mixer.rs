use video_pipeline::{Frame, FrameCtx, Pipeline, Pixel, Process};

/// 複数フレームを 1 枚に合成するノード。[`Mix`] が 2 入力を受けて呼び出す。
pub trait Mixer {
    /// `frames` を合成して 1 枚を返す。`ctx` は第一ソースのフレーム文脈。
    fn mix(&mut self, frames: Vec<Frame>, ctx: FrameCtx) -> Frame;
}

impl<F: FnMut(Vec<Frame>, FrameCtx) -> Frame> Mixer for F {
    fn mix(&mut self, frames: Vec<Frame>, ctx: FrameCtx) -> Frame {
        self(frames, ctx)
    }
}

pub struct Mix<P: Pipeline, M: Mixer> {
    p: P,
    mixer: M,
}

impl<P: Pipeline, M: Mixer> Mix<P, M> {
    /// 2 つの上流ソースと合成器から Mix ノードを構築する。
    pub fn new(p: P, mixer: M) -> Self {
        Self { p, mixer }
    }
}

impl<P: Pipeline, M: Mixer> Process for Mix<P, M> {
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame {
        let b = self.p.next_frame();
        if let Some(b) = b {
            self.mixer.mix(vec![frame, b], ctx)
        } else {
            frame
        }
    }
}

/// ピクセル単位でフレームを合成する [`Mixer`]。
pub struct PerPixelMix {
    f: Box<dyn FnMut(&[Pixel], FrameCtx) -> Pixel>,
}

impl PerPixelMix {
    /// 任意のクロージャからピクセル合成器を構築する。
    pub fn new(f: impl FnMut(&[Pixel], FrameCtx) -> Pixel + 'static) -> Self {
        Self { f: Box::new(f) }
    }

    /// 各入力に `ratio[i]` を掛けて加算ブレンドする。
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
        let mut canvas = Frame::black(min_width, min_height, ctx);
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
