use video_pipeline::{Frame, FrameCtx, Pipeline, Pixel};

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

/// 2 つの上流 [`Pipeline`] を [`Mixer`] で合成し、それ自身も [`Pipeline`] となるノード。
///
/// どちらかの上流が枯渇した時点でストリームが終わる（`Iterator::zip` 相当）。各上流を
/// 独立に `.buffered()` しておけば、2 本のデコードを別スレッドで重ねられる。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{VideoFile, EncodeSettings, Pipeline};
/// use video_sandbox::nodes::{Mix, PerPixelMix};
///
/// Mix::new(
///     VideoFile::new("a.mp4").buffered(4),
///     VideoFile::new("b.mp4").buffered(4),
///     PerPixelMix::new_blend(&[0.5, 0.5]),
/// )
/// .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct Mix<A, B, M> {
    a: A,
    b: B,
    mixer: M,
}

impl<A: Pipeline, B: Pipeline, M: Mixer> Mix<A, B, M> {
    /// 2 つの上流ソースと合成器から Mix ノードを構築する。
    pub fn new(a: A, b: B, mixer: M) -> Self {
        Self { a, b, mixer }
    }
}

impl<A: Pipeline, B: Pipeline, M: Mixer> Pipeline for Mix<A, B, M> {
    fn next_frame(&mut self) -> Option<Frame> {
        let fa = self.a.next_frame()?;
        let fb = self.b.next_frame()?;
        let ctx = fa.ctx();
        Some(self.mixer.mix(vec![fa, fb], ctx))
    }

    fn size_hint(&self) -> Option<u64> {
        match (self.a.size_hint(), self.b.size_hint()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
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
