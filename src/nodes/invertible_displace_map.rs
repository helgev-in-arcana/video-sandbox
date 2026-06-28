//! 外部マップ駆動の可逆ディスプレイスメントノード（2 入力 [`Pipeline`]）。
//!
//! [`InvertibleDisplace`](super::InvertibleDisplace) が内部生成のフラクタルノイズを速度場に使うのに対し、
//! こちらは [`Displace`](super::Displace) と同じく **`source`（ずらされる側）と `map`（変位を与える側）**
//! の 2 つの上流 [`Pipeline`] を受け取り、`map` の **R→vx, G→vy** チャンネルを速度場として読む。
//!
//! その速度場を SVF 積分してから **順ワープ → 内側 [`Process`] → 逆ワープ** を適用するので、
//! 任意の動画をマップに入れても（折りたたみが起きない）可逆な warp になる。フラクタルノイズを
//! 使いたい場合は [`FractalNoise`](crate::videogen::FractalNoise) を `map` として流せる。
//!
//! SVF・ワープのコアは [`InvertibleDisplace`](super::InvertibleDisplace) と共有する。

use video_pipeline::{Frame, Pipeline, Process};

use super::invertible_displace::field::velocity_from_map;
use super::invertible_displace::warp_with_field;
use super::{DisplacementDescriptor, WarpMode};

/// `source` と `map` の 2 上流を受け、`map` を速度場として `source` を可逆ワープする 2 入力ノード。
///
/// `map` は **R→X, G→Y**（`128` が変位ゼロ）として解釈し、[`DisplacementDescriptor::amplitude`] を
/// 掛けて速度場にする。`map` 由来なので、ディスクリプタのノイズ専用フィールド
/// （`field`/`feature_scale`/`time_scale`/`seed`）は無視され、`amplitude`・`squaring_steps`・
/// `supersample`・`field_divisor`・`precision`・`warp_mode` のみが効く。
///
/// [`Displace`](super::Displace) と同じく、どちらかの上流が枯渇した時点でストリームが終わる。各上流を
/// 独立に `.buffered()` しておけば 2 本のデコードを別スレッドで重ねられる。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{EncodeSettings, Pipeline, VideoFile};
/// use video_sandbox::nodes::{DisplacementDescriptor, InvertibleDisplaceMap, Invert};
/// use video_sandbox::videogen::FractalNoise;
///
/// // フラクタルノイズをマップに、歪んだ空間で色反転してから引き戻す。
/// InvertibleDisplaceMap::new(
///     VideoFile::new("source.mp4").buffered(4),
///     FractalNoise::new(1920, 1080, 300).buffered(4),
///     DisplacementDescriptor::new().amplitude(40.0),
///     Invert,
/// )
/// .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct InvertibleDisplaceMap<S, M, P> {
    source: S,
    map: M,
    desc: DisplacementDescriptor,
    inner: P,
}

impl<S: Pipeline, M: Pipeline, P: Process> InvertibleDisplaceMap<S, M, P> {
    /// `source`・`map`・ディスクリプタ・内側 [`Process`] からノードを構築する。
    ///
    /// 純ワープ（処理なし）にしたいなら `inner` に `()` を渡す。
    pub fn new(source: S, map: M, desc: DisplacementDescriptor, inner: P) -> Self {
        Self { source, map, desc, inner }
    }

    /// ディスクリプタを差し替える。
    pub fn descriptor(mut self, desc: DisplacementDescriptor) -> Self {
        self.desc = desc;
        self
    }

    /// ワープ方向（順のみ／逆のみ／両方）を設定する。
    pub fn warp_mode(mut self, m: WarpMode) -> Self {
        self.desc.warp_mode = m;
        self
    }
}

impl<S: Pipeline, M: Pipeline, P: Process> Pipeline for InvertibleDisplaceMap<S, M, P> {
    fn next_frame(&mut self) -> Option<Frame> {
        let source = self.source.next_frame()?;
        let map = self.map.next_frame()?;
        let ctx = source.ctx();

        let (w, h) = (source.width() as usize, source.height() as usize);
        if w == 0 || h == 0 {
            return Some(source);
        }
        let div = self.desc.field_divisor.max(1) as usize;
        let fw = (w / div).max(1);
        let fh = (h / div).max(1);

        // map の R/G を field 解像度の速度場として読み、共有エンジンへ。
        let v = velocity_from_map(&map, fw, fh, div as f32, self.desc.amplitude);
        Some(warp_with_field(&self.desc, &mut self.inner, &source, v, ctx))
    }

    fn size_hint(&self) -> Option<u64> {
        match (self.source.size_hint(), self.map.size_hint()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::IntermediatePrecision;
    use video_pipeline::{Frame, FrameCtx, Pixel};

    fn ctx() -> FrameCtx {
        FrameCtx { index: 0, pts: 0, seconds: 0.0 }
    }

    /// 1 枚のフレームを 1 回だけ流す最小ソース。
    struct Once(Option<Frame>);
    impl Pipeline for Once {
        fn next_frame(&mut self) -> Option<Frame> {
            self.0.take()
        }
    }

    fn pattern(w: u32, h: u32) -> Frame {
        let mut f = Frame::black(w, h, ctx());
        f.per_iter_row(&ctx(), |_c, y, row| {
            for (x, px) in row.iter_mut().enumerate() {
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                let b = (((fx * 6.0).sin() * 0.5 + 0.5) * 255.0) as u8;
                *px = Pixel::rgb((fx * 255.0) as u8, (fy * 255.0) as u8, b);
            }
        });
        f
    }

    /// なめらかな速度マップ（R=vx, G=vy を別位相の正弦で）。128 が変位ゼロ。
    fn velocity_map(w: u32, h: u32) -> Frame {
        let mut f = Frame::black(w, h, ctx());
        f.per_iter_row(&ctx(), |_c, y, row| {
            for (x, px) in row.iter_mut().enumerate() {
                let u = x as f32 / w as f32 * std::f32::consts::TAU;
                let v = y as f32 / h as f32 * std::f32::consts::TAU;
                let r = ((u.sin() * 0.5 + 0.5) * 255.0) as u8;
                let g = ((v.cos() * 0.5 + 0.5) * 255.0) as u8;
                *px = Pixel::rgb(r, g, 128);
            }
        });
        f
    }

    fn interior_mean_abs_diff(a: &Frame, b: &Frame, margin: u32) -> f32 {
        let (w, h) = (a.width(), a.height());
        let mut sum = 0.0f64;
        let mut n = 0u64;
        for y in margin..h - margin {
            for x in margin..w - margin {
                let (pa, pb) = (a.get_pixel(x, y), b.get_pixel(x, y));
                sum += (pa.r as f32 - pb.r as f32).abs() as f64;
                sum += (pa.g as f32 - pb.g as f32).abs() as f64;
                sum += (pa.b as f32 - pb.b as f32).abs() as f64;
                n += 3;
            }
        }
        (sum / n as f64) as f32
    }

    fn run(desc: DisplacementDescriptor) -> (Frame, Frame) {
        let input = pattern(80, 80);
        let node = InvertibleDisplaceMap::new(
            Once(Some(input.clone())),
            Once(Some(velocity_map(80, 80))),
            desc,
            (),
        );
        let mut node = node;
        let out = node.next_frame().unwrap();
        (input, out)
    }

    /// map 駆動でも順→逆ラウンドトリップが内側でほぼ無損失なこと。
    #[test]
    fn map_roundtrip_is_near_identity() {
        let desc = DisplacementDescriptor::new()
            .amplitude(5.0)
            .squaring_steps(4)
            .supersample(2)
            .precision(IntermediatePrecision::F32);
        let (input, out) = run(desc);
        let err = interior_mean_abs_diff(&input, &out, 16);
        assert!(err < 8.0, "map ラウンドトリップ誤差が大きい: {err}");
    }

    /// ForwardOnly は元に戻さない（出力が入力から有意にずれる）。
    #[test]
    fn map_forward_only_displaces() {
        let desc = DisplacementDescriptor::new()
            .amplitude(14.0)
            .squaring_steps(4)
            .warp_mode(WarpMode::ForwardOnly);
        let (input, out) = run(desc);
        let diff = interior_mean_abs_diff(&input, &out, 16);
        assert!(diff > 5.0, "ForwardOnly が変位していない: {diff}");
    }

    /// 両上流の短い方でストリームが終わる（map 1 枚 → 1 フレームで枯渇）。
    #[test]
    fn ends_with_shorter_input() {
        let mut node = InvertibleDisplaceMap::new(
            Once(Some(pattern(32, 32))),
            Once(Some(velocity_map(32, 32))),
            DisplacementDescriptor::new().squaring_steps(3),
            (),
        );
        assert!(node.next_frame().is_some());
        assert!(node.next_frame().is_none());
    }
}
