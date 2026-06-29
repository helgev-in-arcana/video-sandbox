//! 低解像度グリッチ・差分オーバーレイノード（1 入力 [`Process`]）。
//!
//! 元フレームを「モザイク状の低解像度」に落としてから内側 [`Process`]（任意のグリッチ）を
//! かけ、低解像度の処理前後で**有意に色が変化したブロックだけ**を元のフル解像度フレームへ
//! 書き戻す。グリッチの寄与がブロック解像度（低周波）に収まるので、非グリッチ領域は元の
//! 圧縮特性を保ったまま、グリッチ部分も H.264 で圧縮しやすくなる（軽量化）。
//!
//! 構造は [`Feedback`](crate::nodes::Feedback) と同型で、内側 `P: Process` を 1 つ保持する。
//! `source.pipe(LowResGlitch::new(desc, inner))` で連結できる。

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use video_pipeline::{Frame, FrameCtx, Pixel, Process};

/// 低解像度版（モザイク）の作り方。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DownsampleMode {
    /// ブロック内全ピクセルの RGBA 平均（既定）。
    #[default]
    Interpolate,
    /// ブロック中心の 1 ピクセルをそのまま採る（補間なし）。
    Nearest,
    /// ブロック内から 1 点ランダムサンプルする。
    Random,
}

/// 変化ブロックを元解像度へ書き戻すときの合成方法。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OverlayMode {
    /// 出力 = OutputLowRes をブロックごと単色で上書き（既定）。グリッチ領域は完全に低周波化する。
    #[default]
    Replace,
    /// 出力 = 元ピクセル + (OutputLowRes − InputLowRes)。元の高周波ディテールを残しつつ差分だけ乗せる。
    Add,
}

/// [`LowResGlitch`] の設定ディスクリプタ。
///
/// `#[derive(PartialEq)]` のみ（`threshold` が f32 のため `Eq` は付けない）。各フィールドに
/// `const fn` ビルダーを用意する。
///
/// # 使用例
///
/// ```
/// use video_sandbox::nodes::{LowResGlitchDescriptor, DownsampleMode, OverlayMode};
///
/// let desc = LowResGlitchDescriptor::new()
///     .block(8)
///     .downsample(DownsampleMode::Interpolate)
///     .overlay(OverlayMode::Replace)
///     .threshold(8.0);
/// # let _ = desc;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowResGlitchDescriptor {
    /// モザイク 1 ブロックの一辺（px）。低解像度化の粒度。実効値は `max(1)`。
    pub block: u32,
    /// 低解像度版の作り方。
    pub downsample: DownsampleMode,
    /// 書き戻しの合成方法（置換 / 加算）。
    pub overlay: OverlayMode,
    /// 「有意な変化」のしきい値。ブロックの平均 |Δ|（0..255）がこれを超えたら書き戻す。
    pub threshold: f32,
    /// [`DownsampleMode::Random`] 用シード。
    pub seed: u64,
}

impl Default for LowResGlitchDescriptor {
    fn default() -> Self {
        Self {
            block: 8,
            downsample: DownsampleMode::Interpolate,
            overlay: OverlayMode::Replace,
            threshold: 8.0,
            seed: 0,
        }
    }
}

impl LowResGlitchDescriptor {
    /// 既定値のディスクリプタを作る。
    pub fn new() -> Self {
        Self::default()
    }

    /// ブロックサイズ（px）を設定する。
    pub const fn block(mut self, b: u32) -> Self {
        self.block = b;
        self
    }

    /// 低解像度化モードを設定する。
    pub const fn downsample(mut self, m: DownsampleMode) -> Self {
        self.downsample = m;
        self
    }

    /// オーバーレイ（合成）モードを設定する。
    pub const fn overlay(mut self, m: OverlayMode) -> Self {
        self.overlay = m;
        self
    }

    /// 変化しきい値（平均 |Δ|, 0..255）を設定する。
    pub const fn threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    /// `Random` モード用シードを設定する。
    pub const fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

/// 低解像度でグリッチして差分ブロックだけ元解像度へオーバーレイする [`Process`]。
///
/// `inner` は低解像度フレームに対して走るので、処理コストは概ね `1/block²` に減る。内側が
/// `()`（恒等）なら全ブロックが閾値未満となり、出力は入力と完全一致する（無改変）。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{EncodeSettings, Pipeline, VideoFile};
/// use video_sandbox::nodes::{LowResGlitch, LowResGlitchDescriptor, Invert};
///
/// VideoFile::new("in.mp4")
///     .pipe(LowResGlitch::new(LowResGlitchDescriptor::new().block(8), Invert))
///     .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct LowResGlitch<P> {
    desc: LowResGlitchDescriptor,
    inner: P,
}

impl<P: Process> LowResGlitch<P> {
    /// ディスクリプタと内側 [`Process`] からノードを構築する。
    pub fn new(desc: LowResGlitchDescriptor, inner: P) -> Self {
        Self { desc, inner }
    }
}

/// `frame` をブロックサイズ `b` で低解像度化する（`mode` に従って 1 ブロック = 1 px に集約）。
fn downsample(frame: &Frame, b: u32, mode: DownsampleMode, seed: u64) -> Frame {
    let (w, h) = (frame.width(), frame.height());
    let (lw, lh) = (w.div_ceil(b), h.div_ceil(b));
    let ctx = frame.ctx();
    let mut low = Frame::black(lw, lh, ctx);

    low.per_iter_row(&ctx, |_c, by, row| {
        let by = by as u32;
        let y0 = by * b;
        let y1 = (y0 + b).min(h);
        // Random は行ごとに決定的にシードし直す（行は並列・互いに独立）。
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ by as u64);
        for (bx, slot) in row.iter_mut().enumerate() {
            let bx = bx as u32;
            let x0 = bx * b;
            let x1 = (x0 + b).min(w);
            *slot = match mode {
                DownsampleMode::Interpolate => {
                    let (mut sr, mut sg, mut sb, mut sa) = (0u32, 0u32, 0u32, 0u32);
                    let mut n = 0u32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let p = frame.get_pixel(x, y);
                            sr += p.r as u32;
                            sg += p.g as u32;
                            sb += p.b as u32;
                            sa += p.a as u32;
                            n += 1;
                        }
                    }
                    let n = n.max(1);
                    Pixel::new(
                        (sr / n) as u8,
                        (sg / n) as u8,
                        (sb / n) as u8,
                        (sa / n) as u8,
                    )
                }
                DownsampleMode::Nearest => {
                    let cx = (x0 + b / 2).min(w - 1);
                    let cy = (y0 + b / 2).min(h - 1);
                    frame.get_pixel(cx, cy)
                }
                DownsampleMode::Random => {
                    let rx = x0 + (rng.random::<u32>() % (x1 - x0).max(1));
                    let ry = y0 + (rng.random::<u32>() % (y1 - y0).max(1));
                    frame.get_pixel(rx.min(w - 1), ry.min(h - 1))
                }
            };
        }
    });
    low
}

/// ブロックの平均 |Δ|（RGB のみ, 0..255）を返す。
#[inline]
fn block_change(i: Pixel, o: Pixel) -> f32 {
    let d = |a: u8, b: u8| (a as f32 - b as f32).abs();
    (d(i.r, o.r) + d(i.g, o.g) + d(i.b, o.b)) / 3.0
}

impl<P: Process> Process for LowResGlitch<P> {
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame {
        let (w, h) = (frame.width(), frame.height());
        if w == 0 || h == 0 {
            return frame;
        }
        let b = self.desc.block.max(1);

        // 1. 低解像度化 → 2. 内側 Process。
        let low_in = downsample(&frame, b, self.desc.downsample, self.desc.seed);
        let low_out = self.inner.process(low_in.clone(), ctx);

        // 3. 変化ブロックだけ元解像度へオーバーレイ。
        let (threshold, overlay) = (self.desc.threshold, self.desc.overlay);
        let mut out = frame;
        out.per_iter_row(&ctx, |_c, y, row| {
            let by = (y as u32 / b).min(low_in.height() - 1);
            for (x, px) in row.iter_mut().enumerate() {
                let bx = (x as u32 / b).min(low_in.width() - 1);
                let i = low_in.get_pixel(bx, by);
                let o = low_out.get_pixel(bx, by);
                if block_change(i, o) <= threshold {
                    continue;
                }
                *px = match overlay {
                    OverlayMode::Replace => Pixel::new(o.r, o.g, o.b, px.a),
                    OverlayMode::Add => {
                        let add = |base: u8, oc: u8, ic: u8| {
                            (base as i16 + (oc as i16 - ic as i16)).clamp(0, 255) as u8
                        };
                        Pixel::new(
                            add(px.r, o.r, i.r),
                            add(px.g, o.g, i.g),
                            add(px.b, o.b, i.b),
                            px.a,
                        )
                    }
                };
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FrameCtx {
        FrameCtx { index: 0, pts: 0, seconds: 0.0 }
    }

    /// 各画素を `f(x, y)` で塗ったフレーム。
    fn make(w: u32, h: u32, f: impl Fn(u32, u32) -> Pixel) -> Frame {
        let mut frame = Frame::black(w, h, ctx());
        for y in 0..h {
            for x in 0..w {
                frame.set_pixel(x, y, f(x, y));
            }
        }
        frame
    }

    fn frames_equal(a: &Frame, b: &Frame) -> bool {
        if (a.width(), a.height()) != (b.width(), b.height()) {
            return false;
        }
        for y in 0..a.height() {
            for x in 0..a.width() {
                if a.get_pixel(x, y) != b.get_pixel(x, y) {
                    return false;
                }
            }
        }
        true
    }

    /// 内側 `()`（恒等）で出力が入力と完全一致（差分ゼロ → 無改変）。
    #[test]
    fn identity_inner_passes_through() {
        let input = make(32, 24, |x, y| Pixel::rgb(x as u8, y as u8, (x ^ y) as u8));
        let mut node = LowResGlitch::new(LowResGlitchDescriptor::new().block(8), ());
        let out = node.process(input.clone(), ctx());
        assert!(frames_equal(&input, &out));
    }

    /// 単色入力はどの DownsampleMode でもその色にダウンサンプルされる。
    #[test]
    fn flat_block_downsample() {
        let color = Pixel::rgb(40, 160, 200);
        let input = make(24, 16, |_, _| color);
        for mode in [DownsampleMode::Interpolate, DownsampleMode::Nearest, DownsampleMode::Random] {
            let low = downsample(&input, 8, mode, 0);
            assert_eq!((low.width(), low.height()), (3, 2));
            for y in 0..low.height() {
                for x in 0..low.width() {
                    assert_eq!(low.get_pixel(x, y), color, "mode={mode:?} ({x},{y})");
                }
            }
        }
    }

    /// Replace モード: 全面白に塗る内側 ＋ 低 threshold で出力ブロックが白になる。
    #[test]
    fn replace_writes_changed_blocks() {
        let input = make(16, 16, |_, _| Pixel::rgb(0, 0, 0));
        let paint_white = |mut f: Frame, c: FrameCtx| {
            f.per_iter_row(&c, |_c, _y, row| {
                for px in row.iter_mut() {
                    *px = Pixel::rgb(255, 255, 255);
                }
            });
            f
        };
        let desc = LowResGlitchDescriptor::new().block(8).threshold(1.0);
        let mut node = LowResGlitch::new(desc, paint_white);
        let out = node.process(input, ctx());
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(out.get_pixel(x, y), Pixel::rgb(255, 255, 255));
            }
        }
    }

    /// Add モード: 出力 = 元 + (out − in) がチャンネルごとに一致（クランプ込み）。
    #[test]
    fn add_overlays_delta() {
        // 元は単色 100、内側で +50 して 150 に。差分 +50 が元へ加算され 150 になる。
        let input = make(8, 8, |_, _| Pixel::rgb(100, 100, 100));
        let plus50 = |mut f: Frame, c: FrameCtx| {
            f.per_iter_row(&c, |_c, _y, row| {
                for px in row.iter_mut() {
                    *px = Pixel::rgb(150, 150, 150);
                }
            });
            f
        };
        let desc = LowResGlitchDescriptor::new()
            .block(8)
            .overlay(OverlayMode::Add)
            .threshold(1.0);
        let mut node = LowResGlitch::new(desc, plus50);
        let out = node.process(input, ctx());
        assert_eq!(out.get_pixel(0, 0), Pixel::rgb(150, 150, 150));
    }

    /// 微小変化（< threshold）のブロックは書き戻されず元のまま。
    #[test]
    fn threshold_gates_small_change() {
        let input = make(8, 8, |_, _| Pixel::rgb(100, 100, 100));
        // +3 の微小変化。平均 |Δ| = 3 < threshold(10) なのでゲートで弾かれる。
        let plus3 = |mut f: Frame, c: FrameCtx| {
            f.per_iter_row(&c, |_c, _y, row| {
                for px in row.iter_mut() {
                    *px = Pixel::rgb(103, 103, 103);
                }
            });
            f
        };
        let desc = LowResGlitchDescriptor::new().block(8).threshold(10.0);
        let mut node = LowResGlitch::new(desc, plus3);
        let out = node.process(input.clone(), ctx());
        assert!(frames_equal(&input, &out));
    }

    /// 非整除サイズでも出力は W×H を保つ。
    #[test]
    fn output_size_preserved() {
        let input = make(37, 53, |x, y| Pixel::rgb(x as u8, y as u8, 0));
        let mut node = LowResGlitch::new(LowResGlitchDescriptor::new().block(8), ());
        let out = node.process(input, ctx());
        assert_eq!((out.width(), out.height()), (37, 53));
    }
}
