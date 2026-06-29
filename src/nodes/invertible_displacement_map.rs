//! 外部マップ駆動の可逆ディスプレイスメントノード（2 入力 [`Pipeline`]）。
//!
//! **`source`（ずらされる側）と `map`（変位を与える側）** の 2 つの上流 [`Pipeline`] を受け取り、
//! `map` の **R→vx, G→vy** チャンネル（128 = 変位ゼロ）を速度場として読む。
//!
//! その速度場を SVF 積分してから **順ワープ(φ⁻¹) → 内側 [`Process`] → 逆ワープ(φ)** を適用する。
//! SVF（定常速度場の time-1 フロー）を使うため写像 φ は構造的に折りたたみ無し（微分同相）で、
//! 逆写像は速度場 `v` の符号反転 `exp(-v)` だけで得られ、順・逆ワープが定義上整合する。
//!
//! フラクタルノイズを速度場に使いたい場合は [`FractalNoise`](crate::videogen::FractalNoise) を
//! `map` として渡す。アルゴリズムの詳細は `svf_invertible_displacement_spec.md` を参照。

pub(crate) mod field;
pub(crate) mod warp;

use video_pipeline::{Frame, FrameCtx, Pipeline, Process};

use field::{integrate_svf, pad_field_clamp, upsample_field, velocity_from_map, Field};
use warp::{downsample_image, upsample_image, warp_image, FloatImage};

/// ワープの方向。順ワープ(φ⁻¹) と 逆ワープ(φ) のどちらを適用するか。
///
/// 内側 [`Process`] は常に「順と逆の間」に置かれる。[`ForwardOnly`](WarpMode::ForwardOnly) は
/// 歪んだ空間に出力し（元に戻さない＝単純ワープ）、[`InverseOnly`](WarpMode::InverseOnly) は
/// 逆向きの単発ワープになる。φ と φ⁻¹ は速度場 `v` の符号違いなので両者は実質対称。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WarpMode {
    /// 順 → Process → 逆（既定）。可逆ラウンドトリップ。
    #[default]
    Roundtrip,
    /// 順 → Process のみ。出力は歪んだ空間のまま。
    ForwardOnly,
    /// Process → 逆 のみ。
    InverseOnly,
}

impl WarpMode {
    #[inline]
    const fn uses_forward(self) -> bool {
        matches!(self, WarpMode::Roundtrip | WarpMode::ForwardOnly)
    }

    #[inline]
    const fn uses_inverse(self) -> bool {
        matches!(self, WarpMode::Roundtrip | WarpMode::InverseOnly)
    }
}

/// 中間バッファの精度。
///
/// 既定の [`F32`](IntermediatePrecision::F32) は量子化床（スペック §6）を避ける。
/// [`U8`](IntermediatePrecision::U8) はその劣化を意図的に観察する実験用。なお内側 [`Process`] の
/// 境界は trait 仕様上 u8 不可避なので、F32 でもそこで 1 回だけ量子化が挟まる。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IntermediatePrecision {
    /// 中間バッファを 8bit に丸める（劣化観察用）。
    U8,
    /// 中間バッファを f32 のまま通す（既定）。
    #[default]
    F32,
}

/// [`InvertibleDisplacementMap`] の全設定を束ねるディスクリプタ。
///
/// ノイズの形・seed・スケールは `map` 側の [`FractalNoise`](crate::videogen::FractalNoise) で
/// 設定する。このディスクリプタはワープ計算に関わるパラメータのみを持つ。
///
/// # 使用例
///
/// ```
/// use video_sandbox::nodes::{DisplacementDescriptor, IntermediatePrecision, WarpMode};
///
/// let desc = DisplacementDescriptor::new()
///     .amplitude(24.0)
///     .squaring_steps(6)
///     .supersample(2)
///     .precision(IntermediatePrecision::F32)
///     .warp_mode(WarpMode::Roundtrip);
/// # let _ = desc;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplacementDescriptor {
    /// 1×-px 単位の最大変位の目安（map チャンネル `[-1,1]` に掛かる）。
    pub amplitude: f32,
    /// scaling-and-squaring の段数 N（補間段数）。6〜7 で飽和。
    pub squaring_steps: u32,
    /// スーパーサンプル倍率 k。中間と両ワープを k× で回す。2（強変形は 4）。
    pub supersample: u32,
    /// 場の解像度デカップル。場を `1/field_divisor` 解像度で積分して squaring を高速化。
    pub field_divisor: u32,
    /// 中間バッファ精度。
    pub precision: IntermediatePrecision,
    /// 適用するワープ方向（順のみ／逆のみ／両方）。
    pub warp_mode: WarpMode,
    /// 中間キャンバスを外側に広げるパディング量（1×-px、片側）。
    ///
    /// 端をまたぐ変位で生じる引き伸ばし（[`FloatImage::bilinear`](warp::FloatImage::bilinear) の
    /// 端クランプ由来）を抑える。`None`（既定）なら [`amplitude`](Self::amplitude) と同サイズに
    /// 自動設定する（＝変位ピークを概ねカバー）。source は mirror（鏡像反射）で拡張し、
    /// 出力時に中央をクロップして元サイズに戻す。Roundtrip では可視領域の忠実度がほぼ完全に回復し、
    /// ForwardOnly では最外周の滲みが「鏡像の続き」に置き換わって見栄えが改善する。
    pub padding: Option<u32>,
}

impl Default for DisplacementDescriptor {
    fn default() -> Self {
        Self {
            amplitude: 16.0,
            squaring_steps: 6,
            supersample: 2,
            field_divisor: 1,
            precision: IntermediatePrecision::F32,
            warp_mode: WarpMode::Roundtrip,
            padding: None,
        }
    }
}

impl DisplacementDescriptor {
    /// 既定値のディスクリプタを作る。
    pub fn new() -> Self {
        Self::default()
    }

    /// 最大変位の目安（1×-px）を設定する。
    pub const fn amplitude(mut self, a: f32) -> Self {
        self.amplitude = a;
        self
    }

    /// squaring 段数 N を設定する。
    pub const fn squaring_steps(mut self, n: u32) -> Self {
        self.squaring_steps = n;
        self
    }

    /// スーパーサンプル倍率 k を設定する。
    pub const fn supersample(mut self, k: u32) -> Self {
        self.supersample = k;
        self
    }

    /// 場の解像度デカップル（`1/field_divisor`）を設定する。
    pub const fn field_divisor(mut self, d: u32) -> Self {
        self.field_divisor = d;
        self
    }

    /// 中間バッファ精度を設定する。
    pub const fn precision(mut self, p: IntermediatePrecision) -> Self {
        self.precision = p;
        self
    }

    /// ワープ方向（順のみ／逆のみ／両方）を設定する。
    pub const fn warp_mode(mut self, m: WarpMode) -> Self {
        self.warp_mode = m;
        self
    }

    /// パディング量（1×-px、片側）を明示設定する。`0` で無効。
    ///
    /// 未設定（既定 `None`）なら [`amplitude`](Self::amplitude) と同サイズに自動設定される。
    pub const fn padding(mut self, px: u32) -> Self {
        self.padding = Some(px);
        self
    }

    /// 実際に使うパディング量（1×-px）。`None` は `amplitude` 切り上げに解決する。
    fn resolved_padding(&self) -> u32 {
        self.padding.unwrap_or_else(|| self.amplitude.max(0.0).ceil() as u32)
    }
}

/// `source` と `map` の 2 上流を受け、`map` を速度場として `source` を可逆ワープする 2 入力ノード。
///
/// `map` は **R→X, G→Y**（`128` が変位ゼロ）として解釈し、[`DisplacementDescriptor::amplitude`] を
/// 掛けて速度場にする。内側 `()` を渡せば純粋なラウンドトリップ（≈ 恒等）になる。
/// どちらかの上流が枯渇した時点でストリームが終わる。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{EncodeSettings, Pipeline, VideoFile};
/// use video_sandbox::nodes::{DisplacementDescriptor, InvertibleDisplacementMap, Invert};
/// use video_sandbox::videogen::FractalNoise;
///
/// // フラクタルノイズをマップに、歪んだ空間で色反転してから引き戻す。
/// InvertibleDisplacementMap::new(
///     VideoFile::new("source.mp4").buffered(4),
///     FractalNoise::new(1920, 1080, 300).buffered(4),
///     DisplacementDescriptor::new().amplitude(40.0),
///     Invert,
/// )
/// .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct InvertibleDisplacementMap<S, M, P> {
    source: S,
    map: M,
    desc: DisplacementDescriptor,
    inner: P,
}

impl<S: Pipeline, M: Pipeline, P: Process> InvertibleDisplacementMap<S, M, P> {
    /// `source`・`map`・ディスクリプタ・内側 [`Process`] からノードを構築する。
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

impl<S: Pipeline, M: Pipeline, P: Process> Pipeline for InvertibleDisplacementMap<S, M, P> {
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

/// 速度場 `v`（field 解像度・field-grid px 単位）を SVF 積分し、`source` に対して
/// **順ワープ(φ⁻¹) → 内側 Process → 逆ワープ(φ) → ダウンサンプル**を適用する共通エンジン。
///
/// `desc.padding` 分だけ中間キャンバスを外側に広げて計算し、最後に中央をクロップする。これにより
/// 端をまたぐ変位の clamp（引き伸ばし）を抑える。速度場は積分前に端クランプで margin 拡張し、
/// source は mirror で拡張する。
fn warp_with_field<P: Process>(
    desc: &DisplacementDescriptor,
    inner: &mut P,
    source: &Frame,
    v: Field,
    ctx: FrameCtx,
) -> Frame {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let k = desc.supersample.max(1) as usize;
    let div = desc.field_divisor.max(1) as usize;
    let n = desc.squaring_steps;

    // パディング: field 格子マージン mf（セル）と、それに対応する画像 1×-px マージン mpx = mf*div。
    // field と画像で同じ物理領域を覆わせるため div の倍数に揃える。
    let mf = (desc.resolved_padding() as usize).div_ceil(div);
    let mpx = mf * div;

    // 0. 速度場を端クランプで拡張（margin 内は端の変位を保持＝平行移動なので折りたたみを生まない）。
    let v = if mf > 0 { pad_field_clamp(&v, mf) } else { v };

    // 1〜2. φ・φ⁻¹ を warp_mode が要求する向きだけ拡張ドメインで積分。
    let phi = desc.warp_mode.uses_inverse().then(|| integrate_svf(&v, n));
    let phi_inv = desc.warp_mode.uses_forward().then(|| {
        let neg = Field {
            w: v.w,
            h: v.h,
            vx: v.vx.iter().map(|c| -c).collect(),
            vy: v.vy.iter().map(|c| -c).collect(),
        };
        integrate_svf(&neg, n)
    });

    // 3. 拡張画像（1×: (w+2mpx, h+2mpx)）の k× 解像度へアップサンプル（変位ベクトルも k*div 倍）。
    let (wp, hp) = (w + 2 * mpx, h + 2 * mpx);
    let (wk, hk) = (wp * k, hp * k);
    let vec_scale = (k * div) as f32;
    let quantize = desc.precision == IntermediatePrecision::U8;
    let upsample_to_k = |f: Field| upsample_field(&f, wk, hk, vec_scale);
    let phi_k = phi.map(upsample_to_k);
    let phi_inv_k = phi_inv.map(upsample_to_k);

    // 4. 入力を mirror で拡張してから k× へ。
    let src_img = if mpx > 0 {
        FloatImage::from_frame_mirror_padded(source, mpx)
    } else {
        FloatImage::from_frame(source)
    };
    let mut img_k = upsample_image(&src_img, k);
    if quantize {
        img_k.quantize_u8();
    }

    // 5〜6. （任意の）順ワープ → 内側 Process（境界は u8）。
    let j_k = match &phi_inv_k {
        Some(field) => warp_image(&img_k, field),
        None => img_k,
    };
    let j_frame = j_k.to_frame(ctx);
    let p_k = inner.process(j_frame, ctx);

    // 7. （任意の）逆ワープ。
    let mut r_k = match &phi_k {
        Some(field) => warp_image(&FloatImage::from_frame(&p_k), field),
        None => FloatImage::from_frame(&p_k),
    };
    if quantize {
        r_k.quantize_u8();
    }

    // 8. 1×（拡張サイズ）へダウンサンプル → 中央 (w, h) をクロップして返す。
    let down = downsample_image(&r_k, k);
    let out = if mpx > 0 { down.crop(mpx) } else { down };
    out.to_frame(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_pipeline::{Frame, FrameCtx, Pixel};

    fn ctx() -> FrameCtx {
        FrameCtx { index: 0, pts: 0, seconds: 0.0 }
    }

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

    /// 端から `band` px 以内（外周リング）の平均絶対差。引き伸ばしは端で起きるのでそこを測る。
    fn edge_mean_abs_diff(a: &Frame, b: &Frame, band: u32) -> f32 {
        let (w, h) = (a.width(), a.height());
        let mut sum = 0.0f64;
        let mut n = 0u64;
        for y in 0..h {
            for x in 0..w {
                let on_edge = x < band || x >= w - band || y < band || y >= h - band;
                if !on_edge {
                    continue;
                }
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
        let mut node = InvertibleDisplacementMap::new(
            Once(Some(input.clone())),
            Once(Some(velocity_map(80, 80))),
            desc,
            (),
        );
        let out = node.next_frame().unwrap();
        (input, out)
    }

    /// 順→逆ラウンドトリップが内側でほぼ無損失なこと（可逆性の中核）。
    #[test]
    fn roundtrip_is_near_identity() {
        let desc = DisplacementDescriptor::new()
            .amplitude(5.0)
            .squaring_steps(4)
            .supersample(2)
            .precision(IntermediatePrecision::F32);
        let (input, out) = run(desc);
        let err = interior_mean_abs_diff(&input, &out, 16);
        assert!(err < 8.0, "ラウンドトリップ誤差が大きい: {err}");
    }

    /// ForwardOnly は元に戻さない（出力が入力から有意にずれる）。
    #[test]
    fn forward_only_displaces() {
        let desc = DisplacementDescriptor::new()
            .amplitude(14.0)
            .squaring_steps(4)
            .warp_mode(WarpMode::ForwardOnly);
        let (input, out) = run(desc);
        let diff = interior_mean_abs_diff(&input, &out, 16);
        assert!(diff > 5.0, "ForwardOnly が変位していない: {diff}");
    }

    /// ForwardOnly → InverseOnly（同一マップ）でラウンドトリップが復元すること。
    #[test]
    fn forward_then_inverse_recovers() {
        let input = pattern(80, 80);
        let map = velocity_map(80, 80);
        let desc = DisplacementDescriptor::new().amplitude(6.0).squaring_steps(4);

        let fwd = InvertibleDisplacementMap::new(
            Once(Some(input.clone())),
            Once(Some(map.clone())),
            desc.warp_mode(WarpMode::ForwardOnly),
            (),
        )
        .next_frame()
        .unwrap();

        let back = InvertibleDisplacementMap::new(
            Once(Some(fwd)),
            Once(Some(map)),
            desc.warp_mode(WarpMode::InverseOnly),
            (),
        )
        .next_frame()
        .unwrap();

        let err = interior_mean_abs_diff(&input, &back, 16);
        assert!(err < 10.0, "分割ラウンドトリップ誤差が大きい: {err}");
    }

    /// スーパーサンプリングが品質を大きく悪化させないこと（k=2 の誤差が k=1 より 1.0 以上増加しない）。
    ///
    /// 滑らかな変位場では k=1 と k=2 の差が僅少なため厳密な「以下」は保証されないが、
    /// 高周波・大振幅の変位場では k=2 が k=1 以上の忠実度になる（スペック §4.2）。
    #[test]
    fn supersample_improves_or_equal() {
        let input = pattern(80, 80);
        let err_at = |k: u32| {
            // 高周波マップ（4サイクル/80px ≈ 20px スケール）で圧縮領域を生じさせる。
            let map = {
                let mut f = Frame::black(80, 80, ctx());
                f.per_iter_row(&ctx(), |_c, y, row| {
                    for (x, px) in row.iter_mut().enumerate() {
                        let u = x as f32 / 80.0 * std::f32::consts::TAU * 4.0;
                        let v = y as f32 / 80.0 * std::f32::consts::TAU * 4.0;
                        let r = ((u.sin() * 0.5 + 0.5) * 255.0) as u8;
                        let g = ((v.cos() * 0.5 + 0.5) * 255.0) as u8;
                        *px = Pixel::rgb(r, g, 128);
                    }
                });
                f
            };
            let desc =
                DisplacementDescriptor::new().amplitude(14.0).squaring_steps(5).supersample(k);
            let mut node = InvertibleDisplacementMap::new(
                Once(Some(input.clone())),
                Once(Some(map)),
                desc,
                (),
            );
            let out = node.next_frame().unwrap();
            interior_mean_abs_diff(&input, &out, 16)
        };
        let (e1, e2) = (err_at(1), err_at(2));
        assert!(e2 <= e1 + 1.0, "k=2({e2}) が k=1({e1}) より 1.0 以上悪化");
    }

    /// パディングが Roundtrip の端の引き伸ばし（端誤差）を減らすこと。
    ///
    /// 大振幅で端をまたぐ変位を起こし、`padding(0)`（無効）と `padding`=既定（amplitude 相当）で
    /// 外周リングの誤差を比較する。
    #[test]
    fn padding_reduces_edge_stretch() {
        let edge_err = |desc: DisplacementDescriptor| {
            let (input, out) = run(desc);
            edge_mean_abs_diff(&input, &out, 6)
        };
        // 端をまたぐよう大きめの振幅。
        let base = DisplacementDescriptor::new().amplitude(20.0).squaring_steps(5);
        let no_pad = edge_err(base.padding(0));
        let auto_pad = edge_err(base); // padding=None → amplitude(=20) 相当
        assert!(
            auto_pad < no_pad,
            "パディングで端誤差が改善しない: auto={auto_pad}, none={no_pad}"
        );
    }

    /// 出力サイズはパディングに関わらず入力と同じ（クロップで戻る）。
    #[test]
    fn padding_preserves_output_size() {
        let desc = DisplacementDescriptor::new().amplitude(10.0).squaring_steps(4).padding(12);
        let (input, out) = run(desc);
        assert_eq!((out.width(), out.height()), (input.width(), input.height()));
    }

    /// 両上流の短い方でストリームが終わる（map 1 枚 → 1 フレームで枯渇）。
    #[test]
    fn ends_with_shorter_input() {
        let mut node = InvertibleDisplacementMap::new(
            Once(Some(pattern(32, 32))),
            Once(Some(velocity_map(32, 32))),
            DisplacementDescriptor::new().squaring_steps(3),
            (),
        );
        assert!(node.next_frame().is_some());
        assert!(node.next_frame().is_none());
    }
}
