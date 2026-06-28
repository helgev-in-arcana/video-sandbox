//! 可逆ディスプレイスメントマップ（SVF版）ノード。
//!
//! フラクタルノイズ由来の**定常速度場（Stationary Velocity Field）の time-1 フロー写像**で画像を
//! ワープし、その歪んだ空間で内側の [`Process`] を適用してから元の空間へ引き戻す。SVF を使うため
//! 写像 φ は構造的に折りたたみ無し（微分同相）で、逆写像は速度場 `v` の符号反転 `exp(-v)` だけで
//! 得られ、順・逆ワープが定義上整合する（[`InvertibleDisplace`]）。
//!
//! アルゴリズムの根拠は `svf_invertible_displacement_spec.md` を参照。

// map 駆動ノード（[`InvertibleDisplaceMap`](super::InvertibleDisplaceMap)）と SVF/ワープのコアを
// 共有するため crate 内に公開する。
pub(crate) mod field;
pub(crate) mod warp;

use std::sync::Arc;

use video_pipeline::{Frame, FrameCtx, Process};

use crate::videogen::{FractalNoiseDescriptor, Perm};

use field::{generate_velocity, integrate_svf, upsample_field, Field};
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
    /// 順ワープ(φ⁻¹) を適用するか。
    #[inline]
    const fn uses_forward(self) -> bool {
        matches!(self, WarpMode::Roundtrip | WarpMode::ForwardOnly)
    }

    /// 逆ワープ(φ) を適用するか。
    #[inline]
    const fn uses_inverse(self) -> bool {
        matches!(self, WarpMode::Roundtrip | WarpMode::InverseOnly)
    }
}

/// 中間バッファの精度。再標本化を f32 で通すか、各段で 8bit に丸めるか。
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

/// [`InvertibleDisplace`] の全設定を束ねるディスクリプタ。
///
/// 速度場の形は [`FractalNoiseDescriptor`] を埋め込んで再利用する。各 `const fn` ビルダーで
/// 個別に差し替えられる（[`FractalNoiseDescriptor`] と同じスタイル）。
///
/// # 使用例
///
/// ```
/// use video_sandbox::nodes::{DisplacementDescriptor, IntermediatePrecision};
/// use video_sandbox::videogen::{FractalNoiseDescriptor, NoiseKind};
///
/// let desc = DisplacementDescriptor::new()
///     .amplitude(24.0)
///     .feature_scale(96.0)
///     .squaring_steps(6)
///     .supersample(2)
///     .precision(IntermediatePrecision::F32)
///     .field(FractalNoiseDescriptor::new().noise(NoiseKind::Simplex));
/// # let _ = desc;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplacementDescriptor {
    /// 速度場を生成するフラクタルノイズの形。
    pub field: FractalNoiseDescriptor,
    /// 1×-px 単位の最大変位の目安（ノイズ振幅 `[-1,1]` に掛かる）。
    pub amplitude: f32,
    /// 特徴サイズ（px / ノイズ単位）。大きいほど場が滑らか＝粗い。
    pub feature_scale: f32,
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
    /// 時間発展速度（`z = seconds * time_scale`）。0 で静止。
    pub time_scale: f32,
    /// 置換表の seed。
    pub seed: u64,
}

impl Default for DisplacementDescriptor {
    fn default() -> Self {
        Self {
            field: FractalNoiseDescriptor::new(),
            amplitude: 16.0,
            feature_scale: 96.0,
            squaring_steps: 6,
            supersample: 2,
            field_divisor: 1,
            precision: IntermediatePrecision::F32,
            warp_mode: WarpMode::Roundtrip,
            time_scale: 1.0,
            seed: 0,
        }
    }
}

impl DisplacementDescriptor {
    /// 既定値のディスクリプタを作る（表は型ドキュメント参照）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 速度場のフラクタルノイズ形を差し替える。
    pub const fn field(mut self, field: FractalNoiseDescriptor) -> Self {
        self.field = field;
        self
    }

    /// 最大変位の目安（1×-px）を設定する。
    pub const fn amplitude(mut self, a: f32) -> Self {
        self.amplitude = a;
        self
    }

    /// 特徴サイズ（px / ノイズ単位）を設定する。
    pub const fn feature_scale(mut self, s: f32) -> Self {
        self.feature_scale = s;
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

    /// 時間発展速度を設定する。
    pub const fn time_scale(mut self, s: f32) -> Self {
        self.time_scale = s;
        self
    }

    /// 置換表の seed を設定する。
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// **順ワープ → 内側 [`Process`] → 逆ワープ**を行う可逆ディスプレイスメントノード。
///
/// [`Feedback`](crate::nodes::Feedback) と同じく内部に 1 段の [`Process`] を持つ。内側 `()` を渡せば
/// 純粋なラウンドトリップ（≈ 恒等）になる。内側 Process は **k× スーパーサンプル解像度**のフレームを
/// 受け取る点に注意（スペック §4.2）。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{EncodeSettings, Pipeline};
/// use video_sandbox::nodes::{DisplacementDescriptor, InvertibleDisplace};
/// use video_sandbox::nodes::Invert;
/// use video_sandbox::videogen::FractalNoise;
///
/// // フラクタルノイズ動画を、歪んだ空間で色反転してから引き戻す。
/// let node = InvertibleDisplace::new(
///     DisplacementDescriptor::new().amplitude(20.0),
///     Invert,
/// );
/// FractalNoise::new(320, 240, 30)
///     .pipe(node)
///     .encode_to("warp.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct InvertibleDisplace<P> {
    desc: DisplacementDescriptor,
    perm: Arc<Perm>,
    inner: P,
}

impl<P: Process> InvertibleDisplace<P> {
    /// ディスクリプタと内側 [`Process`] からノードを構築する（seed から置換表を作る）。
    pub fn new(desc: DisplacementDescriptor, inner: P) -> Self {
        Self { perm: Perm::new(desc.seed), desc, inner }
    }

}

impl<P: Process> Process for InvertibleDisplace<P> {
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame {
        let (w, h) = (frame.width() as usize, frame.height() as usize);
        if w == 0 || h == 0 {
            return frame;
        }
        let d = self.desc;
        let div = d.field_divisor.max(1) as usize;
        let fw = (w / div).max(1);
        let fh = (h / div).max(1);
        let z = ctx.seconds * d.time_scale;

        // 速度場 `v` を fBm から field 解像度で生成し、共有エンジンへ渡す。
        let v = generate_velocity(
            &self.perm,
            &d.field,
            fw,
            fh,
            div as f32,
            d.feature_scale,
            d.amplitude,
            z,
        );
        warp_with_field(&d, &mut self.inner, &frame, v, ctx)
    }
}

/// 速度場 `v`（field 解像度・field-grid px 単位）を SVF 積分し、`source` に対して
/// **順ワープ(φ⁻¹) → 内側 Process → 逆ワープ(φ) → ダウンサンプル**を適用する共通エンジン。
///
/// noise 駆動の [`InvertibleDisplace`] と map 駆動の
/// [`InvertibleDisplaceMap`](super::InvertibleDisplaceMap) が、速度場の作り方だけを替えてこの
/// コアを共有する。`warp_mode` が要求する向きだけを積分する（片方向時は積分コスト半減）。
pub(crate) fn warp_with_field<P: Process>(
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

    // 1〜2. φ・φ⁻¹ を warp_mode が要求する向きだけ積分。
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

    // 3. k× 画像解像度へアップサンプル（変位ベクトルも k*div 倍）。
    let (wk, hk) = (w * k, h * k);
    let vec_scale = (k * div) as f32;
    let quantize = desc.precision == IntermediatePrecision::U8;
    let upsample_to_k = |f: Field| upsample_field(&f, wk, hk, vec_scale);
    let phi_k = phi.map(upsample_to_k);
    let phi_inv_k = phi_inv.map(upsample_to_k);

    // 4. 入力を k× へ。
    let mut img_k = upsample_image(&FloatImage::from_frame(source), k);
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

    // 8. 1× へダウンサンプルして返す（唯一の不可避な床）。
    downsample_image(&r_k, k).to_frame(ctx)
}

#[cfg(test)]
mod tests {
    use super::field::{generate_velocity, integrate_svf, min_det_jacobian};
    use super::*;
    use video_pipeline::Pixel;

    fn ctx() -> FrameCtx {
        FrameCtx { index: 0, pts: 0, seconds: 0.0 }
    }

    /// なだらかなディテール入りのテスト画像（グラデーション＋軽い正弦）。
    fn pattern(w: u32, h: u32) -> Frame {
        let mut f = Frame::black(w, h, ctx());
        f.per_iter_row(&ctx(), |_c, y, row| {
            for (x, px) in row.iter_mut().enumerate() {
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                let r = (fx * 255.0) as u8;
                let g = (fy * 255.0) as u8;
                let b = (((fx * 6.0).sin() * 0.5 + 0.5) * 255.0) as u8;
                *px = Pixel::rgb(r, g, b);
            }
        });
        f
    }

    /// 内側領域での平均絶対差（端マージン除外）。
    fn interior_mean_abs_diff(a: &Frame, b: &Frame, margin: u32) -> f32 {
        let (w, h) = (a.width(), a.height());
        let mut sum = 0.0f64;
        let mut n = 0u64;
        for y in margin..h - margin {
            for x in margin..w - margin {
                let pa = a.get_pixel(x, y);
                let pb = b.get_pixel(x, y);
                sum += (pa.r as f32 - pb.r as f32).abs() as f64;
                sum += (pa.g as f32 - pb.g as f32).abs() as f64;
                sum += (pa.b as f32 - pb.b as f32).abs() as f64;
                n += 3;
            }
        }
        (sum / n as f64) as f32
    }

    /// 恒等 Process でのラウンドトリップが内側領域でほぼ無損失なこと（可逆性の中核）。
    #[test]
    fn roundtrip_is_near_identity() {
        let input = pattern(80, 80);
        let desc = DisplacementDescriptor::new()
            .amplitude(5.0)
            .feature_scale(32.0)
            .squaring_steps(4)
            .supersample(2);
        let mut node = InvertibleDisplace::new(desc, ());
        let out = node.process(input.clone(), ctx());
        let err = interior_mean_abs_diff(&input, &out, 16);
        assert!(err < 8.0, "ラウンドトリップ誤差が大きい: {err}");
    }

    /// SVF 積分済みの φ は全点で折りたたみ無し（det(I+∇d) > 0）。
    #[test]
    fn no_folding() {
        let perm = Perm::new(3);
        let desc = FractalNoiseDescriptor::new();
        let v = generate_velocity(&perm, &desc, 64, 64, 1.0, 24.0, 20.0, 0.0);
        let phi = integrate_svf(&v, 6);
        let min_det = min_det_jacobian(&phi);
        assert!(min_det > 0.0, "折りたたみが発生: min det = {min_det}");
    }

    /// 速度場が実際に非ゼロの変位を生むこと（同一 seed は決定的、異なる seed は相違）。
    #[test]
    fn velocity_is_deterministic_and_nonzero() {
        let desc = FractalNoiseDescriptor::new();
        let field_for = |seed: u64| {
            generate_velocity(&Perm::new(seed), &desc, 48, 48, 1.0, 24.0, 16.0, 0.0)
        };
        let a = field_for(1);
        let b = field_for(1);
        let c = field_for(2);
        let max_abs = a.vx.iter().chain(a.vy.iter()).fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_abs > 1.0, "変位がほぼゼロ: {max_abs}");
        assert_eq!(a.vx, b.vx, "同一 seed が非決定的");
        assert_ne!(a.vx, c.vx, "異なる seed が同一");
    }

    /// 同一 seed のノードは決定的に同じフレームを返す。
    #[test]
    fn node_is_deterministic() {
        let desc = DisplacementDescriptor::new().amplitude(8.0).squaring_steps(4).seed(7);
        let run = || InvertibleDisplace::new(desc, ()).process(pattern(48, 48), ctx());
        let a = run();
        let b = run();
        for y in 0..a.height() {
            for x in 0..a.width() {
                assert_eq!(a.get_pixel(x, y), b.get_pixel(x, y));
            }
        }
    }

    /// ForwardOnly は元に戻さない（出力が入力から有意にずれる）。
    #[test]
    fn forward_only_displaces() {
        let input = pattern(80, 80);
        let desc = DisplacementDescriptor::new()
            .amplitude(14.0)
            .feature_scale(28.0)
            .squaring_steps(4)
            .warp_mode(WarpMode::ForwardOnly);
        let out = InvertibleDisplace::new(desc, ()).process(input.clone(), ctx());
        let diff = interior_mean_abs_diff(&input, &out, 16);
        assert!(diff > 5.0, "ForwardOnly が変位していない: {diff}");
    }

    /// ForwardOnly → InverseOnly（同一設定）でラウンドトリップが復元すること。
    #[test]
    fn forward_then_inverse_recovers() {
        let input = pattern(80, 80);
        let desc = DisplacementDescriptor::new()
            .amplitude(6.0)
            .feature_scale(32.0)
            .squaring_steps(4)
            .seed(5);
        let fwd = InvertibleDisplace::new(desc.warp_mode(WarpMode::ForwardOnly), ())
            .process(input.clone(), ctx());
        let back = InvertibleDisplace::new(desc.warp_mode(WarpMode::InverseOnly), ())
            .process(fwd, ctx());
        let err = interior_mean_abs_diff(&input, &back, 16);
        assert!(err < 10.0, "分割ラウンドトリップ誤差が大きい: {err}");
    }

    /// スーパーサンプリングが圧縮領域の忠実度を改善する（k=2 の誤差が k=1 以下）。
    #[test]
    fn supersample_improves_or_equal() {
        let input = pattern(80, 80);
        let err_at = |k: u32| {
            let desc = DisplacementDescriptor::new()
                .amplitude(12.0)
                .feature_scale(20.0)
                .squaring_steps(5)
                .supersample(k);
            let out = InvertibleDisplace::new(desc, ()).process(input.clone(), ctx());
            interior_mean_abs_diff(&input, &out, 16)
        };
        let e1 = err_at(1);
        let e2 = err_at(2);
        assert!(e2 <= e1 + 0.01, "k=2({e2}) が k=1({e1}) より悪化");
    }
}
