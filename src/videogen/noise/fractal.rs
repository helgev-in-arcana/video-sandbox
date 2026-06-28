use super::{sample_base, NoiseKind, Perm};

/// フラクタル合成の方式。基底ノイズ（[`NoiseKind`]）の複数オクターブをどう積み上げるか。
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum FractalKind {
    /// fBm（fractional Brownian motion）。素直な重ね合わせ。雲・地形の基本。
    #[default]
    Fbm,
    /// Billow。各オクターブの絶対値を使い、丸い塊（雲・煙）状になる。
    Billow,
    /// Ridged Multi-Fractal。`1 - |n|` を二乗し前段でフィードバックを掛け、鋭い稜線を作る（山稜・稲妻）。
    Ridged,
    /// Domain Warping。座標自体を fBm 場で歪めてから評価し、渦巻く有機的な模様を作る。
    DomainWarp(DomainWarp),
}

/// ドメインワーピングの設定。座標を歪める fBm 場の強さとオクターブ数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainWarp {
    /// 変位の強さ。大きいほど大胆に歪む。
    pub strength: f32,
    /// 歪み場（ワープ用 fBm）のオクターブ数。
    pub octaves: u32,
}

impl Default for DomainWarp {
    fn default() -> Self {
        Self { strength: 4.0, octaves: 4 }
    }
}

/// フラクタルノイズ 1 サンプルの全設定を束ねるディスクリプタ。
///
/// 基底ノイズ（[`NoiseKind`]）とフラクタル合成方式（[`FractalKind`]）、および共通の
/// オクターブパラメータ（octaves / lacunarity / gain）を一括で持つ。[`sample`](Self::sample)
/// に置換表と座標を渡すと、概ね `[-1, 1]` のスカラーを返す。
///
/// # 使用例
///
/// ```
/// use video_sandbox::videogen::{
///     Cellular, DomainWarp, FractalKind, FractalNoiseDescriptor, NoiseKind, Perm,
/// };
///
/// // Simplex を Ridged 合成。
/// let d = FractalNoiseDescriptor::new()
///     .noise(NoiseKind::Simplex)
///     .fractal(FractalKind::Ridged)
///     .octaves(6);
/// let v = d.sample(&Perm::new(0), 1.0, 2.0, 0.0);
/// assert!((-1.0..=1.0).contains(&v));
///
/// // セルラーノイズをドメインワーピング。
/// let warped = FractalNoiseDescriptor::new()
///     .noise(NoiseKind::Cellular(Cellular::new()))
///     .fractal(FractalKind::DomainWarp(DomainWarp::default()));
/// # let _ = warped;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractalNoiseDescriptor {
    /// 各オクターブが評価する基底ノイズ。
    pub noise: NoiseKind,
    /// オクターブの積み上げ方。
    pub fractal: FractalKind,
    /// オクターブ数。
    pub octaves: u32,
    /// オクターブごとの周波数倍率。
    pub lacunarity: f32,
    /// オクターブごとの振幅倍率（persistence）。
    pub gain: f32,
}

impl Default for FractalNoiseDescriptor {
    fn default() -> Self {
        Self {
            noise: NoiseKind::Perlin,
            fractal: FractalKind::Fbm,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }
}

impl FractalNoiseDescriptor {
    /// 既定（Perlin・fBm・octaves=5・lacunarity=2.0・gain=0.5）のディスクリプタを作る。
    pub fn new() -> Self {
        Self::default()
    }

    /// 基底ノイズを差し替える。
    pub const fn noise(mut self, noise: NoiseKind) -> Self {
        self.noise = noise;
        self
    }

    /// フラクタル合成方式を差し替える。
    pub const fn fractal(mut self, fractal: FractalKind) -> Self {
        self.fractal = fractal;
        self
    }

    /// オクターブ数を設定する。
    pub const fn octaves(mut self, n: u32) -> Self {
        self.octaves = n;
        self
    }

    /// オクターブごとの周波数倍率を設定する。
    pub const fn lacunarity(mut self, l: f32) -> Self {
        self.lacunarity = l;
        self
    }

    /// オクターブごとの振幅倍率（persistence）を設定する。
    pub const fn gain(mut self, g: f32) -> Self {
        self.gain = g;
        self
    }

    /// 座標 `(x, y, z)` でフラクタルノイズを 1 サンプル評価する。概ね `[-1, 1]`。
    #[inline]
    pub fn sample(&self, perm: &Perm, x: f32, y: f32, z: f32) -> f32 {
        match self.fractal {
            FractalKind::Fbm => {
                fbm(perm, self.noise, x, y, z, self.octaves, self.lacunarity, self.gain)
            }
            FractalKind::Billow => {
                billow(perm, self.noise, x, y, z, self.octaves, self.lacunarity, self.gain)
            }
            FractalKind::Ridged => {
                ridged(perm, self.noise, x, y, z, self.octaves, self.lacunarity, self.gain)
            }
            FractalKind::DomainWarp(dw) => domain_warp(
                perm,
                self.noise,
                dw,
                x,
                y,
                z,
                self.octaves,
                self.lacunarity,
                self.gain,
            ),
        }
    }
}

/// fBm（fractional Brownian motion）。基底ノイズを `octaves` 回、周波数を `lacunarity` 倍、
/// 振幅を `gain` 倍しながら加算し、総振幅で正規化して概ね `[-1, 1]` に収める。
///
/// 旧 API 互換の自由関数。新規コードでは [`FractalNoiseDescriptor`] を推奨。
#[allow(clippy::too_many_arguments)]
pub fn fbm(
    perm: &Perm,
    noise: NoiseKind,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let mut freq = 1.0;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves.max(1) {
        sum += sample_base(perm, noise, x * freq, y * freq, z * freq) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    if norm > 0.0 { sum / norm } else { 0.0 }
}

/// Billow ノイズ。各オクターブで `2 * |n| - 1` を取り、丸い塊状（雲・煙）にする。概ね `[-1, 1]`。
#[allow(clippy::too_many_arguments)]
pub fn billow(
    perm: &Perm,
    noise: NoiseKind,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let mut freq = 1.0;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves.max(1) {
        let n = sample_base(perm, noise, x * freq, y * freq, z * freq);
        sum += (2.0 * n.abs() - 1.0) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    if norm > 0.0 { sum / norm } else { 0.0 }
}

/// Ridged Multi-Fractal。`(1 - |n|)^2` を前段の出力でフィードバック重み付けし、鋭い稜線を作る。
/// `[0, 1]` を `[-1, 1]` に写して返す。
#[allow(clippy::too_many_arguments)]
pub fn ridged(
    perm: &Perm,
    noise: NoiseKind,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let mut freq = 1.0;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut weight = 1.0f32;
    for _ in 0..octaves.max(1) {
        let n = sample_base(perm, noise, x * freq, y * freq, z * freq);
        // n=0 で 1 になる稜線。二乗で尖らせる。
        let mut signal = 1.0 - n.abs();
        signal *= signal;
        // 前オクターブの強い箇所だけを次に通す（多重フラクタルの特徴）。
        signal *= weight;
        weight = (signal * 2.0).clamp(0.0, 1.0);
        sum += signal * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    let v = if norm > 0.0 { sum / norm } else { 0.0 };
    (v * 2.0 - 1.0).clamp(-1.0, 1.0)
}

/// ドメインワーピング。座標を fBm 由来の変位ベクトルで歪めてから fBm を評価する。概ね `[-1, 1]`。
///
/// 3 軸の歪み場は固定オフセットで互いに相関を消している（Inigo Quilez の手法に準拠）。
#[allow(clippy::too_many_arguments)]
pub fn domain_warp(
    perm: &Perm,
    noise: NoiseKind,
    dw: DomainWarp,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let wo = dw.octaves;
    // 各軸の変位を独立な fBm から取る（座標に固定オフセットを足して相関を断つ）。
    let wx = fbm(perm, noise, x, y, z, wo, lacunarity, gain);
    let wy = fbm(perm, noise, x + 5.2, y + 1.3, z + 2.7, wo, lacunarity, gain);
    let wz = fbm(perm, noise, x + 2.8, y + 7.1, z + 4.4, wo, lacunarity, gain);
    let s = dw.strength;
    fbm(perm, noise, x + s * wx, y + s * wy, z + s * wz, octaves, lacunarity, gain)
}
