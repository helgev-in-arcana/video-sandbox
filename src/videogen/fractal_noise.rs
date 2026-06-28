use std::sync::Arc;

use video_pipeline::{Frame, FrameCtx, Pipeline, Pixel};

use super::noise::{fbm, NoiseKind, Perm};

/// 時間方向の扱い方。第3軸 `z`（時間）の進め方を切り替える。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeMode {
    /// 時間連続。`z = seconds * time_scale` で隣接フレームが近いスライスを引き、滑らかに揺らぐ。
    Continuous,
    /// 非連続。`z = index * GAP` で各フレームが遠く離れたスライスを引き、実質独立になる。
    PerFrame,
}

/// PerFrame 時のフレーム間オフセット。隣接フレームの相関を消すため十分大きく取る。
/// 置換表の周期 512 の倍数だと同一スライスに当たって相関が残るため、非整数かつ
/// 512 と揃わない値にする（格子点にも乗らないので各フレームが完全に独立になる）。
const PER_FRAME_GAP: f32 = 101.137;

/// fBm フラクタルノイズを各フレームに描く、ゼロから映像を生成する [`Pipeline`] ソース。
///
/// 3 次元（x, y, 時間）の格子ノイズを [`NoiseKind`] から選び、[`TimeMode`] で時間方向の
/// 連続／非連続を切り替えられる。出力は既定でグレースケール（fBm 値を RGB に複製）で、
/// [`with_color`](Self::with_color) で任意の色マッピングに差し替えられる。
///
/// [`StillVideo`](crate::videogen::StillVideo) と同じく `index` を進めながらフレームを発行し、
/// 各フレームの [`FrameCtx`] を連番に振り直す。後段の処理ノード（例 [`Displace`] の map 入力）へ
/// そのまま流せる。
///
/// [`Displace`]: crate::nodes::Displace
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{EncodeSettings, Pipeline};
/// use video_sandbox::videogen::{FractalNoise, NoiseKind, TimeMode};
///
/// // 640x480・90 フレーム（30fps で 3 秒）の、時間連続な Perlin フラクタルノイズ。
/// FractalNoise::new(640, 480, 90)
///     .kind(NoiseKind::Perlin)
///     .time_mode(TimeMode::Continuous)
///     .scale(96.0)
///     .encode_to("noise.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct FractalNoise {
    width: u32,
    height: u32,
    frames: u64,
    kind: NoiseKind,
    time_mode: TimeMode,
    /// fBm のオクターブ数。
    octaves: u32,
    /// オクターブごとの周波数倍率。
    lacunarity: f32,
    /// オクターブごとの振幅倍率（persistence）。
    gain: f32,
    /// 特徴サイズ（px）。大きいほど模様が粗くなる。
    scale: f32,
    /// Continuous 時の時間進行速度（秒あたりの z 進み）。
    time_scale: f32,
    /// 秒数計算に使うフレームレート。
    fps: f32,
    /// 置換表の seed。
    seed: u64,
    /// 構築済み置換表（[`seed`](Self::seed) 変更時に作り直す）。
    perm: Arc<Perm>,
    /// fBm 値 `[-1,1]` → ピクセルの色マッピング。`None` ならグレースケール。
    color: Option<Arc<dyn Fn(f32) -> Pixel + Send + Sync>>,
    /// 次に発行するフレーム番号。
    index: u64,
}

impl FractalNoise {
    /// `width`×`height` のフレームを `frames` 枚生成するソースを構築する。
    ///
    /// 既定: Perlin・時間連続・octaves=5・lacunarity=2.0・gain=0.5・scale=64px・
    /// time_scale=1.0・fps=30・seed=0・グレースケール。
    pub fn new(width: u32, height: u32, frames: u64) -> Self {
        Self {
            width,
            height,
            frames,
            kind: NoiseKind::Perlin,
            time_mode: TimeMode::Continuous,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            scale: 64.0,
            time_scale: 1.0,
            fps: 30.0,
            seed: 0,
            perm: Perm::new(0),
            color: None,
            index: 0,
        }
    }

    /// 基本ノイズの種類を選ぶ。
    pub fn kind(mut self, kind: NoiseKind) -> Self {
        self.kind = kind;
        self
    }

    /// 時間方向の連続／非連続を選ぶ。
    pub fn time_mode(mut self, mode: TimeMode) -> Self {
        self.time_mode = mode;
        self
    }

    /// fBm のオクターブ数を設定する。
    pub fn octaves(mut self, n: u32) -> Self {
        self.octaves = n;
        self
    }

    /// オクターブごとの周波数倍率を設定する。
    pub fn lacunarity(mut self, l: f32) -> Self {
        self.lacunarity = l;
        self
    }

    /// オクターブごとの振幅倍率（persistence）を設定する。
    pub fn gain(mut self, g: f32) -> Self {
        self.gain = g;
        self
    }

    /// 特徴サイズ（px）を設定する。
    pub fn scale(mut self, px: f32) -> Self {
        self.scale = px;
        self
    }

    /// Continuous 時の時間進行速度（秒あたりの z 進み）を設定する。
    pub fn time_scale(mut self, s: f32) -> Self {
        self.time_scale = s;
        self
    }

    /// 秒数計算に使うフレームレートを設定する。
    pub fn fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }

    /// 置換表の seed を設定する（置換表を作り直す）。
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.perm = Perm::new(seed);
        self
    }

    /// fBm 値 `[-1, 1]` をピクセルへ写す色マッピングを設定する。既定はグレースケール。
    pub fn with_color(mut self, f: impl Fn(f32) -> Pixel + Send + Sync + 'static) -> Self {
        self.color = Some(Arc::new(f));
        self
    }
}

/// fBm 値 `[-1, 1]` を既定のグレースケールピクセルに写す。
#[inline]
fn gray(v: f32) -> Pixel {
    let g = ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
    Pixel::rgb(g, g, g)
}

impl Pipeline for FractalNoise {
    fn next_frame(&mut self) -> Option<Frame> {
        if self.index >= self.frames {
            return None;
        }
        let index = self.index;
        let seconds = index as f32 / self.fps;
        let ctx = FrameCtx { index, pts: index as i64, seconds };

        // 第3軸（時間）の座標。
        let z = match self.time_mode {
            TimeMode::Continuous => seconds * self.time_scale,
            TimeMode::PerFrame => index as f32 * PER_FRAME_GAP,
        };

        let mut frame = Frame::black(self.width, self.height, ctx);
        // クロージャに渡すため借用をローカルへ。
        let (perm, kind, scale) = (&*self.perm, self.kind, self.scale);
        let (octaves, lacunarity, gain) = (self.octaves, self.lacunarity, self.gain);
        let color = self.color.as_deref();

        frame.per_iter_row(&ctx, |_ctx, y, row| {
            let ny = y as f32 / scale;
            for (x, px) in row.iter_mut().enumerate() {
                let nx = x as f32 / scale;
                let v = fbm(perm, kind, nx, ny, z, octaves, lacunarity, gain);
                *px = match color {
                    Some(f) => f(v),
                    None => gray(v),
                };
            }
        });

        self.index += 1;
        Some(frame)
    }

    fn size_hint(&self) -> Option<u64> {
        Some(self.frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 連続モードは隣接フレーム差が小さく、PerFrame は大きい（非連続）こと。
    #[test]
    fn continuity_differs_by_mode() {
        let avg_abs_diff = |mode: TimeMode| -> f32 {
            let mut src = FractalNoise::new(48, 48, 2).time_mode(mode).scale(16.0);
            let a = src.next_frame().unwrap();
            let b = src.next_frame().unwrap();
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for y in 0..a.height() {
                for x in 0..a.width() {
                    let da = a.get_pixel(x, y).r as f32;
                    let db = b.get_pixel(x, y).r as f32;
                    sum += (da - db).abs();
                    n += 1;
                }
            }
            sum / n as f32
        };

        let cont = avg_abs_diff(TimeMode::Continuous);
        let per = avg_abs_diff(TimeMode::PerFrame);
        assert!(cont < per, "連続({cont}) は非連続({per}) より小さいはず");
    }

    /// 同一 seed は決定的、異なる seed は異なるフレームを生む。
    #[test]
    fn seed_is_deterministic() {
        let first = |seed: u64| {
            FractalNoise::new(32, 32, 1).seed(seed).next_frame().unwrap().get_pixel(10, 10)
        };
        assert_eq!(first(1), first(1));
        assert_ne!(first(1), first(2));
    }
}
