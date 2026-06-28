use rayon::prelude::*;
use video_pipeline::Frame;

/// 2ch（vx, vy）の変位場。SoA で `vx`/`vy` を別バッファに持つ（スペック §2 推奨）。
///
/// フロー写像 φ を**相対変位形** `φ(x) = x + d(x)` で保持する（絶対座標でなく変位）。squaring も
/// gather も変位形が書きやすい。値の単位は格子1セル＝1px（このグリッド基準の px）。
pub struct Field {
    pub w: usize,
    pub h: usize,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
}

impl Field {
    fn zeros(w: usize, h: usize) -> Self {
        Self { w, h, vx: vec![0.0; w * h], vy: vec![0.0; w * h] }
    }

    /// 格子点 `(x, y)` の変位を読む。
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> (f32, f32) {
        let i = y * self.w + x;
        (self.vx[i], self.vy[i])
    }

    /// 連続座標 `(px, py)` の変位をバイリニアサンプルする。端はクランプ（スペック §3.2）。
    #[inline]
    fn sample(&self, px: f32, py: f32) -> (f32, f32) {
        let (w, h) = (self.w, self.h);
        let x = px.clamp(0.0, (w - 1) as f32);
        let y = py.clamp(0.0, (h - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let s = |buf: &[f32]| {
            let top = lerp(buf[y0 * w + x0], buf[y0 * w + x1], fx);
            let bot = lerp(buf[y1 * w + x0], buf[y1 * w + x1], fx);
            lerp(top, bot, fy)
        };
        (s(&self.vx), s(&self.vy))
    }
}

/// 外部マップフレームの **R→vx, G→vy** チャンネルから 2ch 速度場 `v` を生成する。
///
/// `0..=255` → `-1..=1`（128 が変位ゼロ）で正規化し `amplitude`(1×-px) を掛ける。
/// 戻り値は **field-grid px** 単位（`amplitude / field_divisor`）。後段 [`upsample_field`] で
/// `k * field_divisor` 倍して k×-px にすると実効変位 = `amplitude`(1×-px) と一致する。
/// `map` のサイズが field 解像度と違っても比例最近傍サンプルするので任意サイズを受け付ける。
pub fn velocity_from_map(
    map: &Frame,
    w: usize,
    h: usize,
    field_divisor: f32,
    amplitude: f32,
) -> Field {
    let mut field = Field::zeros(w, h);
    let amp = amplitude / field_divisor;
    let (mw, mh) = (map.width(), map.height());
    field
        .vx
        .par_chunks_mut(w)
        .zip(field.vy.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (vx_row, vy_row))| {
            let my = ((y as u32 * mh) / h.max(1) as u32).min(mh - 1);
            for x in 0..w {
                let mx = ((x as u32 * mw) / w.max(1) as u32).min(mw - 1);
                let p = map.get_pixel(mx, my);
                vx_row[x] = (p.r as f32 / 127.5 - 1.0) * amp;
                vy_row[x] = (p.g as f32 / 127.5 - 1.0) * amp;
            }
        });
    field
}

/// squaring 1 ステップ。`φ∘φ` を変位形で書くと `d'(x) = d(x) + d(x + d(x))`（スペック §3.3）。
fn square(src: &Field, dst: &mut Field) {
    let w = src.w;
    dst.vx
        .par_chunks_mut(w)
        .zip(dst.vy.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (vx_row, vy_row))| {
            for x in 0..w {
                let (cx, cy) = src.at(x, y);
                let (sx, sy) = src.sample(x as f32 + cx, y as f32 + cy);
                vx_row[x] = cx + sx;
                vy_row[x] = cy + sy;
            }
        });
}

/// SVF 積分（scaling-and-squaring, ping-pong）。`φ₀ = v / 2^N` から `φ ← φ∘φ` を N 回（スペック §3.4）。
///
/// 逆写像は `v` の符号を反転して同じ関数を呼ぶだけ（`φ⁻¹ = exp(-v)`）。
pub fn integrate_svf(v: &Field, n: u32) -> Field {
    let (w, h) = (v.w, v.h);
    let s = 1.0 / (1u32 << n) as f32;
    let mut a = Field {
        w,
        h,
        vx: v.vx.iter().map(|&c| c * s).collect(),
        vy: v.vy.iter().map(|&c| c * s).collect(),
    };
    let mut b = Field::zeros(w, h);
    for _ in 0..n {
        square(&a, &mut b);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// 場を `(out_w, out_h)` へバイリニア拡大し、**変位ベクトルも `vec_scale` 倍**する。
///
/// グリッド解像度を上げると同じ物理変位の px 数も増えるので、ベクトルのスケーリングが必須
/// （これを忘れると k× で変位が `1/vec_scale` に縮む）。
pub fn upsample_field(field: &Field, out_w: usize, out_h: usize, vec_scale: f32) -> Field {
    let mut out = Field::zeros(out_w, out_h);
    let sx = field.w as f32 / out_w as f32;
    let sy = field.h as f32 / out_h as f32;
    out.vx
        .par_chunks_mut(out_w)
        .zip(out.vy.par_chunks_mut(out_w))
        .enumerate()
        .for_each(|(y, (vx_row, vy_row))| {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            for x in 0..out_w {
                let fx = (x as f32 + 0.5) * sx - 0.5;
                let (dx, dy) = field.sample(fx, fy);
                vx_row[x] = dx * vec_scale;
                vy_row[x] = dy * vec_scale;
            }
        });
    out
}

/// `det(I + ∇d)` の最小値を内部格子点（中心差分）で返す（折りたたみ検証用）。
#[cfg(test)]
pub fn min_det_jacobian(field: &Field) -> f32 {
    let (w, h) = (field.w, field.h);
    if w < 3 || h < 3 {
        return f32::INFINITY;
    }
    (1..h - 1)
        .into_par_iter()
        .map(|y| {
            let mut local = f32::INFINITY;
            for x in 1..w - 1 {
                let (xp, _) = field.at(x + 1, y);
                let (xm, _) = field.at(x - 1, y);
                let (_, yp) = field.at(x, y + 1);
                let (_, ym) = field.at(x, y - 1);
                let (_, vyxp) = field.at(x + 1, y);
                let (_, vyxm) = field.at(x - 1, y);
                let (vxyp, _) = field.at(x, y + 1);
                let (vxym, _) = field.at(x, y - 1);
                let dvx_dx = (xp - xm) * 0.5;
                let dvy_dy = (yp - ym) * 0.5;
                let dvy_dx = (vyxp - vyxm) * 0.5;
                let dvx_dy = (vxyp - vxym) * 0.5;
                let det = (1.0 + dvx_dx) * (1.0 + dvy_dy) - dvx_dy * dvy_dx;
                local = local.min(det);
            }
            local
        })
        .reduce(|| f32::INFINITY, f32::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_pipeline::{Frame, FrameCtx, Pixel};

    fn ctx() -> FrameCtx {
        FrameCtx { index: 0, pts: 0, seconds: 0.0 }
    }

    fn sinusoidal_map(w: u32, h: u32) -> Frame {
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

    /// SVF 積分済みの φ は全点で折りたたみ無し（det(I+∇d) > 0）。
    #[test]
    fn no_folding() {
        let map = sinusoidal_map(64, 64);
        let v = velocity_from_map(&map, 64, 64, 1.0, 20.0);
        let phi = integrate_svf(&v, 6);
        let min_det = min_det_jacobian(&phi);
        assert!(min_det > 0.0, "折りたたみが発生: min det = {min_det}");
    }
}
