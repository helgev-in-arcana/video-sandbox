use rayon::prelude::*;
use video_pipeline::{Frame, FrameCtx, Pixel};

use super::field::Field;

/// f32 精度の中間画像（RGBA, 0..255 レンジで [`Pixel`] と整合）。
///
/// ワープ・アップ／ダウンサンプルの再標本化を f32 で行い、毎ステップの 8bit 量子化を避ける
/// ためのバッファ（スペック §6「中間バッファ量子化」対策）。行優先で `data[y*w + x]`。
pub struct FloatImage {
    pub w: usize,
    pub h: usize,
    /// 各画素 `[r, g, b, a]`（0..255 の連続値）。
    pub data: Vec<[f32; 4]>,
}

impl FloatImage {
    /// 黒（不透明）で初期化した `w`×`h` の画像を作る。
    fn black(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![[0.0, 0.0, 0.0, 255.0]; w * h] }
    }

    /// [`Frame`]（u8 RGBA）から f32 画像へ変換する。
    pub fn from_frame(frame: &Frame) -> Self {
        let (w, h) = (frame.width() as usize, frame.height() as usize);
        let mut data = vec![[0.0f32; 4]; w * h];
        data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, slot) in row.iter_mut().enumerate() {
                let p = frame.get_pixel(x as u32, y as u32);
                *slot = [p.r as f32, p.g as f32, p.b as f32, p.a as f32];
            }
        });
        Self { w, h, data }
    }

    /// [`Frame`]（u8 RGBA）へ書き戻す。各チャンネルを丸めて 0..=255 にクランプ。
    pub fn to_frame(&self, ctx: FrameCtx) -> Frame {
        let mut frame = Frame::black(self.w as u32, self.h as u32, ctx);
        let to_u8 = |v: f32| v.round().clamp(0.0, 255.0) as u8;
        frame.per_iter_row(&ctx, |_ctx, y, row| {
            let base = y * self.w;
            for (x, px) in row.iter_mut().enumerate() {
                let c = self.data[base + x];
                *px = Pixel::new(to_u8(c[0]), to_u8(c[1]), to_u8(c[2]), to_u8(c[3]));
            }
        });
        frame
    }

    /// 中間バッファを 8bit に丸めてから f32 に戻す（U8 精度モードの量子化を再現する実験用）。
    pub fn quantize_u8(&mut self) {
        self.data.par_iter_mut().for_each(|c| {
            for ch in c.iter_mut() {
                *ch = ch.round().clamp(0.0, 255.0);
            }
        });
    }

    /// 連続座標 `(px, py)` をバイリニアサンプルする。端はクランプ。
    #[inline]
    pub fn bilinear(&self, px: f32, py: f32) -> [f32; 4] {
        let (w, h) = (self.w, self.h);
        let x = px.clamp(0.0, (w - 1) as f32);
        let y = py.clamp(0.0, (h - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let a = self.data[y0 * w + x0];
        let b = self.data[y0 * w + x1];
        let c = self.data[y1 * w + x0];
        let d = self.data[y1 * w + x1];
        let mut out = [0.0f32; 4];
        for ch in 0..4 {
            let top = a[ch] + (b[ch] - a[ch]) * fx;
            let bot = c[ch] + (d[ch] - c[ch]) * fx;
            out[ch] = top + (bot - top) * fy;
        }
        out
    }
}

/// 変位場 `field` で `src` を backward gather する。
///
/// 各出力画素 `(x, y)` で変位 `d = field(x, y)` を読み、`src` の `(x+dx, y+dy)` をバイリニア
/// サンプルして書く。scatter 無し・穴無し（スペック §4.1）。`field` と出力は同サイズ前提。
pub fn warp_image(src: &FloatImage, field: &Field) -> FloatImage {
    let (w, h) = (field.w, field.h);
    let mut out = FloatImage::black(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, slot) in row.iter_mut().enumerate() {
            let (dx, dy) = field.at(x, y);
            *slot = src.bilinear(x as f32 + dx, y as f32 + dy);
        }
    });
    out
}

/// `src` をバイリニアで `k` 倍に拡大する。出力画素 `(X, Y)` は元座標 `((X+0.5)/k - 0.5, …)`。
pub fn upsample_image(src: &FloatImage, k: usize) -> FloatImage {
    if k <= 1 {
        return FloatImage { w: src.w, h: src.h, data: src.data.clone() };
    }
    let (w, h) = (src.w * k, src.h * k);
    let inv = 1.0 / k as f32;
    let mut out = FloatImage::black(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5) * inv - 0.5;
        for (x, slot) in row.iter_mut().enumerate() {
            let sx = (x as f32 + 0.5) * inv - 0.5;
            *slot = src.bilinear(sx, sy);
        }
    });
    out
}

/// `src`（k× 解像度）を k×k ボックス平均で 1× に縮小する。平均は f32 で集約（唯一の床, §4.2/§6）。
pub fn downsample_image(src: &FloatImage, k: usize) -> FloatImage {
    if k <= 1 {
        return FloatImage { w: src.w, h: src.h, data: src.data.clone() };
    }
    let (w, h) = (src.w / k, src.h / k);
    let inv = 1.0 / (k * k) as f32;
    let mut out = FloatImage::black(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, slot) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; 4];
            for dy in 0..k {
                let base = ((y * k + dy) * src.w) + x * k;
                for dx in 0..k {
                    let c = src.data[base + dx];
                    for ch in 0..4 {
                        acc[ch] += c[ch];
                    }
                }
            }
            for v in &mut acc {
                *v *= inv;
            }
            *slot = acc;
        }
    });
    out
}
