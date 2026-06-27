//! ピクセルの明示的表現と基本演算。
//!
//! [`Frame`](crate::Frame) の内部バッファは RGBA8 のバイト列のままだが、ピクセル単位で
//! 触る箇所はこの [`Pixel`] を通す。`#[repr(C)]` で `[u8; 4]` と同一レイアウトなので、
//! バイトスライスとの相互変換はコピーなしで行える。
//!
//! アルファは**ストレートアルファ**として扱う（RGB はアルファで事前乗算されていない）。

use bytemuck::{Pod, Zeroable};
use rand::RngExt;
use rand_xoshiro::Xoshiro256PlusPlus;

/// RGBA8 の 1 ピクセル。各チャンネル 0–255、アルファはストレート。
///
/// `#[repr(C)]` によりメモリ上は `r, g, b, a` の順に 4 バイトで、`[u8; 4]` と同一レイアウト。
/// パディングを持たない POD なので、[`Frame`](crate::Frame) の内部バッファ
/// `Vec<Pixel>` とバイト列 `&[u8]` を [`bytemuck`] でコピーなしに相互変換できる。
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, Pod, Zeroable)]
pub struct Pixel {
    /// 赤。
    pub r: u8,
    /// 緑。
    pub g: u8,
    /// 青。
    pub b: u8,
    /// アルファ（ストレート）。255 が不透明。
    pub a: u8,
}

impl Pixel {
    /// 全チャンネルを指定して構築。
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Pixel { r, g, b, a }
    }

    /// 不透明（a = 255）のピクセルを構築。
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Pixel { r, g, b, a: 255 }
    }

    /// 不透明な黒。
    pub const BLACK: Pixel = Pixel::rgb(0, 0, 0);
    /// 不透明な白。
    pub const WHITE: Pixel = Pixel::rgb(255, 255, 255);
    /// 完全透明（全チャンネル 0）。
    pub const TRANSPARENT: Pixel = Pixel::new(0, 0, 0, 0);

    /// `[u8; 4]`（RGBA 順）から構築。
    pub const fn from_array(a: [u8; 4]) -> Self {
        Pixel { r: a[0], g: a[1], b: a[2], a: a[3] }
    }

    /// `[u8; 4]`（RGBA 順）へ変換。
    pub const fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    // --- 算術（RGB チャンネルごと・飽和。アルファは self を保持） ---

    /// チャンネルごとの飽和加算（RGB のみ。アルファは `self` を保持）。
    pub fn saturating_add(self, o: Pixel) -> Pixel {
        Pixel {
            r: self.r.saturating_add(o.r),
            g: self.g.saturating_add(o.g),
            b: self.b.saturating_add(o.b),
            a: self.a,
        }
    }

    /// チャンネルごとの飽和減算（RGB のみ。アルファは `self` を保持）。
    pub fn saturating_sub(self, o: Pixel) -> Pixel {
        Pixel {
            r: self.r.saturating_sub(o.r),
            g: self.g.saturating_sub(o.g),
            b: self.b.saturating_sub(o.b),
            a: self.a,
        }
    }

    /// チャンネルごとの乗算（0–1 正規化した積。modulate）。アルファも乗算する。
    pub fn modulate(self, o: Pixel) -> Pixel {
        let m = |x: u8, y: u8| ((x as u16 * y as u16) / 255) as u8;
        Pixel { r: m(self.r, o.r), g: m(self.g, o.g), b: m(self.b, o.b), a: m(self.a, o.a) }
    }

    /// RGB をスカラー倍して 0–255 にクランプ（明るさ調整）。アルファは保持。
    pub fn scale(self, f: f32) -> Pixel {
        let s = |x: u8| (x as f32 * f).round().clamp(0.0, 255.0) as u8;
        Pixel { r: s(self.r), g: s(self.g), b: s(self.b), a: self.a }
    }

    // --- ブレンド・合成 ---

    /// `self` と `o` を比率 `t`（0=self, 1=o）で線形補間する。全チャンネル（アルファ含む）。
    pub fn lerp(self, o: Pixel, t: f32) -> Pixel {
        let t = t.clamp(0.0, 1.0);
        let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8;
        Pixel { r: l(self.r, o.r), g: l(self.g, o.g), b: l(self.b, o.b), a: l(self.a, o.a) }
    }

    /// `self` を前景、`bg` を背景とした source-over 合成（Porter-Duff, ストレートアルファ）。
    ///
    /// 結果のアルファは `a_s + a_b (1 - a_s)`、色は `(c_s a_s + c_b a_b (1 - a_s)) / a_out`。
    pub fn over(self, bg: Pixel) -> Pixel {
        let sa = self.a as f32 / 255.0;
        let ba = bg.a as f32 / 255.0;
        let out_a = sa + ba * (1.0 - sa);
        if out_a <= f32::EPSILON {
            return Pixel::TRANSPARENT;
        }
        let c = |cs: u8, cb: u8| {
            let v = (cs as f32 * sa + cb as f32 * ba * (1.0 - sa)) / out_a;
            v.round().clamp(0.0, 255.0) as u8
        };
        Pixel {
            r: c(self.r, bg.r),
            g: c(self.g, bg.g),
            b: c(self.b, bg.b),
            a: (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
        }
    }

    // --- 明度・グレースケール ---

    /// Rec.601 輝度（0.0–255.0）。ピクセルソートのキー等に使う。
    pub fn luma(self) -> f32 {
        0.299 * self.r as f32 + 0.587 * self.g as f32 + 0.114 * self.b as f32
    }

    /// 輝度を全 RGB に適用したグレースケール（アルファは保持）。
    pub fn grayscale(self) -> Pixel {
        let y = self.luma().round().clamp(0.0, 255.0) as u8;
        Pixel { r: y, g: y, b: y, a: self.a }
    }

    /// チャンネルごとの反転（255 - 値。RGB のみ。アルファは保持）。
    pub fn invert(self) -> Pixel {
        Pixel { r: 255 - self.r, g: 255 - self.g, b: 255 - self.b, a: self.a }
    }

    // --- HSV ---

    /// HSV へ変換（アルファは引き継ぐ）。
    pub fn to_hsv(self) -> Hsv {
        let r = self.r as f32 / 255.0;
        let g = self.g as f32 / 255.0;
        let b = self.b as f32 / 255.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;

        let h = if d == 0.0 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / d) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / d) + 2.0)
        } else {
            60.0 * (((r - g) / d) + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        let s = if max == 0.0 { 0.0 } else { d / max };
        Hsv { h, s, v: max, a: self.a }
    }

    /// HSV から構築（アルファは [`Hsv::a`] を使う）。
    pub fn from_hsv(hsv: Hsv) -> Pixel {
        hsv.to_pixel()
    }

    /// 一様乱数で RGB を生成（不透明）。並列時はスレッドごとに別の `rng` を渡すこと。
    pub fn random(rng: &mut Xoshiro256PlusPlus) -> Pixel {
        Pixel::rgb(rng.random(), rng.random(), rng.random())
    }

    /// 一様乱数で RGBA を生成（アルファも乱数）。
    pub fn random_rgba(rng: &mut Xoshiro256PlusPlus) -> Pixel {
        Pixel::new(rng.random(), rng.random(), rng.random(), rng.random())
    }
}

/// HSV 色（+ ストレートアルファ）。
///
/// `h` は度（0–360, 巡回）、`s`・`v` は 0.0–1.0。
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Hsv {
    /// 色相（度, 0–360）。
    pub h: f32,
    /// 彩度（0.0–1.0）。
    pub s: f32,
    /// 明度（0.0–1.0）。
    pub v: f32,
    /// アルファ（ストレート, 0–255）。
    pub a: u8,
}

impl Hsv {
    /// 色相を `deg` 度回転させる（0–360 に正規化）。
    pub fn rotate_hue(self, deg: f32) -> Hsv {
        Hsv { h: (self.h + deg).rem_euclid(360.0), ..self }
    }

    /// 彩度をスカラー倍して 0–1 にクランプ。
    pub fn scale_saturation(self, f: f32) -> Hsv {
        Hsv { s: (self.s * f).clamp(0.0, 1.0), ..self }
    }

    /// 明度をスカラー倍して 0–1 にクランプ。
    pub fn scale_value(self, f: f32) -> Hsv {
        Hsv { v: (self.v * f).clamp(0.0, 1.0), ..self }
    }

    /// RGBA ピクセルへ変換。
    pub fn to_pixel(self) -> Pixel {
        let h = self.h.rem_euclid(360.0);
        let s = self.s.clamp(0.0, 1.0);
        let v = self.v.clamp(0.0, 1.0);
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r1, g1, b1) = match h {
            _ if h < 60.0 => (c, x, 0.0),
            _ if h < 120.0 => (x, c, 0.0),
            _ if h < 180.0 => (0.0, c, x),
            _ if h < 240.0 => (0.0, x, c),
            _ if h < 300.0 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        let q = |t: f32| ((t + m) * 255.0).round().clamp(0.0, 255.0) as u8;
        Pixel { r: q(r1), g: q(g1), b: q(b1), a: self.a }
    }
}

impl From<[u8; 4]> for Pixel {
    fn from(a: [u8; 4]) -> Self {
        Pixel::from_array(a)
    }
}

impl From<Pixel> for [u8; 4] {
    fn from(p: Pixel) -> Self {
        p.to_array()
    }
}

impl std::ops::Add for Pixel {
    type Output = Pixel;
    /// 飽和加算（[`Pixel::saturating_add`]）。
    fn add(self, o: Pixel) -> Pixel {
        self.saturating_add(o)
    }
}

impl std::ops::Sub for Pixel {
    type Output = Pixel;
    /// 飽和減算（[`Pixel::saturating_sub`]）。
    fn sub(self, o: Pixel) -> Pixel {
        self.saturating_sub(o)
    }
}

impl std::ops::Mul<f32> for Pixel {
    type Output = Pixel;
    /// スカラー倍（[`Pixel::scale`]）。
    fn mul(self, f: f32) -> Pixel {
        self.scale(f)
    }
}

impl std::ops::Mul<Pixel> for Pixel {
    type Output = Pixel;
    /// チャンネル乗算（[`Pixel::modulate`]）。
    fn mul(self, o: Pixel) -> Pixel {
        self.modulate(o)
    }
}
