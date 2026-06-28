use std::sync::Arc;

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// 基本ノイズの種類。fBm（[`fbm`]）の各オクターブが評価する格子ノイズを選ぶ。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseKind {
    /// 格子点に擬似乱数値を置いて補間する Value ノイズ。最も素朴で軽い。
    Value,
    /// 格子点に擬似乱数勾配を置きドット積を補間する Perlin（勾配）ノイズ。
    Perlin,
}

/// 256 要素の置換表（doubled で 512）。古典 Perlin と同じ仕組みで、座標ハッシュに使う。
///
/// 配列を共有参照で読むだけなので [`Sync`]。rayon の行並列クロージャから安全に参照できる。
/// 構築コストは無視できるが、フレーム間で使い回せるよう [`Arc`] で持つ前提。
pub struct Perm {
    /// `perm[i & 511]` で 0..256 の値を引く（i は 0..512）。
    perm: [u8; 512],
}

impl Perm {
    /// `seed` から Fisher–Yates で 0..256 をシャッフルして置換表を作る。
    pub fn new(seed: u64) -> Arc<Self> {
        let mut p = [0u8; 256];
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = i as u8;
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        // Fisher–Yates（末尾から）。j ∈ [0, i] を一様に選ぶ。
        for i in (1..256).rev() {
            let j = (rng.random::<u32>() as usize) % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Arc::new(Perm { perm })
    }

    #[inline]
    fn hash(&self, i: i32) -> usize {
        self.perm[(i & 511) as usize] as usize
    }
}

/// 改良 fade（quintic）。`6t^5 - 15t^4 + 10t^3`。1 次・2 次導関数が端点で 0。
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 3D 勾配（標準 12 方向 + 重複 4）とのドット積。古典 improved Perlin と同じ。
#[inline]
fn grad3(hash: usize, x: f32, y: f32, z: f32) -> f32 {
    match hash & 15 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 => -y + z,
        10 => y - z,
        11 => -y - z,
        12 => x + y,
        13 => -y + z,
        14 => -x + y,
        _ => -y - z,
    }
}

/// 格子点ハッシュから `[-1, 1)` 付近の擬似乱数値（Value ノイズ用）。
#[inline]
fn value_at(h: usize) -> f32 {
    // 256 段階を [-1, 1] に写す。
    (h as f32 / 255.0) * 2.0 - 1.0
}

impl Perm {
    /// Perlin 3D ノイズ。概ね `[-1, 1]` を返す。
    pub fn perlin3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let aaa = self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi);
        let aba = self.hash(self.hash(self.hash(xi) as i32 + yi + 1) as i32 + zi);
        let aab = self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi + 1);
        let abb = self.hash(self.hash(self.hash(xi) as i32 + yi + 1) as i32 + zi + 1);
        let baa = self.hash(self.hash(self.hash(xi + 1) as i32 + yi) as i32 + zi);
        let bba = self.hash(self.hash(self.hash(xi + 1) as i32 + yi + 1) as i32 + zi);
        let bab = self.hash(self.hash(self.hash(xi + 1) as i32 + yi) as i32 + zi + 1);
        let bbb = self.hash(self.hash(self.hash(xi + 1) as i32 + yi + 1) as i32 + zi + 1);

        let x1 = lerp(grad3(aaa, xf, yf, zf), grad3(baa, xf - 1.0, yf, zf), u);
        let x2 = lerp(grad3(aba, xf, yf - 1.0, zf), grad3(bba, xf - 1.0, yf - 1.0, zf), u);
        let y1 = lerp(x1, x2, v);
        let x3 = lerp(grad3(aab, xf, yf, zf - 1.0), grad3(bab, xf - 1.0, yf, zf - 1.0), u);
        let x4 =
            lerp(grad3(abb, xf, yf - 1.0, zf - 1.0), grad3(bbb, xf - 1.0, yf - 1.0, zf - 1.0), u);
        let y2 = lerp(x3, x4, v);
        lerp(y1, y2, w)
    }

    /// Value 3D ノイズ。概ね `[-1, 1]` を返す。
    pub fn value3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let u = fade(x - x.floor());
        let v = fade(y - y.floor());
        let w = fade(z - z.floor());

        // 8 隅の擬似乱数値を三線形補間。
        let corner = |dx: i32, dy: i32, dz: i32| -> f32 {
            value_at(self.hash(self.hash(self.hash(xi + dx) as i32 + yi + dy) as i32 + zi + dz))
        };
        let x1 = lerp(corner(0, 0, 0), corner(1, 0, 0), u);
        let x2 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
        let y1 = lerp(x1, x2, v);
        let x3 = lerp(corner(0, 0, 1), corner(1, 0, 1), u);
        let x4 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
        let y2 = lerp(x3, x4, v);
        lerp(y1, y2, w)
    }
}

/// fBm（fractional Brownian motion）。基本ノイズ（`kind`）を `octaves` 回、周波数を
/// `lacunarity` 倍、振幅を `gain` 倍しながら加算し、総振幅で正規化して概ね `[-1, 1]` に収める。
pub fn fbm(
    perm: &Perm,
    kind: NoiseKind,
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
        let n = match kind {
            NoiseKind::Value => perm.value3(x * freq, y * freq, z * freq),
            NoiseKind::Perlin => perm.perlin3(x * freq, y * freq, z * freq),
        };
        sum += n * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    if norm > 0.0 { sum / norm } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbm_in_range_and_deterministic() {
        let perm = Perm::new(42);
        let mut max = f32::MIN;
        let mut min = f32::MAX;
        for i in 0..2000 {
            let t = i as f32 * 0.137;
            for &kind in &[NoiseKind::Value, NoiseKind::Perlin] {
                let v = fbm(&perm, kind, t, t * 0.7, t * 0.3, 5, 2.0, 0.5);
                assert!(v.is_finite());
                max = max.max(v);
                min = min.min(v);
            }
        }
        // 正規化済みなので [-1, 1]（数値誤差込み）に収まる。
        assert!(min >= -1.001 && max <= 1.001, "範囲外: min={min}, max={max}");

        // 同一 seed は決定的。
        let a = Perm::new(7).perlin3(1.5, 2.5, 3.5);
        let b = Perm::new(7).perlin3(1.5, 2.5, 3.5);
        assert_eq!(a, b);
        // 異なる seed は（ほぼ確実に）異なる。
        let c = Perm::new(8).perlin3(1.5, 2.5, 3.5);
        assert_ne!(a, c);
    }
}
