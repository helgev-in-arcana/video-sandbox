use std::sync::Arc;

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// 256 要素の置換表（doubled で 512）。古典 Perlin と同じ仕組みで、座標ハッシュに使う。
///
/// 全ての基底ノイズ（[`value`](super::value)・[`perlin`](super::perlin)・
/// [`simplex`](super::simplex)・[`cellular`](super::cellular)）が、この置換表だけを乱数源に
/// 座標を決定的にハッシュする。配列を共有参照で読むだけなので [`Sync`]。rayon の行並列
/// クロージャから安全に参照でき、フレーム間で使い回せるよう [`Arc`] で持つ前提。
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

    /// 1 次元ハッシュ。`i` を周期 512 で畳んで 0..256 の値を引く。
    #[inline]
    pub(super) fn hash(&self, i: i32) -> usize {
        self.perm[(i & 511) as usize] as usize
    }

    /// 格子点 `(xi, yi, zi)` を入れ子ハッシュして 0..256 の値を返す。
    /// 各基底ノイズが格子点の勾配・値・特徴点を引く共通の入口。
    #[inline]
    pub(super) fn hash3(&self, xi: i32, yi: i32, zi: i32) -> usize {
        self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi)
    }
}

/// 改良 fade（quintic）。`6t^5 - 15t^4 + 10t^3`。1 次・2 次導関数が端点で 0。
#[inline]
pub(super) fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
pub(super) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
