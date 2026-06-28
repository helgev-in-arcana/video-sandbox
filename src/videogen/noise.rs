//! 手続き的ノイズの基底アルゴリズムとフラクタル合成。
//!
//! 基底ノイズ（[`NoiseKind`]）は格子／単体／特徴点ベースの 4 種:
//! [`Value`](NoiseKind::Value) / [`Perlin`](NoiseKind::Perlin) /
//! [`Simplex`](NoiseKind::Simplex) / [`Cellular`](NoiseKind::Cellular)。いずれも 3D で評価し、
//! 共有の置換表 [`Perm`] だけを乱数源に座標を決定的にハッシュする。
//!
//! フラクタル合成（[`FractalKind`]）はオクターブの積み上げ方で
//! [`Fbm`](FractalKind::Fbm) / [`Billow`](FractalKind::Billow) / [`Ridged`](FractalKind::Ridged) /
//! [`DomainWarp`](FractalKind::DomainWarp) の 4 種。基底とフラクタルの組み合わせと共通パラメータは
//! [`FractalNoiseDescriptor`] にまとめて設定し、[`sample`](FractalNoiseDescriptor::sample) で評価する。

mod cellular;
mod fractal;
mod perlin;
mod perm;
mod simplex;
mod value;

pub use cellular::{CellDistance, CellFeature, Cellular};
pub use fractal::{
    billow, domain_warp, fbm, ridged, DomainWarp, FractalKind, FractalNoiseDescriptor,
};
pub use perm::Perm;

/// 基本ノイズの種類。フラクタル合成（[`FractalNoiseDescriptor`]）の各オクターブが評価する
/// 3D 格子／単体／特徴点ノイズを選ぶ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseKind {
    /// 格子点に擬似乱数値を置いて補間する Value ノイズ。最も素朴で軽い。
    Value,
    /// 格子点に擬似乱数勾配を置きドット積を補間する Perlin（勾配）ノイズ。
    #[default]
    Perlin,
    /// 単体（四面体）格子で評価する Simplex ノイズ。格子方向アーティファクトが出にくい。
    Simplex,
    /// 特徴点までの距離に基づく Cellular（Worley）ノイズ。細胞・ひび割れ状。距離計量と
    /// 返す特徴量を [`Cellular`] で選ぶ。
    Cellular(Cellular),
}

/// 基底ノイズを 1 サンプル評価する。各種は概ね `[-1, 1]` を返す。
#[inline]
pub(crate) fn sample_base(perm: &Perm, kind: NoiseKind, x: f32, y: f32, z: f32) -> f32 {
    match kind {
        NoiseKind::Value => perm.value3(x, y, z),
        NoiseKind::Perlin => perm.perlin3(x, y, z),
        NoiseKind::Simplex => perm.simplex3(x, y, z),
        NoiseKind::Cellular(cfg) => perm.cellular3(x, y, z, cfg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// すべての基底×フラクタルの組が有限かつ概ね `[-1, 1]` に収まること。
    #[test]
    fn all_combos_in_range_and_finite() {
        let perm = Perm::new(42);
        let noises = [
            NoiseKind::Value,
            NoiseKind::Perlin,
            NoiseKind::Simplex,
            NoiseKind::Cellular(Cellular::new()),
        ];
        let fractals = [
            FractalKind::Fbm,
            FractalKind::Billow,
            FractalKind::Ridged,
            FractalKind::DomainWarp(DomainWarp::default()),
        ];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &noise in &noises {
            for &fractal in &fractals {
                let d = FractalNoiseDescriptor::new().noise(noise).fractal(fractal);
                for i in 0..500 {
                    let t = i as f32 * 0.137;
                    let v = d.sample(&perm, t, t * 0.7, t * 0.3);
                    assert!(v.is_finite(), "非有限: noise={noise:?} fractal={fractal:?}");
                    min = min.min(v);
                    max = max.max(v);
                }
            }
        }
        assert!(min >= -1.001 && max <= 1.001, "範囲外: min={min}, max={max}");
    }

    /// セルラーの特徴量・距離計量を変えると出力が変わること。
    #[test]
    fn cellular_config_affects_output() {
        let perm = Perm::new(1);
        let f1 = perm.cellular3(1.3, 2.7, 0.5, Cellular::new());
        let f2 = perm.cellular3(1.3, 2.7, 0.5, Cellular::new().feature(CellFeature::F2));
        let man = perm.cellular3(
            1.3,
            2.7,
            0.5,
            Cellular::new().distance(CellDistance::Manhattan),
        );
        assert_ne!(f1, f2);
        assert_ne!(f1, man);
    }

    /// 同一 seed は決定的、異なる seed は（ほぼ確実に）異なる。
    #[test]
    fn seed_is_deterministic() {
        // 格子頂点ちょうど（simplex が恒等的に 0 になる点）は避ける。
        let a = Perm::new(7).simplex3(1.7, 2.3, 0.9);
        let b = Perm::new(7).simplex3(1.7, 2.3, 0.9);
        assert_eq!(a, b);
        let c = Perm::new(8).simplex3(1.7, 2.3, 0.9);
        assert_ne!(a, c);
    }
}
