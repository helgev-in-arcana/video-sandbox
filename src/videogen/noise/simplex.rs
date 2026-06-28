use super::perm::Perm;

/// 3D 単体（simplex）の歪み係数。`(sqrt(4) - 1) / 3`。
const F3: f32 = 1.0 / 3.0;
/// 3D 単体の逆歪み係数。`(1 - 1 / sqrt(4)) / 3`。
const G3: f32 = 1.0 / 6.0;

/// 12 本の標準勾配ベクトル（立方体の各辺の中点方向）。Perlin と同じ集合。
#[rustfmt::skip]
const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

/// 勾配ベクトルと変位 `(x, y, z)` のドット積。
#[inline]
fn dot(g: [f32; 3], x: f32, y: f32, z: f32) -> f32 {
    g[0] * x + g[1] * y + g[2] * z
}

impl Perm {
    /// Simplex 3D ノイズ（Gustavson 法）。概ね `[-1, 1]`。
    ///
    /// 立方格子の代わりに単体（四面体）格子で評価するため、Perlin の格子方向アーティファクト
    /// が出にくく、高次元でも評価コストが緩やかに増える。出力は `32.0` 倍で `[-1, 1]` に正規化。
    pub fn simplex3(&self, x: f32, y: f32, z: f32) -> f32 {
        // 入力点を単体格子の歪んだ座標へ。最も近い格子原点 (i, j, k) を求める。
        let s = (x + y + z) * F3;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let k = (z + s).floor();
        let t = (i + j + k) * G3;
        // 単体内のローカル座標（最初の頂点からの変位）。
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let z0 = z - (k - t);

        // (x0, y0, z0) がどの四面体に属するかで、2 番目・3 番目の頂点へのオフセットを決める。
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0) // X Y Z 順
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1) // X Z Y 順
            } else {
                (0, 0, 1, 1, 0, 1) // Z X Y 順
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1) // Z Y X 順
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1) // Y Z X 順
        } else {
            (0, 1, 0, 1, 1, 0) // Y X Z 順
        };

        // 残り 3 頂点のローカル座標（逆歪みを各段階で加える）。
        let x1 = x0 - i1 as f32 + G3;
        let y1 = y0 - j1 as f32 + G3;
        let z1 = z0 - k1 as f32 + G3;
        let x2 = x0 - i2 as f32 + 2.0 * G3;
        let y2 = y0 - j2 as f32 + 2.0 * G3;
        let z2 = z0 - k2 as f32 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let ii = i as i32;
        let jj = j as i32;
        let kk = k as i32;

        // 各頂点の寄与。半径外（t < 0）の頂点は寄与 0。
        let corner = |xc: f32, yc: f32, zc: f32, gi: usize| -> f32 {
            let tt = 0.6 - xc * xc - yc * yc - zc * zc;
            if tt < 0.0 {
                0.0
            } else {
                let tt2 = tt * tt;
                tt2 * tt2 * dot(GRAD3[gi % 12], xc, yc, zc)
            }
        };

        let n0 = corner(x0, y0, z0, self.hash3(ii, jj, kk));
        let n1 = corner(x1, y1, z1, self.hash3(ii + i1, jj + j1, kk + k1));
        let n2 = corner(x2, y2, z2, self.hash3(ii + i2, jj + j2, kk + k2));
        let n3 = corner(x3, y3, z3, self.hash3(ii + 1, jj + 1, kk + 1));

        // 寄与の和を [-1, 1] に正規化。
        32.0 * (n0 + n1 + n2 + n3)
    }
}
