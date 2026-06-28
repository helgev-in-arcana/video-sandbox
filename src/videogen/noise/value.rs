use super::perm::{fade, lerp, Perm};

/// 格子点ハッシュから `[-1, 1]` の擬似乱数値。
#[inline]
fn value_at(h: usize) -> f32 {
    // 256 段階を [-1, 1] に写す。
    (h as f32 / 255.0) * 2.0 - 1.0
}

impl Perm {
    /// Value 3D ノイズ。格子点に擬似乱数値を置き三線形補間する。概ね `[-1, 1]`。
    ///
    /// 最も素朴で軽い基底ノイズ。勾配を持たないため Perlin より角張った見た目になる。
    pub fn value3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let u = fade(x - x.floor());
        let v = fade(y - y.floor());
        let w = fade(z - z.floor());

        // 8 隅の擬似乱数値を三線形補間。
        let corner = |dx: i32, dy: i32, dz: i32| -> f32 {
            value_at(self.hash3(xi + dx, yi + dy, zi + dz))
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
