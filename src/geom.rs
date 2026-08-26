use image::{GrayImage, Rgb, RgbImage};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

pub struct RectAt {
    left: i32,
    top: i32,
}

impl Rect {
    pub fn at(x: i32, y: i32) -> RectAt {
        RectAt { left: x, top: y }
    }

    pub fn left(&self) -> i32 {
        self.left
    }

    pub fn top(&self) -> i32 {
        self.top
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl RectAt {
    pub fn of_size(self, width: u32, height: u32) -> Rect {
        Rect {
            left: self.left,
            top: self.top,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Contour {
    pub points: Vec<Point<i32>>,
    pub parent: Option<usize>,
}

pub fn find_contours(image: &GrayImage) -> Vec<Contour> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let data = image.as_raw();
    let mut visited = vec![false; width * height];
    let mut contours = Vec::new();
    let mut queue: Vec<(usize, usize)> = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            let idx = start_y * width + start_x;
            if visited[idx] || data[idx] == 0 {
                continue;
            }

            queue.clear();
            queue.push((start_x, start_y));
            visited[idx] = true;
            let mut component = Vec::new();
            while let Some((x, y)) = queue.pop() {
                component.push((x, y));
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }
                        let nidx = ny as usize * width + nx as usize;
                        if !visited[nidx] && data[nidx] != 0 {
                            visited[nidx] = true;
                            queue.push((nx as usize, ny as usize));
                        }
                    }
                }
            }

            let mut points = Vec::new();
            for &(x, y) in &component {
                let mut boundary = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
                if !boundary {
                    for dy in -1..=1i32 {
                        for dx in -1..=1i32 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;
                            if data[ny * width + nx] == 0 {
                                boundary = true;
                                break;
                            }
                        }
                        if boundary {
                            break;
                        }
                    }
                }
                if boundary {
                    points.push(Point::new(x as i32, y as i32));
                }
            }

            if points.len() >= 4 {
                contours.push(Contour {
                    points,
                    parent: None,
                });
            }
        }
    }

    contours
}

#[derive(Debug, Clone, Copy)]
pub struct Projection {
    transform: [f32; 9],
    inverse: [f32; 9],
}

fn normalize(mx: [f32; 9]) -> [f32; 9] {
    [
        mx[0] / mx[8],
        mx[1] / mx[8],
        mx[2] / mx[8],
        mx[3] / mx[8],
        mx[4] / mx[8],
        mx[5] / mx[8],
        mx[6] / mx[8],
        mx[7] / mx[8],
        1.0,
    ]
}

fn try_inverse(t: &[f32; 9]) -> Option<[f32; 9]> {
    let [t00, t01, t02, t10, t11, t12, t20, t21, t22] = t;

    let m00 = t11 * t22 - t12 * t21;
    let m01 = t10 * t22 - t12 * t20;
    let m02 = t10 * t21 - t11 * t20;

    let det = t00 * m00 - t01 * m01 + t02 * m02;

    if det.abs() < 1e-10 {
        return None;
    }

    let m10 = t01 * t22 - t02 * t21;
    let m11 = t00 * t22 - t02 * t20;
    let m12 = t00 * t21 - t01 * t20;
    let m20 = t01 * t12 - t02 * t11;
    let m21 = t00 * t12 - t02 * t10;
    let m22 = t00 * t11 - t01 * t10;

    let inv = [
        m00 / det,
        -m10 / det,
        m20 / det,
        -m01 / det,
        m11 / det,
        -m21 / det,
        m02 / det,
        -m12 / det,
        m22 / det,
    ];

    Some(normalize(inv))
}

impl Projection {
    pub fn from_control_points(from: [(f32, f32); 4], to: [(f32, f32); 4]) -> Option<Projection> {
        let (xf1, yf1, xf2, yf2, xf3, yf3, xf4, yf4) = (
            from[0].0 as f64,
            from[0].1 as f64,
            from[1].0 as f64,
            from[1].1 as f64,
            from[2].0 as f64,
            from[2].1 as f64,
            from[3].0 as f64,
            from[3].1 as f64,
        );

        let (x1, y1, x2, y2, x3, y3, x4, y4) = (
            to[0].0 as f64,
            to[0].1 as f64,
            to[1].0 as f64,
            to[1].1 as f64,
            to[2].0 as f64,
            to[2].1 as f64,
            to[3].0 as f64,
            to[3].1 as f64,
        );

        let mut a = [
            [0.0, 0.0, 0.0, -xf1, -yf1, -1.0, y1 * xf1, y1 * yf1],
            [xf1, yf1, 1.0, 0.0, 0.0, 0.0, -x1 * xf1, -x1 * yf1],
            [0.0, 0.0, 0.0, -xf2, -yf2, -1.0, y2 * xf2, y2 * yf2],
            [xf2, yf2, 1.0, 0.0, 0.0, 0.0, -x2 * xf2, -x2 * yf2],
            [0.0, 0.0, 0.0, -xf3, -yf3, -1.0, y3 * xf3, y3 * yf3],
            [xf3, yf3, 1.0, 0.0, 0.0, 0.0, -x3 * xf3, -x3 * yf3],
            [0.0, 0.0, 0.0, -xf4, -yf4, -1.0, y4 * xf4, y4 * yf4],
            [xf4, yf4, 1.0, 0.0, 0.0, 0.0, -x4 * xf4, -x4 * yf4],
        ];
        let mut b = [-y1, x1, -y2, x2, -y3, x3, -y4, x4];

        let n = 8usize;
        for col in 0..n {
            let mut pivot = col;
            for row in (col + 1)..n {
                if a[row][col].abs() > a[pivot][col].abs() {
                    pivot = row;
                }
            }
            if a[pivot][col].abs() < 1e-10 {
                return None;
            }
            a.swap(col, pivot);
            b.swap(col, pivot);

            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                a[row][col] = 0.0;
                for j in (col + 1)..n {
                    a[row][j] -= factor * a[col][j];
                }
                b[row] -= factor * b[col];
            }
        }

        let mut h = [0.0f64; 8];
        for row in (0..n).rev() {
            let mut sum = b[row];
            for j in (row + 1)..n {
                sum -= a[row][j] * h[j];
            }
            h[row] = sum / a[row][row];
        }

        let transform = normalize([
            h[0] as f32,
            h[1] as f32,
            h[2] as f32,
            h[3] as f32,
            h[4] as f32,
            h[5] as f32,
            h[6] as f32,
            h[7] as f32,
            1.0,
        ]);

        try_inverse(&transform).map(|inverse| Projection { transform, inverse })
    }

    pub fn invert(self) -> Projection {
        Projection {
            transform: self.inverse,
            inverse: self.transform,
        }
    }

    #[inline(always)]
    fn map_projective(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;
        let d = t[6] * x + t[7] * y + t[8];
        (
            (t[0] * x + t[1] * y + t[2]) / d,
            (t[3] * x + t[4] * y + t[5]) / d,
        )
    }
}

impl std::ops::Mul<(f32, f32)> for Projection {
    type Output = (f32, f32);

    fn mul(self, rhs: (f32, f32)) -> (f32, f32) {
        self.map_projective(rhs.0, rhs.1)
    }
}

impl std::ops::Mul<&(f32, f32)> for Projection {
    type Output = (f32, f32);

    fn mul(self, rhs: &(f32, f32)) -> (f32, f32) {
        self.map_projective(rhs.0, rhs.1)
    }
}

pub fn warp_bilinear_rgb(image: &RgbImage, projection: &Projection, default: Rgb<u8>, out: &mut RgbImage) {
    let projection = projection.invert();
    let (out_w, out_h) = out.dimensions();
    let (src_w, src_h) = image.dimensions();
    let src = image.as_raw();
    let src_w = src_w as f32;
    let src_h = src_h as f32;
    let mut pixel = [0u8; 3];

    for y in 0..out_h {
        for x in 0..out_w {
            let (sx, sy) = projection.map_projective(x as f32, y as f32);
            let left = sx.floor();
            let right = left + 1.0;
            let top = sy.floor();
            let bottom = top + 1.0;

            let right_weight = sx - left;
            let bottom_weight = sy - top;

            if left >= 0.0 && right < src_w && top >= 0.0 && bottom < src_h {
                let left = left as usize;
                let right = right as usize;
                let top = top as usize;
                let bottom = bottom as usize;
                let tl = (top * src_w as usize + left) * 3;
                let tr = (top * src_w as usize + right) * 3;
                let bl = (bottom * src_w as usize + left) * 3;
                let br = (bottom * src_w as usize + right) * 3;

                for c in 0..3 {
                    let top_v = src[tl + c] as f32 * (1.0 - right_weight)
                        + src[tr + c] as f32 * right_weight;
                    let bottom_v = src[bl + c] as f32 * (1.0 - right_weight)
                        + src[br + c] as f32 * right_weight;
                    let v = top_v * (1.0 - bottom_weight) + bottom_v * bottom_weight;
                    pixel[c] = if v < u8::MAX as f32 {
                        if v > 0.0 {
                            v as u8
                        } else {
                            0
                        }
                    } else {
                        u8::MAX
                    };
                }
                out.put_pixel(x, y, Rgb(pixel));
            } else {
                out.put_pixel(x, y, default);
            }
        }
    }
}
