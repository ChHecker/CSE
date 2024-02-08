use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub enum NodeScheme {
    Uniform,
    Chebychev,
}

impl NodeScheme {
    fn nodes(&self, a: f64, b: f64, n: usize) -> Vec<f64> {
        match &self {
            NodeScheme::Uniform => (0..=n)
                .map(move |k| a + (b - a) * k as f64 / n as f64)
                .collect(),
            NodeScheme::Chebychev => (0..=n)
                .map(move |k| {
                    (b + a) / 2.
                        + (b - a) / 2. * ((2 * k + 1) as f64 / (2 * n + 2) as f64 * PI).cos()
                })
                .rev()
                .collect(),
        }
    }
}

pub trait Interpolation {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self;

    fn from_fn(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize, node_scheme: NodeScheme) -> Self
    where
        Self: Sized,
    {
        let x = node_scheme.nodes(a, b, n);
        let y = x.iter().map(|xi| f(*xi)).collect();
        Self::new(x, y)
    }

    fn update(&mut self, x: f64, y: f64);

    fn interpolate(&self, x: f64) -> f64;
}

pub struct BarycentricLagrange {
    w: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Interpolation for BarycentricLagrange {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        let w = x
            .iter()
            .enumerate()
            .map(|(k, xk)| {
                let prod: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(j, xj)| if j != k { xk - xj } else { 1. })
                    .product();
                1. / prod
            })
            .collect();
        Self { w, x, y }
    }

    fn update(&mut self, x: f64, y: f64) {
        self.w.iter_mut().zip(self.x.iter()).for_each(|(wk, xk)| {
            *wk /= xk - x;
        });

        let prod: f64 = self.x.iter().map(|xi| x - *xi).product();
        self.w.push(1. / prod);

        self.x.push(x);
        self.y.push(y);
    }

    fn interpolate(&self, x: f64) -> f64 {
        // TODO: Slow
        let nearest_index = self
            .x
            .iter()
            .map(|xi| (x - xi).abs())
            .enumerate()
            .min_by(|(_, x), (_, y)| x.partial_cmp(y).expect("encountered NaN"))
            .expect("empty iterator")
            .0;
        let x_near = self.x[nearest_index];

        let l: f64 = self
            .x
            .iter()
            .filter(|&&xi| xi != x_near)
            .map(|xi| x - xi)
            .product();

        let sum: f64 = self
            .x
            .iter()
            .zip(self.y.iter())
            .zip(self.w.iter())
            .map(|((xi, yi), wi)| {
                if *xi == x_near {
                    wi * yi
                } else {
                    wi * (x - x_near) / (x - xi) * yi
                }
            })
            .sum();
        l * sum
    }
}

#[derive(Clone, Debug, Default)]
pub struct AitkenNeville {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Interpolation for AitkenNeville {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self { x, y }
    }

    fn update(&mut self, x: f64, y: f64) {
        self.x.push(x);
        self.y.push(y);
    }

    fn interpolate(&self, x: f64) -> f64 {
        let n = self.x.len();
        let mut f = self.y.clone();

        for order in 1..n {
            for i in 0..n - order {
                let k = i + order;
                f[i] = f[i + 1] + (x - self.x[k]) / (self.x[k] - self.x[i]) * (f[i + 1] - f[i]);
            }
        }

        f[0]
    }
}

#[derive(Clone, Debug)]
pub struct Newton {
    f_diag: Vec<f64>,
    f_last_row: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Newton {
    fn divided_difference(&mut self) {
        let n = self.x.len();
        let mut f = self.y.clone();

        for order in 1..n {
            for i in 0..n - order {
                let k = i + order;
                f[i] = (f[i + 1] - f[i]) / (self.x[k] - self.x[i]);
            }
            self.f_last_row[order] = f[n - order - 1];
            self.f_diag[order] = f[0];
        }
    }
}

impl Interpolation for Newton {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len();
        let mut newton = Self {
            f_diag: vec![y[0]; n],
            f_last_row: vec![y[n - 1]; n],
            x,
            y,
        };
        newton.divided_difference();

        newton
    }

    fn update(&mut self, x: f64, y: f64) {
        let n = self.y.len();

        self.x.push(x);
        self.y.push(y);

        let mut f = vec![y; n + 1];
        for (i, fi) in self.f_last_row.iter().enumerate() {
            f[i + 1] = (f[i] - fi) / (self.x[n] - self.x[n - i - 1]);
        }

        self.f_diag.push(f[n]);
        self.f_last_row = f;
    }

    fn interpolate(&self, x: f64) -> f64 {
        let mut f = self.f_diag[self.x.len() - 1];
        for (xi, fi) in self.x.iter().zip(self.f_diag.iter()).rev().skip(1) {
            f = f * (x - xi) + fi;
        }

        f
    }
}

#[derive(Clone, Debug)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    y_prime: Vec<f64>,
    left_boundary: f64,
    right_boundary: f64,
}

impl CubicSpline {
    pub fn new(x: Vec<f64>, y: Vec<f64>, left_boundary: f64, right_boundary: f64) -> Self {
        let n = x.len();

        let mut ret = Self {
            x,
            y,
            y_prime: vec![0.; n],
            left_boundary,
            right_boundary,
        };
        ret.calculate_yprime();

        ret
    }

    fn calculate_yprime(&mut self) {
        let n = self.x.len();

        let y001 = (self.y[1] - self.left_boundary) / (self.x[1] - self.x[0]);
        let y1nn = (self.right_boundary - self.y[n - 2]) / (self.x[n - 1] - self.x[n - 2]);

        let mut c = vec![1. / 2.; n - 1];
        let mut d = vec![y001 / 2.; n - 1];

        for i in 1..n - 1 {
            let mu = (self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1]);
            let lambda = (self.x[i + 1] - self.x[i]) / (self.x[i + 1] - self.x[i - 1]);

            let y_diff1 = (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1]);
            let y_diff2 = (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i]);
            let y_diff = (y_diff2 - y_diff1) / (self.x[i + 1] - self.x[i - 1]);

            c[i] = lambda / (2. - c[i - 1] * mu);
            d[i] = (y_diff - d[i - 1] * mu) / (2. - c[i - 1] * mu)
        }
        self.y_prime[n - 1] = (y1nn - d[n - 2]) / (2. - c[n - 2]);

        for i in (0..n - 1).rev() {
            self.y_prime[i] = d[i] - c[i] * self.y_prime[i + 1];
        }
    }

    pub fn interpolate(&self, x: f64) -> f64 {
        let mut i_near = None;
        let mut x_near = None;
        for (i, x_window) in self.x.windows(2).enumerate() {
            if x_window[1] >= x {
                i_near = Some(i);
                x_near = Some(x_window[0]);
                break;
            }
        }
        let i = i_near.expect("x out of range");
        let h = self.x[i + 1] - self.x[i];
        let xi = (x - x_near.expect("x out of range")) / h;

        self.y[i] * (1. - xi)
            + self.y[i + 1] * xi
            + h * h * self.y_prime[i] * (-xi / 3. + xi * xi / 2. - xi * xi * xi / 6.)
            + h * h * self.y_prime[i + 1] * (-xi / 6. + xi * xi * xi / 6.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn barycentric_lagrange_const() {
        const D: usize = 10;
        let x: Vec<f64> = (0..D).map(|i| i as f64).collect();
        let y = vec![0.; D];

        let bl = BarycentricLagrange::new(x, y);

        for x in 0..D + 1 {
            let y = bl.interpolate(x as f64 - 0.5);
            assert_abs_diff_eq!(y, 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn barycentric_lagrange_cos() {
        let upper = 5;
        let d = upper * 10;
        let x: Vec<f64> = (0..d).map(|i| upper as f64 * i as f64 / d as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.cos()).collect();

        let bl = BarycentricLagrange::new(x, y);

        let x: Vec<f64> = (0..10 * d)
            .map(|i| upper as f64 * i as f64 / (10 * d) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| bl.interpolate(*xi)).collect();

        for (xi, yi) in x.into_iter().zip(y) {
            assert_abs_diff_eq!(yi, xi.cos(), epsilon = 0.1);
        }
    }

    #[test]
    fn aitken_neville_const() {
        const D: usize = 10;
        let x: Vec<f64> = (0..D).map(|i| i as f64).collect();
        let y = vec![0.; D];

        let an = AitkenNeville::new(x, y);

        for x in 0..D + 1 {
            let y = an.interpolate(x as f64 - 0.5);
            assert_abs_diff_eq!(y, 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn aitken_neville_cos() {
        let upper = 5;
        let d = upper * 10;
        let x: Vec<f64> = (0..d).map(|i| upper as f64 * i as f64 / d as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.cos()).collect();

        let an = AitkenNeville::new(x, y);

        let x: Vec<f64> = (0..10 * d)
            .map(|i| upper as f64 * i as f64 / (10 * d) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| an.interpolate(*xi)).collect();

        for (xi, yi) in x.into_iter().zip(y) {
            assert_abs_diff_eq!(yi, xi.cos(), epsilon = 0.1);
        }
    }

    #[test]
    fn newton_const() {
        const D: usize = 10;
        let x: Vec<f64> = (0..D).map(|i| i as f64).collect();
        let y = vec![0.; D];

        let ne = Newton::new(x, y);

        for x in 0..D + 1 {
            let y = ne.interpolate(x as f64 - 0.5);
            assert_abs_diff_eq!(y, 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn newton_cos() {
        let upper = 5;
        let d = upper * 10;
        let x: Vec<f64> = (0..d).map(|i| upper as f64 * i as f64 / d as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.cos()).collect();

        let ne = Newton::new(x, y);

        let x: Vec<f64> = (0..10 * d)
            .map(|i| upper as f64 * i as f64 / (10 * d) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| ne.interpolate(*xi)).collect();

        for (xi, yi) in x.into_iter().zip(y) {
            assert_abs_diff_eq!(yi, xi.cos(), epsilon = 0.1);
        }
    }

    #[test]
    fn cubic_spline_const() {
        const D: usize = 10;
        let x: Vec<f64> = (0..D).map(|i| i as f64).collect();
        let y = vec![0.; D];

        let bl = CubicSpline::new(x, y, 0., 0.);

        for x in 0..D - 1 {
            let y = bl.interpolate(x as f64 + 0.5);
            assert_abs_diff_eq!(y, 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn cubic_spline_cos() {
        let upper = 5;
        let d = upper * 10;
        let x: Vec<f64> = (0..=d)
            .map(|i| upper as f64 * i as f64 / d as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.cos()).collect();

        let bl = CubicSpline::new(x, y, 0., -(upper as f64).sin());

        let x: Vec<f64> = (0..=10 * d)
            .map(|i| upper as f64 * i as f64 / (10 * d) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| bl.interpolate(*xi)).collect();

        for (xi, yi) in x.into_iter().zip(y) {
            assert_abs_diff_eq!(yi, xi.cos(), epsilon = 0.1);
        }
    }

    #[test]
    fn test_newton_exam() {
        let x = vec![-1., 0., 2.];
        let y = vec![8., -4., 2.];
        let ne = Newton::new(x, y);
        dbg!(&ne.f_diag);
        dbg!(&ne.f_last_row);
        dbg!(ne.interpolate(0.));
        dbg!(ne.interpolate(1.));
    }
}
