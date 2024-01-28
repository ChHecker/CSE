use std::{collections::VecDeque, mem};

pub fn trapezoidal(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    (b - a) / 2. * (f(a) + f(b))
}

pub fn kepler(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    (b - a) / 6. * (f(a) + 4. * f((a + b) / 2.) + f(b))
}

pub fn composite_trapezoidal(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u16) -> f64 {
    let h = (b - a) / n as f64;
    let sum: f64 = (0..=n)
        .map(|i| (if i == 0 || i == n { 1. } else { 2. }, a + i as f64 * h))
        .map(|(w, x)| w * f(x))
        .sum();
    h / 2. * sum
}

pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u16) -> f64 {
    let h = (b - a) / n as f64;
    let sum: f64 = (0..=n)
        .map(|i| {
            (
                if i == 0 || i == n {
                    1.
                } else {
                    2. + 2. * (i % 2) as f64
                },
                a + i as f64 * h,
            )
        })
        .map(|(w, x)| w * f(x))
        .sum();
    h / 3. * sum
}

type Data = Vec<f64>;

pub fn adapative_trapezoidal(f: impl Fn(f64) -> f64, a: f64, b: f64, err: f64) -> f64 {
    let n = 15;
    let h = (b - a) / n as f64;

    /*
    Indexing:
    last 4 bits: from original 16 nodes
    for each depth:
        choose 1 bit left
        1: right node
    e.g. 0...00 1111: 15th node
         0...01 0000: right of the 0th node, i.e. 0 + 1/2th node
    */
    let mut values: Data = (0..=n).map(|i| f(a + i as f64 * h)).collect();
    values.reserve(1000);
    // Indexing: integral starting from point indexed in `values`
    let mut integrals: Data = values.windows(2).map(|f| h / 2. * (f[1] - f[0])).collect();
    integrals.reserve(1000);

    let mut at = AdaptiveTrapezoidal {
        values,
        integrals,
        f,
        a,
        err,
        h,
    };

    // iterate through intervals
    for i in 0..n {
        at.subdivide_trapezoidal(i, 0);
    }

    at.integrals.into_iter().sum()
}

struct AdaptiveTrapezoidal<F: Fn(f64) -> f64> {
    values: Data,
    integrals: Data,
    f: F,
    a: f64,
    err: f64,
    h: f64,
}

impl<F: Fn(f64) -> f64> AdaptiveTrapezoidal<F> {
    fn subdivide_trapezoidal(&mut self, index: usize, depth: usize) {
        let (_, x_middle) = x_lm(index, depth, self.h, self.a);
        let (index_middle, index_right) = index_mr(index, depth);

        // get current depth value and integral
        let int = self.integrals[index];
        let y_left = self.values[index];
        let y_middle = (self.f)(x_middle);
        if self.values.len() <= index_middle {
            self.values.resize(2 * self.values.len(), 0.);
        }
        self.values[index_middle] = y_middle;
        let y_right = self.values[index_right];

        let weight = self.h / 2f64.powi(depth as i32 + 2);
        let int_left = weight * (y_left + y_middle);
        let int_right = weight * (y_middle + y_right);
        let err_ref = (int_left + int_right - int).abs();

        if self.integrals.len() <= index || self.integrals.len() <= index_middle {
            self.integrals.resize(2 * self.integrals.len() + 1, 0.);
        }
        self.integrals[index] = int_left;
        self.integrals[index_middle] = int_right;

        if err_ref > self.err / 2f64.powi(depth as i32 + 1) {
            self.subdivide_trapezoidal(index, depth + 1);
            self.subdivide_trapezoidal(index_middle, depth + 1);
        }
    }
}

fn index_mr(index: usize, depth: usize) -> (usize, usize) {
    let int_part = index & 0b1111;
    let frac_part = index >> 4;

    let frac_middle = frac_part | (1usize << depth);
    let index_middle = int_part + (frac_middle << 4);

    let index_right = if depth == 0 {
        index + 1
    } else {
        let mut int_right = int_part;
        let (mut frac_right, overflowed) = frac_part
            .reverse_bits()
            .overflowing_add(1usize << (8 * mem::size_of::<usize>() - depth));
        frac_right = frac_right.reverse_bits();
        int_right += overflowed as usize;
        int_right + (frac_right << 4)
    };

    (index_middle, index_right)
}

fn index_to_dec(index: usize) -> f64 {
    let mut ret = 0.;
    ret += (index & 0b1111) as f64;
    let frac_part = index >> 4;

    for i in 0..(8 * mem::size_of::<usize>() - 4) {
        if ((frac_part >> i) & 1) == 1 {
            ret += 1. / 2f64.powi(i as i32 + 1);
        }
    }

    ret
}

fn x_lm(index: usize, depth: usize, h: f64, a: f64) -> (f64, f64) {
    let index_dec = index_to_dec(index);
    let x_left = a + index_dec * h;
    let x_middle = x_left + h / 2usize.pow(depth as u32 + 1) as f64;

    (x_left, x_middle)
}

fn aitken_neville_deque(x: &VecDeque<f64>, y: &VecDeque<f64>, x_eval: f64) -> f64 {
    let n = x.len();
    let mut f = y.clone();

    for order in 1..n {
        for i in 0..n - order {
            let k = i + order;
            f[i] = f[i + 1] + (x_eval - x[k]) / (x[k] - x[i]) * (f[i + 1] - f[i]);
        }
    }

    f[0]
}

pub fn romberg(f: impl Fn(f64) -> f64, a: f64, b: f64, err: f64) -> f64 {
    let mut h: VecDeque<f64> = VecDeque::with_capacity(8);
    h.push_back(0.5);
    let mut trapezoidal: VecDeque<f64> = VecDeque::with_capacity(8);
    trapezoidal.push_back(composite_trapezoidal(&f, a, b, 2));
    let mut int = 0.;

    let mut current_depth = 0;
    let mut current_err = f64::INFINITY;

    while current_err > 5. * err.abs() {
        current_depth += 1;
        if current_depth > 7 {
            h.pop_front();
            trapezoidal.pop_front();
        }

        h.push_back(0.5f64.powi(current_depth + 1));
        trapezoidal.push_back(composite_trapezoidal(
            &f,
            a,
            b,
            2u16.pow(current_depth as u32),
        ));

        let new_int = aitken_neville_deque(&h, &trapezoidal, 0.);
        current_err = (new_int - int).abs();
        int = new_int;
    }

    int
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_trapezoidal() {
        let int = trapezoidal(f64::exp, 0., 1.);
        let exact = f64::exp(1.) - 1.;
        assert_abs_diff_eq!(int, exact, epsilon = 0.2);
    }

    #[test]
    fn test_kepler() {
        let int = kepler(f64::exp, 0., 1.);
        let exact = f64::exp(1.) - 1.;
        assert_abs_diff_eq!(int, exact, epsilon = 0.01);
    }

    #[test]
    fn test_composite_trapezoidal() {
        let int = composite_trapezoidal(f64::exp, 0., 1., 1000);
        let exact = f64::exp(1.) - 1.;
        assert_abs_diff_eq!(int, exact, epsilon = 3e-6);
    }

    #[test]
    fn test_simpson() {
        let int = simpson(f64::exp, 0., 1., 1000);
        let exact = f64::exp(1.) - 1.;
        assert_abs_diff_eq!(int, exact, epsilon = 3e-12);
    }

    #[test]
    fn test_adaptive_trapezoidal() {
        let int = adapative_trapezoidal(f64::exp, 0., 1., 1e-10);
        let exact = f64::exp(1.) - 1.;
        assert_abs_diff_eq!(int, exact, epsilon = 1e-8);
    }

    #[test]
    fn test_index() {
        let index = 0b11_1011;
        assert_abs_diff_eq!(index_to_dec(index), 11.75, epsilon = 1e-10);

        let (index_middle, index_right) = index_mr(index, 2);
        assert_abs_diff_eq!(index_to_dec(index_middle), 11.875, epsilon = 1e-10);
        assert_abs_diff_eq!(index_to_dec(index_right), 12., epsilon = 1e-10);
    }

    #[test]
    fn test_romberg() {
        let int = romberg(f64::exp, 0., 1., 1e-10);
        let exact = f64::exp(1.) - 1.;
        assert_abs_diff_eq!(int, exact, epsilon = 1e-10);
    }
}
