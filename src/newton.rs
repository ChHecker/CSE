use crate::{linalg::solve::Lu, IterativeResult};

pub fn newton<const D: usize, V: IntoVector<f64, D>, M: IntoMatrix<f64, D>>(
    f: impl Fn(V) -> V,
    df: impl Fn(V) -> M,
    x0: V,
) -> IterativeResult<V> {
    let nmax = 100;
    let epsilon = 1e-8;

    let mut a = x0;
    let mut n = 0;
    while f(a.clone()).into_vector().norm() > epsilon && n < nmax {
        let lu = match Lu::new(df(a.clone()).into_matrix()) {
            Some(lu) => lu,
            None => return IterativeResult::Failed,
        };
        let sol = match lu.solve(&f(a.clone()).into_vector()) {
            Some(s) => s,
            None => return IterativeResult::Failed,
        };
        a = V::from_vector(a.clone().into_vector() - sol);
        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(a)
    } else {
        IterativeResult::Converged(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_newton() {
        let f = |x: f64| x.powi(3) - 3.;
        let df = |x: f64| 3. * x.powi(2);

        let n = newton(f, df, 1.).unwrap();
        assert_abs_diff_eq!(n, 3f64.powf(1. / 3.), epsilon = 1e-8);
    }
}
