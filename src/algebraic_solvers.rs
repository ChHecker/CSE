use nalgebra::{
    allocator::Allocator, ArrayStorage, Const, DefaultAllocator, DimMin, DimMinimum, ToTypenum,
};

use crate::{linalg::exact_solvers::Lu, IntoMatrix, IntoVector, IterativeResult};

pub fn newton<const D: usize, V: IntoVector<f64, D>, M: IntoMatrix<f64, D>>(
    f: impl Fn(V) -> V,
    df: impl Fn(V) -> M,
    x0: V,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<V>
where
    Const<D>: DimMin<Const<D>, Output = Const<D>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>>
        + Allocator<f64, Const<D>, Const<1>>
        + Allocator<f64, Const<D>, Buffer = ArrayStorage<f64, D, 1>>
        + Allocator<(usize, usize), DimMinimum<Const<D>, Const<D>>>,
{
    let mut a = x0;
    let mut n = 0;

    let mut f_eval = f(a.clone()).into_vector();
    let mut f_norm = f_eval.norm();

    while f_norm > epsilon && n < nmax {
        let lu = match Lu::new(df(a.clone()).into_matrix()) {
            Some(lu) => lu,
            None => return IterativeResult::Failed,
        };
        let sol = match lu.solve(&f_eval) {
            Some(s) => s,
            None => return IterativeResult::Failed,
        };
        a = V::from_vector(a.clone().into_vector() - sol);

        f_eval = f(a.clone()).into_vector();
        f_norm = f_eval.norm();

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(a)
    } else {
        IterativeResult::Converged(a)
    }
}

pub fn modified_newton<const D: usize, V: IntoVector<f64, D>, M: IntoMatrix<f64, D>>(
    f: impl Fn(V) -> V,
    df: impl Fn(V) -> M,
    x0: V,
    epsilon: f64,
    nmax: u32,
    lambda_min: f64,
) -> IterativeResult<V>
where
    Const<D>: DimMin<Const<D>, Output = Const<D>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>>
        + Allocator<f64, Const<D>, Const<1>>
        + Allocator<f64, Const<D>, Buffer = ArrayStorage<f64, D, 1>>
        + Allocator<(usize, usize), DimMinimum<Const<D>, Const<D>>>,
{
    let mut n = 0;
    let mut successes = 0;

    let mut x_1 = x0;
    let mut x_2;

    let mut lambda = 0.2;
    let mut f_1 = f(x_1.clone()).into_vector();
    let mut f_2;

    while f_1.norm() > epsilon && n < nmax {
        loop {
            let lu = match Lu::new(df(x_1.clone()).into_matrix()) {
                Some(lu) => lu,
                None => return IterativeResult::Failed,
            };
            let dx = -match lu.solve(&f_1) {
                Some(s) => s,
                None => return IterativeResult::Failed,
            };
            x_2 = V::from_vector(x_1.clone().into_vector() + lambda * dx);

            f_2 = f(x_2.clone()).into_vector();
            // affine invariant success condition
            let condition = match lu.solve(&f_2) {
                Some(s) => s,
                None => return IterativeResult::Failed,
            }
            .norm();

            n += 1;

            // successful step
            if f_2.norm() <= f_1.norm() && condition <= dx.norm() {
                successes += 1;
                break;
            }
            successes = 0;

            // step not sucessful - repeat with lower lambda
            lambda = (0.5 * lambda).clamp(lambda_min, 1.);
            // reached minimum lambda - this has to be good enough
            if lambda == lambda_min {
                break;
            }
        }

        std::mem::swap(&mut x_1, &mut x_2);
        std::mem::swap(&mut f_1, &mut f_2);

        if successes == 2 {
            lambda = (2. * lambda).clamp(lambda_min, 1.);
            successes = 0;
        }
    }

    if n == nmax {
        IterativeResult::MaxIterations(x_1)
    } else {
        IterativeResult::Converged(x_1)
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

        let n = newton(f, df, 1., 1e-8, 100).unwrap();
        assert_abs_diff_eq!(n, 3f64.powf(1. / 3.), epsilon = 1e-8);
    }

    #[test]
    fn test_modified_newton() {
        let f = |x: f64| x.powi(3) - 3.;
        let df = |x: f64| 3. * x.powi(2);

        let n = modified_newton(f, df, 1., 1e-8, 100, 1e-2).unwrap();
        assert_abs_diff_eq!(n, 3f64.powf(1. / 3.), epsilon = 1e-8);
    }
}
