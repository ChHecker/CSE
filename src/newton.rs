use nalgebra::{
    allocator::Allocator, ArrayStorage, Const, DefaultAllocator, DimMin, DimMinimum, ToTypenum,
};

use crate::{linalg::Lu, IntoMatrix, IntoVector};

pub enum NewtonResult<V> {
    Converged(V),
    MaxIteratoration(V),
    Failed,
}

impl<V> NewtonResult<V> {
    pub fn unwrap(self) -> V {
        match self {
            NewtonResult::Converged(v) => v,
            NewtonResult::MaxIteratoration(v) => v,
            NewtonResult::Failed => panic!("called unwrap on Failed"),
        }
    }

    pub fn successful_or<E>(self, err: E) -> Result<V, E> {
        match self {
            NewtonResult::Converged(v) => Ok(v),
            NewtonResult::MaxIteratoration(v) => Ok(v),
            NewtonResult::Failed => Err(err),
        }
    }

    pub fn convergent_or<E>(self, err: E) -> Result<V, E> {
        match self {
            NewtonResult::Converged(v) => Ok(v),
            NewtonResult::MaxIteratoration(_) => Err(err),
            NewtonResult::Failed => Err(err),
        }
    }
}

pub fn newton<const D: usize, V: IntoVector<f64, D>, M: IntoMatrix<f64, D>>(
    f: impl Fn(V) -> V,
    df: impl Fn(V) -> M,
    x0: V,
) -> NewtonResult<V>
where
    Const<D>: DimMin<Const<D>, Output = Const<D>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>>
        + Allocator<f64, Const<D>, Const<1>>
        + Allocator<f64, Const<D>, Buffer = ArrayStorage<f64, D, 1>>
        + Allocator<(usize, usize), DimMinimum<Const<D>, Const<D>>>,
{
    let nmax = 100;
    let epsilon = 1e-8;

    let mut a = x0;
    let mut n = 0;
    while f(a.clone()).into_vector().norm() > epsilon && n < nmax {
        let lu = match Lu::new(df(a.clone()).into_matrix()) {
            Some(lu) => lu,
            None => return NewtonResult::Failed,
        };
        let sol = match lu.solve(&f(a.clone()).into_vector()) {
            Some(s) => s,
            None => return NewtonResult::Failed,
        };
        a = V::from_vector(a.clone().into_vector() - sol);
        n += 1;
    }

    if n == nmax {
        NewtonResult::MaxIteratoration(a)
    } else {
        NewtonResult::Converged(a)
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
