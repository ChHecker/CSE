use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector};

use crate::linalg::exact_solvers::Lu;

pub fn linear<N: Dim, M: Dim>(
    a: &OMatrix<f64, N, M>,
    b: &OVector<f64, N>,
    epsilon: f64,
) -> Option<OVector<f64, M>>
where
    DefaultAllocator: Allocator<f64, N, M>
        + Allocator<f64, M, M>
        + Allocator<f64, N, Const<1>>
        + Allocator<f64, M, Const<1>>
        + Allocator<f64, Const<1>, M>,
{
    let lhs = a.tr_mul(a);
    let rhs = a.tr_mul(b);
    let lu = Lu::new(lhs)?;
    lu.solve_refine(&rhs, epsilon)
}

// TODO: Gauss-Newton

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{Matrix3x2, Vector2, Vector3};

    use super::*;

    #[test]
    fn test_normal() {
        let a = Matrix3x2::new(1., 0., 0., 1., 1., 1.);
        let b = Vector3::new(1., 1., 0.);
        let sol = linear(&a, &b, 1e-8).unwrap();
        assert_abs_diff_eq!(sol, Vector2::new(1. / 3., 1. / 3.), epsilon = 1e-8);
    }
}
