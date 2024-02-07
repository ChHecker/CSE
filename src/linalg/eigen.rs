use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector};

use crate::linalg::solve::{Lu, Qr};

pub fn power_iteration<D: Dim>(
    a: OMatrix<f64, D, D>,
    q0: OVector<f64, D>,
    err: f64,
) -> Option<(f64, OVector<f64, D>)>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, Const<1>, D>,
{
    let mut current_err = f64::INFINITY;
    let mut z = q0;
    if z.iter().all(|elem| *elem == 0.) {
        println!("Warning: Encountered zero vector as start vector in power iteration!\nReplaced 0 with 0.1.");
        z.fill(0.1);
    }
    let mut nu = 0.;

    let mut i = 0;
    let max_iterations = 10000;

    while 3. * current_err.abs() > err.abs() && i < max_iterations {
        i += 1;
        z = &a * &z;
        z.normalize_mut();
        nu = (z.transpose() * &a * &z).x;
        let r = &a * &z - nu * &z;
        current_err = r.norm() / nu.abs();
    }

    if i == max_iterations {
        return None;
    }

    Some((nu, z))
}

pub fn inverse_power_iteration<D: Dim>(
    a: OMatrix<f64, D, D>,
    q0: OVector<f64, D>,
    mu: f64,
    err: f64,
) -> Option<(f64, OVector<f64, D>)>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, Const<1>, D>,
{
    assert!(a.shape().0 == a.shape().1 && a.shape().0 == q0.shape().0);
    let n_gen = a.shape_generic().0;

    let mut current_err = f64::INFINITY;
    let mut z = q0;
    let mut sigma = 0.;
    let lu = Lu::new(&a - mu * OMatrix::<f64, D, D>::identity_generic(n_gen, n_gen))?;

    let mut i = 0;
    let max_iterations = 10000;

    while 3. * current_err.abs() > err.abs() && i < max_iterations {
        i += 1;

        z = lu.solve(&z)?;
        z.normalize_mut();
        sigma = (z.transpose() * &a * &z).x;
        let r = &a * &z - sigma * &z;
        current_err = r.norm() / sigma.abs();
    }

    if i == max_iterations {
        return None;
    }

    Some((sigma, z))
}

pub fn qr_algorithm<D: Dim>(a: OMatrix<f64, D, D>, n: usize) -> OVector<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, Const<1>, D>,
{
    let mut t = a;

    for _ in 0..n {
        let qr = Qr::new(t);
        t = qr.r * qr.q;
    }

    t.diagonal()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector, Vector3};

    use super::*;

    #[test]
    fn test_power_iteration() {
        let a = OMatrix::<f64, Const<3>, Const<3>>::new(1., 2., 3., 4., 5., 6., 7., 8., 9.);
        let q0 = OVector::<f64, Const<3>>::new(1., 2., 3.);
        let (nu, _) = power_iteration(a, q0, 0.01).unwrap();

        assert_abs_diff_eq!(nu, 16.1168, epsilon = 0.1);
    }

    #[test]
    fn test_power_iteration_dynamic() {
        let a = DMatrix::from_vec(3, 3, (1..=9).map(|i| i as f64).collect::<Vec<_>>());
        let q0 = DVector::zeros(3);
        let (nu, _) = power_iteration(a, q0, 0.01).unwrap();

        assert_abs_diff_eq!(nu, 16.1168, epsilon = 0.1);
    }

    #[test]
    fn test_inverse_power_iteration() {
        let a = OMatrix::<f64, Const<3>, Const<3>>::new(1., 2., 3., 4., 5., 6., 7., 8., 9.);
        let q0 = OVector::<f64, Const<3>>::new(1., 2., 3.);
        let (nu, q) = inverse_power_iteration(a, q0, 15., 1e-6).unwrap();

        assert_abs_diff_eq!(nu, 16.1168, epsilon = 1e-4);
        assert_abs_diff_eq!(
            q,
            Vector3::new(0.231971, 0.525322, 0.818673),
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_qr_algorithm() {
        let a = OMatrix::<f64, Const<3>, Const<3>>::new(1., 2., 3., 4., 5., 6., 7., 8., 9.);
        let t = qr_algorithm(a, 100);

        assert_abs_diff_eq!(t, Vector3::new(16.1168, -1.11684, 0.), epsilon = 1e-3);
    }
}
