use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector};

use crate::IterativeResult;

pub fn jacobi<D: Dim>(
    a: &OMatrix<f64, D, D>,
    b: &OVector<f64, D>,
    x0: OVector<f64, D>,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = a.shape().0;
    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut x = x0;

    while res > epsilon && n < nmax {
        let mut x_prime: OVector<f64, D> = OVector::zeros_generic(x.shape_generic().0, Const);

        for i in 0..dim {
            let mut sum = 0.;
            for j in 0..dim {
                sum += a[(i, j)] * x[j];
            }
            x_prime[i] = x[i] + (b[i] - sum) / a[(i, i)];
        }

        x = x_prime;
        res = (b - a * &x).norm();

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(x)
    } else {
        IterativeResult::Converged {
            result: x,
            iterations: n,
        }
    }
}

pub fn gauss_seidel<D: Dim>(
    a: &OMatrix<f64, D, D>,
    b: &OVector<f64, D>,
    x0: OVector<f64, D>,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    successive_overrelaxation(a, b, x0, epsilon, nmax, 1.)
}

pub fn successive_overrelaxation<D: Dim>(
    a: &OMatrix<f64, D, D>,
    b: &OVector<f64, D>,
    x0: OVector<f64, D>,
    epsilon: f64,
    nmax: u32,
    omega: f64,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    assert!(
        omega > 0. && omega < 2.,
        "Omega may only be between 0 and 2"
    );

    let dim = a.shape().0;
    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut x = x0;

    while res > epsilon && n < nmax {
        for i in 0..dim {
            let mut sum = 0.;
            for j in 0..dim {
                sum += a[(i, j)] * x[j];
            }
            x[i] += omega * (b[i] - sum) / a[(i, i)];
        }

        res = (b - a * &x).norm();

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(x)
    } else {
        IterativeResult::Converged {
            result: x,
            iterations: n,
        }
    }
}

pub fn gradient<D: Dim>(
    a: &OMatrix<f64, D, D>,
    b: &OVector<f64, D>,
    x0: OVector<f64, D>,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let mut n = 0;
    let mut res_norm = epsilon + 1.;
    let mut x = x0;

    while res_norm > epsilon && n < nmax {
        let res = b - a * &x;
        let alpha = res.norm_squared() / (res.tr_mul(a) * &res)[0];
        x += alpha * &res;

        res_norm = res.norm();
        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(x)
    } else {
        IterativeResult::Converged {
            result: x,
            iterations: n,
        }
    }
}

pub fn conjugate_gradient<D: Dim>(
    a: &OMatrix<f64, D, D>,
    b: &OVector<f64, D>,
    x0: OVector<f64, D>,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D, Const<1>>
        + Allocator<f64, Const<1>, D>
        + Allocator<f64, Const<1>, Const<1>>,
{
    let mut n = 0;
    let mut res_norm = epsilon + 1.;
    let mut x = x0;
    let mut r = b - a * &x;
    let mut d = r.clone();

    while res_norm > epsilon && n < nmax {
        let r_old = r.clone();
        let z = a * &d;

        let alpha = r.norm_squared() / d.tr_mul(&z)[0];
        x += alpha * &d;
        r -= alpha * z;

        let beta = r.norm_squared() / r_old.norm_squared();
        d = &r + beta * &d;

        res_norm = r.norm();
        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(x)
    } else {
        IterativeResult::Converged {
            result: x,
            iterations: n,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3};

    use super::*;

    fn random_pos_def_matrix() -> Matrix3<f64> {
        let rand: Matrix3<f64> = Matrix3::new_random();
        0.5 * (rand + rand.transpose()) + 3. * Matrix3::identity()
    }

    #[test]
    fn test_jacobi() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let x = jacobi(&a, &b, Vector3::zeros(), 1e-6, 100);
        assert!(x.is_converged())
    }

    #[test]
    fn test_sor() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let x = successive_overrelaxation(&a, &b, Vector3::zeros(), 1e-6, 100, 1.7);
        assert!(x.is_converged())
    }

    #[test]
    fn test_gradient() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let x = gradient(&a, &b, Vector3::zeros(), 1e-6, 100);
        match x {
            IterativeResult::Converged {
                result: _,
                iterations,
            } => {
                dbg!(iterations);
            }
            _ => panic!(),
        }
        assert!(x.is_converged())
    }

    #[test]
    fn test_conjugate_gradient() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let x = conjugate_gradient(&a, &b, Vector3::zeros(), 1e-6, 100);
        match x {
            IterativeResult::Converged {
                result: _,
                iterations,
            } => {
                dbg!(iterations);
            }
            _ => panic!(),
        }
        assert!(x.is_converged())
    }
}
