use nalgebra::{allocator::Allocator, Const, DVector, DefaultAllocator, Dim, Dyn, OVector};

use crate::IterativeResult;

fn residual<D: Dim>(f: &OVector<f64, D>, u: &OVector<f64, D>, h: f64) -> OVector<f64, D>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = f.shape().0;
    let mut res: OVector<f64, D> = OVector::zeros_generic(u.shape_generic().0, Const);

    res[0] = f[0] - (2. * u[0] - u[1]) / (h * h);
    for i in 1..dim - 1 {
        res[i] = f[i] - (2. * u[i] - u[i - 1] - u[i + 1]) / (h * h);
    }
    res[dim - 1] = f[dim - 1] - (2. * u[dim - 1] - u[dim - 2]) / (h * h);

    res
}

pub fn jacobi<D: Dim>(
    f: &OVector<f64, D>,
    u0: OVector<f64, D>,
    h: f64,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = f.shape().0;
    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut u = u0;

    while res > epsilon && n < nmax {
        let mut u_prime: OVector<f64, D> = OVector::zeros_generic(u.shape_generic().0, Const);

        u_prime[0] = (h * h * f[0] + u[1]) / 2.;
        for i in 1..dim - 1 {
            u_prime[i] = (h * h * f[i] + u[i - 1] + u[i + 1]) / 2.;
        }
        u_prime[dim - 1] = (h * h * f[dim - 1] + u[dim - 2]) / 2.;

        u = u_prime;
        res = residual(f, &u, h).norm();

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(u)
    } else {
        IterativeResult::Converged{ result: u, iterations: n }
    }
}

pub fn gauss_seidel<D: Dim>(
    f: &OVector<f64, D>,
    u0: OVector<f64, D>,
    h: f64,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = f.shape().0;
    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut u = u0;

    while res > epsilon && n < nmax {
        u[0] = (h * h * f[0] + u[1]) / 2.;
        for i in 1..dim - 1 {
            u[i] = (h * h * f[i] + u[i - 1] + u[i + 1]) / 2.;
        }
        u[dim - 1] = (h * h * f[dim - 1] + u[dim - 2]) / 2.;

        res = residual(f, &u, h).norm();

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(u)
    } else {
        IterativeResult::Converged{ result: u, iterations: n }
    }
}

pub fn successive_overrelaxation<D: Dim>(
    f: &OVector<f64, D>,
    u0: OVector<f64, D>,
    h: f64,
    omega: f64,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = f.shape().0;
    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut u = u0;

    while res > epsilon && n < nmax {
        u[0] = omega * (h * h * f[0] + u[1]) / 2. + (1. - omega) * u[0];
        for i in 1..dim - 1 {
            u[i] = omega * (h * h * f[i] + u[i - 1] + u[i + 1]) / 2. + (1. - omega) * u[i];
        }
        u[dim - 1] = omega * (h * h * f[dim - 1] + u[dim - 2]) / 2. + (1. - omega) * u[dim - 1];

        res = residual(f, &u, h).norm();

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(u)
    } else {
        IterativeResult::Converged{ result: u, iterations: n }
    }
}

fn restriction<D: Dim>(x: &OVector<f64, D>) -> DVector<f64>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim_fine = x.shape().0;
    let dim_coarse = dim_fine / 2;
    let mut x_coarse: DVector<f64> = DVector::zeros(dim_coarse);

    if dim_fine == 1 {
        x_coarse[0] = x[0];
        return x_coarse;
    }

    // dim_fine = 7, dim_coarse = 3
    // x_coarse[0] = (x[0] + 2 x[1] + x[2]) / 4
    // x_coarse[1] = (x[2] + 2 x[3] + x[4]) / 4
    // x_coarse[2] = (x[4] + 2 x[5] + x[6]) / 4
    for i in 0..dim_coarse {
        x_coarse[i] = (x[2 * i] + 2. * x[2 * i + 1] + x[2 * i + 2]) / 4.;
    }

    x_coarse
}

fn prolongation<D: Dim>(x: &DVector<f64>, d: D) -> OVector<f64, D>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim_coarse = x.shape().0;

    let mut x_fine: OVector<f64, D> = OVector::zeros_generic(d, Const);

    for i in 0..dim_coarse {
        x_fine[2 * i + 1] = x[i];
    }

    x_fine[0] = x[0] / 2.;
    for i in 1..dim_coarse {
        x_fine[2 * i] = (x[i - 1] + x[i]) / 2.;
    }
    x_fine[d.value() - 1] = x[dim_coarse - 1] / 2.;

    x_fine
}

pub fn multigrid<D: Dim>(
    f: &OVector<f64, D>,
    u0: OVector<f64, D>,
    h: f64,
    omega: f64,
    epsilon: f64,
    nmax: u32,
    nsmoothing: u32,
) -> IterativeResult<OVector<f64, D>>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = f.shape().0;
    assert!(dim % 2 == 1);

    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut u = u0;

    while res > epsilon && n < nmax {
        u = multigrid_recursive(f, u, h, omega, nsmoothing);
        res = residual(f, &u, h).norm();
        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(u)
    } else {
        IterativeResult::Converged{ result: u, iterations: n }
    }
}

fn multigrid_recursive<D: Dim>(
    f: &OVector<f64, D>,
    u: OVector<f64, D>,
    h: f64,
    omega: f64,
    nsmoothing: u32,
) -> OVector<f64, D>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let dim = f.shape().0;

    let presmoothed = gauss_seidel(f, u, h, 0., nsmoothing).unwrap();
    let res_fine = residual(f, &presmoothed, h);

    let h_coarse = 2. * h;
    let res_coarse = restriction(&res_fine);
    let error_coarse = if dim > 7 {
        multigrid_recursive::<Dyn>(
            &res_coarse,
            DVector::zeros(dim / 2),
            h_coarse,
            omega,
            nsmoothing,
        )
    } else {
        successive_overrelaxation::<Dyn>(
            &res_coarse,
            DVector::zeros(dim / 2),
            h_coarse,
            omega,
            1e-12,
            100,
        )
        .unwrap()
    };
    let error = prolongation(&error_coarse, f.shape_generic().0);

    let corrected = presmoothed + error;
    gauss_seidel(f, corrected, h, 0., nsmoothing).unwrap()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::SVector;

    use super::*;

    #[test]
    fn test_jacobi() {
        let f: SVector<f64, 100> = SVector::new_random();
        let u0: SVector<f64, 100> = SVector::new_random();
        let x = jacobi(&f, u0, 1., 1e-5, 100_000);
        assert!(x.is_converged());

        let res = residual(&f, &x.unwrap(), 1.).norm();
        dbg!(res);

        assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
    }

    #[test]
    fn test_gauss_seidel() {
        let f: SVector<f64, 100> = SVector::new_random();
        let u0: SVector<f64, 100> = SVector::new_random();
        let x = gauss_seidel(&f, u0, 1., 1e-5, 100_000);
        assert!(x.is_converged());

        let res = residual(&f, &x.unwrap(), 1.).norm();
        dbg!(res);

        assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
    }

    #[test]
    fn test_sor() {
        let f: SVector<f64, 100> = SVector::new_random();
        let u0: SVector<f64, 100> = SVector::new_random();
        let x = successive_overrelaxation(&f, u0, 1., 1.7, 1e-5, 100_000);
        assert!(x.is_converged());

        let res = residual(&f, &x.unwrap(), 1.).norm();
        dbg!(res);

        assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
    }

    #[test]
    fn test_multigrid() {
        // let f: SVector<f64, 127> = SVector::new_random();
        // let u0: SVector<f64, 127> = SVector::new_random();
        for i in 5..10 {
            let size = 2usize.pow(i) - 1;

            let mut f: DVector<f64> = DVector::zeros(size);
            for (i, elem) in f.iter_mut().enumerate() {
                *elem = (4. * i as f64 / size as f64).sin();
            }
            let u0: DVector<f64> = DVector::zeros(size);

            println!("grid size {size}:");
            let x = multigrid(&f, u0, 1., 1.7, 1e-5, 1000, 2);
            assert!(x.is_converged());

            let res = residual(&f, &x.unwrap(), 1.).norm();
            println!();

            assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
        }
    }
}
