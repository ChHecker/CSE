use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OVector};

use crate::IterativeResult;

fn residual<D: Dim>(f: &OVector<f64, D>, u: &OVector<f64, D>, h: f64) -> OVector<f64, D>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    let mut res: OVector<f64, D> = OVector::zeros_generic(u.shape_generic().0, Const);
    residual_into(f.as_slice(), u.as_slice(), h, res.as_mut_slice());
    res
}

fn residual_into(f: &[f64], u: &[f64], h: f64, res: &mut [f64]) {
    let dim = f.len();
    let h2 = h * h;

    if dim == 1 {
        res[0] = f[0] - 2. * u[0] / h2;
        return;
    }

    res[0] = f[0] - (2. * u[0] - u[1]) / h2;
    for i in 1..dim - 1 {
        res[i] = f[i] - (2. * u[i] - u[i - 1] - u[i + 1]) / h2;
    }
    res[dim - 1] = f[dim - 1] - (2. * u[dim - 1] - u[dim - 2]) / h2;
}

fn residual_norm(f: &[f64], u: &[f64], h: f64) -> f64 {
    let dim = f.len();
    let h2 = h * h;

    let norm_sq = if dim == 1 {
        let residual = f[0] - 2. * u[0] / h2;
        residual * residual
    } else {
        let residual = f[0] - (2. * u[0] - u[1]) / h2;
        let mut norm_sq = residual * residual;
        for i in 1..dim - 1 {
            let residual = f[i] - (2. * u[i] - u[i - 1] - u[i + 1]) / h2;
            norm_sq += residual * residual;
        }
        let residual = f[dim - 1] - (2. * u[dim - 1] - u[dim - 2]) / h2;
        norm_sq + residual * residual
    };

    norm_sq.sqrt()
}

pub fn jacobi(
    f: &[f64],
    u: &mut [f64],
    workspace: &mut [f64],
    h: f64,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<()> {
    let dim = f.len();
    assert_eq!(u.len(), dim);
    assert!(workspace.len() > dim);

    let mut n = 0;
    let mut res = epsilon + 1.;
    let h2 = h * h;
    let mut wrote_to_workspace = false;

    let mut u = u;
    let mut u_prime = workspace;

    while res > epsilon && n < nmax {
        if dim == 1 {
            u_prime[0] = h2 * f[0] / 2.;
        } else {
            u_prime[0] = (h2 * f[0] + u[1]) / 2.;
            u_prime[dim - 1] = (h2 * f[dim - 1] + u[dim - 2]) / 2.;
        }

        for i in 1..dim - 1 {
            u_prime[i] = (h2 * f[i] + u[i - 1] + u[i + 1]) / 2.;
        }

        std::mem::swap(&mut u, &mut u_prime);
        res = residual_norm(f, u, h);
        wrote_to_workspace.toggle();

        n += 1;
    }

    if wrote_to_workspace {
        u_prime.copy_from_slice(u);
    }

    if n == nmax {
        IterativeResult::MaxIterations(())
    } else {
        IterativeResult::Converged {
            result: (),
            iterations: n,
        }
    }
}

pub fn gauss_seidel(
    f: &[f64],
    u0: &mut [f64],
    h: f64,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<()> {
    let dim = f.len();
    assert_eq!(u0.len(), dim);

    let mut n = 0;
    let mut res = epsilon + 1.;
    let h2 = h * h;

    let u = u0;

    while res > epsilon && n < nmax {
        if dim == 1 {
            u[0] = h2 * f[0] / 2.;
        } else {
            u[0] = (h2 * f[0] + u[1]) / 2.;
            for i in 1..dim - 1 {
                u[i] = (h2 * f[i] + u[i - 1] + u[i + 1]) / 2.;
            }
            u[dim - 1] = (h2 * f[dim - 1] + u[dim - 2]) / 2.;
        }

        res = residual_norm(f, u, h);

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(())
    } else {
        IterativeResult::Converged {
            result: (),
            iterations: n,
        }
    }
}

pub fn successive_overrelaxation(
    f: &[f64],
    u0: &mut [f64],
    h: f64,
    omega: f64,
    epsilon: f64,
    nmax: u32,
) -> IterativeResult<()> {
    let dim = f.len();
    assert_eq!(u0.len(), dim);

    let mut n = 0;
    let mut res = epsilon + 1.;
    let h2 = h * h;

    let u = u0;

    while res > epsilon && n < nmax {
        if dim == 1 {
            u[0] = omega * h2 * f[0] / 2. + (1. - omega) * u[0];
        } else {
            u[0] = omega * (h2 * f[0] + u[1]) / 2. + (1. - omega) * u[0];
            for i in 1..dim - 1 {
                u[i] = omega * (h2 * f[i] + u[i - 1] + u[i + 1]) / 2. + (1. - omega) * u[i];
            }
            u[dim - 1] = omega * (h2 * f[dim - 1] + u[dim - 2]) / 2. + (1. - omega) * u[dim - 1];
        }

        res = residual_norm(f, u, h);

        n += 1;
    }

    if n == nmax {
        IterativeResult::MaxIterations(())
    } else {
        IterativeResult::Converged {
            result: (),
            iterations: n,
        }
    }
}

fn restriction_into(x: &[f64], x_coarse: &mut [f64]) {
    let dim_fine = x.len();
    if dim_fine == 1 {
        x_coarse[0] = x[0];
        return;
    }

    for i in 0..x_coarse.len() {
        x_coarse[i] = (x[2 * i] + 2. * x[2 * i + 1] + x[2 * i + 2]) / 4.;
    }
}

fn prolongation_into(x: &[f64], x_fine: &mut [f64]) {
    let dim_coarse = x.len();
    let dim_fine = x_fine.len();
    x_fine.fill(0.);

    for i in 0..dim_coarse {
        x_fine[2 * i + 1] = x[i];
    }

    x_fine[0] = x[0] / 2.;
    for i in 1..dim_coarse {
        x_fine[2 * i] = (x[i - 1] + x[i]) / 2.;
    }
    x_fine[dim_fine - 1] = x[dim_coarse - 1] / 2.;
}

pub fn multigrid(
    f: &[f64],
    u0: &mut [f64],
    h: f64,
    omega: f64,
    epsilon: f64,
    nmax: u32,
    nsmoothing: u32,
) -> IterativeResult<()> {
    let dim = f.len();
    assert!(dim % 2 == 1);
    assert_eq!(u0.len(), dim);

    let mut n = 0;
    let mut res = epsilon + 1.;
    let mut storage = MgStorage::new(f, u0);

    while res > epsilon && n < nmax {
        storage.multigrid_recursive(h, omega, nsmoothing, 0);
        res = residual_norm(f, storage.u[0].as_slice(), h);
        n += 1;
    }

    u0.copy_from_slice(storage.u[0].as_slice());
    if n == nmax {
        IterativeResult::MaxIterations(())
    } else {
        IterativeResult::Converged {
            result: (),
            iterations: n,
        }
    }
}

struct MgStorage {
    u: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    workspace: Vec<f64>,
}

impl MgStorage {
    fn new(f0: &[f64], u0: &[f64]) -> Self {
        let mut dim = u0.len();

        let mut u = vec![vec![0.; dim]];
        u[0].copy_from_slice(u0);
        let mut f = vec![vec![0.; dim]];
        f[0].copy_from_slice(f0);
        let workspace = vec![0.; dim];

        dim /= 2;
        while dim > 1 {
            u.push(vec![0.; dim]);
            f.push(vec![0.; dim]);

            dim /= 2;
        }

        Self { u, f, workspace }
    }

    fn multigrid_recursive(&mut self, h_fine: f64, omega: f64, nsmoothing: u32, depth: usize) {
        let dim_fine = self.u[depth].len();
        let h_coarse = 2. * h_fine;

        let u = &mut self.u[depth];

        assert!(dim_fine % 2 == 1);
        assert_eq!(u.len(), dim_fine);

        // mappings:
        // GS: f -> u
        // residual: f, u -> res_fine
        // restriction: res_fine -> res_coarse
        // recursion: res_coarse -> err_coarse
        // prolongation: err_coarse -> err_fine
        // correction: err_fine -> u
        // GS: f -> u
        gauss_seidel(&self.f[depth], u, h_fine, 0., nsmoothing).unwrap();

        residual_into(&self.f[depth], u, h_fine, &mut self.workspace[..dim_fine]);

        restriction_into(&self.workspace[..dim_fine], &mut self.f[depth + 1]);

        self.u[depth + 1].fill(0.);
        if dim_fine > 7 {
            self.multigrid_recursive(h_coarse, omega, nsmoothing, depth + 1);
        } else {
            successive_overrelaxation(
                &self.f[depth + 1],
                &mut self.u[depth + 1],
                h_coarse,
                omega,
                1e-12,
                100,
            )
            .unwrap();
        }

        prolongation_into(&self.u[depth + 1], &mut self.workspace[..dim_fine]);
        let u = &mut self.u[depth];
        u.iter_mut()
            .zip(self.workspace[..dim_fine].iter())
            .for_each(|(u, e)| *u += e);

        gauss_seidel(&self.f[depth], u, h_fine, 0., nsmoothing).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DVector, SVector};

    use super::*;

    #[test]
    fn test_jacobi() {
        let f: SVector<f64, 100> = SVector::new_random();
        let mut u0: SVector<f64, 100> = SVector::new_random();
        let mut workspace: SVector<f64, 100> = SVector::zeros();
        let x = jacobi(
            f.as_slice(),
            u0.as_mut_slice(),
            workspace.as_mut_slice(),
            1.,
            1e-5,
            100_000,
        );
        assert!(x.is_converged());

        let res = residual(&f, &u0, 1.).norm();
        dbg!(res);

        assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
    }

    #[test]
    fn test_gauss_seidel() {
        let f: SVector<f64, 100> = SVector::new_random();
        let mut u0: SVector<f64, 100> = SVector::new_random();
        let x = gauss_seidel(f.as_slice(), u0.as_mut_slice(), 1., 1e-5, 100_000);
        assert!(x.is_converged());

        let res = residual(&f, &u0, 1.).norm();
        dbg!(res);

        assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
    }

    #[test]
    fn test_sor() {
        let f: SVector<f64, 100> = SVector::new_random();
        let mut u0: SVector<f64, 100> = SVector::new_random();
        let x = successive_overrelaxation(f.as_slice(), u0.as_mut_slice(), 1., 1.7, 1e-5, 100_000);
        assert!(x.is_converged());

        let res = residual(&f, &u0, 1.).norm();
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
            let mut u0: DVector<f64> = DVector::zeros(size);

            println!("grid size {size}:");
            let x = multigrid(f.as_slice(), u0.as_mut_slice(), 1., 1.7, 1e-5, 1000, 2);
            assert!(x.is_converged());

            let res = residual(&f, &u0, 1.).norm();
            println!("res {res} after {} iterations\n", x.n_iterations().unwrap());

            assert_abs_diff_eq!(res, 0., epsilon = 1e-3);
        }
    }
}
