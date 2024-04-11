use std::f64::consts::PI;

pub fn fft_oop<C>(mut arr: Vec<C>) -> Vec<Complex<f64>>
where
    C: ComplexField<RealField = f64> + Copy,
    Complex<f64>: From<C>,
{
    let pow2 = 2usize.pow(arr.len().ilog2());

    if arr.len() - pow2 != 0 {
        arr.resize(2 * pow2 - arr.len(), C::zero());
    }

    let n = arr.len();
    let exp = Complex::exp(-Complex::i() * 2. * PI / n as f64);

    fft_oop_rec(arr, exp)
}

pub fn fft_inv_oop<C>(mut arr: Vec<C>) -> Vec<Complex<f64>>
where
    C: ComplexField<RealField = f64> + Copy,
    Complex<f64>: From<C>,
{
    let pow2 = 2usize.pow(arr.len().ilog2());

    if arr.len() - pow2 != 0 {
        arr.resize(2 * pow2 - arr.len(), C::zero());
    }

    let n = arr.len();
    let exp = Complex::exp(Complex::i() * 2. * PI / n as f64);

    let mut inv_fft = fft_oop_rec(arr, exp);
    inv_fft.iter_mut().for_each(|x| *x /= n as f64);
    inv_fft
}

pub(crate) fn fft_oop_rec<C>(arr: Vec<C>, exp: Complex<f64>) -> Vec<Complex<f64>>
where
    C: ComplexField<RealField = f64> + Copy,
    Complex<f64>: From<C>,
{
    let n = arr.len();
    if n <= 1 {
        return arr.iter().map(|x| (*x).into()).collect();
    }

    let even = fft_oop_rec(arr.clone().into_iter().step_by(2).collect(), exp.powu(2));
    let odd = fft_oop_rec(
        arr.clone().into_iter().skip(1).step_by(2).collect(),
        exp.powu(2),
    );

    let mut vec = vec![Complex::from_real(0.); n];
    for (k, (e, o)) in even.into_iter().zip(odd).enumerate() {
        let exp = exp.powu(k as u32);
        vec[k] = e + exp * o;
        vec[n / 2 + k] = e - exp * o;
    }

    vec
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rand::{rngs::ThreadRng, Rng};

    use super::*;

    #[test]
    fn test_fft_oop() {
        let mut vec = vec![0.; 15];
        vec[0] = 1.;

        let fft = fft_oop(vec);
        for f in fft {
            assert_eq!(f, Complex::new(1., 0.));
        }
    }

    #[test]
    fn test_fft_oop_2() {
        let mut vec = vec![0.; 8];
        vec[2] = 1.;

        let fft = fft_oop(vec);

        assert_abs_diff_eq!(fft[0], Complex::new(1., 0.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[1], Complex::new(0., -1.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[2], Complex::new(-1., 0.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[3], Complex::new(0., 1.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[4], Complex::new(1., 0.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[5], Complex::new(0., -1.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[6], Complex::new(-1., 0.), epsilon = 1e-10);
        assert_abs_diff_eq!(fft[7], Complex::new(0., 1.), epsilon = 1e-10);
    }

    #[test]
    fn test_fft_inv_oop() {
        let mut rng = ThreadRng::default();

        let arr: Vec<f64> = (0..16).map(|_| rng.gen()).collect();
        let fft = fft_oop(arr.clone());
        let inv_fft = fft_inv_oop(fft);

        assert_eq!(arr.len(), inv_fft.len());
        for (x, y) in arr.into_iter().zip(inv_fft) {
            assert_abs_diff_eq!(x, y.re, epsilon = 1e-10);
            assert_abs_diff_eq!(y.im, 0., epsilon = 1e-10);
        }
    }
}
