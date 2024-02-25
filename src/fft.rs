use std::ops::{Deref, DerefMut};

use nalgebra::{Complex, ComplexField, RealField};

enum Borrowed<'a, T> {
    Owned(Vec<T>),
    Borrowed(&'a mut [T]),
}

impl<'a, T> Deref for Borrowed<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Borrowed::Owned(vec) => vec,
            Borrowed::Borrowed(slice) => slice,
        }
    }
}

impl<'a, T> DerefMut for Borrowed<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Borrowed::Owned(vec) => vec,
            Borrowed::Borrowed(slice) => slice,
        }
    }
}

pub fn fft_out_of_place<F: RealField + Copy>(arr: &mut [F]) -> Vec<Complex<F>> {
    let pow2 = 2usize.pow(arr.len().ilog2());

    let mut arr = if arr.len() - pow2 != 0 {
        let mut new_vec = vec![F::zero(); 2 * pow2];
        for (a, f) in arr.iter().zip(new_vec.iter_mut()) {
            *f = *a;
        }

        Borrowed::Owned(new_vec)
    } else {
        Borrowed::Borrowed(arr)
    };

    let n = arr.len();
    let exp = (-Complex::i() * F::two_pi() / F::from_usize(n).unwrap()).exp();

    fft_out_of_place_rec(arr.deref_mut(), exp)
}

fn fft_out_of_place_rec<F: RealField + Copy>(arr: &mut [F], exp: Complex<F>) -> Vec<Complex<F>> {
    let n = arr.len();
    if n <= 1 {
        return arr.iter().map(|f| f.into()).collect();
    }

    for i in 0..n / 2 {
        arr.swap(i, 2 * i);
    }

    let even = fft_out_of_place_rec(&mut arr[0..n / 2], exp);
    let odd = fft_out_of_place_rec(&mut arr[n / 2..n], exp);

    let mut vec = vec![F::zero().into(); n];
    for (i, (e, o)) in even.into_iter().zip(odd).enumerate() {
        let exp = exp.powu(i as u32);
        vec[i] = e + exp * o;
        vec[n / 2 + i] = e - exp * o;
    }

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_out_of_place() {
        let mut arr = [0.; 15];
        arr[0] = 1.;

        let fft = fft_out_of_place(&mut arr);
        for f in fft {
            assert_eq!(f, Complex::new(1., 0.));
        }
    }

    #[test]
    fn test_fft_out_of_place_2() {
        let mut arr = [0.; 4];
        arr[2] = 1.;

        let fft = fft_out_of_place(&mut arr);

        assert_eq!(fft[0], Complex::new(1., 0.));
        assert_eq!(fft[1], Complex::new(-1., 0.));
        assert_eq!(fft[2], Complex::new(1., 0.));
        assert_eq!(fft[3], Complex::new(-1., 0.));
    }
}
