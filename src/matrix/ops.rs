use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use super::*;

macro_rules! impl_binary_op {
    ($trait:tt, $fn:tt, $trait_assign:tt, $fn_assign:tt) => {
        impl<'a, 'b, T, R1, C1, S1, R2, C2, S2> $trait<&'b Matrix<T, R2, C2, S2>>
            for &'a Matrix<T, R1, C1, S1>
        where
            &'a T: $trait<&'b T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            R2: SameDim<R1>,
            C2: SameDim<C1>,
            S2: Storage<T, R2, C2>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: &'b Matrix<T, R2, C2, S2>) -> Self::Output {
                self.data.assert_same_shape(&rhs.data);

                let (rows, columns) = self.shape_generic();
                let mut storage = DefaultAllocator::new_uninit(rows, columns);
                for ((a, b), s) in self.iter().zip(rhs.iter()).zip(storage.iter_mut()) {
                    s.write(a.$fn(b));
                }

                unsafe { Matrix::from_data_unchecked(DefaultAllocator::assume_init(storage)) }
            }
        }

        impl<'a, T, R1, C1, S1, R2, C2, S2> $trait<&'a Matrix<T, R2, C2, S2>>
            for Matrix<T, R1, C1, S1>
        where
            for<'b> &'b T: $trait<&'a T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            R2: SameDim<R1>,
            C2: SameDim<C1>,
            S2: Storage<T, R2, C2>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: &'a Matrix<T, R2, C2, S2>) -> Self::Output {
                (&self).$fn(rhs)
            }
        }

        impl<'a, T, R1, C1, S1, R2, C2, S2> $trait<Matrix<T, R2, C2, S2>>
            for &'a Matrix<T, R1, C1, S1>
        where
            &'a T: for<'b> $trait<&'b T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            R2: SameDim<R1>,
            C2: SameDim<C1>,
            S2: Storage<T, R2, C2>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: Matrix<T, R2, C2, S2>) -> Self::Output {
                self.$fn(&rhs)
            }
        }

        impl<T, R1, C1, S1, R2, C2, S2> $trait<Matrix<T, R2, C2, S2>> for Matrix<T, R1, C1, S1>
        where
            for<'a, 'b> &'a T: $trait<&'b T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            R2: SameDim<R1>,
            C2: SameDim<C1>,
            S2: Storage<T, R2, C2>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: Matrix<T, R2, C2, S2>) -> Self::Output {
                self.$fn(&rhs)
            }
        }

        impl<'a, T, R1, C1, S1, R2, C2, S2> $trait_assign<&'a Matrix<T, R2, C2, S2>>
            for Matrix<T, R1, C1, S1>
        where
            T: $trait_assign<&'a T>,
            R1: Dim,
            C1: Dim,
            S1: StorageMut<T, R1, C1>,
            R2: SameDim<R1>,
            C2: SameDim<C1>,
            S2: Storage<T, R2, C2>,
        {
            fn $fn_assign(&mut self, rhs: &'a Matrix<T, R2, C2, S2>) {
                for (a, b) in self.iter_mut().zip(rhs.iter()) {
                    a.$fn_assign(b);
                }
            }
        }

        impl<T, R1, C1, S1, R2, C2, S2> $trait_assign<Matrix<T, R2, C2, S2>>
            for Matrix<T, R1, C1, S1>
        where
            T: for<'a> $trait_assign<&'a T>,
            R1: Dim,
            C1: Dim,
            S1: StorageMut<T, R1, C1>,
            R2: SameDim<R1>,
            C2: SameDim<C1>,
            S2: Storage<T, R2, C2>,
        {
            fn $fn_assign(&mut self, rhs: Matrix<T, R2, C2, S2>) {
                self.$fn_assign(&rhs)
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign);
impl_binary_op!(Sub, sub, SubAssign, sub_assign);
impl_binary_op!(Mul, mul, MulAssign, mul_assign);
impl_binary_op!(Div, div, DivAssign, div_assign);

macro_rules! impl_scalar_op {
    ($trait:tt, $fn:tt, $trait_assign:tt, $fn_assign:tt) => {
        impl<'a, 'b, T, R1, C1, S1> $trait<&'b T> for &'a Matrix<T, R1, C1, S1>
        where
            T: Clone + $trait<T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: &'b T) -> Self::Output {
                let (rows, columns) = self.shape_generic();
                let mut storage = DefaultAllocator::new_uninit(rows, columns);
                for (a, s) in self.iter().zip(storage.iter_mut()) {
                    s.write(a.clone().$fn(rhs.clone()));
                }

                unsafe { Matrix::from_data_unchecked(DefaultAllocator::assume_init(storage)) }
            }
        }

        impl<'a, T, R1, C1, S1> $trait<&'a T> for Matrix<T, R1, C1, S1>
        where
            T: Clone + $trait<T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: &'a T) -> Self::Output {
                (&self).$fn(rhs)
            }
        }

        impl<'a, T, R1, C1, S1> $trait<T> for &'a Matrix<T, R1, C1, S1>
        where
            T: Clone + $trait<T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: T) -> Self::Output {
                self.$fn(&rhs)
            }
        }

        impl<T, R1, C1, S1> $trait<T> for Matrix<T, R1, C1, S1>
        where
            T: Clone + $trait<T, Output = T>,
            R1: Dim,
            C1: Dim,
            S1: Storage<T, R1, C1>,
            DefaultAllocator: Allocator<T, R1, C1>,
        {
            type Output = Matrix<T, R1, C1, <DefaultAllocator as Allocator<T, R1, C1>>::Storage>;

            fn $fn(self, rhs: T) -> Self::Output {
                self.$fn(&rhs)
            }
        }

        impl<'a, T, R, C, S> $trait_assign<&'a T> for Matrix<T, R, C, S>
        where
            T: Clone + $trait_assign<T>,
            R: Dim,
            C: Dim,
            S: StorageMut<T, R, C>,
        {
            fn $fn_assign(&mut self, rhs: &'a T) {
                for a in self.iter_mut() {
                    a.$fn_assign(rhs.clone());
                }
            }
        }

        impl<T, R, C, S> $trait_assign<T> for Matrix<T, R, C, S>
        where
            T: Clone + $trait_assign<T>,
            R: Dim,
            C: Dim,
            S: StorageMut<T, R, C>,
        {
            fn $fn_assign(&mut self, rhs: T) {
                self.$fn_assign(&rhs)
            }
        }
    };
}

impl_scalar_op!(Mul, mul, MulAssign, mul_assign);
impl_scalar_op!(Div, div, DivAssign, div_assign);

macro_rules! impl_left_scalar_mul {
    ($type:tt) => {
        impl<'a, 'b, R, C, S> Mul<&'b Matrix<$type, R, C, S>> for &'a $type
        where
            &'a $type: Mul<&'b $type, Output = $type>,
            R: Dim,
            C: Dim,
            S: Storage<$type, R, C>,
            DefaultAllocator: Allocator<$type, R, C>,
        {
            type Output =
                Matrix<$type, R, C, <DefaultAllocator as Allocator<$type, R, C>>::Storage>;

            fn mul(self, rhs: &'b Matrix<$type, R, C, S>) -> Self::Output {
                rhs.mul(self)
            }
        }

        impl<'a, R, C, S> Mul<&'a Matrix<$type, R, C, S>> for $type
        where
            for<'b> &'b $type: Mul<&'a $type, Output = $type>,
            R: Dim,
            C: Dim,
            S: Storage<$type, R, C>,
            DefaultAllocator: Allocator<$type, R, C>,
        {
            type Output =
                Matrix<$type, R, C, <DefaultAllocator as Allocator<$type, R, C>>::Storage>;

            fn mul(self, rhs: &'a Matrix<$type, R, C, S>) -> Self::Output {
                rhs.mul(self)
            }
        }

        impl<'a, R, C, S> Mul<Matrix<$type, R, C, S>> for &'a $type
        where
            &'a $type: for<'b> Mul<&'b $type, Output = $type>,
            R: Dim,
            C: Dim,
            S: Storage<$type, R, C>,
            DefaultAllocator: Allocator<$type, R, C>,
        {
            type Output =
                Matrix<$type, R, C, <DefaultAllocator as Allocator<$type, R, C>>::Storage>;

            fn mul(self, rhs: Matrix<$type, R, C, S>) -> Self::Output {
                rhs.mul(self)
            }
        }

        impl<R, C, S> Mul<Matrix<$type, R, C, S>> for $type
        where
            for<'a, 'b> &'a $type: Mul<&'b $type, Output = $type>,
            R: Dim,
            C: Dim,
            S: Storage<$type, R, C>,
            DefaultAllocator: Allocator<$type, R, C>,
        {
            type Output =
                Matrix<$type, R, C, <DefaultAllocator as Allocator<$type, R, C>>::Storage>;

            fn mul(self, rhs: Matrix<$type, R, C, S>) -> Self::Output {
                rhs.mul(self)
            }
        }
    };
}

impl_left_scalar_mul!(f32);
impl_left_scalar_mul!(f64);

impl_left_scalar_mul!(u8);
impl_left_scalar_mul!(u16);
impl_left_scalar_mul!(u32);
impl_left_scalar_mul!(u64);
impl_left_scalar_mul!(usize);

impl_left_scalar_mul!(i8);
impl_left_scalar_mul!(i16);
impl_left_scalar_mul!(i32);
impl_left_scalar_mul!(i64);
impl_left_scalar_mul!(isize);
