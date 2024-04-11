use super::*;

pub trait VectorStorage<T, R: Dim>: Storage<T, R, Const<1>> {}
impl<T, R, S> VectorStorage<T, R> for S
where
    R: Dim,
    S: Storage<T, R, Const<1>>,
{
}

pub type Vector<T, R, S> = Matrix<T, R, Const<1>, S>;

pub type OVector<T, R> = OMatrix<T, R, Const<1>>;
pub type SVector<T, const R: usize> = Vector<T, Const<R>, ArrayStorage<T, R, 1>>;
pub type DVector<T, R> = Vector<T, R, VecStorage<T, R, Const<1>>>;

#[macro_export]
macro_rules! vector {
    [$($elem:expr),* $(,)*] => {
        {
            let arr = $crate::matrix::storage::ArrayStorage::from([$([$elem]),*]);
            unsafe { $crate::matrix::Matrix::from_data_unchecked(arr) }
        }
    };
}

pub type RowVector<T, C, S> = Matrix<T, Const<1>, C, S>;

pub type RowVectorView<'a, T, C> = RowVector<T, C, &'a ViewStorage<T, Const<1>, C>>;
pub type RowVectorViewMut<'a, T, C> = RowVector<T, C, &'a mut ViewStorage<T, Const<1>, C>>;
