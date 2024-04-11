pub mod allocator;

pub use allocator::*;

pub mod array;
pub use array::ArrayStorage;

pub mod vec;
pub use vec::VecStorage;

pub mod view;
pub use view::ViewStorage;

use super::*;

pub trait RawStorage<T, R: Dim, C: Dim>:
    Index<[usize; 2], Output = T> + Index<usize, Output = T>
{
    fn shape(&self) -> (R, C);

    fn assert_same_shape<R2, C2, S2>(&self, other: &S2)
    where
        R2: SameDim<R>,
        C2: SameDim<C>,
        S2: Storage<T, R2, C2>;

    fn row(&self, index: usize) -> RowVectorView<'_, T, C>;
}

pub trait RawStorageMut<T, R: Dim, C: Dim>:
    RawStorage<T, R, C> + IndexMut<[usize; 2], Output = T> + IndexMut<usize, Output = T>
{
    fn row_mut(&mut self, index: usize) -> RowVectorViewMut<'_, T, C>;
}

pub trait StorageIterator<T, R: Dim, C: Dim>: RawStorage<T, R, C> {
    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;

    type RowIter<'a>: Iterator<Item = RowVectorView<'a, T, C>>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_>;

    fn row_iter(&self) -> Self::RowIter<'_>;
}

pub trait StorageIteratorMut<T, R: Dim, C: Dim>:
    RawStorageMut<T, R, C> + StorageIterator<T, R, C>
{
    type IterMut<'a>: Iterator<Item = &'a mut T>
    where
        Self: 'a,
        T: 'a;

    type RowIterMut<'a>: Iterator<Item = RowVectorViewMut<'a, T, C>>
    where
        Self: 'a,
        T: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn row_iter_mut(&mut self) -> Self::RowIterMut<'_>;
}

pub trait Storage<T, R: Dim, C: Dim>: RawStorage<T, R, C> + StorageIterator<T, R, C> {}
impl<T, R, C, S> Storage<T, R, C> for S
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C> + StorageIterator<T, R, C>,
{
}

pub trait StorageMut<T, R: Dim, C: Dim>: StorageIteratorMut<T, R, C> {}
impl<T, R, C, S> StorageMut<T, R, C> for S
where
    R: Dim,
    C: Dim,
    S: StorageIteratorMut<T, R, C>,
{
}

pub trait OwnedStorage<T, R: Dim, C: Dim>:
    Storage<T, R, C> + StorageIteratorMut<T, R, C> + IntoIterator<Item = T> + 'static
{
}
impl<T, R, C, S> OwnedStorage<T, R, C> for S
where
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + StorageIteratorMut<T, R, C> + IntoIterator<Item = T> + 'static,
{
}
