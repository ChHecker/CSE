#![allow(clippy::needless_range_loop)]

use core::slice;
use std::{
    borrow::{Borrow, BorrowMut},
    iter::Flatten,
};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayStorage<T, const R: usize, const C: usize> {
    pub array: [[T; C]; R],
}

impl<T, const R: usize, const C: usize> ArrayStorage<T, R, C> {
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i >= R || j >= C {
            return None;
        }
        Some(self.index([i, j]))
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i >= R || j >= C {
            return None;
        }
        Some(self.index_mut([i, j]))
    }
}

impl<T, const R: usize, const C: usize> RawStorage<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
{
    fn shape(&self) -> (Const<R>, Const<C>) {
        (Const, Const)
    }

    #[inline(always)]
    fn assert_same_shape<R2, C2, S2>(&self, _other: &S2)
    where
        R2: SameDim<Const<R>>,
        C2: SameDim<Const<C>>,
        S2: Storage<T, R2, C2>,
    {
        // automatically fulfilled by const
    }

    fn row(&self, index: usize) -> RowVectorView<'_, T, Const<C>> {
        unsafe {
            RowVector::from_data_unchecked(ViewStorage::from_raw_parts_generic(
                &self.array[index][0],
                Const,
                Const,
            ))
        }
    }
}

impl<T, const R: usize, const C: usize> RawStorageMut<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
{
    fn row_mut(&mut self, index: usize) -> RowVectorViewMut<'_, T, Const<C>> {
        unsafe {
            RowVector::from_data_unchecked(ViewStorage::from_raw_parts_mut_generic(
                &mut self.array[index][0],
                Const,
                Const,
            ))
        }
    }
}

impl<T, const R: usize, const C: usize> IntoIterator for ArrayStorage<T, R, C> {
    type Item = T;

    type IntoIter = IntoIter<T, R, C>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.array.into_iter().flatten(),
        }
    }
}

impl<T, const R: usize, const C: usize> StorageIterator<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
{
    type Iter<'a> = Iter<'a, T, R, C> where T: 'a;

    type RowIter<'a> = RowIter<'a, T, C> where T: 'a;

    fn iter(&self) -> Iter<'_, T, R, C> {
        Iter {
            data: self,
            index: 0,
        }
    }

    fn row_iter(&self) -> Self::RowIter<'_> {
        RowIter {
            iter: self.array.iter(),
        }
    }
}

impl<T, const R: usize, const C: usize> StorageIteratorMut<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
{
    type IterMut<'a> = IterMut<'a, T, R, C> where T: 'a;

    type RowIterMut<'a> = RowIterMut<'a, T, C> where T: 'a;

    fn iter_mut(&mut self) -> IterMut<'_, T, R, C> {
        IterMut {
            data: self,
            index: 0,
        }
    }

    fn row_iter_mut(&mut self) -> Self::RowIterMut<'_> {
        RowIterMut {
            iter: self.array.iter_mut(),
        }
    }
}

impl<T, const R: usize, const C: usize> Index<[usize; 2]> for ArrayStorage<T, R, C> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &T {
        &self.array[index[0]][index[1]]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<[usize; 2]> for ArrayStorage<T, R, C> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        &mut self.array[index[0]][index[1]]
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for ArrayStorage<T, R, C> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        let i = index / C;
        let j = index % C;

        &self.array[i][j]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for ArrayStorage<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let i = index / C;
        let j = index % C;

        &mut self.array[i][j]
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for ArrayStorage<T, R, C> {
    fn from(array: [[T; C]; R]) -> Self {
        Self { array }
    }
}

impl<T, const R: usize, const C: usize> From<ArrayStorage<T, R, C>> for [[T; C]; R] {
    fn from(storage: ArrayStorage<T, R, C>) -> Self {
        storage.array
    }
}

pub struct IntoIter<T, const R: usize, const C: usize> {
    iter: Flatten<core::array::IntoIter<[T; C], R>>,
}

impl<T, const R: usize, const C: usize> Iterator for IntoIter<T, R, C> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct Iter<'a, T, const R: usize, const C: usize> {
    data: &'a ArrayStorage<T, R, C>,
    index: usize,
}

impl<'a, T, const R: usize, const C: usize> Iterator for Iter<'a, T, R, C> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let i = self.index / C;
        let j = self.index % C;
        let out = self.data.get(i, j);

        self.index += 1;

        out
    }
}

pub struct IterMut<'a, T, const R: usize, const C: usize> {
    data: &'a mut ArrayStorage<T, R, C>,
    index: usize,
}

impl<'a, T, const R: usize, const C: usize> Iterator for IterMut<'a, T, R, C> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        let i = self.index / C;
        let j = self.index % C;

        let ptr = self.data.get_mut(i, j)? as *mut T;
        let out = unsafe { &mut *ptr };

        self.index += 1;

        Some(out)
    }
}

pub struct RowIter<'a, T, const C: usize> {
    iter: slice::Iter<'a, [T; C]>,
}

impl<'a, T, const C: usize> Iterator for RowIter<'a, T, C> {
    type Item = RowVectorView<'a, T, Const<C>>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            Some(Matrix::from_data_unchecked(
                ViewStorage::from_raw_parts_generic(&self.iter.next()?[0], Const, Const),
            ))
        }
    }
}

pub struct RowIterMut<'a, T, const C: usize> {
    iter: slice::IterMut<'a, [T; C]>,
}

impl<'a, T, const C: usize> Iterator for RowIterMut<'a, T, C> {
    type Item = RowVectorViewMut<'a, T, Const<C>>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            Some(Matrix::from_data_unchecked(
                ViewStorage::from_raw_parts_mut_generic(&mut self.iter.next()?[0], Const, Const),
            ))
        }
    }
}

impl<T, const R: usize, const C: usize> Borrow<ViewStorage<T, Const<R>, Const<C>>>
    for ArrayStorage<T, R, C>
{
    fn borrow(&self) -> &ViewStorage<T, Const<R>, Const<C>> {
        unsafe { ViewStorage::from_raw_parts_generic(self.array.as_ptr().cast(), Const, Const) }
    }
}

impl<T, const R: usize, const C: usize> BorrowMut<ViewStorage<T, Const<R>, Const<C>>>
    for ArrayStorage<T, R, C>
{
    fn borrow_mut(&mut self) -> &mut ViewStorage<T, Const<R>, Const<C>> {
        unsafe {
            ViewStorage::from_raw_parts_mut_generic(self.array.as_mut_ptr().cast(), Const, Const)
        }
    }
}

impl<T, const R: usize, const C: usize> AsRef<ViewStorage<T, Const<R>, Const<C>>>
    for ArrayStorage<T, R, C>
{
    fn as_ref(&self) -> &ViewStorage<T, Const<R>, Const<C>> {
        unsafe { ViewStorage::from_raw_parts_generic(self.array.as_ptr().cast(), Const, Const) }
    }
}

impl<T, const R: usize, const C: usize> AsMut<ViewStorage<T, Const<R>, Const<C>>>
    for ArrayStorage<T, R, C>
{
    fn as_mut(&mut self) -> &mut ViewStorage<T, Const<R>, Const<C>> {
        unsafe {
            ViewStorage::from_raw_parts_mut_generic(self.array.as_mut_ptr().cast(), Const, Const)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(arr[[0, 0]], 1);
        assert_eq!(arr[[0, 1]], 2);
        assert_eq!(arr[[0, 2]], 3);
        assert_eq!(arr[[1, 0]], 4);
        assert_eq!(arr[[1, 1]], 5);
        assert_eq!(arr[[1, 2]], 6);
    }

    #[test]
    fn linear_index() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);
        assert_eq!(arr[3], 4);
        assert_eq!(arr[4], 5);
        assert_eq!(arr[5], 6);
    }

    #[test]
    fn iter() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        for (i, elem) in arr.iter().enumerate() {
            assert_eq!(i + 1, *elem);
        }
    }

    #[test]
    fn iter_mut() {
        let mut arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        for (i, elem) in arr.iter_mut().enumerate() {
            assert_eq!(i + 1, *elem);
        }
    }

    #[test]
    fn row_iter() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let mut iter = arr.row_iter();

        assert_eq!(iter.next().unwrap().data.as_slice(), &[1, 2, 3]);
        assert_eq!(iter.next().unwrap().data.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn row_iter_mut() {
        let mut arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let mut iter = arr.row_iter_mut();

        assert_eq!(iter.next().unwrap().data.as_slice(), &mut [1, 2, 3]);
        assert_eq!(iter.next().unwrap().data.as_slice(), &mut [4, 5, 6]);
    }
}
