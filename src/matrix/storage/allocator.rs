use std::fmt::Debug;
use std::mem::MaybeUninit;

use super::*;

pub trait Allocator<T, R: Dim, C: Dim> {
    type Storage: OwnedStorage<T, R, C> + Clone + Debug;
    type StorageUninit: OwnedStorage<MaybeUninit<T>, R, C>;

    fn new_uninit(rows: R, columns: C) -> Self::StorageUninit;

    unsafe fn assume_init(uninit: Self::StorageUninit) -> Self::Storage;

    fn from_iter<I: IntoIterator<Item = T>>(iter: I, rows: R, columns: C) -> Self::Storage {
        let mut storage = Self::new_uninit(rows, columns);
        storage.iter_mut().zip(iter).for_each(|(a, i)| {
            a.write(i);
        });
        unsafe { Self::assume_init(storage) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DefaultAllocator;

impl<T: 'static + Clone + Debug, const R: usize, const C: usize> Allocator<T, Const<R>, Const<C>>
    for DefaultAllocator
{
    type Storage = ArrayStorage<T, R, C>;
    type StorageUninit = ArrayStorage<MaybeUninit<T>, R, C>;

    fn new_uninit(_rows: Const<R>, _columns: Const<C>) -> Self::StorageUninit {
        let array: [[MaybeUninit<T>; C]; R] = unsafe { MaybeUninit::uninit().assume_init() };
        array.into()
    }

    unsafe fn assume_init(uninit: Self::StorageUninit) -> Self::Storage {
        std::mem::transmute_copy(&uninit.array)
    }
}

impl<T: 'static + Clone + Debug> Allocator<T, Dyn, Dyn> for DefaultAllocator {
    type Storage = VecStorage<T, Dyn, Dyn>;
    type StorageUninit = VecStorage<MaybeUninit<T>, Dyn, Dyn>;

    fn new_uninit(rows: Dyn, columns: Dyn) -> Self::StorageUninit {
        let mut vec = Vec::new();
        let length = rows.dim() * columns.dim();
        vec.reserve_exact(length);
        vec.resize_with(length, MaybeUninit::uninit);
        unsafe { VecStorage::new_unchecked(vec, rows, columns) }
    }

    unsafe fn assume_init(uninit: Self::StorageUninit) -> Self::Storage {
        std::mem::transmute_copy(&uninit.vec)
    }
}
