use std::{fmt::Debug, marker::PhantomData};

pub mod node_storage;
pub mod vec_storage;

pub trait Storage<V, P: Ord> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, value: V, priority: P);
    fn sift_up_last_node(&mut self);
    fn sift_down(&mut self, node: usize);
    fn pop(&mut self) -> Option<V>;
    fn min(&self) -> Option<&V>;
}

#[derive(Clone, Debug)]
pub struct BinaryHeap<V, P, S>
where
    P: Ord,
    S: Storage<V, P>,
{
    storage: S,
    phantom: PhantomData<(V, P)>,
}

pub type NodeBinaryHeap<V, P> = BinaryHeap<V, P, node_storage::NodeStorage<V, P>>;
pub type VecBinaryHeap<V, P> = BinaryHeap<V, P, vec_storage::VecStorage<V, P>>;

impl<V, P, S> BinaryHeap<V, P, S>
where
    P: Ord,
    S: Default + Storage<V, P>,
{
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V, P, S> BinaryHeap<V, P, S>
where
    P: Ord,
    S: Storage<V, P>,
{
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    fn insert_no_sifting(&mut self, value: V, priority: P) {
        self.storage.push(value, priority);
    }

    pub fn insert(&mut self, value: V, priority: P) {
        self.insert_no_sifting(value, priority);
        self.storage.sift_up_last_node();
    }

    pub fn pop(&mut self) -> Option<V> {
        self.storage.pop()
    }

    pub fn min(&self) -> Option<&V> {
        self.storage.min()
    }
}

impl<V, P, S> Default for BinaryHeap<V, P, S>
where
    P: Ord,
    S: Default + Storage<V, P>,
{
    fn default() -> Self {
        Self {
            storage: S::default(),
            phantom: PhantomData,
        }
    }
}

impl<V, P, S> FromIterator<(V, P)> for BinaryHeap<V, P, S>
where
    P: Ord,
    S: Storage<V, P> + FromIterator<(V, P)>,
{
    fn from_iter<T: IntoIterator<Item = (V, P)>>(iter: T) -> Self {
        println!("Collecting into iter.");
        let mut storage: S = iter.into_iter().collect();

        if storage.len() > 1 {
            for node in (0..storage.len() / 2).rev() {
                storage.sift_down(node);
            }
        }

        Self {
            storage,
            phantom: PhantomData,
        }
    }
}

impl<P, S> FromIterator<P> for BinaryHeap<P, P, S>
where
    P: Clone + Ord,
    S: Storage<P, P> + FromIterator<(P, P)>,
{
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        Self::from_iter(iter.into_iter().map(|p| (p.clone(), p)))
    }
}

impl<V, P, S, const N: usize> From<[(V, P); N]> for BinaryHeap<V, P, S>
where
    P: Ord,
    S: Storage<V, P> + FromIterator<(V, P)>,
{
    fn from(value: [(V, P); N]) -> Self {
        Self::from_iter(value)
    }
}

impl<V, P, S> From<Vec<(V, P)>> for BinaryHeap<V, P, S>
where
    P: Ord,
    S: Storage<V, P> + FromIterator<(V, P)>,
{
    fn from(value: Vec<(V, P)>) -> Self {
        Self::from_iter(value)
    }
}

impl<P, S, const N: usize> From<[P; N]> for BinaryHeap<P, P, S>
where
    P: Clone + Ord,
    S: Storage<P, P> + FromIterator<(P, P)>,
{
    fn from(value: [P; N]) -> Self {
        Self::from_iter(value)
    }
}

impl<P, S> From<Vec<P>> for BinaryHeap<P, P, S>
where
    P: Clone + Ord,
    S: Storage<P, P> + FromIterator<(P, P)>,
{
    fn from(value: Vec<P>) -> Self {
        Self::from_iter(value)
    }
}
