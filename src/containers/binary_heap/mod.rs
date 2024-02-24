use std::{fmt::Debug, marker::PhantomData};

pub mod node_storage;
pub mod vec_storage;

pub trait ValuePriorityPair<V, P: Ord> {
    fn value(&self) -> &V;
    fn priority(&self) -> &P;

    fn value_mut(&mut self) -> &mut V;
    fn priority_mut(&mut self) -> &mut P;

    fn into_value(self) -> V;

    fn swap(&mut self, other: &mut Self);
}

impl<P: Ord> ValuePriorityPair<P, P> for P {
    fn value(&self) -> &P {
        self
    }
    fn priority(&self) -> &P {
        self
    }

    fn value_mut(&mut self) -> &mut P {
        self
    }
    fn priority_mut(&mut self) -> &mut P {
        self
    }

    fn into_value(self) -> P {
        self
    }

    fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }
}

impl<V, P: Ord> ValuePriorityPair<V, P> for (V, P) {
    fn value(&self) -> &V {
        &self.0
    }
    fn priority(&self) -> &P {
        &self.1
    }

    fn value_mut(&mut self) -> &mut V {
        &mut self.0
    }
    fn priority_mut(&mut self) -> &mut P {
        &mut self.1
    }

    fn into_value(self) -> V {
        self.0
    }

    fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.0, &mut other.0);
        std::mem::swap(&mut self.1, &mut other.1);
    }
}

pub trait Storage<V, P: Ord, Vp: ValuePriorityPair<V, P>> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn sift_up_last_node(&mut self);
    fn sift_down(&mut self, node: usize);
    fn push(&mut self, value_priority_pair: Vp);
    fn pop(&mut self) -> Option<V>;
    fn min(&self) -> Option<&V>;
}

#[derive(Clone, Debug)]
pub struct BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp>,
{
    storage: S,
    phantom: PhantomData<(V, P, Vp)>,
}

pub type NodeBinaryHeap<V, P = V, Vp = V> =
    BinaryHeap<V, P, Vp, node_storage::NodeStorage<V, P, Vp>>;
pub type VecBinaryHeap<V, P = V, Vp = V> = BinaryHeap<V, P, Vp, vec_storage::VecStorage<V, P, Vp>>;

impl<V, P, Vp, S> BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V, P, Vp, S> BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp>,
{
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    fn insert_no_sifting(&mut self, value_priority_pair: Vp) {
        self.storage.push(value_priority_pair);
    }

    pub fn insert(&mut self, value_priority_pair: Vp) {
        self.insert_no_sifting(value_priority_pair);
        self.storage.sift_up_last_node();
    }

    pub fn pop(&mut self) -> Option<V> {
        self.storage.pop()
    }

    pub fn min(&self) -> Option<&V> {
        self.storage.min()
    }
}

impl<V, P, Vp, S> Default for BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp> + Default,
{
    fn default() -> Self {
        Self {
            storage: S::default(),
            phantom: PhantomData,
        }
    }
}

impl<V, P, Vp, S> FromIterator<Vp> for BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp> + FromIterator<Vp>,
{
    fn from_iter<T: IntoIterator<Item = Vp>>(iter: T) -> Self {
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

impl<V, P, Vp, S, const N: usize> From<[Vp; N]> for BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp> + FromIterator<Vp>,
{
    fn from(value: [Vp; N]) -> Self {
        Self::from_iter(value)
    }
}

impl<V, P, Vp, S> From<Vec<Vp>> for BinaryHeap<V, P, Vp, S>
where
    P: Ord,
    Vp: ValuePriorityPair<V, P>,
    S: Storage<V, P, Vp> + FromIterator<Vp>,
{
    fn from(value: Vec<Vp>) -> Self {
        Self::from_iter(value)
    }
}
