use super::*;

#[derive(Clone, Debug, Default)]
pub struct VecStorage<V, P: Ord, Vp: ValuePriorityPair<V, P>> {
    pub(crate) vec: Vec<Vp>,
    phantom: PhantomData<(V, P)>,
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> VecStorage<V, P, Vp> {
    pub(crate) fn new(vec: Vec<Vp>) -> Self {
        Self {
            vec,
            phantom: PhantomData,
        }
    }

    fn parent(node: usize) -> Option<usize> {
        node.checked_sub(1).map(|i| i / 2)
    }

    fn left(&self, node: usize) -> Option<usize> {
        let left = 2 * node + 1;
        if left < self.len() {
            Some(left)
        } else {
            None
        }
    }

    fn right(&self, node: usize) -> Option<usize> {
        let right = 2 * node + 2;
        if right < self.len() {
            Some(right)
        } else {
            None
        }
    }
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> Storage<V, P, Vp> for VecStorage<V, P, Vp> {
    fn len(&self) -> usize {
        self.vec.len()
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn push(&mut self, value_priority_pair: Vp) {
        self.vec.push(value_priority_pair);
    }

    fn sift_up_last_node(&mut self) {
        let mut current_node = self.len() - 1;
        let mut opt_parent = Self::parent(current_node);

        while let Some(parent) = opt_parent {
            if self.vec[current_node].priority() > self.vec[parent].priority() {
                break;
            }
            self.vec.swap(current_node, parent);

            current_node = parent;
            opt_parent = Self::parent(current_node);
        }
    }

    fn sift_down(&mut self, node: usize) {
        let mut current_node = node;

        loop {
            let left = self.left(current_node);
            let right = self.right(current_node);

            match (left, right) {
                (None, None) => break,
                (None, Some(right)) => {
                    if self.vec[right].priority() < self.vec[current_node].priority() {
                        self.vec.swap(current_node, right);
                        current_node = right;
                    } else {
                        break;
                    }
                }
                (Some(left), None) => {
                    if self.vec[left].priority() < self.vec[current_node].priority() {
                        self.vec.swap(current_node, left);
                        current_node = left;
                    } else {
                        break;
                    }
                }
                (Some(left), Some(right)) => {
                    let new = if self.vec[left].priority() <= self.vec[right].priority() {
                        left
                    } else {
                        right
                    };

                    if self.vec[new].priority() < self.vec[current_node].priority() {
                        self.vec.swap(current_node, new);
                        current_node = new;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    fn pop(&mut self) -> Option<V> {
        if self.len() <= 1 {
            return self.vec.pop().map(|vp| vp.into_value());
        }

        let last_node = self.len() - 1;
        self.vec.swap(0, last_node);
        let root = self.vec.pop()?;

        self.sift_down(0);

        Some(root.into_value())
    }

    fn min(&self) -> Option<&V> {
        self.vec.first().map(|vp| vp.value())
    }
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> FromIterator<Vp> for VecStorage<V, P, Vp> {
    fn from_iter<T: IntoIterator<Item = Vp>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<P: Debug + Ord, V, Vp: ValuePriorityPair<V, P>> VecStorage<V, P, Vp> {
        fn assert_order(&self) {
            for (i, vp) in self.vec.iter().enumerate() {
                let p = vp.priority();

                let left = self.left(i);
                let right = self.right(i);

                if let Some(left) = left {
                    assert!(
                        p < self.vec[left].priority(),
                        "own priority: {:?}, left priority: {:?}",
                        *p,
                        self.vec[left].priority()
                    );
                }

                if let Some(right) = right {
                    assert!(
                        p < self.vec[right].priority(),
                        "own priority: {:?}, right priority: {:?}",
                        *p,
                        self.vec[right].priority()
                    );
                }
            }
        }
    }

    #[test]
    fn build() {
        let values = vec![15, 20, 9, 1, 11, 8, 4, 13];
        let len = values.len();

        let bh: VecBinaryHeap<u32> = BinaryHeap::from(values);
        assert_eq!(bh.len(), len);
        bh.storage.assert_order();
    }

    #[test]
    fn from_iter() {
        let bh: VecBinaryHeap<u32> = (0..10).rev().collect();
        bh.storage.assert_order();
    }

    #[test]
    fn insert() {
        let mut bh: VecBinaryHeap<u32> = BinaryHeap::default();
        for i in (0..10).rev() {
            bh.insert(i);
            bh.storage.assert_order();
        }
    }

    #[test]
    fn pop() {
        let mut bh: VecBinaryHeap<u32> = (0..10).rev().collect();

        for _ in 0..9 {
            bh.pop();
            bh.storage.assert_order();
        }
    }
}
