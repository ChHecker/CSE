use super::*;

#[derive(Clone, Debug, Default)]
pub(super) struct VecStorage<V, P: Ord> {
    pub(super) vec: Vec<(V, P)>,
}

impl<V, P: Ord> VecStorage<V, P> {
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

impl<V, P: Ord> Storage<V, P> for VecStorage<V, P> {
    fn len(&self) -> usize {
        self.vec.len()
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn push(&mut self, value: V, priority: P) {
        self.vec.push((value, priority));
    }

    fn sift_up_last_node(&mut self) {
        let mut current_node = self.len() - 1;
        let mut opt_parent = Self::parent(current_node);

        while let Some(parent) = opt_parent {
            if self.vec[current_node].1 > self.vec[parent].1 {
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
                    if self.vec[right].1 < self.vec[current_node].1 {
                        self.vec.swap(current_node, right);
                        current_node = right;
                    } else {
                        break;
                    }
                }
                (Some(left), None) => {
                    if self.vec[left].1 < self.vec[current_node].1 {
                        self.vec.swap(current_node, left);
                        current_node = left;
                    } else {
                        break;
                    }
                }
                (Some(left), Some(right)) => {
                    let new = if self.vec[left].1 <= self.vec[right].1 {
                        left
                    } else {
                        right
                    };

                    if self.vec[new].1 < self.vec[current_node].1 {
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
            return self.vec.pop().map(|(v, _)| v);
        }

        let last_node = self.len() - 1;
        self.vec.swap(0, last_node);
        let root = self.vec.pop()?;

        self.sift_down(0);

        Some(root.0)
    }

    fn min(&self) -> Option<&V> {
        self.vec.first().map(|(v, _)| v)
    }
}

impl<V, P: Ord> FromIterator<(V, P)> for VecStorage<V, P> {
    fn from_iter<T: IntoIterator<Item = (V, P)>>(iter: T) -> Self {
        Self {
            vec: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<P: Debug + Ord, V> VecStorage<V, P> {
        fn assert_order(&self) {
            for (i, (_, p)) in self.vec.iter().enumerate() {
                let left = self.left(i);
                let right = self.right(i);

                if let Some(left) = left {
                    assert!(
                        *p < self.vec[left].1,
                        "own priority: {:?}, left priority: {:?}",
                        *p,
                        self.vec[left].1
                    );
                }

                if let Some(right) = right {
                    assert!(
                        *p < self.vec[right].1,
                        "own priority: {:?}, right priority: {:?}",
                        *p,
                        self.vec[right].1
                    );
                }
            }
        }
    }

    #[test]
    fn build() {
        let values = vec![15, 20, 9, 1, 11, 8, 4, 13];
        let len = values.len();

        let bh: BinaryHeap<u32, u32, VecStorage<u32, u32>> = BinaryHeap::from(values);
        assert_eq!(bh.len(), len);
        bh.storage.assert_order();
    }

    #[test]
    fn from_iter() {
        let bh: BinaryHeap<u32, u32, VecStorage<u32, u32>> = (0..10).rev().collect();
        bh.storage.assert_order();
    }

    #[test]
    fn insert() {
        let mut bh: BinaryHeap<u32, u32, VecStorage<u32, u32>> = BinaryHeap::default();
        for i in (0..10).rev() {
            bh.insert(i, i);
            bh.storage.assert_order();
        }
    }

    #[test]
    fn pop() {
        let mut bh: BinaryHeap<u32, u32, VecStorage<u32, u32>> = (0..10).rev().collect();

        for _ in 0..9 {
            bh.pop();
            bh.storage.assert_order();
        }
    }
}
