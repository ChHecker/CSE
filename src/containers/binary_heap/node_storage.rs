use super::*;

#[derive(Clone, Debug)]
pub struct NodeStorage<V, P: Ord, Vp: ValuePriorityPair<V, P>> {
    pub(super) root: Option<Node<V, P, Vp>>,
    len: usize,
    phantom: PhantomData<(V, P, Vp)>,
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> NodeStorage<V, P, Vp> {
    fn level(len: usize) -> usize {
        ((len + 1) as f64).log2().ceil() as usize - 1 // 1: 0, 2: 1, 3: 1
    }

    // Get level and number in level of last node of tree
    fn last_node(len: usize) -> (usize, usize) {
        Self::index_to_level(len - 1)
    }

    // Get level and number in level of first free slot in tree
    fn next_node(len: usize) -> (usize, usize) {
        let (height, last_node) = Self::last_node(len);
        if last_node == 2usize.pow(height as u32) - 1 {
            (height + 1, 0)
        } else {
            (height, last_node + 1)
        }
    }

    fn index_to_level(node: usize) -> (usize, usize) {
        let level = Self::level(node + 1);
        let target_node = node + 1 - 2usize.pow(level as u32);

        (level, target_node)
    }
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> Storage<V, P, Vp> for NodeStorage<V, P, Vp> {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn push(&mut self, value_priority_pair: Vp) {
        match &mut self.root {
            Some(root) => {
                let (height, next_node) = Self::next_node(self.len);
                root.insert(value_priority_pair, next_node, height - 1);
            }
            None => self.root = Some(Node::new(value_priority_pair)),
        }

        self.len += 1;
    }

    fn sift_up_last_node(&mut self) {
        let (height, last_node) = Self::last_node(self.len);
        if self.len() > 1 {
            self.root.as_mut().unwrap().sift_up(last_node, height - 1);
        }
    }

    fn sift_down(&mut self, node: usize) {
        if let Some(root) = &mut self.root {
            if node == 0 {
                root.sift_down();
            } else {
                let (level, node) = Self::index_to_level(node);
                root.sift_down_specific_node(node, level - 1);
            }
        }
    }

    fn pop(&mut self) -> Option<V> {
        if let Some(root) = &self.root {
            if root.left.is_none() && root.right.is_none() {
                return self.root.take().map(|r| r.into_value());
            }
        }

        let root = self.root.as_mut()?;

        // Get last node
        let (height, last_node) = Self::last_node(self.len);
        let mut last_node = root.pop(last_node, height - 1);

        // Replace root with last node
        std::mem::swap(root.value_mut(), last_node.value_mut());
        std::mem::swap(root.priority_mut(), last_node.priority_mut());

        root.sift_down();

        self.len -= 1;

        Some(last_node.into_value())
    }

    fn min(&self) -> Option<&V> {
        self.root.as_ref().map(|r| r.value())
    }
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> Default for NodeStorage<V, P, Vp> {
    fn default() -> Self {
        Self {
            root: None,
            len: 0,
            phantom: PhantomData,
        }
    }
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> FromIterator<Vp> for NodeStorage<V, P, Vp> {
    fn from_iter<T: IntoIterator<Item = Vp>>(iter: T) -> Self {
        let mut ret = Self::default();
        for vp in iter {
            ret.push(vp);
        }

        ret
    }
}

#[derive(Clone, Debug)]
pub(super) struct Node<V, P: Ord, Vp: ValuePriorityPair<V, P>> {
    pub(super) value_priority_pair: Vp,
    pub(super) left: Option<Box<Node<V, P, Vp>>>,
    pub(super) right: Option<Box<Node<V, P, Vp>>>,
    phantom: PhantomData<(V, P)>,
}

impl<V, P: Ord, Vp: ValuePriorityPair<V, P>> Node<V, P, Vp> {
    fn new(value_priority_pair: Vp) -> Self {
        Self {
            value_priority_pair,
            left: None,
            right: None,
            phantom: PhantomData,
        }
    }

    fn priority(&self) -> &P {
        self.value_priority_pair.priority()
    }

    fn value(&self) -> &V {
        self.value_priority_pair.value()
    }

    fn priority_mut(&mut self) -> &mut P {
        self.value_priority_pair.priority_mut()
    }

    fn value_mut(&mut self) -> &mut V {
        self.value_priority_pair.value_mut()
    }

    fn into_value(self) -> V {
        self.value_priority_pair.into_value()
    }

    fn recursion_step(&mut self, target_node: usize, current_height: usize) -> (bool, bool) {
        let next_step = (target_node >> current_height) & 1;
        (current_height > 0, next_step == 0)
    }

    fn sift_down(&mut self) {
        match (self.left.as_mut(), self.right.as_mut()) {
            (None, None) => (),
            (None, Some(right)) => {
                if right.priority() < self.value_priority_pair.priority() {
                    self.value_priority_pair
                        .swap(&mut right.value_priority_pair);
                    right.sift_down();
                }
            }
            (Some(left), None) => {
                if left.priority() < self.value_priority_pair.priority() {
                    self.value_priority_pair.swap(&mut left.value_priority_pair);
                    left.sift_down();
                }
            }
            (Some(left), Some(right)) => {
                let new = if left.priority() <= right.priority() {
                    left
                } else {
                    right
                };
                if new.priority() < self.value_priority_pair.priority() {
                    self.value_priority_pair.swap(&mut new.value_priority_pair);
                    new.sift_down();
                }
            }
        }
    }

    #[allow(clippy::collapsible_else_if)]
    fn sift_down_specific_node(&mut self, target_node: usize, current_height: usize) {
        let (recurse, left) = self.recursion_step(target_node, current_height);
        // Haven't reached next to last level
        if recurse {
            if left {
                self.left
                    .as_mut()
                    .unwrap()
                    .sift_down_specific_node(target_node, current_height - 1);
            } else {
                self.right
                    .as_mut()
                    .unwrap()
                    .sift_down_specific_node(target_node, current_height - 1);
            }
        // Reached next to last level
        } else {
            if left {
                if let Some(left) = &mut self.left {
                    left.sift_down();
                }
            } else {
                if let Some(right) = &mut self.right {
                    right.sift_down();
                }
            }
        }
    }

    #[allow(clippy::collapsible_else_if)]
    fn sift_up(&mut self, target_node: usize, current_height: usize) -> Option<&mut Vp> {
        let (recurse, left) = self.recursion_step(target_node, current_height);
        // Haven't reached next to last level
        let vp = if recurse {
            if left {
                self.left
                    .as_mut()
                    .unwrap()
                    .sift_up(target_node, current_height - 1)
            } else {
                self.right
                    .as_mut()
                    .unwrap()
                    .sift_up(target_node, current_height - 1)
            }
        // Reached next to last level
        } else {
            if left {
                let left = self.left.as_mut().unwrap();
                Some(&mut left.value_priority_pair)
            } else {
                let right = self.right.as_mut().unwrap();
                Some(&mut right.value_priority_pair)
            }
        }?;

        if vp.priority() < self.value_priority_pair.priority() {
            self.value_priority_pair.swap(vp);
            Some(&mut self.value_priority_pair)
        } else {
            None
        }
    }

    #[allow(clippy::collapsible_else_if)]
    fn insert(&mut self, value_priority_pair: Vp, target_node: usize, current_height: usize) {
        let (recurse, left) = self.recursion_step(target_node, current_height);
        // Haven't reached next to last level
        if recurse {
            if left {
                self.left.as_mut().unwrap().insert(
                    value_priority_pair,
                    target_node,
                    current_height - 1,
                );
            } else {
                self.right.as_mut().unwrap().insert(
                    value_priority_pair,
                    target_node,
                    current_height - 1,
                );
            }
        // Reached next to last level
        } else {
            let node = Box::new(Node::new(value_priority_pair));
            if left {
                self.left = Some(node);
            } else {
                self.right = Some(node);
            }
        }
    }

    #[allow(clippy::collapsible_else_if)]
    fn pop(&mut self, target_node: usize, current_height: usize) -> Box<Node<V, P, Vp>> {
        let (recurse, left) = self.recursion_step(target_node, current_height);
        // Haven't reached next to last level
        if recurse {
            // Go left
            if left {
                self.left
                    .as_mut()
                    .unwrap()
                    .pop(target_node, current_height - 1)
            // Go right
            } else {
                self.right
                    .as_mut()
                    .unwrap()
                    .pop(target_node, current_height - 1)
            }
        // Reached next to last level
        } else {
            // Go left
            if left {
                self.left.take().unwrap()
            // Go right
            } else {
                self.right.take().unwrap()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<P: Debug + Ord, V, Vp: ValuePriorityPair<V, P>> Node<V, P, Vp> {
        fn assert_order(&self) {
            if let Some(left) = &self.left {
                assert!(
                    left.priority() >= self.priority(),
                    "own priority: {:?}, left priority: {:?}",
                    self.priority(),
                    left.priority()
                );
                left.assert_order();
            }
            if let Some(right) = &self.right {
                assert!(
                    right.priority() >= self.priority(),
                    "own priority: {:?}, right priority: {:?}",
                    self.priority(),
                    right.priority()
                );
                right.assert_order();
            }
        }
    }

    #[test]
    fn build() {
        let values = vec![15, 20, 9, 1, 11, 8, 4, 13];
        let len = values.len();

        let bh: NodeBinaryHeap<u32, u32, u32> = BinaryHeap::from(values);
        assert_eq!(bh.len(), len);
        bh.storage.root.unwrap().assert_order();
    }

    #[test]
    fn from_iter() {
        let bh: NodeBinaryHeap<u32, u32, u32> = (0..10).rev().collect();
        bh.storage.root.unwrap().assert_order();
    }

    #[test]
    fn insert() {
        let mut bh: NodeBinaryHeap<u32, u32, u32> = BinaryHeap::default();
        for i in (0..10).rev() {
            bh.insert(i);
            bh.storage.root.as_ref().unwrap().assert_order();
        }
    }

    #[test]
    fn pop() {
        let mut bh: NodeBinaryHeap<u32, u32, u32> = (0..10).rev().collect();

        for _ in 0..9 {
            bh.pop();
            bh.storage.root.as_ref().unwrap().assert_order();
        }
    }
}
