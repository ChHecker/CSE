use super::*;

#[derive(Clone, Debug)]
pub(super) struct NodeStorage<V, P: Ord> {
    pub(super) root: Option<Node<V, P>>,
    len: usize,
}

impl<V, P: Ord> NodeStorage<V, P> {
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

impl<V, P: Ord> Storage<V, P> for NodeStorage<V, P> {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn push(&mut self, value: V, priority: P) {
        match &mut self.root {
            Some(root) => {
                let (height, next_node) = Self::next_node(self.len);
                root.insert(value, priority, next_node, height - 1);
            }
            None => self.root = Some(Node::new(value, priority)),
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
                return self.root.take().map(|r| r.value);
            }
        }

        let root = self.root.as_mut()?;

        // Get last node
        let (height, last_node) = Self::last_node(self.len);
        let mut last_node = root.pop(last_node, height - 1);

        // Replace root with last node
        std::mem::swap(&mut root.value, &mut last_node.value);
        std::mem::swap(&mut root.priority, &mut last_node.priority);

        root.sift_down();

        self.len -= 1;

        Some(last_node.value)
    }

    fn min(&self) -> Option<&V> {
        self.root.as_ref().map(|r| &r.value)
    }
}

impl<V, P: Ord> Default for NodeStorage<V, P> {
    fn default() -> Self {
        Self { root: None, len: 0 }
    }
}

impl<V, P: Ord> FromIterator<(V, P)> for NodeStorage<V, P> {
    fn from_iter<T: IntoIterator<Item = (V, P)>>(iter: T) -> Self {
        let mut ret = Self::default();
        for (val, prio) in iter {
            ret.push(val, prio);
        }

        ret
    }
}

#[derive(Clone, Debug)]
pub(super) struct Node<V, P: Ord> {
    pub(super) priority: P,
    pub(super) value: V,
    pub(super) left: Option<Box<Node<V, P>>>,
    pub(super) right: Option<Box<Node<V, P>>>,
}

impl<V, P: Ord> Node<V, P> {
    fn new(value: V, priority: P) -> Self {
        Self {
            priority,
            value,
            left: None,
            right: None,
        }
    }

    fn recursion_step(&mut self, target_node: usize, current_height: usize) -> (bool, bool) {
        let next_step = (target_node >> current_height) & 1;
        (current_height > 0, next_step == 0)
    }

    fn sift_down(&mut self) {
        match (self.left.as_mut(), self.right.as_mut()) {
            (None, None) => (),
            (None, Some(right)) => {
                if right.priority < self.priority {
                    std::mem::swap(&mut self.value, &mut right.value);
                    std::mem::swap(&mut self.priority, &mut right.priority);
                    right.sift_down();
                }
            }
            (Some(left), None) => {
                if left.priority < self.priority {
                    std::mem::swap(&mut self.value, &mut left.value);
                    std::mem::swap(&mut self.priority, &mut left.priority);
                    left.sift_down();
                }
            }
            (Some(left), Some(right)) => {
                let new = if left.priority <= right.priority {
                    left
                } else {
                    right
                };
                if new.priority < self.priority {
                    std::mem::swap(&mut self.value, &mut new.value);
                    std::mem::swap(&mut self.priority, &mut new.priority);
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
    fn sift_up(&mut self, target_node: usize, current_height: usize) -> Option<(&mut V, &mut P)> {
        let (recurse, left) = self.recursion_step(target_node, current_height);
        // Haven't reached next to last level
        let (val, prio) = if recurse {
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
                Some((&mut left.value, &mut left.priority))
            } else {
                let right = self.right.as_mut().unwrap();
                Some((&mut right.value, &mut right.priority))
            }
        }?;

        if *prio < self.priority {
            std::mem::swap(&mut self.value, val);
            std::mem::swap(&mut self.priority, prio);
            Some((&mut self.value, &mut self.priority))
        } else {
            None
        }
    }

    #[allow(clippy::collapsible_else_if)]
    fn insert(&mut self, value: V, priority: P, target_node: usize, current_height: usize) {
        let (recurse, left) = self.recursion_step(target_node, current_height);
        // Haven't reached next to last level
        if recurse {
            if left {
                self.left.as_mut().unwrap().insert(
                    value,
                    priority,
                    target_node,
                    current_height - 1,
                );
            } else {
                self.right.as_mut().unwrap().insert(
                    value,
                    priority,
                    target_node,
                    current_height - 1,
                );
            }
        // Reached next to last level
        } else {
            let node = Box::new(Node::new(value, priority));
            if left {
                self.left = Some(node);
            } else {
                self.right = Some(node);
            }
        }
    }

    #[allow(clippy::collapsible_else_if)]
    fn pop(&mut self, target_node: usize, current_height: usize) -> Box<Node<V, P>> {
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

    impl<P: Debug + Ord, V> Node<V, P> {
        fn assert_order(&self) {
            if let Some(left) = &self.left {
                assert!(
                    left.priority >= self.priority,
                    "own priority: {:?}, left priority: {:?}",
                    self.priority,
                    left.priority
                );
                left.assert_order();
            }
            if let Some(right) = &self.right {
                assert!(
                    right.priority >= self.priority,
                    "own priority: {:?}, right priority: {:?}",
                    self.priority,
                    right.priority
                );
                right.assert_order();
            }
        }
    }

    #[test]
    fn build() {
        let values = vec![15, 20, 9, 1, 11, 8, 4, 13];
        let len = values.len();

        let bh: BinaryHeap<u32, u32, NodeStorage<u32, u32>> = BinaryHeap::from(values);
        assert_eq!(bh.len(), len);
        bh.storage.root.unwrap().assert_order();
    }

    #[test]
    fn from_iter() {
        let bh: BinaryHeap<u32, u32, NodeStorage<u32, u32>> = (0..10).rev().collect();
        bh.storage.root.unwrap().assert_order();
    }

    #[test]
    fn insert() {
        let mut bh: BinaryHeap<u32, u32, NodeStorage<u32, u32>> = BinaryHeap::default();
        for i in (0..10).rev() {
            bh.insert(i, i);
            bh.storage.root.as_ref().unwrap().assert_order();
        }
    }

    #[test]
    fn pop() {
        let mut bh: BinaryHeap<u32, u32, NodeStorage<u32, u32>> = (0..10).rev().collect();

        for _ in 0..9 {
            bh.pop();
            bh.storage.root.as_ref().unwrap().assert_order();
        }
    }
}
