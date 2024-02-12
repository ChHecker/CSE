use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct BinaryHeap<V, P: Ord> {
    root: Option<Box<Node<V, P>>>,
    len: usize,
}

impl<V, P: Ord> BinaryHeap<V, P> {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert_no_sifting(&mut self, value: V, priority: P) {
        match &mut self.root {
            Some(root) => {
                let (height, next_node) = Self::next_node(self.len);
                root.insert(value, priority, next_node, height - 1);
            }
            None => self.root = Some(Box::new(Node::new(value, priority))),
        }

        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    // Get level and number in level of last node of tree
    fn last_node(len: usize) -> (usize, usize) {
        let height = ((len + 1) as f64).log2().ceil() as usize - 1; // 1: 0, 2: 1, 3: 1
        let target_node = len - 2usize.pow(height as u32); // 1: 0, 2: 0, 3: 1
        (height, target_node)
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

    pub fn insert(&mut self, value: V, priority: P) {
        self.insert_no_sifting(value, priority);
        let (height, last_node) = Self::last_node(self.len());
        if height > 0 {
            if let Some(root) = &mut self.root {
                root.sift_up(last_node, height - 1);
            }
        }
    }

    pub fn pop(&mut self) -> Option<V> {
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

    pub fn min(&self) -> Option<&V> {
        self.root.as_ref().map(|r| &r.value)
    }
}

impl<V, P: Ord> Default for BinaryHeap<V, P> {
    fn default() -> Self {
        Self { root: None, len: 0 }
    }
}

impl<V, P: Ord, const N: usize> From<[(V, P); N]> for BinaryHeap<V, P> {
    fn from(value: [(V, P); N]) -> Self {
        Self::from_iter(value)
    }
}

impl<V, P: Ord> From<Vec<(V, P)>> for BinaryHeap<V, P> {
    fn from(value: Vec<(V, P)>) -> Self {
        Self::from_iter(value)
    }
}

impl<P: Clone + Ord, const N: usize> From<[P; N]> for BinaryHeap<P, P> {
    fn from(value: [P; N]) -> Self {
        Self::from_iter(value)
    }
}

impl<P: Clone + Ord> From<Vec<P>> for BinaryHeap<P, P> {
    fn from(value: Vec<P>) -> Self {
        Self::from_iter(value)
    }
}

impl<V, P: Ord> FromIterator<(V, P)> for BinaryHeap<V, P> {
    fn from_iter<T: IntoIterator<Item = (V, P)>>(iter: T) -> Self {
        let mut ret = Self::default();
        for (val, prio) in iter {
            ret.insert_no_sifting(val, prio);
        }

        let (height, _) = Self::last_node(ret.len());
        if let Some(root) = &mut ret.root {
            if height > 0 {
                for level in (1..height).rev() {
                    for node in (0..2usize.pow(level as u32)).rev() {
                        root.sift_down_specific_node(node, level - 1);
                    }
                }
            }

            root.sift_down();
        }

        ret
    }
}

impl<P: Clone + Ord> FromIterator<P> for BinaryHeap<P, P> {
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        Self::from_iter(iter.into_iter().map(|p| (p.clone(), p)))
    }
}

#[derive(Clone, Debug)]
struct Node<V, P: Ord> {
    priority: P,
    value: V,
    left: Option<Box<Node<V, P>>>,
    right: Option<Box<Node<V, P>>>,
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
    use std::{borrow::Cow, fmt::Display};

    use ptree::TreeItem;

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

    impl<P: Clone + Ord, V: Clone + Display> TreeItem for Node<V, P> {
        type Child = Self;

        fn write_self<W: std::io::Write>(
            &self,
            f: &mut W,
            style: &ptree::Style,
        ) -> std::io::Result<()> {
            write!(f, "{}", style.paint(self.value.clone()))
        }

        fn children(&self) -> Cow<[Self::Child]> {
            let mut children = vec![];
            if let Some(left) = self.left.clone() {
                children.push(*left);
            }
            if let Some(right) = self.right.clone() {
                children.push(*right);
            }

            Cow::from(children)
        }
    }

    #[test]
    fn build() {
        let values = vec![15, 20, 9, 1, 11, 8, 4, 13];

        let bh = BinaryHeap::from(values);
        bh.root.unwrap().assert_order();
    }

    #[test]
    fn insert() {
        let mut bh = BinaryHeap::default();
        for i in (0..10).rev() {
            bh.insert(i, i);
            bh.root.as_ref().unwrap().assert_order();
        }
    }

    #[test]
    fn remove_min() {
        let mut bh = BinaryHeap::default();
        for i in (0..10).rev() {
            bh.insert(i, i);
        }

        for _ in 0..9 {
            bh.pop();
            bh.root.as_ref().unwrap().assert_order();
        }
    }
}
