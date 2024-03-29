use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct SortedBinaryTree<K: Ord, V> {
    root: Option<Box<Node<K, V>>>,
    len: usize,
}

impl<K: Ord, V> SortedBinaryTree<K, V> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.root.as_ref()?.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.root.as_mut()?.get_mut(key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let old_value = match &mut self.root {
            Some(root) => root.insert(key, value),
            None => {
                self.root = Some(Box::new(Node::new(key, value)));
                None
            }
        };

        if old_value.is_none() {
            self.len += 1;
        }

        old_value
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let value = Node::remove(&mut self.root, key);

        if value.is_some() {
            self.len -= 1;
        }

        value
    }

    pub fn contains(&self, key: &K) -> bool {
        if let Some(root) = &self.root {
            root.contains(key)
        } else {
            false
        }
    }
}

impl<K: Ord, V> Default for SortedBinaryTree<K, V> {
    fn default() -> Self {
        Self { root: None, len: 0 }
    }
}

impl<K: Ord, V> Index<&K> for SortedBinaryTree<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).expect("key not in tree")
    }
}

impl<K: Ord, V> IndexMut<&K> for SortedBinaryTree<K, V> {
    fn index_mut(&mut self, index: &K) -> &mut Self::Output {
        self.get_mut(index).expect("key not in tree")
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for SortedBinaryTree<K, V> {
    fn from(value: [(K, V); N]) -> Self {
        Self::from_iter(value)
    }
}

impl<K: Ord, V> From<Vec<(K, V)>> for SortedBinaryTree<K, V> {
    fn from(value: Vec<(K, V)>) -> Self {
        Self::from_iter(value)
    }
}

impl<K: Clone + Ord, const N: usize> From<[K; N]> for SortedBinaryTree<K, K> {
    fn from(value: [K; N]) -> Self {
        Self::from_iter(value)
    }
}

impl<K: Clone + Ord> From<Vec<K>> for SortedBinaryTree<K, K> {
    fn from(value: Vec<K>) -> Self {
        Self::from_iter(value)
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for SortedBinaryTree<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut ret = Self::default();
        for (key, val) in iter {
            ret.insert(key, val);
        }

        ret
    }
}

impl<K: Clone + Ord> FromIterator<K> for SortedBinaryTree<K, K> {
    fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
        Self::from_iter(iter.into_iter().map(|p| (p.clone(), p)))
    }
}

#[derive(Clone, Debug)]
struct Node<K: Ord, V> {
    key: K,
    value: V,
    left: Option<Box<Node<K, V>>>,
    right: Option<Box<Node<K, V>>>,
}

impl<K: Ord, V> Node<K, V> {
    fn new(key: K, value: V) -> Self {
        Self {
            key,
            value,
            left: None,
            right: None,
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        match self.key.cmp(key) {
            std::cmp::Ordering::Equal => Some(&self.value),
            std::cmp::Ordering::Less => self.left.as_ref()?.get(key),
            std::cmp::Ordering::Greater => self.right.as_ref()?.get(key),
        }
    }

    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.key.cmp(key) {
            std::cmp::Ordering::Equal => Some(&mut self.value),
            std::cmp::Ordering::Less => self.left.as_mut()?.get_mut(key),
            std::cmp::Ordering::Greater => self.right.as_mut()?.get_mut(key),
        }
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        match key.cmp(&self.key) {
            std::cmp::Ordering::Equal => {
                let mut value = value;
                std::mem::swap(&mut self.value, &mut value);
                Some(value)
            }
            std::cmp::Ordering::Less => match self.left.as_mut() {
                Some(left) => left.insert(key, value),
                None => {
                    self.left = Some(Box::new(Node::new(key, value)));
                    None
                }
            },
            std::cmp::Ordering::Greater => match self.right.as_mut() {
                Some(right) => right.insert(key, value),
                None => {
                    self.right = Some(Box::new(Node::new(key, value)));
                    None
                }
            },
        }
    }

    fn insert_rec(&mut self, other: Box<Node<K, V>>) {
        match other.key.cmp(&self.key) {
            std::cmp::Ordering::Equal => panic!("encountered to equal keys in tree"),
            std::cmp::Ordering::Less => match self.left.as_mut() {
                Some(left) => left.insert_rec(other),
                None => self.left = Some(other),
            },
            std::cmp::Ordering::Greater => match self.right.as_mut() {
                Some(right) => right.insert_rec(other),
                None => self.right = Some(other),
            },
        }
    }

    fn remove(node: &mut Option<Box<Node<K, V>>>, key: &K) -> Option<V> {
        match key.cmp(&node.as_ref()?.key) {
            std::cmp::Ordering::Equal => {
                let current = node.take()?;
                let ret = current.value;
                match (current.left, current.right) {
                    (None, None) => (),
                    (None, Some(right)) => {
                        node.replace(right);
                    }
                    (Some(left), None) => {
                        node.replace(left);
                    }
                    (Some(left), Some(right)) => {
                        node.replace(left);
                        node.as_mut()?.insert_rec(right);
                    }
                }
                Some(ret)
            }
            std::cmp::Ordering::Less => Self::remove(&mut node.as_mut()?.left, key),
            std::cmp::Ordering::Greater => Self::remove(&mut node.as_mut()?.right, key),
        }
    }

    fn contains(&self, key: &K) -> bool {
        match key.cmp(&self.key) {
            std::cmp::Ordering::Equal => true,
            std::cmp::Ordering::Less => match &self.left {
                Some(left) => left.contains(key),
                None => false,
            },
            std::cmp::Ordering::Greater => match &self.right {
                Some(right) => right.contains(key),
                None => false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert() {
        let mut ll = SortedBinaryTree::new();
        ll.insert(2, 2);
        ll.insert(1, 1);
        ll.insert(3, 3);
        ll.insert(4, 4);

        let root = ll.clone().root.unwrap();
        assert_eq!(root.value, 2);
        assert_eq!(root.clone().left.unwrap().value, 1);
        let right = root.clone().right.unwrap();
        assert_eq!(right.value, 3);
        assert_eq!(right.right.unwrap().value, 4);
    }

    #[test]
    fn remove() {
        let mut ll = SortedBinaryTree::from([2, 1, 3, 4]);

        ll.remove(&2);

        let root = ll.clone().root.unwrap();
        assert_eq!(root.value, 1);
        assert_eq!(root.right.clone().unwrap().value, 3);
        assert_eq!(root.right.unwrap().right.unwrap().value, 4);
    }

    #[test]
    fn contains() {
        let ll = SortedBinaryTree::from([2, 1, 3, 4]);

        for i in 1..=4 {
            assert!(ll.contains(&i));
        }

        assert!(!ll.contains(&5))
    }
}
