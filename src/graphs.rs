use std::collections::{HashMap, VecDeque};

/// A weighted, directed graph.
#[derive(Clone, Debug)]
pub struct Graph<W> {
    neighbors: Vec<Vec<(usize, W)>>,
    /// Store index into inner vec for each edge
    search_hash: HashMap<(usize, usize), usize>,
}

impl<W> Graph<W> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn num_nodes(&self) -> usize {
        self.neighbors.len()
    }

    /// Insert a new node. Returns its number.
    pub fn insert_node(&mut self) -> usize {
        let index = self.num_nodes();
        self.neighbors.push(vec![]);
        index
    }

    /// Remove a node by its number.
    pub fn remove_node(&mut self, node: usize) -> bool {
        if node >= self.num_nodes() {
            return false;
        }
        let num_nodes = self.num_nodes();

        let neighbors = self.neighbors.remove(node);
        // Remove outgoing edges
        for (n, _) in neighbors {
            self.search_hash.remove(&(node, n));
        }
        // Remove incoming edges
        for i in 0..num_nodes {
            self.remove_edge((i, node));
        }

        // Update edges
        for i in 0..node {
            for j in node + 1..num_nodes {
                if let Some(idx) = self.search_hash.remove(&(i, j)) {
                    self.search_hash.insert((i, j - 1), idx);
                    self.neighbors[i][idx].0 -= 1;
                }
            }
        }

        for i in node + 1..num_nodes {
            for j in 0..node {
                if let Some(idx) = self.search_hash.remove(&(i, j)) {
                    self.search_hash.insert((i - 1, j), idx);
                }
            }

            for j in node + 1..num_nodes {
                if let Some(idx) = self.search_hash.remove(&(i, j)) {
                    self.search_hash.insert((i - 1, j - 1), idx);
                    self.neighbors[i - 1][idx].0 -= 1;
                }
            }
        }

        true
    }

    /// Insert an edge by the connected nodes and its weight.
    pub fn insert_edge(&mut self, nodes: (usize, usize), weight: W) -> bool {
        let (node, neighbor) = nodes;

        assert!(node < self.num_nodes());
        assert!(neighbor < self.num_nodes());

        // Check if edge already in graph
        if self.search_hash.contains_key(&nodes) {
            return false;
        }

        let neighbor_index = self.neighbors[node].len();
        self.neighbors[node].push((neighbor, weight));

        self.search_hash.insert(nodes, neighbor_index);
        true
    }

    /// Remove an edge by the connected nodes.
    pub fn remove_edge(&mut self, edge: (usize, usize)) -> Option<W> {
        let neighbor_idx = self.search_hash.remove(&edge)?;
        let weight = self.neighbors[edge.0].remove(neighbor_idx).1;

        for (idx, neighbor) in self.neighbors[edge.0].iter().enumerate().skip(neighbor_idx) {
            if let Some(old_idx) = self.search_hash.get_mut(&(edge.0, neighbor.0)) {
                *old_idx = idx;
            }
        }

        Some(weight)
    }

    /// Get all neighbors and corresponding edge weights of a node.
    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = &(usize, W)> {
        self.neighbors[node].iter()
    }

    /// Get the weight of an edge.
    pub fn weight(&self, edge: (usize, usize)) -> Option<&W> {
        let idx = self.search_hash.get(&edge)?;
        Some(&self.neighbors[edge.0][*idx].1)
    }

    pub fn breadth_first_search(&self, start: usize) -> Vec<usize> {
        let mut out = vec![];

        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(u) = queue.pop_front() {
            for (v, _) in self.neighbors(u) {
                if !out.contains(v) {
                    queue.push_back(*v);
                    out.push(*v);
                }
            }
        }

        out
    }

    fn dfs_recursion(&self, node: usize, visited: &mut [bool], out: &mut Vec<usize>) {
        for (v, _) in self.neighbors(node) {
            if !visited[*v] {
                visited[*v] = true;
                self.dfs_recursion(*v, visited, out);
                out.push(*v);
            }
        }
    }

    pub fn depth_first_search(&self, start: usize) -> Vec<usize> {
        let mut out = vec![];
        let mut visited = vec![false; self.num_nodes()];

        self.dfs_recursion(start, &mut visited, &mut out);

        out
    }
}

impl<T> Default for Graph<T> {
    fn default() -> Self {
        Self {
            neighbors: Default::default(),
            search_hash: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph() {
        let mut graph = Graph::new();

        for i in 0..5 {
            assert_eq!(graph.insert_node(), i);
        }

        let edges = [(0, 1), (0, 2), (1, 2), (3, 4), (4, 2)];
        for (i, n) in edges.into_iter().enumerate() {
            graph.insert_edge(n, i);
        }

        assert_eq!(graph.neighbors[0], vec![(1, 0), (2, 1)]);
        assert_eq!(graph.neighbors[1], vec![(2, 2)]);
        assert_eq!(graph.neighbors[2], vec![]);
        assert_eq!(graph.neighbors[3], vec![(4, 3)]);
        assert_eq!(graph.neighbors[4], vec![(2, 4)]);

        for n in &edges {
            assert!(graph.search_hash.contains_key(n))
        }

        assert!(graph.remove_node(0));

        assert_eq!(graph.neighbors[0], vec![(1, 2)]);
        assert_eq!(graph.neighbors[1], vec![]);
        assert_eq!(graph.neighbors[2], vec![(3, 3)]);
        assert_eq!(graph.neighbors[3], vec![(1, 4)]);
        assert!(graph.search_hash.contains_key(&(0, 1)));
    }

    #[test]
    fn breadth_first_search() {
        let mut graph = Graph::new();
        for _ in 0..7 {
            graph.insert_node();
        }

        let edges = [(3, 1), (3, 5), (1, 0), (1, 2), (5, 4), (5, 6)];
        for edge in edges.iter() {
            graph.insert_edge(*edge, 1);
        }

        let bfs = graph.breadth_first_search(3);
        for (i, (_, j)) in bfs.into_iter().zip(edges) {
            assert_eq!(i, j);
        }
    }

    #[test]
    fn depth_first_search() {
        let mut graph = Graph::new();
        for _ in 0..7 {
            graph.insert_node();
        }

        let edges = [(3, 1), (3, 5), (1, 0), (1, 2), (5, 4), (5, 6)];
        for edge in edges.iter() {
            graph.insert_edge(*edge, 1);
        }

        let dfs = graph.depth_first_search(3);
        assert_eq!(dfs[0], 0);
        assert_eq!(dfs[1], 2);
        assert_eq!(dfs[2], 1);
        assert_eq!(dfs[3], 4);
        assert_eq!(dfs[4], 6);
        assert_eq!(dfs[5], 5);
    }
}
