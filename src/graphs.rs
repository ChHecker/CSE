use std::{
    collections::{HashMap, VecDeque},
    ops::Add,
};

use num_traits::Zero;

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

    pub fn get_weight(&self, edge: (usize, usize)) -> Option<&W> {
        let neighbor_idx = self.search_hash.get(&edge)?;
        Some(&self.neighbors[edge.0].get(*neighbor_idx).as_ref()?.1)
    }

    /// Get all neighbors and corresponding edge weights of a node.
    pub fn out_neighbors(&self, node: usize) -> &[(usize, W)] {
        &self.neighbors[node]
    }

    pub fn out_degree(&self, node: usize) -> usize {
        self.out_neighbors(node).len()
    }

    pub fn in_neighbors(&self, node: usize) -> Vec<&(usize, W)> {
        let mut out = vec![];

        for i in 0..self.num_nodes() {
            if let Some(&n) = self.search_hash.get(&(i, node)) {
                out.push(&self.neighbors[i][n]);
            }
        }

        out
    }

    pub fn in_degree(&self, node: usize) -> usize {
        let mut count = 0;

        for i in 0..self.num_nodes() {
            if self.search_hash.contains_key(&(i, node)) {
                count += 1;
            }
        }

        count
    }

    /// Get the weight of an edge.
    pub fn weight(&self, edge: (usize, usize)) -> Option<&W> {
        let idx = self.search_hash.get(&edge)?;
        Some(&self.neighbors[edge.0][*idx].1)
    }

    pub fn breadth_first_search(&self, start: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.num_nodes());

        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(u) = queue.pop_front() {
            for (v, _) in self.out_neighbors(u) {
                if !out.contains(v) {
                    queue.push_back(*v);
                    out.push(*v);
                }
            }
        }

        out
    }

    fn dfs_recursion(&self, node: usize, visited: &mut [bool], out: &mut Vec<usize>) {
        for (v, _) in self.out_neighbors(node) {
            if !visited[*v] {
                visited[*v] = true;
                self.dfs_recursion(*v, visited, out);
                out.push(*v);
            }
        }
    }

    pub fn depth_first_search(&self, start: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.num_nodes());
        let mut visited = vec![false; self.num_nodes()];

        self.dfs_recursion(start, &mut visited, &mut out);

        out
    }

    fn topological_order(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.num_nodes());
        let mut in_degree: Vec<usize> = (0..self.num_nodes()).map(|i| self.in_degree(i)).collect();
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, i)| **i == 0)
            .map(|(i, _)| i)
            .collect();

        while let Some(v) = queue.pop_front() {
            out.push(v);
            for (w, _) in self.out_neighbors(v) {
                if in_degree[*w] > 0 {
                    in_degree[*w] -= 1;
                } else {
                    queue.push_back(*w);
                }
            }
        }

        out
    }
}

impl<W> Graph<W>
where
    W: Add<W, Output = W> + Copy + PartialOrd + Zero + 'static,
{
    fn sssp_dag_rec(
        &self,
        node: usize,
        orders: &[usize],
        dist: &mut [Option<W>],
        parents: &mut [usize],
    ) {
        let mut sorted: Vec<usize> = self
            .out_neighbors(node)
            .iter()
            .map(|(i, _)| i)
            .copied()
            .collect();
        sorted.sort_by(|i, j| orders[*i].cmp(&orders[*j]));

        for neighbor in sorted {
            let weight = *self.get_weight((node, neighbor)).unwrap();
            let d_node = dist[node].unwrap();
            let d_neighbor = dist[neighbor].unwrap_or(d_node + weight);
            if d_neighbor > d_node + weight {
                dist[neighbor] = Some(d_node + weight);
            }
        }

        todo!()
    }

    pub fn sssp_dag(&self, start: usize) -> Vec<usize> {
        let mut parents: Vec<usize> = (0..self.num_nodes()).collect();
        let mut dist: Vec<Option<W>> = (0..self.num_nodes()).map(|_| None).collect();
        dist[start] = Some(W::zero());
        let order = self.topological_order();

        self.sssp_dag_rec(start, &order, &mut dist, &mut parents);

        parents
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
