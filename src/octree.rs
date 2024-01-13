use nalgebra::Vector3;

#[derive(Clone, Debug)]
pub(super) struct Octree {
    root: Node,
    theta: f64,
}

impl Octree {
    pub(super) fn new(particles: Vec<Mass>, theta: f64) -> Self {
        Self {
            root: Node::from_particles(particles),
            theta,
        }
    }

    pub(super) fn calculate_acceleration<F>(&self, particle: &Mass, force_fn: &F) -> Vector3<f64>
    where
        F: Fn(Vector3<f64>, f64, f64) -> Vector3<f64>,
    {
        self.root
            .calculate_acceleration(particle, force_fn, self.theta)
    }
}

#[derive(Clone, Debug)]
pub struct Mass {
    pub mass: f64,
    pub position: Vector3<f64>,
}

#[derive(Clone, Debug)]
struct Node {
    subnodes: Option<[Option<Box<Node>>; 8]>,
    mass: Option<Mass>,
    contains_particle: bool,
    center: Vector3<f64>,
    width: f64,
}

impl Node {
    fn new(center: Vector3<f64>, width: f64) -> Self {
        Self {
            subnodes: None,
            mass: None,
            contains_particle: false,
            center,
            width,
        }
    }

    fn from_particles(particles: Vec<Mass>) -> Self {
        let mut v_min = Vector3::zeros();
        let mut v_max = Vector3::zeros();
        for particle in particles.iter() {
            for (i, elem) in particle.position.iter().enumerate() {
                if *elem > v_max[i] {
                    v_max[i] = *elem;
                }
                if *elem < v_min[i] {
                    v_min[i] = *elem;
                }
            }
        }
        let width = (v_max - v_min).max();

        let mut node = Self {
            subnodes: None,
            mass: None,
            contains_particle: false,
            center: Vector3::zeros(),
            width,
        };

        for particle in particles {
            node.insert_particle(particle);
        }

        node.calculate_mass();

        node
    }

    fn insert_particle(&mut self, particle: Mass) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode = Node::choose_subnode(&self.center, &particle.position);

                subnodes[new_subnode]
                    .get_or_insert_with(|| {
                        Box::new(Node::new(
                            Self::center_from_subnode_static(self.width, self.center, new_subnode),
                            self.width / 2.,
                        ))
                    })
                    .insert_particle(particle);

                self.calculate_mass();
            }

            // Self is outer node
            None => match self.contains_particle {
                // Self contains a particle, subdivide
                true => {
                    let mass = self
                        .mass
                        .take()
                        .expect("node contains particle, but has no mass");

                    let previous_index = Node::choose_subnode(&self.center, &mass.position);
                    let mut previous_node = Box::new(Node::new(
                        self.center_from_subnode(previous_index),
                        self.width / 2.,
                    ));
                    previous_node.insert_particle(mass);

                    let new_index = Node::choose_subnode(&self.center, &particle.position);
                    let mut new_node = Box::new(Node::new(
                        self.center_from_subnode(new_index),
                        self.width / 2.,
                    ));
                    new_node.insert_particle(particle);

                    let mut new_nodes: [Option<Box<Node>>; 8] = Default::default();
                    new_nodes[previous_index] = Some(previous_node);
                    new_nodes[new_index] = Some(new_node);

                    self.subnodes = Some(new_nodes);
                    self.contains_particle = false;
                    self.calculate_mass();
                }

                // Self doesn't contain a particle, add mass of particle
                false => {
                    self.mass = Some(particle);
                    self.contains_particle = true;
                }
            },
        }
    }

    fn calculate_mass(&mut self) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, center_of_mass) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .filter(|node| node.subnodes.is_some() || node.mass.is_some())
                .map(|node| {
                    let node_mass = node.mass.as_ref().unwrap();
                    (node_mass.mass, node_mass.position)
                })
                .reduce(|(m_acc, pos_acc), (m, pos)| {
                    (m_acc + m, (m_acc * pos_acc + m * pos) / (m_acc + m))
                })
                .unwrap();
            self.mass = Some(Mass {
                mass,
                position: center_of_mass,
            });
        }
    }

    fn calculate_acceleration<F>(&self, particle: &Mass, acc_fn: &F, theta: f64) -> Vector3<f64>
    where
        F: Fn(Vector3<f64>, f64, f64) -> Vector3<f64>,
    {
        let mut acc = Vector3::zeros();

        if let Some(mass) = &self.mass {
            let r = mass.position - particle.position;

            if self.contains_particle || self.width / r.norm() < theta {
                // leaf nodes or node is far enough away
                acc += acc_fn(r, particle.mass, mass.mass);
            } else {
                // near field forces, go deeper into tree
                for node in self
                    .subnodes
                    .as_ref()
                    .expect("node has neither particle nor subnodes")
                {
                    if let Some(node) = &node {
                        acc += node.calculate_acceleration(particle, acc_fn, theta);
                    }
                }
            }
        }

        acc
    }

    fn choose_subnode(center: &Vector3<f64>, position: &Vector3<f64>) -> usize {
        if position.x > center.x {
            if position.y > center.y {
                if position.z > center.z {
                    return 0;
                }
                return 4;
            }
            if position.z > center.z {
                return 3;
            }
            return 7;
        }
        if position.y > center.y {
            if position.z > center.z {
                return 1;
            }
            return 5;
        }
        if position.z > center.z {
            return 2;
        }
        6
    }

    fn center_from_subnode(&self, i: usize) -> Vector3<f64> {
        Self::center_from_subnode_static(self.width, self.center, i)
    }

    fn center_from_subnode_static(width: f64, center: Vector3<f64>, i: usize) -> Vector3<f64> {
        let step_size = width / 2.;
        if i == 0 {
            return center + Vector3::new(step_size, step_size, step_size);
        }
        if i == 1 {
            return center + Vector3::new(-step_size, step_size, step_size);
        }
        if i == 2 {
            return center + Vector3::new(-step_size, -step_size, step_size);
        }
        if i == 3 {
            return center + Vector3::new(step_size, -step_size, step_size);
        }
        if i == 4 {
            return center + Vector3::new(step_size, step_size, -step_size);
        }
        if i == 5 {
            return center + Vector3::new(-step_size, step_size, -step_size);
        }
        if i == 6 {
            return center + Vector3::new(-step_size, -step_size, -step_size);
        }
        center + Vector3::new(step_size, -step_size, -step_size)
    }
}
