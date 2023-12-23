use nalgebra::Vector3;

#[derive(Clone, Debug)]
pub struct Octree {
    root: Node,
    theta: f64,
}

impl Octree {
    pub(crate) fn new(particles: &[Mass], theta: f64) -> Self {
        Self {
            root: Node::new(particles),
            theta,
        }
    }

    pub(crate) fn calculate_force<F>(&self, particle: &Mass, force_fn: &F) -> Vector3<f64>
    where
        F: Fn(Vector3<f64>, f64, f64) -> Vector3<f64>,
    {
        self.root.calculate_force(particle, force_fn, self.theta)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Mass {
    pub mass: f64,
    pub center_of_mass: Vector3<f64>,
}

impl Mass {
    pub fn new(mass: f64, position: Vector3<f64>) -> Self {
        Self {
            mass,
            center_of_mass: position,
        }
    }
}

#[derive(Clone, Debug)]
struct Node {
    subnodes: Option<[Box<Node>; 8]>,
    mass: Option<Mass>,
    center: Vector3<f64>,
    width: f64,
}

impl Node {
    fn new(particles: &[Mass]) -> Self {
        let mut v_min = Vector3::zeros();
        let mut v_max = Vector3::zeros();
        for particle in particles {
            for (i, elem) in particle.center_of_mass.iter().enumerate() {
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
            center: Vector3::zeros(),
            width,
        };

        for particle in particles {
            node.insert_particle(particle);
        }

        node
    }

    fn insert_particle(&mut self, particle: &Mass) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode = Node::choose_subnode(&self.center, &particle.center_of_mass);
                subnodes[new_subnode].insert_particle(particle);
                self.calculate_mass();
            }
            // Self is outer node
            None => match &self.mass {
                // Self contains a particle, subdivide
                Some(mass) => {
                    let previous_particle_subnode =
                        Node::choose_subnode(&self.center, &mass.center_of_mass);
                    let new_particle_subnode =
                        Node::choose_subnode(&self.center, &particle.center_of_mass);

                    let new_nodes = core::array::from_fn(|i| {
                        let mut new_node = Node {
                            subnodes: None,
                            mass: None,
                            center: self.center_from_subnode(i),
                            width: self.width / 4.,
                        };
                        if i == previous_particle_subnode {
                            new_node.insert_particle(mass);
                        }
                        if i == new_particle_subnode {
                            new_node.insert_particle(particle);
                        }
                        Box::new(new_node)
                    });

                    self.subnodes = Some(new_nodes);
                    self.mass = None;
                    self.calculate_mass()
                }
                // Self doesn't contain a particle, add mass of particle
                None => self.mass = Some(particle.clone()),
            },
        }
    }

    fn calculate_mass(&mut self) {
        if let Some(subnodes) = &mut self.subnodes {
            let mut mass = 0.;
            let mut center_of_mass = Vector3::zeros();
            for node in subnodes {
                if node.subnodes.is_some() || node.mass.is_some() {
                    node.calculate_mass();
                    let node_mass = node.mass.as_ref().unwrap();
                    center_of_mass = (center_of_mass * mass
                        + node_mass.center_of_mass * node_mass.mass)
                        / (mass + node_mass.mass);
                    mass += node_mass.mass;
                }
            }
            self.mass = Some(Mass {
                mass,
                center_of_mass,
            });
        }
    }

    fn calculate_force<F>(&self, particle: &Mass, force_fn: &F, theta: f64) -> Vector3<f64>
    where
        F: Fn(Vector3<f64>, f64, f64) -> Vector3<f64>,
    {
        let mut force = Vector3::zeros();

        // check if node has a center of mass (empty leaf node)
        if let Some(mass) = &self.mass {
            if mass.center_of_mass == particle.center_of_mass {
                return force;
            }

            let r = mass.center_of_mass - particle.center_of_mass;

            if self.mass.is_some() || self.width / r.norm() < theta {
                // leaf nodes or node is far enough away
                force += force_fn(r, particle.mass, mass.mass);
            } else {
                // near field forces, go deeper into tree
                for node in self
                    .subnodes
                    .as_ref()
                    .expect("node has neither mass nor subnodes")
                {
                    force += node.calculate_force(particle, force_fn, theta);
                }
            }
        }

        force
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
        let step_size = self.width / 4.;
        if i == 0 {
            return self.center + Vector3::new(step_size, step_size, step_size);
        }
        if i == 1 {
            return self.center + Vector3::new(-step_size, step_size, step_size);
        }
        if i == 2 {
            return self.center + Vector3::new(-step_size, -step_size, step_size);
        }
        if i == 3 {
            return self.center + Vector3::new(step_size, -step_size, step_size);
        }
        if i == 4 {
            return self.center + Vector3::new(step_size, step_size, -step_size);
        }
        if i == 5 {
            return self.center + Vector3::new(-step_size, step_size, -step_size);
        }
        if i == 6 {
            return self.center + Vector3::new(-step_size, -step_size, -step_size);
        }
        self.center + Vector3::new(step_size, -step_size, -step_size)
    }
}

#[cfg(test)]
mod tests {}
