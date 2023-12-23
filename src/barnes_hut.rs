use crate::octree::{Mass, Octree};
use nalgebra::Vector3;

const G: f64 = 6.6743015e-11;

fn gravitational_force(r: Vector3<f64>, m1: f64, m2: f64, epsilon: f64) -> Vector3<f64> {
    let r_square = r.norm_squared();
    G * m1 * m2 / (r_square + epsilon).powf(3. / 2.) * r
}

#[derive(Clone, Debug)]
pub struct Particle {
    pub mass: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
}

pub fn barnes_hut(
    particles: &[Particle],
    time_step: f64,
    num_steps: usize,
    theta: f64,
    epsilon: f64,
) -> Vec<Vec<Vector3<f64>>> {
    let n = particles.len();

    let mut positions = vec![vec![Vector3::zeros(); n]; num_steps + 1];
    positions[0] = particles.iter().map(|p| p.position).collect();

    let mut velocity = vec![Vector3::zeros(); n];
    let mut acceleration = vec![Vector3::zeros(); n];

    for t in 0..num_steps {
        let masses: Vec<Mass> = particles
            .iter()
            .map(|p| p.mass)
            .zip(positions[t].iter())
            .map(|(m, p)| Mass::new(m, *p))
            .collect();
        let octree = Octree::new(&masses, theta);

        for (a, m) in acceleration.iter_mut().zip(masses.iter()) {
            let force = octree.calculate_force(m, &|r: Vector3<f64>, m1: f64, m2: f64| {
                gravitational_force(r, m1, m2, epsilon)
            });
            *a = force / m.mass;
        }
        for i in 0..n {
            velocity[i] += acceleration[i] * time_step;
            positions[t + 1][i] = positions[t][i] + velocity[i] * time_step;
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    use super::{barnes_hut, Particle};

    #[test]
    fn test_barnes_hut() {
        let particles = [
            Particle {
                mass: 1e10,
                position: Vector3::new(1., 0., 0.),
                velocity: Vector3::zeros(),
            },
            Particle {
                mass: 1e10,
                position: Vector3::new(-1., 0., 0.),
                velocity: Vector3::zeros(),
            },
        ];

        let positions = barnes_hut(&particles, 0.1, 1000, 1.5, 0.1);
        let last = positions.last().unwrap();

        assert_abs_diff_eq!(last[0][0], -last[1][0], epsilon = 1e-8);
        for p in last {
            assert_abs_diff_eq!(p[1], 0., epsilon = 1e-8);
            assert_abs_diff_eq!(p[2], 0., epsilon = 1e-8);
        }
    }
}
