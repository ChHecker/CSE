use std::ops::{Deref, DerefMut};

use crate::octree::{Mass, Octree};
use nalgebra::Vector3;
use rayon::prelude::*;

const G: f64 = 6.6743015e-11;

/// Calculate the gravitational acceleration of m2 on m1.
pub fn gravitational_acceleration(
    r: Vector3<f64>,
    _m1: f64,
    m2: f64,
    epsilon: f64,
) -> Vector3<f64> {
    let r_square = r.norm_squared();
    G * m2 / (r_square + epsilon).sqrt().powi(3) * r
}

#[derive(Clone, Debug)]
pub struct Particle {
    mass_obj: Mass,
    velocity: Vector3<f64>,
}

impl Particle {
    pub fn new(mass: f64, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Self {
            mass_obj: Mass { mass, position },
            velocity,
        }
    }
}

impl Deref for Particle {
    type Target = Mass;

    fn deref(&self) -> &Self::Target {
        &self.mass_obj
    }
}

impl DerefMut for Particle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mass_obj
    }
}

pub fn barnes_hut(
    particles: &mut [Particle],
    force: impl Fn(Vector3<f64>, f64, f64) -> Vector3<f64> + Send + Sync,
    time_step: f64,
    num_steps: usize,
    theta: f64,
) -> Vec<Vec<Vector3<f64>>> {
    let n = particles.len();

    let mut positions = vec![vec![Vector3::zeros(); n]; num_steps + 1];
    positions[0] = particles.iter().map(|p| p.position).collect();

    let mut acceleration = vec![Vector3::zeros(); n];

    for t in 0..num_steps {
        let masses: Vec<Mass> = particles.iter().map(|p| p.deref()).cloned().collect();
        let octree = Octree::new(masses, theta);

        // Calculate accelerations
        acceleration.par_iter_mut().enumerate().for_each(|(i, a)| {
            *a = particles
                .iter()
                .enumerate()
                .filter(|(j, _)| i != *j)
                .map(|(_, p)| octree.calculate_acceleration(p.deref(), &force))
                .sum();
        });

        /*
         * Leapfrog integration:
         * v_(i + 1/2) = v_(i - 1/2) + a_i dt
         * x_(i + 1) = x_i + v_(i + 1/2) dt
         */
        for i in 0..n {
            // in first time step, need to get from v_0 to v_(1/2)
            if t == 0 {
                particles[i].velocity += acceleration[i] * time_step / 2.;
            } else {
                particles[i].velocity += acceleration[i] * time_step;
            }

            let v = particles[i].velocity;
            particles[i].position += v * time_step;
            positions[t + 1][i] = particles[i].position;

            // in last step, need to get from v_(n_steps - 1/2) to v_(n_steps)
            if t == n - 1 {
                particles[i].velocity += acceleration[i] * time_step / 2.;
            }
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use rand::Rng;

    use crate::barnes_hut::gravitational_acceleration;

    use super::{barnes_hut, Particle};

    #[test]
    fn test_barnes_hut() {
        let mut particles = vec![
            Particle::new(1e10, Vector3::new(1., 0., 0.), Vector3::zeros()),
            Particle::new(1e10, Vector3::new(-1., 0., 0.), Vector3::zeros()),
        ];

        let positions = barnes_hut(
            &mut particles,
            |r, m1, m2| gravitational_acceleration(r, m1, m2, 1e-5),
            0.1,
            1000,
            1.5,
        );

        let last = positions.last().unwrap();

        assert_abs_diff_eq!(last[0][0], -last[1][0], epsilon = 1e-8);
        for p in last {
            assert_abs_diff_eq!(p[1], 0., epsilon = 1e-8);
            assert_abs_diff_eq!(p[2], 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn test_stack_overflow() {
        let mut rng = rand::thread_rng();

        let mut particles: Vec<Particle> = (0..1000)
            .map(|_| {
                Particle::new(
                    rng.gen_range(0.0..1000.0),
                    Vector3::new_random(),
                    Vector3::new_random(),
                )
            })
            .collect();

        barnes_hut(
            &mut particles,
            |r, m1, m2| gravitational_acceleration(r, m1, m2, 1e-4),
            1.,
            100,
            1.5,
        );
    }
}
