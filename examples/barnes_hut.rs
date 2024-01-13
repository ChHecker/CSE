use cse::barnes_hut::{barnes_hut, gravitational_acceleration, Particle};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut particles = (0..100)
        .map(|_| {
            Particle::new(
                rng.gen_range(0.0..100.0),
                1000. * Vector3::new_random() - Vector3::new(500., 500., 500.),
                10. * Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    barnes_hut(
        &mut particles,
        |r, m1, m2| gravitational_acceleration(r, m1, m2, 1e-5),
        0.1,
        100,
        1.5,
    );
}
