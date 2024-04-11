pub mod containers;
// #[cfg(test)]
// pub(crate) mod csv;
// pub mod fft;
// pub mod graphs;
// pub mod interpolation;
// pub mod linalg;
pub mod matrix;
// pub mod newton;
// pub mod ode;
// pub mod parser;
// pub mod quadrature;
pub mod sorting;

pub enum IterativeResult<V> {
    Converged(V),
    MaxIterations(V),
    Failed,
}

impl<V> IterativeResult<V> {
    pub fn unwrap(self) -> V {
        match self {
            IterativeResult::Converged(v) => v,
            IterativeResult::MaxIterations(v) => v,
            IterativeResult::Failed => panic!("called unwrap on Failed"),
        }
    }

    pub fn successful_or<E>(self, err: E) -> Result<V, E> {
        match self {
            IterativeResult::Converged(v) => Ok(v),
            IterativeResult::MaxIterations(v) => Ok(v),
            IterativeResult::Failed => Err(err),
        }
    }

    pub fn converged_or<E>(self, err: E) -> Result<V, E> {
        match self {
            IterativeResult::Converged(v) => Ok(v),
            IterativeResult::MaxIterations(_) => Err(err),
            IterativeResult::Failed => Err(err),
        }
    }

    pub fn is_converged(&self) -> bool {
        match &self {
            IterativeResult::Converged(_) => true,
            IterativeResult::MaxIterations(_) | IterativeResult::Failed => false,
        }
    }

    pub fn is_successful(&self) -> bool {
        match &self {
            IterativeResult::Converged(_) | IterativeResult::MaxIterations(_) => true,
            IterativeResult::Failed => false,
        }
    }
}
