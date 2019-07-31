use std::marker::PhantomData;

use ndarray::{Array, Dimension, Zip};

use crate::Float;

pub trait Optimizer<D> {
    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>, batch_size: usize);
}

#[derive(Debug, Clone)]
pub struct SGD<D> {
    learning_rate: Float,
    dim: PhantomData<D>,
}

impl<D> SGD<D> {
    pub fn new(learning_rate: Float) -> SGD<D> {
        SGD {
            learning_rate,
            dim: PhantomData,
        }
    }
}

impl<D> Default for SGD<D> {
    fn default() -> SGD<D> {
        SGD::new(1e-3)
    }
}

impl<D> Optimizer<D> for SGD<D>
where
    D: Dimension,
{
    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>, batch_size: usize) {
        let lr = self.learning_rate / batch_size as Float;
        Zip::from(param).and(grad).apply(|p, &g| {
            *p -= g * lr;
        })
    }
}
