use std::marker::PhantomData;

use ndarray::{Array, Dimension, Zip};

use crate::Float;

pub trait Optimizer<D> {
    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>);
}

#[derive(Debug, Clone)]
pub struct SGD<D> {
    learning_rate: Float,
    dim: PhantomData<D>,
}

impl<D> SGD<D> {
    pub fn new(learning_rate: Float, batch_size: usize) -> SGD<D> {
        SGD {
            learning_rate: learning_rate / batch_size as Float,
            dim: PhantomData,
        }
    }
}

impl<D> Optimizer<D> for SGD<D>
where
    D: Dimension,
{
    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>) {
        Zip::from(param).and(grad).apply(|p, &g| {
            *p -= g * self.learning_rate;
        })
    }
}
