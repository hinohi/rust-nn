use std::marker::PhantomData;

use ndarray::{Array, Dimension, ShapeBuilder, Zip};

use crate::Float;

pub trait Optimizer<D>
where
    D: Dimension,
{
    fn init<Sh>(&mut self, shape: Sh, batch_size: usize)
    where
        Sh: ShapeBuilder<Dim = D>;

    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>);
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
    fn init<Sh>(&mut self, _: Sh, batch_size: usize)
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        self.learning_rate /= batch_size as Float;
    }

    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>) {
        Zip::from(param).and(grad).apply(|p, &g| {
            *p -= g * self.learning_rate;
        })
    }
}

#[derive(Debug, Clone)]
pub struct Adam<D>
where
    D: Dimension,
{
    alpha: Float,
    beta1: Float,
    beta2: Float,
    eps: Float,
    batch_factor: Float,
    time_step: Float,
    m: Array<Float, D>,
    v: Array<Float, D>,
}

impl<D> Adam<D>
where
    D: Dimension,
{
    pub fn new(alpha: Float, beta1: Float, beta2: Float, eps: Float) -> Adam<D> {
        Adam {
            alpha,
            beta1,
            beta2,
            eps,
            batch_factor: 1.0,
            time_step: 0.0,
            m: Default::default(),
            v: Default::default(),
        }
    }

    fn learning_rate(&self) -> Float {
        let fix1 = 1.0 - self.beta1.powf(self.time_step);
        let fix2 = 1.0 - self.beta2.powf(self.time_step);
        self.alpha as f64 * fix1 / fix2
    }
}

impl<D> Default for Adam<D>
where
    D: Dimension,
{
    fn default() -> Adam<D> {
        Adam::new(0.01, 0.9, 0.999, 1e-8)
    }
}

impl<D> Optimizer<D> for Adam<D>
where
    D: Dimension,
{
    fn init<Sh>(&mut self, shape: Sh, batch_size: usize)
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape();
        self.m = Array::zeros(shape.clone());
        self.v = Array::zeros(shape);
        self.batch_factor = 1.0 / batch_size as Float;
    }

    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>) {
        // incr time
        self.time_step += 1.0;
        // m_{t+1} = beta1 * m_{t} + (1 - beta1) * grad
        let beta = self.beta1;
        let beta_g = (1.0 - self.beta1) * self.batch_factor;
        Zip::from(&mut self.m)
            .and(grad)
            .apply(|m, &g| *m = beta * *m + beta_g * g);
        // v_{t+1} = beta1 * v_{t} + (1 - beta2) * grad * grad
        let beta = self.beta1;
        let beta_g = (1.0 - self.beta1) * self.batch_factor * self.batch_factor;
        Zip::from(&mut self.v)
            .and(grad)
            .apply(|v, &g| *v = beta * *v + beta_g * g * g);
        // param -= a * m / sqrt(v)
        let alpha = self.learning_rate();
        let eps = self.eps;
        Zip::from(param)
            .and(&self.m)
            .and(&self.v)
            .apply(|p, &m, &v| {
                *p -= alpha * m / (v.sqrt() + eps);
            });
    }
}
