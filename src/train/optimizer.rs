use std::marker::PhantomData;

use ndarray::{Array, Dimension, ShapeBuilder, Zip};
use ndarray_parallel::prelude::*;

use crate::Float;

pub trait Optimizer<D>: Sync + Send
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

    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
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
        Zip::from(param).and(grad).par_apply(|p, &g| {
            *p -= g * self.learning_rate;
        })
    }
}

#[derive(Debug, Clone)]
pub struct AdaDelta<D>
where
    D: Dimension,
{
    rho: Float,
    eps: Float,
    batch_factor: Float,
    h: Array<Float, D>,
    s: Array<Float, D>,
}

impl<D> AdaDelta<D>
where
    D: Dimension,
{
    pub fn new(rho: Float, eps: Float) -> AdaDelta<D> {
        AdaDelta {
            rho,
            eps,
            batch_factor: 1.0,
            h: Default::default(),
            s: Default::default(),
        }
    }

    pub fn rho(mut self, rho: Float) -> Self {
        self.rho = rho;
        self
    }

    pub fn eps(mut self, eps: Float) -> Self {
        self.eps = eps;
        self
    }
}

impl<D> Default for AdaDelta<D>
where
    D: Dimension,
{
    fn default() -> AdaDelta<D> {
        AdaDelta::new(0.95, 1e-6)
    }
}

impl<D> Optimizer<D> for AdaDelta<D>
where
    D: Dimension,
{
    fn init<Sh>(&mut self, shape: Sh, batch_size: usize)
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape();
        self.h = Array::zeros(shape.clone());
        self.s = Array::zeros(shape);
        self.batch_factor = 1.0 / batch_size as Float;
    }

    fn optimize(&mut self, param: &mut Array<Float, D>, grad: &Array<Float, D>) {
        // h = ρ * h + (1 - ρ) * grad * grad
        let rho = self.rho;
        let rho_g = (1.0 - self.rho) * self.batch_factor * self.batch_factor;
        Zip::from(&mut self.h).and(grad).par_apply(|h, &g| {
            *h = rho * *h + rho_g * g * g;
        });
        // v = sqrt(s + e) / sqrt(h + e) * grad
        // s = ρ * s + (1 - ρ) * v * v
        // p -= v
        let eps = self.eps;
        let rho1 = 1.0 - self.rho;
        let batch_factor = self.batch_factor;
        Zip::from(param)
            .and(&mut self.s)
            .and(&self.h)
            .and(grad)
            .par_apply(|p, s, &h, &g| {
                let v = ((*s + eps) / (h + eps)).sqrt() * g * batch_factor;
                *s = rho * *s + rho1 * v * v;
                *p -= v;
            });
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
    time_step_delta: Float,
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
            time_step_delta: 1e-3,
            m: Default::default(),
            v: Default::default(),
        }
    }

    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta1(mut self, beta1: Float) -> Self {
        self.beta1 = beta1;
        self
    }

    pub fn beta2(mut self, beta2: Float) -> Self {
        self.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: Float) -> Self {
        self.eps = eps;
        self
    }

    pub fn time_step_delta(mut self, time_step_delta: Float) -> Self {
        self.time_step_delta = time_step_delta;
        self
    }

    fn learning_rate(&self) -> Float {
        let fix1 = 1.0 - self.beta1.powf(self.time_step);
        let fix2 = 1.0 - self.beta2.powf(self.time_step);
        self.alpha * fix2.sqrt() / fix1
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
        self.time_step += self.time_step_delta;
        // m_{t+1} = beta1 * m_{t} + (1 - beta1) * grad
        let beta = self.beta1;
        let beta_g = (1.0 - self.beta1) * self.batch_factor;
        Zip::from(&mut self.m)
            .and(grad)
            .par_apply(|m, &g| *m = beta * *m + beta_g * g);
        // v_{t+1} = beta2 * v_{t} + (1 - beta2) * grad * grad
        let beta = self.beta2;
        let beta_g = (1.0 - self.beta2) * self.batch_factor * self.batch_factor;
        Zip::from(&mut self.v)
            .and(grad)
            .par_apply(|v, &g| *v = beta * *v + beta_g * g * g);
        // param -= a * m / sqrt(v)
        let alpha = self.learning_rate();
        let eps = self.eps;
        Zip::from(param)
            .and(&self.m)
            .and(&self.v)
            .par_apply(|p, &m, &v| {
                *p -= alpha * m / (v.sqrt() + eps);
            });
    }
}
