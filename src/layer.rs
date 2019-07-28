use ndarray::{Array1, Array2, Zip};
use rand::Rng;
use rand_distr::{Distribution, Normal};

type Float = f64;

pub trait Layer {
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>);

    fn input_len(&self) -> Option<usize> {
        None
    }

    fn output_len(&self) -> Option<usize> {
        None
    }

    fn append<L: Layer>(self, other: L) -> Layers<Self, L>
    where
        Self: Sized;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    layer1: L1,
    layer2: L2,
    temporary: Array1<Float>,
}

impl<L1, L2> Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    fn new(layer1: L1, layer2: L2) -> Layers<L1, L2> {
        let size = match (layer1.output_len(), layer2.input_len()) {
            (Some(size), None) | (None, Some(size)) => size,
            (Some(s1), Some(s2)) if s1 == s2 => s1,
            _ => panic!(),
        };
        Layers {
            layer1,
            layer2,
            temporary: Array1::zeros(size),
        }
    }
}

impl<L1, L2> Layer for Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>) {
        self.layer1.forward(input, &mut self.temporary);
        self.layer2.forward(&self.temporary, output);
    }

    fn input_len(&self) -> Option<usize> {
        self.layer1.input_len()
    }

    fn output_len(&self) -> Option<usize> {
        if let Some(size) = self.layer2.output_len() {
            Some(size)
        } else {
            self.layer1.output_len()
        }
    }

    fn append<L: Layer>(self, other: L) -> Layers<Self, L> {
        Layers::new(self, other)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    w: Array2<Float>,
    b: Array1<Float>,
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>) {
        Zip::from(output)
            .and(self.w.genrows())
            .and(&self.b)
            .apply(|y, w, &b| {
                *y = w.dot(input) + b;
            });
    }

    fn input_len(&self) -> Option<usize> {
        Some(self.w.shape()[1])
    }

    fn output_len(&self) -> Option<usize> {
        Some(self.w.shape()[0])
    }

    fn append<L: Layer>(self, other: L) -> Layers<Self, L> {
        Layers::new(self, other)
    }
}

impl Dense {
    pub fn from_normal<R: Rng>(random: &mut R, shape: (usize, usize), sig: Float) -> Dense {
        let normal = Normal::new(0.0, sig).unwrap();
        Dense {
            w: Array2::from_shape_fn(shape, |_| normal.sample(random)),
            b: Array1::zeros(shape.1),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReLU;

impl Layer for ReLU {
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>) {
        Zip::from(output).and(input).apply(|y, &x| {
            if 0.0 < x {
                *y = x;
            } else {
                *y = 0.0;
            }
        });
    }

    fn append<L: Layer>(self, other: L) -> Layers<Self, L> {
        Layers::new(self, other)
    }
}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn smoke_dense() {
        let mut dense = Dense {
            w: arr2(&[[1.0, 2.0], [3.0, -4.0]]),
            b: arr1(&[2.0, -3.0]),
        };
        let input = arr1(&[0.5, -1.5]);
        let mut output = arr1(&[0.0, 0.0]);
        dense.forward(&input, &mut output);
        assert_eq!(output, arr1(&[0.5 - 3.0 + 2.0, 1.5 + 6.0 - 3.0]));
    }

    #[test]
    fn smoke_relu() {
        let mut r = ReLU::new();
        let input = arr1(&[0.5, -1.5, 0.0, 2.0]);
        let mut output = arr1(&[1.0; 4]);
        r.forward(&input, &mut output);
        assert_eq!(output, arr1(&[0.5, 0.0, 0.0, 2.0]));
    }

    #[test]
    fn dense_relu() {
        let dense = Dense {
            w: arr2(&[[1.0, 2.0], [3.0, -4.0], [5.0, 0.0]]),
            b: arr1(&[2.0, -3.0, 1.0]),
        };
        let mut l = dense.append(ReLU::new());
        let input = arr1(&[0.5, 1.5]);
        let mut output = arr1(&[0.0, 0.0, 0.0]);
        l.forward(&input, &mut output);
        assert_eq!(output, arr1(&[0.5 + 3.0 + 2.0, 0.0, 3.5]));
        assert_eq!(l.input_len(), Some(2));
        assert_eq!(l.output_len(), Some(3));
    }
}
