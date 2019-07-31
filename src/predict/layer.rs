use std::io::Read;

use ndarray::{Array1, Array2, Zip};

use crate::Float;

/// 予測用のレイヤインターフェース
pub trait Layer: Sized {
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>);

    fn input_size(&self) -> Option<usize> {
        None
    }

    fn output_size(&self) -> Option<usize> {
        None
    }

    fn decode<R: Read>(reader: &mut R) -> Self;
}

/// レイヤを合成したレイヤ
#[derive(Debug, Clone, PartialEq)]
pub struct Layers<L1, L2> {
    layer1: L1,
    layer2: L2,
    temporary: Array1<Float>,
}

impl<L1, L2> Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    pub fn new(layer1: L1, layer2: L2) -> Layers<L1, L2> {
        let size = match (layer1.output_size(), layer2.input_size()) {
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

    fn input_size(&self) -> Option<usize> {
        self.layer1.input_size()
    }

    fn output_size(&self) -> Option<usize> {
        if let Some(size) = self.layer2.output_size() {
            Some(size)
        } else {
            self.layer1.output_size()
        }
    }

    fn decode<R: Read>(reader: &mut R) -> Self {
        let layer1 = L1::decode(reader);
        let layer2 = L2::decode(reader);
        Layers::new(layer1, layer2)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    w: Array2<Float>,
}

impl Layer for Affine {
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>) {
        Zip::from(output).and(self.w.genrows()).apply(|y, w| {
            *y = w.dot(input);
        });
    }

    fn input_size(&self) -> Option<usize> {
        Some(self.w.shape()[1])
    }

    fn output_size(&self) -> Option<usize> {
        Some(self.w.shape()[0])
    }

    fn decode<R: Read>(reader: &mut R) -> Self {
        let w = rmp_serde::decode::from_read(reader).unwrap();
        Affine { w }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bias {
    b: Array1<Float>,
}

impl Layer for Bias {
    fn forward(&mut self, input: &Array1<Float>, output: &mut Array1<Float>) {
        Zip::from(output)
            .and(input)
            .and(&self.b)
            .apply(|y, &x, &b| {
                *y = x + b;
            });
    }

    fn input_size(&self) -> Option<usize> {
        Some(self.b.len())
    }

    fn output_size(&self) -> Option<usize> {
        Some(self.b.len())
    }

    fn decode<R: Read>(reader: &mut R) -> Self {
        let b = rmp_serde::decode::from_read(reader).unwrap();
        Bias { b }
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

    fn input_size(&self) -> Option<usize> {
        Some(self.w.shape()[1])
    }

    fn output_size(&self) -> Option<usize> {
        Some(self.w.shape()[0])
    }

    fn decode<R: Read>(reader: &mut R) -> Self {
        let a = Affine::decode(reader);
        let b = Bias::decode(reader);
        a.synthesize(b)
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

    fn decode<R: Read>(_: &mut R) -> Self {
        ReLU {}
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

/// レイヤ合成のインターフェース
pub trait Synthesize<Other> {
    type Output;
    fn synthesize(self, other: Other) -> Self::Output;
}

impl Synthesize<Affine> for Affine {
    type Output = Affine;
    fn synthesize(self, other: Affine) -> Self::Output {
        Affine {
            w: self.w.dot(&other.w),
        }
    }
}

impl Synthesize<Bias> for Affine {
    type Output = Dense;
    fn synthesize(self, other: Bias) -> Self::Output {
        Dense {
            w: self.w,
            b: other.b,
        }
    }
}
impl Synthesize<Dense> for Affine {
    type Output = Dense;
    fn synthesize(self, other: Dense) -> Self::Output {
        Dense {
            w: self.w.dot(&other.w),
            b: other.b,
        }
    }
}

impl<L1, L2, L3> Synthesize<L3> for Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
    L3: Layer,
{
    type Output = Layers<Self, L3>;
    fn synthesize(self, other: L3) -> Self::Output {
        Layers::new(self, other)
    }
}

macro_rules! syn_layers {
    ($a:tt, $b:tt) => {
        impl Synthesize<$b> for $a {
            type Output = Layers<Self, $b>;
            fn synthesize(self, other: $b) -> Self::Output {
                Layers::new(self, other)
            }
        }
    };
}

syn_layers!(Affine, ReLU);
syn_layers!(Dense, ReLU);

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
        let mut l = dense.synthesize(ReLU::new());
        let input = arr1(&[0.5, 1.5]);
        let mut output = arr1(&[0.0, 0.0, 0.0]);
        l.forward(&input, &mut output);
        assert_eq!(output, arr1(&[0.5 + 3.0 + 2.0, 0.0, 3.5]));
        assert_eq!(l.input_size(), Some(2));
        assert_eq!(l.output_size(), Some(3));
    }
}
