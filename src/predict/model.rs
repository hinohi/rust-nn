use std::io::Read;

use ndarray::Array1;

use super::layer::*;
use crate::Float;

type DenseNN1 = Layers<Dense, ReLU>;
type DenseNN2 = Layers<Layers<DenseNN1, Dense>, ReLU>;
type DenseNN3 = Layers<Layers<DenseNN2, Dense>, ReLU>;
type DenseNN4 = Layers<Layers<DenseNN3, Dense>, ReLU>;
type DenseNN5 = Layers<Layers<DenseNN4, Dense>, ReLU>;
type DenseNN6 = Layers<Layers<DenseNN5, Dense>, ReLU>;

#[derive(Debug, Clone, PartialEq)]
pub struct NN1Regression {
    nn: Layers<DenseNN1, Dense>,
    output: Array1<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN2Regression {
    nn: Layers<DenseNN2, Dense>,
    output: Array1<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN3Regression {
    nn: Layers<DenseNN3, Dense>,
    output: Array1<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN4Regression {
    nn: Layers<DenseNN4, Dense>,
    output: Array1<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN5Regression {
    nn: Layers<DenseNN5, Dense>,
    output: Array1<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN6Regression {
    nn: Layers<DenseNN6, Dense>,
    output: Array1<Float>,
}

fn build_nn1<R: Read>(reader: &mut R) -> DenseNN1 {
    Dense::decode(reader).synthesize(ReLU::decode(reader))
}

macro_rules! clone_build_nn {
    ($func:ident, $ret:ident, $call:ident) => {
        fn $func<R: Read>(reader: &mut R) -> $ret {
            $call(reader)
                .synthesize(Dense::decode(reader))
                .synthesize(ReLU::decode(reader))
        }
    };
}

clone_build_nn!(build_nn2, DenseNN2, build_nn1);
clone_build_nn!(build_nn3, DenseNN3, build_nn2);
clone_build_nn!(build_nn4, DenseNN4, build_nn3);
clone_build_nn!(build_nn5, DenseNN5, build_nn4);
clone_build_nn!(build_nn6, DenseNN6, build_nn5);

impl NN1Regression {
    pub fn new<R: Read>(reader: &mut R) -> NN1Regression {
        let nn = build_nn1(reader).synthesize(Dense::decode(reader));
        NN1Regression {
            nn,
            output: Array1::zeros(1),
        }
    }

    pub fn predict(&mut self, x: &Array1<Float>) -> Float {
        self.nn.forward(x, &mut self.output);
        self.output[0]
    }
}

macro_rules! impl_nn {
    ($name:ident, $builder:ident) => {
        impl $name {
            pub fn new<R: Read>(reader: &mut R) -> $name {
                let nn = $builder(reader).synthesize(Dense::decode(reader));
                $name {
                    nn,
                    output: Array1::zeros(1),
                }
            }

            pub fn predict(&mut self, x: &Array1<Float>) -> Float {
                self.nn.forward(x, &mut self.output);
                self.output[0]
            }
        }
    };
}

impl_nn!(NN2Regression, build_nn2);
impl_nn!(NN3Regression, build_nn3);
impl_nn!(NN4Regression, build_nn4);
impl_nn!(NN5Regression, build_nn5);
impl_nn!(NN6Regression, build_nn6);
