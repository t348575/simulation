use macros::{DNeuronInfo, SubTraits};
use serde::{Deserialize, Serialize};
use std::{f32::consts::E, fmt::Debug};

use crate::{
    nn::{Edge, NeuronSubTraits, OutputNeuron},
    NeuronInfo,
};

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct Sigmoid {
    value: f32,
    id: usize,
    _type: String,
}

#[typetag::serde]
impl OutputNeuron for Sigmoid {
    fn step(&self, edge: &Edge, input: f32) -> f32 {
        edge.weight * input
    }

    fn finish_and_save(&mut self, partial: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + E.powf(-partial));
        self.value = sigmoid;
        sigmoid
    }

    fn value(&self) -> f32 {
        self.value
    }
}

impl Sigmoid {
    pub fn new(value: f32, id: usize, _type: String) -> Box<dyn OutputNeuron> {
        Box::new(Sigmoid { value, id, _type })
    }
}
