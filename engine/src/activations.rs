use std::{f32::consts::E, fmt::Debug, marker::PhantomData};

use serde::Serialize;

use crate::nn::{Edge, OutputNeuron};

#[derive(Debug, Serialize, Clone)]
pub struct Sigmoid<Identifier: Debug + Clone + Serialize + 'static> {
    value: f32,
    marker: PhantomData<Identifier>,
}

impl<Identifier: Debug + Clone + Serialize + 'static> OutputNeuron for Sigmoid<Identifier> {
    fn step(&self, edge: &Edge, input: f32) -> f32 {
        edge.0 * input
    }

    fn finish_and_save(&mut self, partial: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + E.powf(-partial));
        self.value = sigmoid;
        sigmoid
    }
}

impl<Identifier: Debug + Clone + Serialize + 'static> Sigmoid<Identifier> {
    pub fn new(value: f32) -> Box<dyn OutputNeuron> {
        Box::new(Sigmoid {
            value,
            marker: PhantomData::<Identifier>,
        })
    }
}
