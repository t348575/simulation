use macros::{Name, SubTraits};
use std::{f32::consts::E, fmt::Debug, marker::PhantomData};

use crate::{
    nn::{Edge, NeuronSubTraits, OutputNeuron},
    NeuronName,
};

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct Sigmoid<Identifier: Debug + Clone + Serialize + 'static + Send + Sync> {
    value: f32,
    marker: PhantomData<Identifier>,
}

impl<Identifier: Debug + Clone + Serialize + 'static + Send + Sync> OutputNeuron
    for Sigmoid<Identifier>
{
    fn step(&self, edge: &Edge, input: f32) -> f32 {
        edge.weight * input
    }

    fn finish_and_save(&mut self, partial: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + E.powf(-partial));
        self.value = sigmoid;
        sigmoid
    }
}

impl<Identifier: Debug + Clone + Serialize + 'static + Send + Sync> Sigmoid<Identifier> {
    pub fn new(value: f32) -> Box<dyn OutputNeuron> {
        Box::new(Sigmoid {
            value,
            marker: PhantomData::<Identifier>,
        })
    }
}
