use engine::{
    nn::{InputNeuron, NeuronSubTraits},
    typetag, NeuronInfo,
};
use macros::{DNeuronInfo, SubTraits};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct Hunger {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Hunger {
    fn as_standard(&self) -> f32 {
        self.value
    }

    fn set_value(&mut self, value: f32) {
        self.value = value;
    }
}

impl Hunger {
    pub fn new(value: f32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct Age {
    value: usize,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Age {
    fn as_standard(&self) -> f32 {
        self.value as f32
    }

    fn set_value(&mut self, value: f32) {
        self.value = value as usize;
    }
}

impl Age {
    pub fn new(value: usize, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct Health {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Health {
    fn as_standard(&self) -> f32 {
        self.value as f32
    }

    fn set_value(&mut self, value: f32) {
        self.value = value;
    }
}

impl Health {
    pub fn new(value: f32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct Speed {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Speed {
    fn as_standard(&self) -> f32 {
        self.value
    }

    fn set_value(&mut self, value: f32) {
        self.value = value;
    }
}

impl Speed {
    pub fn new(value: f32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct BlankInput {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for BlankInput {
    fn as_standard(&self) -> f32 {
        self.value
    }

    fn set_value(&mut self, value: f32) {
        self.value = value;
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct TestNeuronA {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl engine::nn::Neuron for TestNeuronA {
    fn step(&self, edge: &engine::nn::Edge, input: f32) -> f32 {
        edge.weight * input
    }

    fn finish(&self, partial: f32) -> f32 {
        partial + self.value
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
pub struct TestNeuronB {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl engine::nn::Neuron for TestNeuronB {
    fn step(&self, edge: &engine::nn::Edge, input: f32) -> f32 {
        edge.weight * input
    }

    fn finish(&self, partial: f32) -> f32 {
        partial + self.value
    }
}
