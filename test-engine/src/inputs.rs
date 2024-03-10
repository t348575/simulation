use engine::{
    nn::{InputNeuron, NeuronSubTraits},
    typetag, NeuronInfo,
};
use macros::{Name, SubTraits};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct Hunger {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Hunger {
    fn as_standard(&self) -> f32 {
        self.value
    }
}

impl Hunger {
    pub fn new(value: f32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct Age {
    value: u32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Age {
    fn as_standard(&self) -> f32 {
        self.value as f32
    }
}

impl Age {
    pub fn new(value: u32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct Health {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Health {
    fn as_standard(&self) -> f32 {
        self.value
    }
}

impl Health {
    pub fn new(value: f32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct Speed {
    value: f32,
    id: usize,
}

#[typetag::serde]
impl InputNeuron for Speed {
    fn as_standard(&self) -> f32 {
        self.value
    }
}

impl Speed {
    pub fn new(value: f32, id: usize) -> Box<dyn InputNeuron> {
        Box::new(Self { value, id })
    }
}
