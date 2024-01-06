use engine::nn::InputNeuron;
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct Hunger(f32);

impl InputNeuron for Hunger {
    fn as_standard(&self) -> f32 {
        self.0
    }
}

impl Hunger {
    pub fn new(value: f32) -> Box<dyn InputNeuron> {
        Box::new(Self(value))
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct Age(u32);

impl InputNeuron for Age {
    fn as_standard(&self) -> f32 {
        self.0 as f32
    }
}

impl Age {
    pub fn new(value: u32) -> Box<dyn InputNeuron> {
        Box::new(Self(value))
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct Health(u8);

impl InputNeuron for Health {
    fn as_standard(&self) -> f32 {
        self.0 as f32
    }
}

impl Health {
    pub fn new(value: u8) -> Box<dyn InputNeuron> {
        Box::new(Self(value))
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct Speed(f32);

impl InputNeuron for Speed {
    fn as_standard(&self) -> f32 {
        self.0
    }
}

impl Speed {
    pub fn new(value: f32) -> Box<dyn InputNeuron> {
        Box::new(Self(value))
    }
}