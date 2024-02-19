pub mod activations;
pub mod nn;
pub mod sim;

pub trait NeuronName {
    fn name(&self) -> &'static str;
}
