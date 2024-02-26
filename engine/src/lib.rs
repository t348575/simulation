pub mod activations;
pub mod nn;
pub mod sim;

pub use typetag;

pub trait NeuronInfo {
    fn _type(&self) -> &'static str;
    fn id(&self) -> usize;
}
