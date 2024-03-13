pub mod activations;
pub mod nn;

pub use typetag;

pub trait NeuronInfo {
    fn _type(&self) -> &'static str;
    fn id(&self) -> usize;
}
