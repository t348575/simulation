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

#[cfg(test)]
mod test {
    use super::*;
    use crate::nn::Edge;

    fn make_sigmoid(value: f32) -> Sigmoid {
        Sigmoid {
            value,
            id: 0,
            _type: "Sigmoid".to_owned(),
        }
    }

    #[test]
    fn step_multiplies_weight_by_input() {
        let s = make_sigmoid(0.0);
        let edge = Edge {
            weight: 3.0,
            enabled: true,
        };
        let result = s.step(&edge, 2.0);
        assert!((result - 6.0).abs() < 1e-6, "step should return weight * input = 6.0, got {result}");
    }

    #[test]
    fn step_with_zero_input_returns_zero() {
        let s = make_sigmoid(0.0);
        let edge = Edge {
            weight: 5.0,
            enabled: true,
        };
        assert_eq!(s.step(&edge, 0.0), 0.0);
    }

    #[test]
    fn finish_and_save_at_zero_gives_half() {
        // sigmoid(0) = 0.5
        let mut s = make_sigmoid(0.0);
        let result = s.finish_and_save(0.0);
        assert!((result - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {result}");
    }

    #[test]
    fn finish_and_save_positive_input_correct_math() {
        // sigmoid(1.0) = 1 / (1 + e^-1) ≈ 0.7310586
        let mut s = make_sigmoid(0.0);
        let result = s.finish_and_save(1.0);
        let expected = 1.0 / (1.0 + E.powf(-1.0));
        assert!(
            (result - expected).abs() < 1e-6,
            "sigmoid(1.0) should be ~{expected}, got {result}"
        );
    }

    #[test]
    fn finish_and_save_negative_input_less_than_half() {
        // sigmoid(-2.0) < 0.5
        let mut s = make_sigmoid(0.0);
        let result = s.finish_and_save(-2.0);
        assert!(result < 0.5, "sigmoid of negative input should be < 0.5, got {result}");
        let expected = 1.0 / (1.0 + E.powf(2.0));
        assert!((result - expected).abs() < 1e-6, "Expected {expected}, got {result}");
    }

    #[test]
    fn value_persists_after_finish_and_save() {
        let mut s = make_sigmoid(0.0);
        s.finish_and_save(2.0);
        let expected = 1.0 / (1.0 + E.powf(-2.0));
        assert!(
            (s.value() - expected).abs() < 1e-6,
            "value() should return last saved result ~{expected}, got {}",
            s.value()
        );
    }

    #[test]
    fn value_updated_on_each_finish_and_save() {
        let mut s = make_sigmoid(0.0);
        s.finish_and_save(0.0); // 0.5
        s.finish_and_save(2.0); // ~0.881
        let expected = 1.0 / (1.0 + E.powf(-2.0));
        assert!(
            (s.value() - expected).abs() < 1e-6,
            "value() should reflect most recent finish_and_save, got {}",
            s.value()
        );
    }
}