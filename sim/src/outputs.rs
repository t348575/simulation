use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Forward;
#[derive(Debug, Clone, Serialize)]
pub struct Backward;
#[derive(Debug, Clone, Serialize)]
pub struct Left;
#[derive(Debug, Clone, Serialize)]
pub struct Right;
#[derive(Debug, Clone, Serialize)]
pub struct OutputSpeed;
