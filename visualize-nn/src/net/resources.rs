use std::path::PathBuf;

use bevy::prelude::*;
use egui_file::FileDialog;
use engine::nn::{GraphLocation, GraphNode, Net, Node};
use serde::{Deserialize, Serialize};

#[derive(Resource, Debug)]
pub struct InspectorWindowId(pub Entity);

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
}

#[derive(Resource, Debug, Default, Serialize, Deserialize)]
pub struct Nn {
    pub net: Net,
    pub node_positions: Vec<(GraphLocation, NodePosition)>,
}

#[derive(Resource, Debug, Default)]
pub struct WindowInfo {
    pub inspect_window_pos: (f32, f32),
}

#[derive(Resource, Debug, Default)]
pub struct InspectInfo(pub (GraphLocation, GraphNode));

#[derive(Resource, Debug)]
pub struct ControlPanel {
    pub num_nets: String,
    pub view_net: String,
    pub view_net_int: usize,
    pub mate: (String, String),
    pub file_name: PathBuf,
    pub save_dialog: Option<FileDialog>,
    pub open_dialog: Option<FileDialog>,
}

impl Default for ControlPanel {
    fn default() -> Self {
        let mut file_name = std::env::current_dir().unwrap();
        file_name.push("net.bin");
        ControlPanel {
            num_nets: "0".to_string(),
            view_net: "0".to_string(),
            view_net_int: 0,
            mate: ("0".to_owned(), "0".to_owned()),
            file_name,
            save_dialog: None,
            open_dialog: None,
        }
    }
}
