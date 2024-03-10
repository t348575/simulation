use bevy::{prelude::*, render::{camera::RenderTarget, view::RenderLayers}, window::{PrimaryWindow, WindowRef}};
use bevy_egui::{
    egui::{self, Layout, Pos2},
    EguiContext,
};
use bevy_vector_shapes::prelude::*;
use egui_file::FileDialog;
use engine::nn::{
    reproduce::{Crossover, DefaultIterator},
    GraphLocation, Net,
};

use super::{resources::*, InspectorWindowId};
use crate::TabState;

const CIRCLE_RADIUS: f32 = 20.0;
const SPACING: f32 = 10.0;

// fn x_pos(layer: usize) -> f32 {
//     ((CIRCLE_RADIUS * 2.0) + SPACING) * layer as f32
// }

// fn y_pos(node: usize) -> f32 {
//     ((CIRCLE_RADIUS * 2.0) + SPACING) * node as f32
// }

pub fn setup(mut commands: Commands) {
    let inspect_net_window = commands
        .spawn(Window {
            title: "Inspector".to_string(),
            position: WindowPosition::Centered(MonitorSelection::Index(1)),
            ..Default::default()
        })
        .id();

    let render_layer = RenderLayers::layer(1);
    commands.spawn((Camera2dBundle {
        camera: Camera {
            target: RenderTarget::Window(WindowRef::Entity(inspect_net_window)),
            ..Default::default()
        },
        ..Default::default()
    }, render_layer));
    commands.insert_resource(InspectorWindowId(inspect_net_window));
}

pub fn test(mut shapes: ShapeCommands) {
    shapes.render_layers = Some(RenderLayers::layer(1));
    shapes.color = Color::hex("1b1b1b").unwrap();
    shapes.circle(15.0);
}

pub fn exit_inspector(mut commands: Commands, inspector_window_id: Res<InspectorWindowId>) {
    commands.entity(inspector_window_id.0).despawn_recursive();
}

// pub fn draw_neural_net(
//     data: &mut Simulation,
//     shapes: &mut ShapeCommands,
//     window_query: &Query<&Window, With<PrimaryWindow>>,
//     commands: &mut Commands,
//     circles: &Query<Entity, With<DiscComponent>>,
//     lines: &Query<Entity, With<LineComponent>>,
//     control_panel: &mut ResMut<ControlPanel>,
// ) {
//     for item in circles.iter() {
//         commands.entity(item).despawn_recursive();
//     }
//     for item in lines.iter() {
//         commands.entity(item).despawn_recursive();
//     }

//     data.nets.iter_mut().for_each(|x| x.node_positions.clear());

//     let view_net = if let Ok(view_net) = control_panel.view_net.parse::<usize>() {
//         if view_net == 0 {
//             0
//         } else if view_net > data.nets.len() {
//             data.nets.len() - 1
//         } else {
//             view_net - 1
//         }
//     } else {
//         0
//     };

//     control_panel.view_net_int = view_net;

//     let window = window_query.get_single().unwrap();
//     let mut nodes = Vec::new();
//     let item = &mut data.nets[view_net];

//     shapes.color = Color::hex("1b1b1b").unwrap();
//     for (num, layer) in item.net.graph.layers.iter().enumerate() {
//         let count = layer.iter().count();
//         let start_x =
//             (-1.0 * window.width() / 3.0) + ((CIRCLE_RADIUS * 2.0) + SPACING) * num as f32;
//         let start_y = -1.0
//             * (((CIRCLE_RADIUS * 2.0 * count as f32) + (SPACING * (count as f32 - 1.0))) / 2.0);
//         for (node_num, node) in layer.iter().enumerate() {
//             shapes.transform =
//                 Transform::from_xyz(start_x + x_pos(num), start_y + y_pos(node_num), 0.0);
//             nodes.push((
//                 GraphLocation {
//                     layer: num as u16,
//                     node: node_num as u16,
//                 },
//                 NodePosition {
//                     x: start_x + x_pos(num),
//                     y: start_y + y_pos(node_num),
//                 },
//             ));
//             shapes.circle(CIRCLE_RADIUS);

//             for c in node.connections.iter() {
//                 let to_x = (-1.0 * window.width() / 3.0)
//                     + ((CIRCLE_RADIUS * 2.0) + SPACING) * c.to.layer as f32;
//                 let to_layer_count = item
//                     .net
//                     .graph
//                     .layers
//                     .iter()
//                     .skip(c.to.layer as usize)
//                     .next()
//                     .unwrap()
//                     .len();
//                 let to_y: f32 = -1.0
//                     * (((CIRCLE_RADIUS * 2.0 * to_layer_count as f32)
//                         + (SPACING * (to_layer_count as f32 - 1.0)))
//                         / 2.0);

//                 if c.value.enabled {
//                     shapes.color = Color::hex("1b1b1b").unwrap();
//                 } else {
//                     shapes.color = Color::hex("808080").unwrap();
//                 }

//                 shapes.thickness = 5.0;
//                 shapes.set_translation(Vec3::NEG_Z);
//                 shapes.line(
//                     Vec3::new(start_x + x_pos(num), start_y + y_pos(node_num), -1.0),
//                     Vec3::new(
//                         to_x + x_pos(c.to.layer as usize),
//                         to_y + y_pos(c.to.node as usize),
//                         -1.0,
//                     ),
//                 );
//                 shapes.thickness = 0.0;
//                 shapes.color = Color::hex("1b1b1b").unwrap();
//             }
//         }
//     }
//     item.node_positions = nodes;
// }

// pub fn toggle_inspect_window(
//     buttons: Res<ButtonInput<MouseButton>>,
//     q_windows: Query<&Window, With<PrimaryWindow>>,
//     data: Res<Simulation>,
//     keyboard: Res<ButtonInput<KeyCode>>,
//     control_panel: Res<ControlPanel>,
//     mut next_iw_state: ResMut<NextState<InspectWindowState>>,
//     mut inspect_info: ResMut<InspectInfo>,
// ) {
//     if keyboard.just_pressed(KeyCode::Escape) {
//         next_iw_state.set(InspectWindowState::None);
//         return;
//     }

//     if buttons.just_pressed(MouseButton::Left) {
//         if data.nets.len() <= control_panel.view_net_int {
//             return;
//         }

//         let w = q_windows.single();
//         if let Some(position) = w.cursor_position() {
//             let x = position.x - w.width() / 2.0;
//             let y = (position.y - w.height() / 2.0) * -1.0;

//             for node in &data.nets[control_panel.view_net_int].node_positions {
//                 let pos = (x - node.1.x).powi(2) + (y - node.1.y).powi(2);
//                 if pos <= CIRCLE_RADIUS.powi(2) {
//                     inspect_info.0 .0 = node.0.clone();
//                     inspect_info.0 .1 = data.nets[control_panel.view_net_int]
//                         .net
//                         .graph
//                         .get_node(&node.0)
//                         .unwrap()
//                         .clone();
//                     next_iw_state.set(InspectWindowState::Display);
//                     break;
//                 }
//             }
//         }
//     }
// }

// pub fn inspect_window(
//     mut egui_ctx: Query<&mut EguiContext, With<PrimaryWindow>>,
//     inspect_info: Res<InspectInfo>,
//     mut window_state: ResMut<WindowInfo>,
// ) {
//     let mut style = (*egui_ctx.single_mut().get_mut().style()).clone();

//     let window = egui::Window::new(format!(
//         "Layer {} Node {}",
//         inspect_info.0 .0.layer, inspect_info.0 .0.node
//     ))
//     .default_pos(Pos2::new(
//         window_state.inspect_window_pos.0,
//         window_state.inspect_window_pos.1,
//     ))
//     .show(egui_ctx.single_mut().get_mut(), |ui| {
//         ui.horizontal(|ui| {
//             *style.text_styles.get_mut(&egui::TextStyle::Body).unwrap() =
//                 egui::FontId::new(20.0, egui::FontFamily::Proportional);
//             ui.set_style(style);
//             ui.label(format!("{:#?}", inspect_info.0 .1));
//         });
//     })
//     .unwrap();

//     window_state.inspect_window_pos = (window.response.rect.left(), window.response.rect.top());
// }

// pub fn control_panel(
//     window_query: Query<&Window, With<PrimaryWindow>>,
//     mut egui_ctx: Query<&mut EguiContext, With<PrimaryWindow>>,
//     mut data: ResMut<Simulation>,
//     mut control_panel: ResMut<ControlPanel>,
//     mut shapes: ShapeCommands,
//     mut commands: Commands,
//     circles: Query<Entity, With<DiscComponent>>,
//     lines: Query<Entity, With<LineComponent>>,
//     mut next_tab_state: ResMut<NextState<TabState>>,
// ) {
//     let mut t = egui_ctx.single_mut();
//     let ctx = t.get_mut();
//     egui::SidePanel::right("Control panel").show(ctx, |ui| {
//         ui.horizontal(|ui| {
//             ui.label("Number of nets: ");
//             ui.text_edit_singleline(&mut control_panel.num_nets);
//         });

//         let button = ui.add_sized(
//             (ui.available_width(), 0.0),
//             egui::Button::new("Update nets"),
//         );
//         if button.clicked() {
//             if let Ok(num_nets) = control_panel.num_nets.parse::<usize>() {
//                 data.nets = (0..num_nets)
//                     .map(|_| Nn {
//                         net: Net::gen(&data.input_nodes, &data.output_nodes).unwrap(),
//                         node_positions: Vec::new(),
//                     })
//                     .collect::<Vec<_>>();
//                 draw_neural_net(
//                     &mut data,
//                     &mut shapes,
//                     &window_query,
//                     &mut commands,
//                     &circles,
//                     &lines,
//                     &mut control_panel,
//                 );
//             }
//         }

//         ui.separator();

//         ui.horizontal(|ui| {
//             ui.label("Visualize net: ");
//             ui.text_edit_singleline(&mut control_panel.view_net);
//         });
//         let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("View net"));
//         if button.clicked() {
//             draw_neural_net(
//                 &mut data,
//                 &mut shapes,
//                 &window_query,
//                 &mut commands,
//                 &circles,
//                 &lines,
//                 &mut control_panel,
//             );
//         }

//         ui.separator();
//         ui.label("Operations");
//         ui.separator();
//         ui.horizontal(|ui| {
//             ui.label("Mate");
//             egui::Grid::new("mate").show(ui, |ui| {
//                 ui.text_edit_singleline(&mut control_panel.mate.0);
//                 ui.text_edit_singleline(&mut control_panel.mate.1);
//                 ui.end_row();
//             });
//         });
//         let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Mate"));
//         if button.clicked() {
//             if let Ok(net_a) = control_panel.mate.0.parse::<usize>() {
//                 if let Ok(net_b) = control_panel.mate.1.parse::<usize>() {
//                     let n = Net::reproduce(
//                         &data.nets[net_a].net,
//                         &data.nets[net_b].net,
//                         &[Crossover::default()],
//                         DefaultIterator::new(),
//                     )
//                     .unwrap();
//                     data.nets.push(Nn {
//                         net: n,
//                         node_positions: Vec::new(),
//                     });
//                 }
//             }
//         }

//         ui.separator();
//         let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Save nets"));
//         if button.clicked() {
//             let filter = Box::new({
//                 let ext = Some(std::ffi::OsStr::new("bin"));
//                 move |path: &std::path::Path| -> bool { path.extension() == ext }
//             });
//             let mut dialog = FileDialog::save_file(Some(control_panel.file_name.clone()))
//                 .show_files_filter(filter)
//                 .default_filename("nets.bin");
//             dialog.set_path(std::env::current_dir().unwrap());
//             dialog.open();
//             control_panel.save_dialog = Some(dialog);
//         } else if let Some(dialog) = &mut control_panel.save_dialog {
//             if dialog.show(ctx).selected() {
//                 if let Some(file) = dialog.path() {
//                     control_panel.file_name = file.to_path_buf();
//                     std::fs::write(
//                         control_panel.file_name.clone(),
//                         bincode::serialize(data.as_ref()).unwrap(),
//                     )
//                     .unwrap();
//                 }
//             }
//         }

//         ui.separator();
//         let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Load net"));
//         if button.clicked() {
//             let filter = Box::new({
//                 let ext = Some(std::ffi::OsStr::new("bin"));
//                 move |path: &std::path::Path| -> bool { path.extension() == ext }
//             });
//             let mut dialog = FileDialog::open_file(Some(control_panel.file_name.clone()))
//                 .show_files_filter(filter)
//                 .default_filename("nets.bin");
//             dialog.set_path(std::env::current_dir().unwrap());
//             dialog.open();
//             control_panel.open_dialog = Some(dialog);
//         } else if let Some(dialog) = &mut control_panel.open_dialog {
//             if dialog.show(ctx).selected() {
//                 if let Some(file) = dialog.path() {
//                     let file_original = file.to_path_buf();
//                     let mut temp = file.to_path_buf();
//                     temp.pop();
//                     control_panel.file_name = temp;
//                     let sim: Simulation =
//                         bincode::deserialize(&std::fs::read(file_original).unwrap()).unwrap();
//                     control_panel.num_nets = sim.nets.len().to_string();

//                     commands.remove_resource::<Simulation>();
//                     commands.insert_resource(sim);
//                 }
//             }
//         }

//         ui.with_layout(Layout::bottom_up(egui::Align::Center), |ui| {
//             let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Main menu"));
//             if button.clicked() {
//                 next_tab_state.set(TabState::MainMenu);
//             }
//         });
//     });
// }

// pub fn gen_new_graph(
//     keyboard: Res<Input<KeyCode>>,
//     mut data: ResMut<Simulation>,
//     mut commands: Commands,
//     mut shapes: ShapeCommands,
//     window_query: Query<&Window, With<PrimaryWindow>>,
//     discs_q: Query<Entity, With<Disc>>,
//     lines_q: Query<Entity, With<Line>>,
// ) {
//     if keyboard.just_pressed(KeyCode::R) {
//         let net =
//             Net::gen(&data.input_nodes, &data.output_nodes).expect("Could not create neural net");
//         data.net = net;
//         data.node_positions = Vec::new();

//         for item in discs_q.iter() {
//             commands.entity(item).despawn_recursive();
//         }

//         for item in lines_q.iter() {
//             commands.entity(item).despawn_recursive();
//         }

//         // draw_neural_net(&mut data, &mut shapes, &window_query);
//     }
// }
