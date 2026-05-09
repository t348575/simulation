use super::resources::InspectorCamera;
use bevy::{
    camera::{visibility::RenderLayers, RenderTarget},
    ecs::schedule::ScheduleLabel,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{WindowClosed, WindowRef},
};
use bevy_egui::{
    egui::{self, Pos2},
    EguiContext, EguiMultipassSchedule,
};
use bevy_vector_shapes::prelude::*;
use engine::{
    nn::{GraphLocation, Net, Node},
    NeuronInfo,
};

use super::resources::*;

const CIRCLE_RADIUS: f32 = 20.0;
const SPACING: f32 = 10.0;
const INSPECTOR_ZOOM_MIN: f32 = 0.05;
const INSPECTOR_ZOOM_MAX: f32 = 20.0;

#[derive(Component)]
pub struct InspectWindow;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InspectWindowContextPass;

pub fn setup(mut commands: Commands) {
    let inspect_net_window = commands
        .spawn((
            Window {
                title: "Inspector".to_string(),
                position: WindowPosition::Centered(MonitorSelection::Index(1)),
                ..Default::default()
            },
            InspectWindow,
        ))
        .id();

    let render_layer = RenderLayers::layer(1);
    commands.spawn((
        Camera2d,
        RenderTarget::Window(WindowRef::Entity(inspect_net_window)),
        EguiMultipassSchedule::new(InspectWindowContextPass),
        render_layer,
        InspectorCamera,
    ));
    commands.insert_resource(InspectorWindowId(inspect_net_window));
}

pub fn exit_inspector(
    mut commands: Commands,
    inspector_window_id: Res<InspectorWindowId>,
    cameras: Query<Entity, With<InspectorCamera>>,
) {
    commands.entity(inspector_window_id.0).despawn();
    for cam in &cameras {
        commands.entity(cam).despawn();
    }
}

pub fn on_inspector_closed(
    mut events: MessageReader<WindowClosed>,
    inspector_window_id: Res<InspectorWindowId>,
    mut next_inspect_state: ResMut<NextState<InspectWindowState>>,
) {
    for event in events.read() {
        if event.window == inspector_window_id.0 {
            next_inspect_state.set(InspectWindowState::None);
        }
    }
}

pub fn get_inspect_net(
    mut commands: Commands,
    mut event_reader: MessageReader<InspectNet>,
    window: Query<&Window, With<InspectWindow>>,
    circles: Query<Entity, (With<DiscComponent>, With<InspectWindow>)>,
    lines: Query<Entity, (With<LineComponent>, With<InspectWindow>)>,
) {
    let Ok(w) = window.single() else {
        // Inspect window has been closed — drain pending events and skip drawing.
        for _ in event_reader.read() {}
        return;
    };
    let dims = (w.width(), w.height());
    for item in event_reader.read() {
        let nn = draw_neural_net(item.0.clone(), &circles, &lines, dims, &mut commands);

        commands.insert_resource(nn);
        break;
    }
}

fn x_pos(layer: usize) -> f32 {
    ((CIRCLE_RADIUS * 2.0) + SPACING) * layer as f32
}

fn y_pos(node: usize) -> f32 {
    ((CIRCLE_RADIUS * 2.0) + SPACING) * node as f32
}

fn node_color(node: &Node) -> Color {
    match node {
        Node::Input(_) => Color::from(Srgba::hex("17c3b2").unwrap()), // teal
        Node::Output(_) => Color::from(Srgba::hex("fe6d73").unwrap()), // coral
        Node::Neuron(_) => Color::from(Srgba::hex("c77dff").unwrap()), // purple
        Node::None => Color::from(Srgba::hex("888888").unwrap()),
    }
}

fn edge_color_and_thickness(weight: f32) -> (Color, f32) {
    let magnitude = weight.abs().min(3.0) / 3.0; // 0..1
    let thickness = 1.0 + magnitude * 5.0; // 1..6
    let color = if weight >= 0.0 {
        // positive: pale blue → vivid blue
        let g = 0.6 - magnitude * 0.4;
        Color::from(Srgba::new(
            0.1,
            g,
            0.9 + magnitude * 0.1,
            0.4 + magnitude * 0.6,
        ))
    } else {
        // negative: pale red → vivid red
        let gb = 0.6 - magnitude * 0.5;
        Color::from(Srgba::new(
            0.9 + magnitude * 0.1,
            gb,
            gb * 0.5,
            0.4 + magnitude * 0.6,
        ))
    };
    (color, thickness)
}

pub fn draw_neural_net(
    net: Net,
    circles: &Query<Entity, (With<DiscComponent>, With<InspectWindow>)>,
    lines: &Query<Entity, (With<LineComponent>, With<InspectWindow>)>,
    dims: (f32, f32),
    commands: &mut Commands,
) -> Nn {
    for item in circles.iter() {
        commands.entity(item).despawn();
    }
    for item in lines.iter() {
        commands.entity(item).despawn();
    }

    let render_layer = RenderLayers::layer(1);
    let mut nodes = Vec::new();
    let mut config = ShapeConfig::default_2d();

    // Precompute layer positions
    let layer_positions: Vec<(f32, f32, usize)> = net
        .graph
        .layers
        .iter()
        .enumerate()
        .map(|(num, layer)| {
            let count = layer.len();
            let start_x = (-1.0 * dims.0 / 3.0) + ((CIRCLE_RADIUS * 2.0) + SPACING) * num as f32;
            let start_y = -1.0
                * (((CIRCLE_RADIUS * 2.0 * count as f32) + (SPACING * (count as f32 - 1.0))) / 2.0);
            (start_x, start_y, count)
        })
        .collect();

    // Pass 1: edges (drawn at Z=-1, behind nodes)
    for (num, layer) in net.graph.layers.iter().enumerate() {
        let (start_x, start_y, _) = layer_positions[num];
        for (node_num, node) in layer.iter().enumerate() {
            let from = Vec3::new(start_x + x_pos(num), start_y + y_pos(node_num), -1.0);
            for c in node.connections.iter() {
                let to_idx = c.to.layer as usize;
                if to_idx >= layer_positions.len() {
                    continue;
                }
                let (to_sx, to_sy, _) = layer_positions[to_idx];
                let to = Vec3::new(
                    to_sx + x_pos(to_idx),
                    to_sy + y_pos(c.to.node as usize),
                    -1.0,
                );
                if c.value.enabled {
                    let (color, thickness) = edge_color_and_thickness(c.value.weight);
                    config.color = color;
                    config.thickness = thickness;
                } else {
                    config.color = Color::from(Srgba::new(0.5, 0.5, 0.5, 0.3));
                    config.thickness = 0.5;
                }
                config.transform = Transform::IDENTITY;
                commands.spawn((
                    ShapeBundle::line(&config, from, to),
                    InspectWindow,
                    render_layer.clone(),
                ));
            }
        }
    }

    // Pass 2: nodes (at Z=0, in front of edges)
    config.hollow = false;
    config.thickness = 0.1;
    for (num, layer) in net.graph.layers.iter().enumerate() {
        let (start_x, start_y, _) = layer_positions[num];
        for (node_num, node) in layer.iter().enumerate() {
            let wx = start_x + x_pos(num);
            let wy = start_y + y_pos(node_num);
            nodes.push((
                GraphLocation {
                    layer: num as u16,
                    node: node_num as u16,
                },
                NodePosition { x: wx, y: wy },
            ));
            config.color = node_color(&node.value);
            config.transform = Transform::from_xyz(wx, wy, 0.0);
            commands.spawn((
                ShapeBundle::circle(&config, CIRCLE_RADIUS),
                InspectWindow,
                render_layer.clone(),
            ));
        }
    }

    Nn {
        net,
        node_positions: nodes,
    }
}

pub fn draw_node_labels(
    mut egui_ctx: Single<&mut EguiContext, With<InspectorCamera>>,
    nn: Res<Nn>,
    window: Query<&Window, With<InspectWindow>>,
    camera: Query<&Transform, With<InspectorCamera>>,
) {
    let Ok(w) = window.single() else { return };
    let dims = (w.width(), w.height());
    let cam = camera.single().ok();
    let ctx = egui_ctx.get_mut();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("nn_labels"),
    ));

    let scale = cam.map(|t| t.scale.x.max(1e-6)).unwrap_or(1.0);
    let cam_x = cam.map(|t| t.translation.x).unwrap_or(0.0);
    let cam_y = cam.map(|t| t.translation.y).unwrap_or(0.0);

    for (loc, pos) in &nn.node_positions {
        // World → window via inverse of Camera2d transform (Camera2d uses scale
        // as units-per-pixel for orthographic projection).
        let screen_x = (pos.x - cam_x) / scale + dims.0 / 2.0;
        let screen_y = dims.1 / 2.0 - (pos.y - cam_y) / scale;

        let node = nn.net.graph.get_node(loc);
        let label = match &node {
            Some(n) => n.value.label().to_string(),
            None => continue,
        };

        let color = match &node.unwrap().value {
            Node::Input(_) => egui::Color32::from_rgb(0x17, 0xc3, 0xb2),
            Node::Output(_) => egui::Color32::from_rgb(0xfe, 0x6d, 0x73),
            Node::Neuron(_) => egui::Color32::from_rgb(0xc7, 0x7d, 0xff),
            Node::None => egui::Color32::GRAY,
        };

        let radius_px = CIRCLE_RADIUS / scale;
        painter.text(
            egui::pos2(screen_x, screen_y - radius_px - 3.0),
            egui::Align2::CENTER_BOTTOM,
            &label,
            egui::FontId::proportional((11.0 / scale).clamp(8.0, 22.0)),
            color,
        );
    }
}

pub fn inspector_camera_controls(
    mut camera: Query<&mut Transform, With<InspectorCamera>>,
    scroll: Res<AccumulatedMouseScroll>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    keyboard: Res<ButtonInput<KeyCode>>,
    inspector_window: Query<&Window, With<InspectWindow>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };
    let focused = inspector_window
        .single()
        .map(|w| w.focused)
        .unwrap_or(false);
    if !focused {
        return;
    }

    let mut zoom = transform.scale.x;

    // Scroll wheel zooms.
    if scroll.delta.y != 0.0 {
        let factor = if scroll.delta.y > 0.0 {
            0.85
        } else {
            1.0 / 0.85
        };
        zoom *= factor;
    }

    // Middle-mouse or right-mouse drag pans (mouse_motion is in screen pixels;
    // multiply by zoom because translation is in world units).
    if mouse_button.pressed(MouseButton::Middle) || mouse_button.pressed(MouseButton::Right) {
        transform.translation.x -= mouse_motion.delta.x * zoom;
        transform.translation.y += mouse_motion.delta.y * zoom;
    }

    // R resets view.
    if keyboard.just_pressed(KeyCode::KeyR) {
        zoom = 1.0;
        transform.translation = Vec3::ZERO;
    }

    zoom = zoom.clamp(INSPECTOR_ZOOM_MIN, INSPECTOR_ZOOM_MAX);
    transform.scale = Vec3::splat(zoom);
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

//     shapes.render_layers = Some(RenderLayers::layer(1));
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

pub fn toggle_inspect_window(
    buttons: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<InspectWindow>>,
    camera: Query<&Transform, With<InspectorCamera>>,
    data: Res<Nn>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_iw_state: ResMut<NextState<InspectNodeState>>,
    mut inspect_info: ResMut<InspectInfo>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        next_iw_state.set(InspectNodeState::None);
        return;
    }

    if buttons.just_pressed(MouseButton::Left) {
        let w = q_windows.single().unwrap();
        if let Some(position) = w.cursor_position() {
            let cam = camera.single().ok();
            let scale = cam.map(|t| t.scale.x.max(1e-6)).unwrap_or(1.0);
            let cam_x = cam.map(|t| t.translation.x).unwrap_or(0.0);
            let cam_y = cam.map(|t| t.translation.y).unwrap_or(0.0);
            // Window cursor → world (Y inverted; Camera2d translation/scale).
            let x = (position.x - w.width() / 2.0) * scale + cam_x;
            let y = (w.height() / 2.0 - position.y) * scale + cam_y;

            let hit_radius2 = (CIRCLE_RADIUS).powi(2);
            for node in &data.node_positions {
                let pos = (x - node.1.x).powi(2) + (y - node.1.y).powi(2);
                if pos <= hit_radius2 {
                    inspect_info.0 .0 = node.0.clone();
                    inspect_info.0 .1 = data.net.graph.get_node(&node.0).unwrap().clone();
                    next_iw_state.set(InspectNodeState::Display);
                    break;
                }
            }
        }
    }
}

pub fn inspect_window(
    mut egui_ctx: Single<&mut EguiContext, With<InspectorCamera>>,
    inspect_info: Res<InspectInfo>,
    mut window_state: ResMut<WindowInfo>,
) {
    let egui_ctx = egui_ctx.get_mut();
    let mut style = (*egui_ctx.style()).clone();

    *style.text_styles.get_mut(&egui::TextStyle::Body).unwrap() =
        egui::FontId::new(14.0, egui::FontFamily::Proportional);
    egui_ctx.set_style(style);

    let window = egui::Window::new(format!(
        "Node [{}, {}]",
        inspect_info.0 .0.layer, inspect_info.0 .0.node
    ))
    .default_pos(Pos2::new(
        window_state.inspect_window_pos.0,
        window_state.inspect_window_pos.1,
    ))
    .show(egui_ctx, |ui| {
        let node = &inspect_info.0 .1;
        match &node.value {
            Node::Input(n) => {
                ui.colored_label(
                    egui::Color32::from_rgb(0x17, 0xc3, 0xb2),
                    format!("Input: {}", n.label()),
                );
                ui.label(format!("Value: {:.4}", n.as_standard()));
            }
            Node::Output(n) => {
                ui.colored_label(
                    egui::Color32::from_rgb(0xfe, 0x6d, 0x73),
                    format!("Output: {}", n.label()),
                );
                ui.label(format!("Activation: {:.4}", n.value()));
            }
            Node::Neuron(n) => {
                ui.colored_label(
                    egui::Color32::from_rgb(0xc7, 0x7d, 0xff),
                    format!("Neuron: {}", n.label()),
                );
            }
            Node::None => {
                ui.label("None");
            }
        }

        ui.separator();
        ui.label(format!("Connections: {}", node.connections.len()));
        for c in &node.connections {
            let status = if c.value.enabled { "✓" } else { "✗" };
            ui.label(format!(
                "  {} → [{},{}]  w={:.3}",
                status, c.to.layer, c.to.node, c.value.weight
            ));
        }
    })
    .unwrap();

    window_state.inspect_window_pos = (window.response.rect.left(), window.response.rect.top());
}
