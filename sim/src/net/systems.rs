use bevy::{
    prelude::*,
    render::{camera::RenderTarget, view::RenderLayers},
    window::WindowRef,
};
use bevy_egui::{
    egui::{self, Pos2},
    EguiContext,
};
use bevy_vector_shapes::prelude::*;
use engine::nn::{GraphLocation, Net};

use super::resources::*;

const CIRCLE_RADIUS: f32 = 20.0;
const SPACING: f32 = 10.0;

#[derive(Component)]
pub struct InspectWindow;

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
        Camera2dBundle {
            camera: Camera {
                target: RenderTarget::Window(WindowRef::Entity(inspect_net_window)),
                ..Default::default()
            },
            ..Default::default()
        },
        render_layer,
    ));
    commands.insert_resource(InspectorWindowId(inspect_net_window));
}

pub fn exit_inspector(mut commands: Commands, inspector_window_id: Res<InspectorWindowId>) {
    commands.entity(inspector_window_id.0).despawn_recursive();
}

pub fn get_inspect_net(
    mut shapes: ShapeCommands,
    mut commands: Commands,
    mut event_reader: EventReader<InspectNet>,
    window: Query<&Window, With<InspectWindow>>,
    circles: Query<Entity, (With<DiscComponent>, With<InspectWindow>)>,
    lines: Query<Entity, (With<LineComponent>, With<InspectWindow>)>,
) {
    let w = window.get_single().unwrap();
    let dims = (w.width(), w.height());
    for item in event_reader.read() {
        let nn = draw_neural_net(
            item.0.clone(),
            &circles,
            &lines,
            dims,
            &mut commands,
            &mut shapes,
        );

        commands.insert_resource(nn);
    }
}

fn x_pos(layer: usize) -> f32 {
    ((CIRCLE_RADIUS * 2.0) + SPACING) * layer as f32
}

fn y_pos(node: usize) -> f32 {
    ((CIRCLE_RADIUS * 2.0) + SPACING) * node as f32
}

pub fn draw_neural_net(
    net: Net,
    circles: &Query<Entity, (With<DiscComponent>, With<InspectWindow>)>,
    lines: &Query<Entity, (With<LineComponent>, With<InspectWindow>)>,
    dims: (f32, f32),
    commands: &mut Commands,
    shapes: &mut ShapeCommands,
) -> Nn {
    for item in circles.iter() {
        commands.entity(item).despawn_recursive();
    }
    for item in lines.iter() {
        commands.entity(item).despawn_recursive();
    }

    let render_layer = RenderLayers::layer(1);
    let mut nodes = Vec::new();

    shapes.color = Color::hex("1b1b1b").unwrap();
    for (num, layer) in net.graph.layers.iter().enumerate() {
        let count = layer.iter().count();
        let start_x = (-1.0 * dims.0 / 3.0) + ((CIRCLE_RADIUS * 2.0) + SPACING) * num as f32;
        let start_y = -1.0
            * (((CIRCLE_RADIUS * 2.0 * count as f32) + (SPACING * (count as f32 - 1.0))) / 2.0);
        for (node_num, node) in layer.iter().enumerate() {
            shapes.transform =
                Transform::from_xyz(start_x + x_pos(num), start_y + y_pos(node_num), 0.0);
            nodes.push((
                GraphLocation {
                    layer: num as u16,
                    node: node_num as u16,
                },
                NodePosition {
                    x: start_x + x_pos(num),
                    y: start_y + y_pos(node_num),
                },
            ));
            commands.spawn((
                ShapeBundle::circle(shapes.config(), CIRCLE_RADIUS),
                InspectWindow,
                render_layer,
            ));

            for c in node.connections.iter() {
                let to_x =
                    (-1.0 * dims.0 / 3.0) + ((CIRCLE_RADIUS * 2.0) + SPACING) * c.to.layer as f32;
                let to_layer_count = net
                    .graph
                    .layers
                    .iter()
                    .skip(c.to.layer as usize)
                    .next()
                    .unwrap()
                    .len();
                let to_y: f32 = -1.0
                    * (((CIRCLE_RADIUS * 2.0 * to_layer_count as f32)
                        + (SPACING * (to_layer_count as f32 - 1.0)))
                        / 2.0);

                if c.value.enabled {
                    shapes.color = Color::hex("1b1b1b").unwrap();
                } else {
                    shapes.color = Color::hex("808080").unwrap();
                }

                shapes.thickness = 5.0;
                shapes.set_translation(Vec3::NEG_Z);
                commands.spawn((
                    ShapeBundle::line(
                        shapes.config(),
                        Vec3::new(start_x + x_pos(num), start_y + y_pos(node_num), -1.0),
                        Vec3::new(
                            to_x + x_pos(c.to.layer as usize),
                            to_y + y_pos(c.to.node as usize),
                            -1.0,
                        ),
                    ),
                    InspectWindow,
                    render_layer,
                ));
                shapes.thickness = 0.0;
                shapes.color = Color::hex("1b1b1b").unwrap();
            }
        }
    }

    Nn {
        net,
        node_positions: nodes,
    }
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
        let w = q_windows.single();
        if let Some(position) = w.cursor_position() {
            let x = position.x - w.width() / 2.0;
            let y = (position.y - w.height() / 2.0) * -1.0;

            for node in &data.node_positions {
                let pos = (x - node.1.x).powi(2) + (y - node.1.y).powi(2);
                if pos <= CIRCLE_RADIUS.powi(2) {
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
    mut egui_ctx: Query<&mut EguiContext, With<InspectWindow>>,
    inspect_info: Res<InspectInfo>,
    mut window_state: ResMut<WindowInfo>,
) {
    let mut style = (*egui_ctx.single_mut().get_mut().style()).clone();

    let window = egui::Window::new(format!(
        "Layer {} Node {}",
        inspect_info.0 .0.layer, inspect_info.0 .0.node
    ))
    .default_pos(Pos2::new(
        window_state.inspect_window_pos.0,
        window_state.inspect_window_pos.1,
    ))
    .show(egui_ctx.single_mut().get_mut(), |ui| {
        ui.horizontal(|ui| {
            *style.text_styles.get_mut(&egui::TextStyle::Body).unwrap() =
                egui::FontId::new(20.0, egui::FontFamily::Proportional);
            ui.set_style(style);
            ui.label(format!("{:#?}", inspect_info.0 .1));
        });
    })
    .unwrap();

    window_state.inspect_window_pos = (window.response.rect.left(), window.response.rect.top());
}
