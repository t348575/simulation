use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{
    egui::{self, Layout},
    EguiContext,
};
use bevy_vector_shapes::prelude::*;
use rand::Rng;
use engine::nn::{Net, Node};

use crate::{BaseNodes, TabState};

use super::resources::*;

const CREATURE_DIM: f32 = 5.0;

pub fn setup(mut control_panel: ResMut<ControlPanel>, window_query: Query<&Window, With<PrimaryWindow>>) {
    let window = window_query.get_single().unwrap();
    control_panel.width = window.width().to_string();
    control_panel.height = window.height().to_string();
}

pub fn control_panel(
    mut egui_ctx: Query<&mut EguiContext, With<PrimaryWindow>>,
    mut next_tab_state: ResMut<NextState<TabState>>,
    mut control_panel: ResMut<ControlPanel>,
    base_nodes: Res<BaseNodes>,
    mut data: ResMut<Simulation>,
    mut next_sim_state: ResMut<NextState<SimulationState>>,
    sim_state: Res<State<SimulationState>>,
    mut commands: Commands,
    rects: Query<Entity, With<RectangleComponent>>
) {
    egui::SidePanel::right("Control panel").show(egui_ctx.single_mut().get_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.label("Number of creatures");
            ui.text_edit_singleline(&mut control_panel.initial_num_creatures);
        });

        ui.separator();

        ui.vertical(|ui| {
            ui.label("World dimensions");
            ui.columns(2, |ui| {
                ui[0].text_edit_singleline(&mut control_panel.width);
                ui[1].text_edit_singleline(&mut control_panel.height);
            });
        });

        ui.horizontal(|ui| {
            ui.set_enabled(control_panel.can_create_sim);
            let button = ui.add_sized(
                (ui.available_width(), 0.0),
                egui::Button::new("Create new simulation"),
            );
            if button.clicked() {
                generate_simulation(&control_panel, &base_nodes, &mut data);
                control_panel.can_create_sim = false;
                next_sim_state.set(SimulationState::Paused);
            }
        });

        ui.horizontal(|ui| {
            ui.set_enabled(!control_panel.can_create_sim);

            let button_txt = match sim_state.get() {
                SimulationState::None => "Start simulation",
                SimulationState::Paused => "Resume simulation",
                SimulationState::Running => "Pause simulation",
            };

            let button = ui.add_sized(
                (ui.available_width(), 0.0),
                egui::Button::new(button_txt),
            );

            if button.clicked() {
                match sim_state.get() {
                    SimulationState::Paused => next_sim_state.set(SimulationState::Running),
                    _ => next_sim_state.set(SimulationState::Paused),
                }
            }
        });

        match sim_state.get() {
            SimulationState::None => {},
            _ => {
                let button = ui.add_sized(
                    (ui.available_width(), 0.0),
                    egui::Button::new("Stop & reset simulation"),
                );

                if button.clicked() {
                    clear_screen(&mut commands, rects);
                    data.creatures.clear();
                    control_panel.can_create_sim = true;
                    next_sim_state.set(SimulationState::Paused);
                    next_sim_state.set(SimulationState::None);
                }
            },
        }

        ui.with_layout(Layout::bottom_up(egui::Align::Center), |ui| {
            let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Main menu"));
            if button.clicked() {
                next_tab_state.set(TabState::MainMenu);
            }
        });
    });
}

fn generate_simulation(control_panel: &ControlPanel, base_nodes: &BaseNodes, data: &mut Simulation) {
    let num_creatures: usize = control_panel.initial_num_creatures.parse().unwrap();

    let mut rng = rand::thread_rng();
    let height = control_panel.height.parse::<f32>().unwrap() / 2.0;
    let width = control_panel.width.parse::<f32>().unwrap() / 2.0;

    let creatures = (0..num_creatures).map(|_| Creature {
        brain: Net::gen(&base_nodes.input_nodes, &base_nodes.output_nodes).unwrap(),
        position: (rng.gen_range((-1.0 * width)..width), rng.gen_range((-1.0 * height)..height)),
    }).collect();

    data.world_dim = (width, height);
    data.creatures = creatures;
}

#[derive(Component)]
struct Target;

pub fn initialize_world(mut shapes: ShapeCommands, data: Res<Simulation>) {
    data.creatures.iter().for_each(|creature| {
        shapes.color = Color::hex("1b1b1b").unwrap();
        shapes.transform = Transform::from_xyz(creature.position.0, creature.position.1, 0.0);
        shapes.rect(Vec2::new(CREATURE_DIM, CREATURE_DIM));
    });
}

fn clear_screen(commands: &mut Commands, rects: Query<Entity, With<RectangleComponent>>) {
    for item in rects.iter() {
        commands.entity(item).despawn_recursive();
    }
}

pub fn run_simulation(mut data: ResMut<Simulation>, mut shapes: ShapeCommands, mut commands: Commands, rects: Query<Entity, With<RectangleComponent>>, time: Res<Time>) {
    clear_screen(&mut commands, rects);

    data.creatures.iter_mut().for_each(|c| {
        let speed = c.brain.graph.layers[c.brain.graph.layers.len() - 1][4].value.clone();
        let inputs = &mut c.brain.graph.layers[c.brain.input_layer as usize];
        if let Node::Input(i) = &mut inputs[0].value {
            i.set_value(i.as_standard() + 0.01); // hunger
        }

        if let Node::Input(i) = &mut inputs[0].value {
            i.set_value(i.as_standard() + 1.0); // age
        }

        if let Node::Input(i) = &mut inputs[0].value {
            i.set_value(i.as_standard() - 0.00001); // health
        }

        if let Node::Input(i) = &mut inputs[0].value {
            if let Node::Output(o) = speed {
                i.set_value(o.value()); // speed
            }
        }

        c.brain.tick();

        let output_layer = &c.brain.graph.layers[c.brain.graph.layers.len() - 1];
        let forward = get_output_value(&output_layer[0].value);
        let backward = get_output_value(&output_layer[1].value);
        let left = get_output_value(&output_layer[2].value);
        let right = get_output_value(&output_layer[3].value);

        let movement_vec = dirs_to_vec(forward, backward, left, right);
        let speed = get_output_value(&output_layer[4].value);

        let t = movement_vec * speed * time.delta_seconds();
        c.position.0 += t.x;
        c.position.1 += t.y;
        
        shapes.color = Color::hex("1b1b1b").unwrap();
        shapes.transform.translation = Vec3::new(c.position.0, c.position.1, 0.0);
        shapes.rect(Vec2::new(CREATURE_DIM, CREATURE_DIM));
    });

    // next_sim_state.set(SimulationState::Paused);
}

fn get_output_value(value: &Node) -> f32 {
    if let Node::Output(o) = value {
        o.value()
    } else {
        0.0
    }
}

fn dirs_to_vec(forward: f32, backward: f32, left: f32, right: f32) -> Vec2 {
    let x = Vec2::new(right, 0.0) + Vec2::new(left * -1.0, 0.0);
    let y = Vec2::new(0.0, forward) + Vec2::new(0.0, backward * -1.0);
    x + y
}

pub fn inspect_creature(buttons: Res<ButtonInput<MouseButton>>, q_windows: Query<&Window, With<PrimaryWindow>>) {
    if buttons.just_pressed(MouseButton::Left) {
        if let Some(position) = q_windows.single().cursor_position() {
            println!("({}, {})", position.x, position.y);
        }
    }
}