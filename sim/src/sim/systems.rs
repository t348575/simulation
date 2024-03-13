use std::thread;

use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{
    egui::{self, collapsing_header::CollapsingState, Layout},
    EguiContext,
};
use bevy_vector_shapes::prelude::*;

use super::sim::{Generate, Runner, RunnerReq, RunnerRes};

use crate::{net::resources::InspectNet, BaseNodes, TabState};

use super::resources::*;

const CREATURE_DIM: f32 = 5.0;
const CREATURE_COLOR: &str = "3686ff";
const FOOD_COLOR: &str = "54ff71";

pub fn init_runner(mut commands: Commands) {
    let (r, tx, rx) = Runner::new();
    let t = thread::spawn(move || {
        r.run();
    });
    commands.insert_resource(RunnerResource { tx, rx, thread: t });
}

pub fn setup(
    mut control_panel: ResMut<ControlPanel>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut data: ResMut<Simulation>,
) {
    let window = window_query.get_single().unwrap();
    control_panel.width = window.width().to_string();
    control_panel.height = window.height().to_string();
    data.window_dims = (window.width(), window.height());
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
    rects: Query<Entity, With<RectangleComponent>>,
    time: Res<Time>,
    runner: Res<RunnerResource>,
) {
    egui::SidePanel::right("Control panel").show(egui_ctx.single_mut().get_mut(), |ui| {
        let id = ui.make_persistent_id("start/stop");
        CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label("start/stop");
            })
            .body(|ui| {
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
                        runner
                            .tx
                            .send(RunnerReq::Generate(Generate {
                                num_creatures: control_panel.initial_num_creatures.parse().unwrap(),
                                dims: (
                                    control_panel.width.parse().unwrap(),
                                    control_panel.height.parse().unwrap(),
                                ),
                                input_nodes: base_nodes.input_nodes.clone(),
                                output_nodes: base_nodes.output_nodes.clone(),
                            }))
                            .expect("Could not send pause request");
                        control_panel.can_create_sim = false;
                        next_sim_state.set(SimulationState::Paused);
                        runner
                            .tx
                            .send(RunnerReq::Pause)
                            .expect("Could not send pause request");
                    }
                });

                ui.horizontal(|ui| {
                    ui.set_enabled(!control_panel.can_create_sim);

                    let button_txt = match sim_state.get() {
                        SimulationState::None => "Start simulation",
                        SimulationState::Paused => "Resume simulation",
                        SimulationState::Running => "Pause simulation",
                    };

                    let button =
                        ui.add_sized((ui.available_width(), 0.0), egui::Button::new(button_txt));

                    if button.clicked() {
                        match sim_state.get() {
                            SimulationState::Paused => {
                                next_sim_state.set(SimulationState::Running);
                                runner
                                    .tx
                                    .send(RunnerReq::Resume)
                                    .expect("Could not send resume request");
                            }
                            _ => {
                                next_sim_state.set(SimulationState::Paused);
                                runner
                                    .tx
                                    .send(RunnerReq::Pause)
                                    .expect("Could not send pause request");
                            }
                        }
                    }
                });

                match sim_state.get() {
                    SimulationState::None => {}
                    _ => {
                        let button = ui.add_sized(
                            (ui.available_width(), 0.0),
                            egui::Button::new("Stop & reset simulation"),
                        );

                        if button.clicked() {
                            clear_screen(&mut commands, rects);
                            data.creatures.clear();
                            control_panel.can_create_sim = true;
                            data.ticks = 0;
                            runner
                                .tx
                                .send(RunnerReq::Pause)
                                .expect("Could not send pause request");
                            next_sim_state.set(SimulationState::Paused);
                            next_sim_state.set(SimulationState::None);
                        }
                    }
                }
            });

        ui.separator();
        ui.label(format!("Ticks: {}", data.ticks));
        ui.label(format!("FPS: {}", 1.0 / time.delta_seconds_f64()));

        ui.with_layout(Layout::bottom_up(egui::Align::Center), |ui| {
            let button = ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Main menu"));
            if button.clicked() {
                next_tab_state.set(TabState::MainMenu);
            }
        });
    });
}

pub fn initialize_world(
    mut shapes: ShapeCommands,
    mut data: ResMut<Simulation>,
    runner: Res<RunnerResource>,
) {
    runner
        .tx
        .send(RunnerReq::GetPositions)
        .expect("Could not send positions request");
    match runner.rx.recv().expect("Could not receive positions") {
        RunnerRes::Positions(p) => {
            data.creatures = p.creatures;
            data.food = p.food;
        }
        _ => unreachable!(),
    }

    render_world(&mut data, &mut shapes);
}

fn clear_screen(commands: &mut Commands, rects: Query<Entity, With<RectangleComponent>>) {
    for item in rects.iter() {
        commands.entity(item).despawn_recursive();
    }
}

pub fn run_simulation(
    mut data: ResMut<Simulation>,
    mut shapes: ShapeCommands,
    mut commands: Commands,
    rects: Query<Entity, With<RectangleComponent>>,
    runner: Res<RunnerResource>,
) {
    clear_screen(&mut commands, rects);

    runner
        .tx
        .send(RunnerReq::GetPositions)
        .expect("Could not send positions request");
    match runner.rx.recv().expect("Could not receive positions") {
        RunnerRes::Positions(p) => {
            data.creatures = p.creatures;
            data.food = p.food;
        }
        _ => unreachable!(),
    }

    render_world(&mut data, &mut shapes);
    data.ticks += 1;
}

fn convert_bottom_left_to_center_coords(pos: Vec2, dims: (f32, f32)) -> Vec2 {
    let half_width = dims.0 / 2.0;
    let half_height = dims.1 / 2.0;
    Vec2::new(pos.x - half_width, half_height - pos.y)
}

// (c.position.0 + 5.0 >= position.x && c.position.0 - 5.0 <= position.x)
//                     && (c.position.1 + 5.0 >= position.y && c.position.1 - 5.0 <= position.y)

pub fn inspect_creature(
    buttons: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    data: Res<Simulation>,
    mut inspect_net: EventWriter<InspectNet>,
    runner: Res<RunnerResource>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        let window = q_windows.single();
        if let Some(position) = window.cursor_position() {
            // let position =
            //     convert_bottom_left_to_center_coords(position, (window.width(), window.height()));
            if let Some(c) = data.creatures.iter().find(|c| {
                (c.position.0 + 5.0 >= position.x && c.position.0 - 5.0 <= position.x)
                    && (c.position.1 + 5.0 >= position.y && c.position.1 - 5.0 <= position.y)
            }) {
                runner
                    .tx
                    .send(RunnerReq::GetNet(c.id))
                    .expect("Could not send net request");
                match runner.rx.recv().expect("Could not receive net") {
                    RunnerRes::Net(n) => inspect_net.send(InspectNet(n.unwrap())),
                    _ => unreachable!(),
                };
            }
        }
    }
}

fn render_world(data: &mut Simulation, shapes: &mut ShapeCommands) {
    shapes.color = Color::hex(CREATURE_COLOR).unwrap();
    data.creatures.iter().for_each(|c| {
        // if speed > 0.0 && (movement_vec.x > 0.0 || movement_vec.y > 0.0) {
        //     // inspect_net.send(InspectNet(c.brain.clone()));
        //     // next_sim_state.set(SimulationState::Paused);
        //     shapes.color = Color::hex("eb4034").unwrap();
        //     shapes.transform.translation = Vec3::new(c.position.0, c.position.1, -1.0);
        //     shapes.circle(15.0);

        //     shapes.color = Color::hex("ffffff").unwrap();
        //     shapes.transform.translation = Vec3::new(c.position.0, c.position.1, -0.5);
        //     shapes.circle(10.0);
        // }

        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.window_dims,
        );
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes.rect(Vec2::new(CREATURE_DIM, CREATURE_DIM));
    });

    shapes.color = Color::hex(FOOD_COLOR).unwrap();
    data.food.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(Vec2::new(c.0, c.1), data.window_dims);
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes.rect(Vec2::new(CREATURE_DIM, CREATURE_DIM));
    });
}
