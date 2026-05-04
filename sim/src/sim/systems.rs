use std::thread;

use bevy::{
    camera::{visibility::RenderLayers, RenderTarget},
    ecs::schedule::ScheduleLabel,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{PrimaryWindow, WindowRef},
};
use bevy_egui::{
    egui,
    EguiContext, EguiMultipassSchedule,
};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SystemWindowContextPass;
use bevy_vector_shapes::prelude::*;

use super::sim::{Generate, Runner, RunnerReq, RunnerRes};

use crate::{net::resources::{InspectNet, InspectWindowState}, BaseNodes};

use super::resources::*;

const POISON_DIM: f32 = 5.0;
const CREATURE_COLOR: &str = "3686ff";
const FOOD_COLOR: &str = "54ff71";
const POISON_COLOR: &str = "ff3864";

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
    mut commands: Commands,
    mut system_window: ResMut<SystemWindow>,
) {
    let window = window_query.single().unwrap();
    control_panel.width = window.width().to_string();
    control_panel.height = window.height().to_string();
    data.window_dims = (window.width(), window.height());

    // Primary sim camera
    commands.spawn((Camera2d, SimCamera));

    // Spawn system window
    let window_entity = commands.spawn((
        Window {
            title: "System".to_string(),
            resolution: (520u32, 680u32).into(),
            ..default()
        },
    )).id();

    commands.spawn((
        Camera2d,
        RenderTarget::Window(WindowRef::Entity(window_entity)),
        EguiMultipassSchedule::new(SystemWindowContextPass),
        RenderLayers::layer(2),
        SystemWindowCamera,
    ));

    system_window.0 = Some(window_entity);
}

pub fn system_window(
    mut egui_ctx: Single<&mut EguiContext, With<SystemWindowCamera>>,
    mut control_panel: ResMut<ControlPanel>,
    base_nodes: Res<BaseNodes>,
    mut data: ResMut<Simulation>,
    mut next_sim_state: ResMut<NextState<SimulationState>>,
    sim_state: Res<State<SimulationState>>,
    mut commands: Commands,
    rects: Query<Entity, With<RectangleComponent>>,
    time: Res<Time>,
    runner: Res<RunnerResource>,
    mut config: ResMut<SimulationConfig>,
    stats: Res<SimulationStats>,
) {
    let ctx = egui_ctx.get_mut();

    egui::CentralPanel::default().show(ctx, |ui| {
        // ── Toolbar ─────────────────────────────────────────────────────
        ui.heading("Simulation");
        ui.separator();

        let mut apply_changes = false;

        egui::ScrollArea::vertical().show(ui, |ui| {
            // ── Sim state strip ──────────────────────────────────────────
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 8.0) / 2.0;
                ui.add_enabled_ui(control_panel.can_create_sim, |ui| {
                    if ui.add_sized((w, 36.0), egui::Button::new("🚀 Create")).clicked() {
                        let world_w = control_panel.width.parse::<f32>().unwrap_or(data.window_dims.0);
                        let world_h = control_panel.height.parse::<f32>().unwrap_or(data.window_dims.1);
                        data.world_dim = (world_w, world_h);
                        runner
                            .tx
                            .send(RunnerReq::Generate(Generate {
                                num_creatures: control_panel.initial_num_creatures.parse().unwrap(),
                                dims: (world_w, world_h),
                                input_nodes: base_nodes.input_nodes.clone(),
                                output_nodes: base_nodes.output_nodes.clone(),
                                config: config.clone(),
                            }))
                            .expect("Could not send generate request");
                        control_panel.can_create_sim = false;
                    }
                });
                ui.add_enabled_ui(!control_panel.can_create_sim, |ui| {
                    let (icon, label) = match sim_state.get() {
                        SimulationState::None    => ("⏵", "Start"),
                        SimulationState::Paused  => ("⏵", "Resume"),
                        SimulationState::Running => ("⏸", "Pause"),
                    };
                    if ui.add_sized((w, 36.0), egui::Button::new(format!("{icon} {label}"))).clicked() {
                        match sim_state.get() {
                            SimulationState::Paused => {
                                next_sim_state.set(SimulationState::Running);
                                runner.tx.send(RunnerReq::Resume).expect("send resume");
                            }
                            _ => {
                                next_sim_state.set(SimulationState::Paused);
                                runner.tx.send(RunnerReq::Pause).expect("send pause");
                            }
                        }
                    }
                });
            });

            if !control_panel.can_create_sim && *sim_state.get() == SimulationState::None {
                ui.label(egui::RichText::new("Generating…").italics().color(egui::Color32::YELLOW));
            }

            if *sim_state.get() != SimulationState::None {
                if ui.add_sized((ui.available_width(), 28.0), egui::Button::new("⏹ Stop & Reset")).clicked() {
                    clear_screen(&mut commands, rects);
                    data.creatures.clear();
                    control_panel.can_create_sim = true;
                    data.ticks = 0;
                    runner.tx.send(RunnerReq::Pause).expect("send pause");
                    next_sim_state.set(SimulationState::Paused);
                    next_sim_state.set(SimulationState::None);
                }
            }

            ui.separator();

            // ── Setup ────────────────────────────────────────────────────
            egui::Grid::new("setup_grid")
                .num_columns(4)
                .spacing([6.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Creatures");
                    ui.add_sized((80.0, 20.0), egui::TextEdit::singleline(&mut control_panel.initial_num_creatures));
                    ui.label("Width");
                    ui.add_sized((80.0, 20.0), egui::TextEdit::singleline(&mut control_panel.width));
                    ui.end_row();
                    ui.label("");
                    ui.label("");
                    ui.label("Height");
                    ui.add_sized((80.0, 20.0), egui::TextEdit::singleline(&mut control_panel.height));
                    ui.end_row();
                });

            ui.separator();

            // ── Stats — two-column wide grid ─────────────────────────────
            section_header(ui, "Live Stats");

            egui::Grid::new("stats_grid")
                .num_columns(4)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    stat_row4(ui, "Ticks", &data.ticks.to_string(),
                                  "FPS",   &format!("{:.0}", 1.0 / time.delta_secs_f64()));
                    stat_row4(ui, "Population", &format!("{}/{}", stats.current_population, stats.target_population),
                                  "Survival",   &format!("{:.1}%", stats.survival_rate * 100.0));
                    stat_row4(ui, "Births", &stats.births_this_tick.to_string(),
                                  "Deaths", &stats.deaths_this_tick.to_string());
                    stat_row4(ui, "Total spawned", &stats.total_spawned.to_string(),
                                  "Food",          &stats.food_count.to_string());
                    stat_row4(ui, "Poison", &stats.poison_count.to_string(),
                                  "Eaten/tick",    &stats.food_eaten_this_tick.to_string());
                    stat_row4(ui, "Total eaten", &stats.total_food_eaten.to_string(),
                                  "Avg energy",    &format!("{:.1}", stats.avg_energy));
                    stat_row4(ui, "Avg age", &format!("{:.0}", stats.avg_age),
                                  "Avg size",      &format!("{:.1}", stats.avg_size));
                    if let Some(id) = stats.selected_creature_id {
                        ui.label(egui::RichText::new("Selected").weak());
                        ui.label(id.to_string());
                        ui.label("");
                        ui.label("");
                        ui.end_row();
                    }
                });

            ui.separator();

            // ── Config — four-column grid (label, val, label, val) ───────
            section_header(ui, "Energy");
            config_grid4(ui, "energy", |ui| {
                ui.label("Start"); ui.add(egui::DragValue::new(&mut config.start_energy).speed(1.0).range(0.0..=200.0));
                ui.label("Max");   ui.add(egui::DragValue::new(&mut config.max_energy).speed(1.0).range(1.0..=200.0));
                ui.end_row();
                ui.label("Base cost"); ui.add(egui::DragValue::new(&mut config.base_energy_cost).speed(0.01).range(0.0..=1.0));
                ui.label("Move cost"); ui.add(egui::DragValue::new(&mut config.move_energy_cost).speed(0.01).range(0.0..=1.0));
                ui.end_row();
                ui.label("Min move"); ui.add(egui::DragValue::new(&mut config.min_move_energy).speed(1.0).range(0.0..=100.0));
                ui.label("Full spd"); ui.add(egui::DragValue::new(&mut config.full_speed_energy).speed(1.0).range(0.0..=200.0));
                ui.end_row();
            });

            section_header(ui, "Creatures");
            config_grid4(ui, "creatures", |ui| {
                ui.label("Max age");  ui.add(egui::DragValue::new(&mut config.max_age).speed(100.0).range(100..=50000));
                ui.label("Min size"); ui.add(egui::DragValue::new(&mut config.min_creature_size).speed(0.1).range(1.0..=20.0));
                ui.end_row();
                ui.label("Max size"); ui.add(egui::DragValue::new(&mut config.max_creature_size).speed(0.1).range(1.0..=50.0));
                ui.label(""); ui.label("");
                ui.end_row();
            });

            section_header(ui, "Reproduction");
            config_grid4(ui, "repro", |ui| {
                ui.label("Threshold");  ui.add(egui::DragValue::new(&mut config.mate_threshold).speed(0.01).range(0.0..=1.0));
                ui.label("Cooldown");   ui.add(egui::DragValue::new(&mut config.mate_cooldown_ticks).speed(1.0).range(0..=500));
                ui.end_row();
                ui.label("Min energy"); ui.add(egui::DragValue::new(&mut config.min_mate_energy).speed(1.0).range(0.0..=200.0));
                ui.label("Mate cost");  ui.add(egui::DragValue::new(&mut config.mate_energy_cost).speed(1.0).range(0.0..=100.0));
                ui.end_row();
                ui.label("Atmp cost");  ui.add(egui::DragValue::new(&mut config.mate_attempt_cost).speed(0.01).range(0.0..=1.0));
                ui.label("Child E");    ui.add(egui::DragValue::new(&mut config.child_energy).speed(1.0).range(0.0..=100.0));
                ui.end_row();
                ui.label("Mut rate");   ui.add(egui::DragValue::new(&mut config.mutation_rate).speed(0.01).range(0.0..=1.0));
                ui.label("Mut amt");    ui.add(egui::DragValue::new(&mut config.mutation_amount).speed(0.01).range(0.0..=5.0));
                ui.end_row();
                ui.label("Max births"); ui.add(egui::DragValue::new(&mut config.max_births_per_tick).speed(1.0).range(1..=100));
                ui.label(""); ui.label("");
                ui.end_row();
            });

            section_header(ui, "Food");
            config_grid4(ui, "food", |ui| {
                ui.label("Per creature"); ui.add(egui::DragValue::new(&mut config.food_per_creature).speed(1.0).range(1..=50));
                ui.label("Spawn mult");   ui.add(egui::DragValue::new(&mut config.food_spawn_multiplier).speed(0.1).range(0.1..=10.0));
                ui.end_row();
                ui.label("Min size"); ui.add(egui::DragValue::new(&mut config.min_food_size).speed(0.1).range(1.0..=20.0));
                ui.label("Max size"); ui.add(egui::DragValue::new(&mut config.max_food_size).speed(0.1).range(1.0..=50.0));
                ui.end_row();
                ui.label("E/size");     ui.add(egui::DragValue::new(&mut config.food_energy_per_size).speed(0.5).range(1.0..=50.0));
                ui.label("Ticks/size"); ui.add(egui::DragValue::new(&mut config.food_eat_ticks_per_size).speed(0.1).range(0.0..=20.0));
                ui.end_row();
                ui.label("Act cost");   ui.add(egui::DragValue::new(&mut config.eat_action_cost).speed(0.1).range(0.0..=10.0));
                ui.label("Base ticks"); ui.add(egui::DragValue::new(&mut config.eat_action_base_ticks).speed(1.0).range(0..=50));
                ui.end_row();
                ui.label("Atmp cost"); ui.add(egui::DragValue::new(&mut config.eat_attempt_cost).speed(0.01).range(0.0..=1.0));
                ui.label(""); ui.label("");
                ui.end_row();
            });

            section_header(ui, "Poison");
            config_grid4(ui, "poison", |ui| {
                ui.label("Per creature"); ui.add(egui::DragValue::new(&mut config.poison_per_creature).speed(1.0).range(1..=50));
                ui.label("Damage");       ui.add(egui::DragValue::new(&mut config.poison_damage).speed(1.0).range(0.0..=200.0));
                ui.end_row();
            });

            section_header(ui, "Spawning");
            ui.checkbox(&mut config.creature_spawning_enabled, "Creature spawning enabled");

            ui.add_space(8.0);
            if ui.add_sized((ui.available_width(), 32.0), egui::Button::new("Apply Config")).clicked() {
                apply_changes = true;
            }
            ui.add_space(4.0);
        });

        if apply_changes && *sim_state.get() != SimulationState::None {
            runner
                .tx
                .send(RunnerReq::UpdateConfig(config.clone()))
                .expect("Could not send config update");
        }
    });
}

fn section_header(ui: &mut egui::Ui, label: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(label).strong().size(12.0));
    ui.separator();
}

fn stat_row4(ui: &mut egui::Ui, l1: &str, v1: &str, l2: &str, v2: &str) {
    ui.label(egui::RichText::new(l1).weak());
    ui.label(v1);
    ui.label(egui::RichText::new(l2).weak());
    ui.label(v2);
    ui.end_row();
}

fn config_grid4(ui: &mut egui::Ui, id: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(4)
        .spacing([6.0, 4.0])
        .striped(true)
        .min_col_width(60.0)
        .show(ui, add_contents);
}


pub fn poll_generated_world(
    mut shapes: ShapeCommands,
    mut data: ResMut<Simulation>,
    mut next_sim_state: ResMut<NextState<SimulationState>>,
    sim_state: Res<State<SimulationState>>,
    control_panel: Res<ControlPanel>,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    mut pending_net: ResMut<PendingNet>,
    mut camera: Query<&mut Transform, With<SimCamera>>,
    mut view_state: ResMut<ViewState>,
) {
    if control_panel.can_create_sim || *sim_state.get() != SimulationState::None {
        return;
    }

    while let Ok(res) = runner.rx.try_recv() {
        match res {
            RunnerRes::Positions(p) => {
                data.creatures = p.creatures;
                data.food = p.food;
                data.poison = p.poison;
                *stats = p.stats;

                // Reset camera to fit the entire world in view
                if let Ok(mut cam) = camera.single_mut() {
                    let (world_w, world_h) = data.world_dim;
                    let (win_w, win_h) = data.window_dims;
                    if world_w > 0.0 && win_w > 0.0 {
                        let scale = (world_w / win_w).max(world_h / win_h);
                        cam.scale = Vec3::splat(scale);
                        view_state.zoom = scale;
                    }
                    cam.translation = Vec3::ZERO;
                    view_state.pan = Vec2::ZERO;
                }

                render_world(&mut data, &mut shapes, None);
                next_sim_state.set(SimulationState::Paused);
                return;
            }
            RunnerRes::Net(Some(n)) => pending_net.0 = Some(n),
            RunnerRes::Net(None) => {}
        }
    }
}

fn clear_screen(commands: &mut Commands, rects: Query<Entity, With<RectangleComponent>>) {
    for item in rects.iter() {
        commands.entity(item).despawn();
    }
}

pub fn run_simulation(
    mut data: ResMut<Simulation>,
    mut shapes: ShapeCommands,
    mut commands: Commands,
    rects: Query<Entity, With<RectangleComponent>>,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    mut pending_net: ResMut<PendingNet>,
) {
    let mut positions = None;
    while let Ok(res) = runner.rx.try_recv() {
        match res {
            RunnerRes::Positions(p) => positions = Some(p),
            RunnerRes::Net(Some(n)) => pending_net.0 = Some(n),
            RunnerRes::Net(None) => {}
        }
    }

    let Some(p) = positions else {
        return;
    };

    let selected_id = stats.selected_creature_id;
    clear_screen(&mut commands, rects);
    data.creatures = p.creatures;
    data.food = p.food;
    data.poison = p.poison;
    *stats = p.stats;
    stats.selected_creature_id = selected_id;

    render_world(&mut data, &mut shapes, selected_id);
    data.ticks += 1;
}

pub fn camera_controls(
    mut camera: Query<&mut Transform, With<SimCamera>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut view_state: ResMut<ViewState>,
    data: Res<Simulation>,
    time: Res<Time>,
) {
    let Ok(mut transform) = camera.single_mut() else { return };

    let mut zoom = transform.scale.x;

    // Scroll wheel to zoom
    if scroll.delta.y != 0.0 {
        let factor = if scroll.delta.y > 0.0 { 0.85 } else { 1.0 / 0.85 };
        zoom *= factor;
    }

    // +/- keys to zoom
    if keyboard.pressed(KeyCode::Equal) || keyboard.pressed(KeyCode::NumpadAdd) {
        zoom *= 1.0 - 1.5 * time.delta_secs();
    }
    if keyboard.pressed(KeyCode::Minus) || keyboard.pressed(KeyCode::NumpadSubtract) {
        zoom *= 1.0 + 1.5 * time.delta_secs();
    }

    zoom = zoom.clamp(0.01, 100.0);

    // WASD / arrow keys to pan
    let pan_speed = 400.0 * zoom * time.delta_secs();
    if keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA) {
        transform.translation.x -= pan_speed;
    }
    if keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD) {
        transform.translation.x += pan_speed;
    }
    if keyboard.pressed(KeyCode::ArrowUp) || keyboard.pressed(KeyCode::KeyW) {
        transform.translation.y += pan_speed;
    }
    if keyboard.pressed(KeyCode::ArrowDown) || keyboard.pressed(KeyCode::KeyS) {
        transform.translation.y -= pan_speed;
    }

    // Middle-mouse drag to pan
    if mouse_button.pressed(MouseButton::Middle) {
        transform.translation.x -= mouse_motion.delta.x * zoom;
        transform.translation.y += mouse_motion.delta.y * zoom;
    }

    // R to reset camera to fit world
    if keyboard.just_pressed(KeyCode::KeyR) {
        let (world_w, world_h) = data.world_dim;
        let (win_w, win_h) = data.window_dims;
        zoom = if world_w > 0.0 && win_w > 0.0 {
            (world_w / win_w).max(world_h / win_h)
        } else {
            1.0
        };
        transform.translation = Vec3::ZERO;
    }

    transform.scale = Vec3::splat(zoom);
    view_state.zoom = zoom;
    view_state.pan = transform.translation.truncate();
}

fn convert_bottom_left_to_center_coords(pos: Vec2, dims: (f32, f32)) -> Vec2 {
    let half_width = dims.0 / 2.0;
    let half_height = dims.1 / 2.0;
    Vec2::new(pos.x - half_width, half_height - pos.y)
}

pub fn inspect_creature(
    buttons: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    data: Res<Simulation>,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    view_state: Res<ViewState>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let window = q_windows.single().unwrap();
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    // Convert cursor (window top-left origin) → Bevy world space → sim space
    let (win_w, win_h) = data.window_dims;
    let zoom = view_state.zoom;
    let pan = view_state.pan;
    let world_x = (cursor.x - win_w * 0.5) * zoom + pan.x;
    let world_y = (win_h * 0.5 - cursor.y) * zoom + pan.y;
    let (world_w, world_h) = data.world_dim;
    let sim_x = world_x + world_w * 0.5;
    let sim_y = world_h * 0.5 - world_y;

    const SEARCH_RADIUS: f32 = 12.0;
    let radius_world = SEARCH_RADIUS * zoom;
    let nearest = data.creatures.iter()
        .filter_map(|c| {
            let d2 = (c.position.0 - sim_x).powi(2) + (c.position.1 - sim_y).powi(2);
            (d2 <= radius_world * radius_world).then_some((c, d2))
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((c, _)) = nearest {
        stats.selected_creature_id = Some(c.id);
        runner.tx.send(RunnerReq::GetNet(c.id)).expect("send net request");
    }
}

pub fn deliver_inspect_net(
    mut pending: ResMut<PendingNet>,
    mut inspect_net: MessageWriter<InspectNet>,
    mut next_inspect_state: ResMut<NextState<InspectWindowState>>,
    inspect_state: Res<State<InspectWindowState>>,
) {
    if let Some(n) = pending.0.take() {
        if *inspect_state.get() != InspectWindowState::Display {
            next_inspect_state.set(InspectWindowState::Display);
        }
        inspect_net.write(InspectNet(n));
    }
}

fn render_world(data: &mut Simulation, shapes: &mut ShapeCommands, selected_id: Option<usize>) {
    shapes.thickness = 0.0;
    data.creatures.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.world_dim,
        );
        if selected_id == Some(c.id) {
            shapes.color = Color::from(Srgba::new(1.0, 1.0, 1.0, 0.9));
            shapes.thickness = 2.0;
            shapes.transform.translation = Vec3::new(coords.x, coords.y, 1.0);
            shapes.circle(c.size + 5.0);
            shapes.thickness = 0.0;
        }
        shapes.color = Color::from(Srgba::hex(CREATURE_COLOR).unwrap());
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes.rect(Vec2::new(c.size, c.size));
    });

    shapes.color = Color::from(Srgba::hex(FOOD_COLOR).unwrap());
    data.food.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.world_dim,
        );
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes.rect(Vec2::new(c.size, c.size));
    });

    shapes.color = Color::from(Srgba::hex(POISON_COLOR).unwrap());
    data.poison.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(Vec2::new(c.0, c.1), data.world_dim);
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes.rect(Vec2::new(POISON_DIM, POISON_DIM));
    });
}


