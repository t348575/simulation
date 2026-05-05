use std::thread;

use bevy::{
    camera::{visibility::RenderLayers, RenderTarget},
    ecs::schedule::ScheduleLabel,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{PrimaryWindow, WindowClosed, WindowRef},
};
use bevy_egui::{egui, EguiContext, EguiMultipassSchedule};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SystemWindowContextPass;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CreatureExplorerContextPass;
use bevy_vector_shapes::prelude::*;

use super::sim::{BasicCreature, Generate, Runner, RunnerReq, RunnerRes};

use crate::{
    net::resources::{InspectNet, InspectWindowState},
    BaseNodes,
};

use super::resources::*;

const POISON_DIM: f32 = 5.0;
const CREATURE_COLOR: &str = "3686ff";
const FOOD_COLOR: &str = "54ff71";
const POISON_COLOR: &str = "ff3864";
const OBSTACLE_COLOR: &str = "555555";

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

    // Only set width/height from window if they are "0" or empty
    if control_panel.width == "0" || control_panel.width.is_empty() {
        control_panel.width = window.width().to_string();
    }
    if control_panel.height == "0" || control_panel.height.is_empty() {
        control_panel.height = window.height().to_string();
    }

    data.window_dims = (window.width(), window.height());

    // Primary sim camera
    commands.spawn((Camera2d, SimCamera));

    // Spawn system window
    let window_entity = commands
        .spawn((Window {
            title: "System".to_string(),
            resolution: (520u32, 680u32).into(),
            ..default()
        },))
        .id();

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
    rects: Query<
        Entity,
        Or<(
            With<RectangleComponent>,
            With<DiscComponent>,
            With<RegularPolygonComponent>,
        )>,
    >,
    time: Res<Time>,
    runner: Res<RunnerResource>,
    mut config: ResMut<SimulationConfig>,
    stats: Res<SimulationStats>,
    mut explorer: ResMut<CreatureExplorer>,
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
                    if ui
                        .add_sized((w, 36.0), egui::Button::new("🚀 Create"))
                        .clicked()
                    {
                        let world_w = control_panel
                            .width
                            .parse::<f32>()
                            .unwrap_or(data.window_dims.0);
                        let world_h = control_panel
                            .height
                            .parse::<f32>()
                            .unwrap_or(data.window_dims.1);
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
                        SimulationState::None => ("⏵", "Start"),
                        SimulationState::Paused => ("⏵", "Resume"),
                        SimulationState::Running => ("⏸", "Pause"),
                    };
                    if ui
                        .add_sized((w, 36.0), egui::Button::new(format!("{icon} {label}")))
                        .clicked()
                    {
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
                ui.label(
                    egui::RichText::new("Generating…")
                        .italics()
                        .color(egui::Color32::YELLOW),
                );
            }

            if *sim_state.get() != SimulationState::None {
                if ui
                    .add_sized(
                        (ui.available_width(), 28.0),
                        egui::Button::new("⏹ Stop & Reset"),
                    )
                    .clicked()
                {
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

            // ── Tools ────────────────────────────────────────────────────
            ui.horizontal(|ui| {
                let can_explore = !control_panel.can_create_sim
                    && *sim_state.get() != SimulationState::None;
                ui.add_enabled_ui(can_explore, |ui| {
                    if ui
                        .add_sized(
                            (ui.available_width(), 26.0),
                            egui::Button::new("👥 Creature Explorer"),
                        )
                        .on_hover_text("Open creature browser: snapshot, sort, inspect brains")
                        .clicked()
                    {
                        explorer.open = true;
                        explorer.snapshot = data.creatures.clone();
                        open_creature_explorer(&mut commands, &mut explorer);
                    }
                });
            });

            ui.separator();

            // ── Save / Load ──────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label("File");
                ui.add_sized(
                    (ui.available_width() - 8.0, 20.0),
                    egui::TextEdit::singleline(&mut control_panel.save_path),
                );
            });
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 8.0) / 2.0;
                let can_save = !control_panel.can_create_sim && *sim_state.get() != SimulationState::None;
                ui.add_enabled_ui(can_save, |ui| {
                    if ui
                        .add_sized((w, 26.0), egui::Button::new("💾 Save"))
                        .on_hover_text("Save full simulation (world, food, poison, obstacles, all creatures)")
                        .clicked()
                    {
                        let path = std::path::PathBuf::from(&control_panel.save_path);
                        runner.tx.send(RunnerReq::SaveSim(path)).expect("send save");
                        next_sim_state.set(SimulationState::Paused);
                    }
                });
                if ui
                    .add_sized((w, 26.0), egui::Button::new("📂 Load"))
                    .on_hover_text("Load full simulation (replaces world + creatures)")
                    .clicked()
                {
                    let path = std::path::PathBuf::from(&control_panel.save_path);
                    clear_screen(&mut commands, rects);
                    data.creatures.clear();
                    runner.tx.send(RunnerReq::LoadSim(path)).expect("send load");
                    control_panel.can_create_sim = false;
                    next_sim_state.set(SimulationState::Paused);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Top %").on_hover_text("Percentage of oldest creatures to save (creatures-only export)");
                ui.add_sized(
                    (60.0, 20.0),
                    egui::TextEdit::singleline(&mut control_panel.save_top_percent),
                );
            });
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 8.0) / 2.0;
                let can_save = !control_panel.can_create_sim && *sim_state.get() != SimulationState::None;
                ui.add_enabled_ui(can_save, |ui| {
                    if ui
                        .add_sized((w, 26.0), egui::Button::new("💾 Save Top X%"))
                        .on_hover_text("Save only the oldest X% of creatures (brains only — no world/food)")
                        .clicked()
                    {
                        let path = std::path::PathBuf::from(&control_panel.save_path);
                        let pct = control_panel
                            .save_top_percent
                            .trim()
                            .parse::<f32>()
                            .unwrap_or(100.0)
                            .clamp(0.0, 100.0);
                        runner.tx.send(RunnerReq::SaveCreatures(path, pct)).expect("send save creatures");
                        next_sim_state.set(SimulationState::Paused);
                    }
                });
                ui.add_enabled_ui(can_save, |ui| {
                    if ui
                        .add_sized((w, 26.0), egui::Button::new("📂 Load Creatures"))
                        .on_hover_text("Replace current creatures with creatures from file (preserves current world/food/config)")
                        .clicked()
                    {
                        let path = std::path::PathBuf::from(&control_panel.save_path);
                        runner.tx.send(RunnerReq::LoadCreatures(path)).expect("send load creatures");
                        next_sim_state.set(SimulationState::Paused);
                    }
                });
            });
            if !control_panel.status.is_empty() {
                ui.label(egui::RichText::new(&control_panel.status).italics().weak());
            }

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Config");
                ui.add_sized(
                    (ui.available_width() - 8.0, 20.0),
                    egui::TextEdit::singleline(&mut control_panel.config_path),
                );
            });
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 8.0) / 3.0;
                if ui
                    .add_sized((w, 26.0), egui::Button::new("💾 Save Config"))
                    .on_hover_text("Save current simulation configuration to JSON")
                    .clicked()
                {
                    let path = std::path::PathBuf::from(&control_panel.config_path);
                    match serde_json::to_string_pretty(&*config)
                        .map_err(|e| format!("serialize: {e}"))
                        .and_then(|s| std::fs::write(&path, s).map_err(|e| format!("write: {e}")))
                    {
                        Ok(()) => {
                            control_panel.status =
                                format!("Saved config → {}", path.display());
                        }
                        Err(e) => {
                            control_panel.status = format!("Save config failed: {e}");
                        }
                    }
                }
                if ui
                    .add_sized((w, 26.0), egui::Button::new("📂 Load Config"))
                    .on_hover_text("Load simulation configuration from JSON and apply")
                    .clicked()
                {
                    let path = std::path::PathBuf::from(&control_panel.config_path);
                    match std::fs::read_to_string(&path)
                        .map_err(|e| format!("read: {e}"))
                        .and_then(|s| {
                            serde_json::from_str::<SimulationConfig>(&s)
                                .map_err(|e| format!("parse: {e}"))
                        }) {
                        Ok(loaded) => {
                            *config = loaded.clone();
                            let _ = runner.tx.send(RunnerReq::UpdateConfig(loaded));
                            save_cached_config(&config);
                            control_panel.status =
                                format!("Loaded config ← {}", path.display());
                        }
                        Err(e) => {
                            control_panel.status = format!("Load config failed: {e}");
                        }
                    }
                }
                if ui
                    .add_sized((w, 26.0), egui::Button::new("↺ Reset"))
                    .on_hover_text("Reset config to defaults and clear cached config")
                    .clicked()
                {
                    *config = SimulationConfig::default();
                    let _ = runner.tx.send(RunnerReq::UpdateConfig(config.clone()));
                    save_cached_config(&config);
                    control_panel.status = "Config reset to defaults".to_owned();
                }
            });

            ui.separator();

            // ── Setup ────────────────────────────────────────────────────
            egui::Grid::new("setup_grid")
                .num_columns(4)
                .spacing([6.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Creatures");
                    ui.add_sized(
                        (80.0, 20.0),
                        egui::TextEdit::singleline(&mut control_panel.initial_num_creatures),
                    );
                    ui.label("Width");
                    ui.add_sized(
                        (80.0, 20.0),
                        egui::TextEdit::singleline(&mut control_panel.width),
                    );
                    ui.end_row();
                    ui.label("");
                    ui.label("");
                    ui.label("Height");
                    ui.add_sized(
                        (80.0, 20.0),
                        egui::TextEdit::singleline(&mut control_panel.height),
                    );
                    ui.end_row();
                });

            ui.separator();

            // ── Stats — two-column wide grid ─────────────────────────────
            section_header(ui, "World");
            egui::Grid::new("world_grid")
                .num_columns(4)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    stat_row4_tip(
                        ui,
                        "Tick", &stats.ticks.to_string(), "Simulation steps elapsed",
                        "FPS", &format!("{:.0}", 1.0 / time.delta_secs_f64()), "",
                    );
                    stat_row4_tip(
                        ui,
                        "Population",
                        &format!("{}/{}", stats.current_population, stats.target_population),
                        "Alive creatures / starting count",
                        "Alive %",
                        &format!(
                            "{:.1}%",
                            if stats.target_population > 0 {
                                stats.current_population as f32 / stats.target_population as f32 * 100.0
                            } else { 0.0 }
                        ),
                        "Alive now / starting (target) population",
                    );
                    stat_row4_tip(
                        ui,
                        "Food", &stats.food_count.to_string(), "Food items currently in world",
                        "Poison", &stats.poison_count.to_string(), "Poison items currently in world",
                    );
                    stat_row4_tip(
                        ui,
                        "Generation", &stats.generation.to_string(), "Number of extinction restarts completed (0 = original run)",
                        "", "", "",
                    );
                });

            section_header(ui, "Population (cumulative)");
            egui::Grid::new("pop_total_grid")
                .num_columns(4)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    stat_row4_tip(
                        ui,
                        "Total spawned", &stats.total_spawned.to_string(),
                        "All creatures ever created (initial spawn + extinction restarts + random spawns)",
                        "Born via mating", &stats.total_births.to_string(),
                        "Creatures produced by two parents mating",
                    );
                    stat_row4_tip(
                        ui,
                        "Total deaths", &stats.total_deaths.to_string(), "",
                        "Net (births − deaths)",
                        &(stats.total_births as i64 - stats.total_deaths as i64).to_string(),
                        "Positive = births outpace deaths",
                    );
                });

            section_header(ui, "Recent activity");
            egui::Grid::new("recent_grid")
                .num_columns(4)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    stat_row4_tip(
                        ui,
                        "Births/tick", &format!("{:.2}", stats.birth_rate_avg),
                        "Rolling average births per tick over the last 120 ticks",
                        "Deaths/tick", &format!("{:.2}", stats.death_rate_avg),
                        "Rolling average deaths per tick over the last 120 ticks",
                    );
                    stat_row4(
                        ui,
                        "Births this tick",
                        &stats.births_this_tick.to_string(),
                        "Deaths this tick",
                        &stats.deaths_this_tick.to_string(),
                    );
                });

            section_header(ui, "Feeding");
            egui::Grid::new("eat_grid")
                .num_columns(4)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    stat_row4_tip(
                        ui,
                        "Total eaten", &stats.total_food_eaten.to_string(), "",
                        "Eaten/tick", &format!("{:.2}", stats.eat_rate_avg),
                        "Rolling average food items eaten per tick over the last 120 ticks",
                    );
                    stat_row4(
                        ui,
                        "Eaten this tick",
                        &stats.food_eaten_this_tick.to_string(),
                        "",
                        "",
                    );
                });

            section_header(ui, "Averages");
            egui::Grid::new("averages_grid")
                .num_columns(4)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    stat_row4_tip(
                        ui,
                        "Avg energy", &format!("{:.1}", stats.avg_energy),
                        "Mean energy across all alive creatures (max energy set in config)",
                        "Avg age", &format!("{:.0}", stats.avg_age),
                        "Mean age in simulation ticks",
                    );
                    stat_row4_tip(ui, "Avg size", &format!("{:.1}", stats.avg_size),
                        "Mean visual size; scales with energy relative to max energy",
                        "", "", "");
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
                ui.label("Start");
                ui.add(
                    egui::DragValue::new(&mut config.start_energy)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
                ui.label("Max");
                ui.add(
                    egui::DragValue::new(&mut config.max_energy)
                        .speed(1.0)
                        .range(1.0..=200.0),
                );
                ui.end_row();
                ui.label("Base cost").on_hover_text("Energy lost per tick regardless of action");
                ui.add(
                    egui::DragValue::new(&mut config.base_energy_cost)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.label("Move cost").on_hover_text("Additional energy lost per unit of speed");
                ui.add(
                    egui::DragValue::new(&mut config.move_energy_cost)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.end_row();
                ui.label("Min move").on_hover_text("Energy below which a creature cannot move at all");
                ui.add(
                    egui::DragValue::new(&mut config.min_move_energy)
                        .speed(1.0)
                        .range(0.0..=100.0),
                );
                ui.label("Full spd").on_hover_text("Energy level at which movement speed is fully uncapped");
                ui.add(
                    egui::DragValue::new(&mut config.full_speed_energy)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
                ui.end_row();
            });

            section_header(ui, "Creatures");
            config_grid4(ui, "creatures", |ui| {
                ui.label("Ignore max age");
                ui.add(
                    egui::Checkbox::new(&mut config.ignore_max_age, "")
                );
                ui.label("Max age");
                ui.add(
                    egui::DragValue::new(&mut config.max_age)
                        .speed(100.0)
                        .range(100..=50000),
                );
                ui.label("Min size");
                ui.add(
                    egui::DragValue::new(&mut config.min_creature_size)
                        .speed(0.1)
                        .range(1.0..=20.0),
                );
                ui.end_row();
                ui.label("Max size");
                ui.add(
                    egui::DragValue::new(&mut config.max_creature_size)
                        .speed(0.1)
                        .range(1.0..=50.0),
                );
                ui.label("");
                ui.label("");
                ui.end_row();
            });

            section_header(ui, "Reproduction");
            config_grid4(ui, "repro", |ui| {
                ui.label("Threshold");
                ui.add(
                    egui::DragValue::new(&mut config.mate_threshold)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.label("Cooldown");
                ui.add(
                    egui::DragValue::new(&mut config.mate_cooldown_ticks)
                        .speed(1.0)
                        .range(0..=500),
                );
                ui.end_row();
                ui.label("Min energy");
                ui.add(
                    egui::DragValue::new(&mut config.min_mate_energy)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
                ui.label("Mate cost");
                ui.add(
                    egui::DragValue::new(&mut config.mate_energy_cost)
                        .speed(1.0)
                        .range(0.0..=100.0),
                );
                ui.end_row();
                ui.label("Atmp cost").on_hover_text("Energy lost per tick while the creature is trying to mate");
                ui.add(
                    egui::DragValue::new(&mut config.mate_attempt_cost)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.label("Child E").on_hover_text("Energy given to each newborn creature");
                ui.add(
                    egui::DragValue::new(&mut config.child_energy)
                        .speed(1.0)
                        .range(0.0..=100.0),
                );
                ui.end_row();
                ui.label("Mut rate").on_hover_text("Probability (0–1) that each brain weight mutates per birth");
                ui.add(
                    egui::DragValue::new(&mut config.mutation_rate)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.label("Mut amt").on_hover_text("Max magnitude of weight change when a mutation occurs");
                ui.add(
                    egui::DragValue::new(&mut config.mutation_amount)
                        .speed(0.01)
                        .range(0.0..=5.0),
                );
                ui.end_row();
                ui.label("Max births").on_hover_text("Hard cap on new creatures born per simulation tick");
                ui.add(
                    egui::DragValue::new(&mut config.max_births_per_tick)
                        .speed(1.0)
                        .range(1..=100),
                );
                ui.label("");
                ui.label("");
                ui.end_row();
            });

            section_header(ui, "Food");
            config_grid4(ui, "food", |ui| {
                ui.label("Per creature");
                ui.add(
                    egui::DragValue::new(&mut config.food_per_creature)
                        .speed(1.0)
                        .range(1..=50),
                );
                ui.label("Spawn mult").on_hover_text("Multiplier on the base food spawn rate (food_per_creature × population)");
                ui.add(
                    egui::DragValue::new(&mut config.food_spawn_multiplier)
                        .speed(0.1)
                        .range(0.1..=10.0),
                );
                ui.end_row();
                ui.label("Min size");
                ui.add(
                    egui::DragValue::new(&mut config.min_food_size)
                        .speed(0.1)
                        .range(1.0..=20.0),
                );
                ui.label("Max size");
                ui.add(
                    egui::DragValue::new(&mut config.max_food_size)
                        .speed(0.1)
                        .range(1.0..=50.0),
                );
                ui.end_row();
                ui.label("E/size").on_hover_text("Energy gained per unit of food size when the food is eaten");
                ui.add(
                    egui::DragValue::new(&mut config.food_energy_per_size)
                        .speed(0.5)
                        .range(1.0..=50.0),
                );
                ui.label("Ticks/size").on_hover_text("Eating duration added per unit of food size (larger food takes longer)");
                ui.add(
                    egui::DragValue::new(&mut config.food_eat_ticks_per_size)
                        .speed(0.1)
                        .range(0.0..=20.0),
                );
                ui.end_row();
                ui.label("Act cost").on_hover_text("Energy deducted when a creature starts eating a food item");
                ui.add(
                    egui::DragValue::new(&mut config.eat_action_cost)
                        .speed(0.1)
                        .range(0.0..=10.0),
                );
                ui.label("Base ticks").on_hover_text("Minimum ticks to finish eating, before size scaling is applied");
                ui.add(
                    egui::DragValue::new(&mut config.eat_action_base_ticks)
                        .speed(1.0)
                        .range(0..=50),
                );
                ui.end_row();
                ui.label("Atmp cost").on_hover_text("Energy lost per tick while the creature is trying to eat");
                ui.add(
                    egui::DragValue::new(&mut config.eat_attempt_cost)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.label("");
                ui.label("");
                ui.end_row();
            });

            section_header(ui, "Poison");
            config_grid4(ui, "poison", |ui| {
                ui.label("Per creature");
                ui.add(
                    egui::DragValue::new(&mut config.poison_per_creature)
                        .speed(1.0)
                        .range(1..=50),
                );
                ui.label("Damage");
                ui.add(
                    egui::DragValue::new(&mut config.poison_damage)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
                ui.end_row();
            });

            section_header(ui, "Vision");
            config_grid4(ui, "vision", |ui| {
                ui.label("FOV°").on_hover_text("Field of view cone in degrees (creature only senses food/mate/poison/obstacle inside this cone, centered on facing direction)");
                let mut fov_deg = config.fov_angle.to_degrees();
                if ui.add(egui::DragValue::new(&mut fov_deg).speed(1.0).range(10.0..=360.0)).changed() {
                    config.fov_angle = fov_deg.to_radians();
                }
                ui.label("Distance").on_hover_text("Maximum sensing range in world units");
                ui.add(
                    egui::DragValue::new(&mut config.vision_distance)
                        .speed(5.0)
                        .range(10.0..=2000.0),
                );
                ui.end_row();
            });

            section_header(ui, "Obstacles");
            ui.label(egui::RichText::new("Applies on next Create").italics().weak());
            config_grid4(ui, "obstacles", |ui| {
                ui.label("Count").on_hover_text("Number of obstacles to spawn in the world");
                ui.add(
                    egui::DragValue::new(&mut config.num_obstacles)
                        .speed(1.0)
                        .range(0..=500),
                );
                ui.label("");
                ui.label("");
                ui.end_row();
                ui.label("Min size").on_hover_text("Minimum obstacle width/height (world units)");
                ui.add(
                    egui::DragValue::new(&mut config.min_obstacle_size)
                        .speed(1.0)
                        .range(5.0..=500.0),
                );
                ui.label("Max size").on_hover_text("Maximum obstacle width/height (world units)");
                ui.add(
                    egui::DragValue::new(&mut config.max_obstacle_size)
                        .speed(1.0)
                        .range(5.0..=1000.0),
                );
                ui.end_row();
            });

            section_header(ui, "Movement");
            config_grid4(ui, "movement", |ui| {
                ui.label("Turn rate").on_hover_text("Maximum radians per tick when turn outputs are saturated");
                ui.add(
                    egui::DragValue::new(&mut config.max_turn_rate)
                        .speed(0.005)
                        .range(0.0..=1.0),
                );
                ui.label("Turn cost").on_hover_text("Energy cost per radian turned");
                ui.add(
                    egui::DragValue::new(&mut config.turn_energy_cost)
                        .speed(0.0005)
                        .range(0.0..=0.1),
                );
                ui.end_row();
                ui.label("Size speed penalty").on_hover_text("Fraction of speed lost at max size (0 = no penalty, 1 = stop at max size)");
                ui.add(
                    egui::DragValue::new(&mut config.size_speed_penalty)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.label("Size move cost").on_hover_text("Extra movement cost multiplier at max size (0 = flat, 1 = 2× cost at max size)");
                ui.add(
                    egui::DragValue::new(&mut config.size_move_cost_factor)
                        .speed(0.05)
                        .range(0.0..=5.0),
                );
                ui.end_row();
            });

            section_header(ui, "Combat");
            config_grid4(ui, "combat", |ui| {
                ui.label("Range").on_hover_text("Attack reach in world units (only creatures in front are valid targets)");
                ui.add(
                    egui::DragValue::new(&mut config.attack_range)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
                ui.label("Damage").on_hover_text("Energy removed from target per successful attack");
                ui.add(
                    egui::DragValue::new(&mut config.attack_damage)
                        .speed(0.5)
                        .range(0.0..=200.0),
                );
                ui.end_row();
                ui.label("Cost").on_hover_text("Energy spent by attacker per attack output activation");
                ui.add(
                    egui::DragValue::new(&mut config.attack_cost)
                        .speed(0.05)
                        .range(0.0..=10.0),
                );
                ui.label("Steal").on_hover_text("Fraction of damage gained as attacker energy (0 = no steal, 1 = full steal)");
                ui.add(
                    egui::DragValue::new(&mut config.attack_steal_ratio)
                        .speed(0.05)
                        .range(0.0..=1.0),
                );
                ui.end_row();
            });

            section_header(ui, "Spawning");
            ui.checkbox(
                &mut config.creature_spawning_enabled,
                "Creature spawning enabled",
            ).on_hover_text("Randomly spawn new creatures when population falls below starting count");

            section_header(ui, "Extinction Restart");
            ui.checkbox(
                &mut config.extinction_restart_enabled,
                "Extinction restart enabled",
            ).on_hover_text("When population drops below the threshold, cross-breed survivors into a full new generation");
            config_grid4(ui, "extinction", |ui| {
                ui.label("Threshold").on_hover_text("Population fraction that triggers restart (e.g. 0.05 = 5% of starting count)");
                ui.add(
                    egui::DragValue::new(&mut config.extinction_threshold)
                        .speed(0.01)
                        .range(0.01..=0.5),
                );
                ui.label("");
                ui.label("");
                ui.end_row();
            });
            ui.add_space(4.0);
            let can_new_gen = !control_panel.can_create_sim
                && *sim_state.get() != SimulationState::None;
            ui.add_enabled_ui(can_new_gen, |ui| {
                if ui
                    .add_sized(
                        (ui.available_width(), 26.0),
                        egui::Button::new("🔄 Start New Generation"),
                    )
                    .on_hover_text("Cross-breed current survivors into a full new generation now")
                    .clicked()
                {
                    runner.tx.send(RunnerReq::NewGeneration).expect("send new gen");
                }
                if ui
                    .add_sized(
                        (ui.available_width(), 26.0),
                        egui::Button::new("🌍 Recreate World"),
                    )
                    .on_hover_text("Regenerate obstacles + food + poison and respawn existing creatures at random locations (brains/ages/energy preserved)")
                    .clicked()
                {
                    let w = control_panel.width.trim().parse::<f32>().unwrap_or(2000.0).max(100.0);
                    let h = control_panel.height.trim().parse::<f32>().unwrap_or(2000.0).max(100.0);
                    data.world_dim = (w, h);
                    runner.tx.send(RunnerReq::RecreateWorld((w, h))).expect("send recreate world");
                }
            });

            ui.add_space(8.0);
            if ui
                .add_sized(
                    (ui.available_width(), 32.0),
                    egui::Button::new("Apply Config"),
                )
                .clicked()
            {
                apply_changes = true;
            }
            ui.add_space(4.0);
        });

        if apply_changes {
            info!("Apply Config: sending UpdateConfig to runner");
            runner
                .tx
                .send(RunnerReq::UpdateConfig(config.clone()))
                .expect("Could not send config update");
            save_cached_config(&config);
        }
    });
}

fn open_creature_explorer(commands: &mut Commands, explorer: &mut CreatureExplorer) {
    if explorer.window.is_some() {
        return;
    }

    let window_entity = commands
        .spawn((Window {
            title: "Creature Explorer".to_string(),
            resolution: (760u32, 540u32).into(),
            ..default()
        },))
        .id();

    commands.spawn((
        Camera2d,
        RenderTarget::Window(WindowRef::Entity(window_entity)),
        EguiMultipassSchedule::new(CreatureExplorerContextPass),
        RenderLayers::layer(3),
        CreatureExplorerCamera,
    ));

    explorer.window = Some(window_entity);
}

pub fn on_creature_explorer_closed(
    mut events: MessageReader<WindowClosed>,
    mut explorer: ResMut<CreatureExplorer>,
    cameras: Query<Entity, With<CreatureExplorerCamera>>,
    mut commands: Commands,
) {
    let Some(window_entity) = explorer.window else {
        return;
    };

    for event in events.read() {
        if event.window == window_entity {
            explorer.open = false;
            explorer.window = None;
            for camera in &cameras {
                commands.entity(camera).despawn();
            }
            return;
        }
    }
}

pub fn creature_explorer_window(
    mut egui_ctx: Single<&mut EguiContext, With<CreatureExplorerCamera>>,
    data: Res<Simulation>,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    mut explorer: ResMut<CreatureExplorer>,
) {
    let ctx = egui_ctx.get_mut();

    egui::CentralPanel::default().show(ctx, |ui| {
        show_creature_explorer_contents(ui, &data, &runner, &mut stats, &mut explorer);
    });
}

fn show_creature_explorer_contents(
    ui: &mut egui::Ui,
    data: &Simulation,
    runner: &RunnerResource,
    stats: &mut SimulationStats,
    explorer: &mut CreatureExplorer,
) {
    ui.horizontal(|ui| {
        if ui.button("🔄 Refresh Snapshot").clicked() {
            explorer.snapshot = data.creatures.clone();
        }
        ui.label(format!("Creatures: {}", explorer.snapshot.len()));
        ui.separator();
        ui.label("Filter id");
        ui.add(egui::TextEdit::singleline(&mut explorer.filter_id).desired_width(80.0));
    });
    ui.separator();

    // Header row with sort buttons.
    ui.horizontal(|ui| {
        let cols: [(&str, ExplorerSort, f32); 7] = [
            ("ID", ExplorerSort::Id, 60.0),
            ("Age", ExplorerSort::Age, 70.0),
            ("Energy", ExplorerSort::Energy, 80.0),
            ("Size", ExplorerSort::Size, 70.0),
            ("MateCD", ExplorerSort::MateCd, 70.0),
            ("Lock", ExplorerSort::ActionLock, 60.0),
            ("Pos", ExplorerSort::Id, 130.0),
        ];
        for (label, sort, w) in cols {
            let active = explorer.sort_by == sort;
            let arrow = if active {
                if explorer.sort_desc {
                    " ▼"
                } else {
                    " ▲"
                }
            } else {
                ""
            };
            let btn = egui::Button::new(format!("{label}{arrow}")).min_size(egui::vec2(w, 22.0));
            if ui.add(btn).clicked() {
                if explorer.sort_by == sort {
                    explorer.sort_desc = !explorer.sort_desc;
                } else {
                    explorer.sort_by = sort;
                    explorer.sort_desc = true;
                }
            }
        }
    });
    ui.separator();

    let filter = explorer.filter_id.trim().parse::<usize>().ok();
    let mut rows: Vec<&BasicCreature> = explorer
        .snapshot
        .iter()
        .filter(|c| filter.map(|f| c.id == f).unwrap_or(true))
        .collect();
    let sort_by = explorer.sort_by;
    let desc = explorer.sort_desc;
    rows.sort_by(|a, b| {
        let ord = match sort_by {
            ExplorerSort::Id => a.id.cmp(&b.id),
            ExplorerSort::Age => a.age.cmp(&b.age),
            ExplorerSort::Energy => a
                .energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal),
            ExplorerSort::Size => a
                .size
                .partial_cmp(&b.size)
                .unwrap_or(std::cmp::Ordering::Equal),
            ExplorerSort::MateCd => a.mate_cooldown.cmp(&b.mate_cooldown),
            ExplorerSort::ActionLock => a.action_lock.cmp(&b.action_lock),
        };
        if desc {
            ord.reverse()
        } else {
            ord
        }
    });

    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            for c in rows {
                ui.horizontal(|ui| {
                    ui.add_sized((60.0, 18.0), egui::Label::new(c.id.to_string()));
                    ui.add_sized((70.0, 18.0), egui::Label::new(c.age.to_string()));
                    ui.add_sized((80.0, 18.0), egui::Label::new(format!("{:.1}", c.energy)));
                    ui.add_sized((70.0, 18.0), egui::Label::new(format!("{:.1}", c.size)));
                    ui.add_sized((70.0, 18.0), egui::Label::new(c.mate_cooldown.to_string()));
                    ui.add_sized((60.0, 18.0), egui::Label::new(c.action_lock.to_string()));
                    ui.add_sized(
                        (130.0, 18.0),
                        egui::Label::new(format!("({:.0},{:.0})", c.position.0, c.position.1)),
                    );
                    if ui.button("🧠 Inspect").clicked() {
                        stats.selected_creature_id = Some(c.id);
                        runner
                            .tx
                            .send(RunnerReq::GetNet(c.id))
                            .expect("send getnet");
                    }
                });
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

fn stat_row4_tip(ui: &mut egui::Ui, l1: &str, v1: &str, t1: &str, l2: &str, v2: &str, t2: &str) {
    let r1 = ui.label(egui::RichText::new(l1).weak());
    if !t1.is_empty() {
        r1.on_hover_text(t1);
    }
    ui.label(v1);
    let r2 = ui.label(egui::RichText::new(l2).weak());
    if !t2.is_empty() {
        r2.on_hover_text(t2);
    }
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
    mut control_panel: ResMut<ControlPanel>,
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
                data.obstacles = p.obstacles;
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
            RunnerRes::SaveResult(r) => set_status(&mut control_panel.status, "Save", r),
            RunnerRes::LoadResult(r) => set_status(&mut control_panel.status, "Load", r),
        }
    }
}

fn set_status(status: &mut String, op: &str, r: Result<std::path::PathBuf, String>) {
    *status = match r {
        Ok(p) => format!("{op} OK: {}", p.display()),
        Err(e) => format!("{op} failed: {e}"),
    };
}

fn clear_screen(
    commands: &mut Commands,
    shapes: Query<
        Entity,
        Or<(
            With<RectangleComponent>,
            With<DiscComponent>,
            With<RegularPolygonComponent>,
        )>,
    >,
) {
    for item in shapes.iter() {
        commands.entity(item).despawn();
    }
}

pub fn run_simulation(
    mut data: ResMut<Simulation>,
    mut shapes: ShapeCommands,
    mut commands: Commands,
    rects: Query<
        Entity,
        Or<(
            With<RectangleComponent>,
            With<DiscComponent>,
            With<RegularPolygonComponent>,
        )>,
    >,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    mut pending_net: ResMut<PendingNet>,
    mut control_panel: ResMut<ControlPanel>,
) {
    let mut positions = None;
    while let Ok(res) = runner.rx.try_recv() {
        match res {
            RunnerRes::Positions(p) => positions = Some(p),
            RunnerRes::Net(Some(n)) => pending_net.0 = Some(n),
            RunnerRes::Net(None) => {}
            RunnerRes::SaveResult(r) => set_status(&mut control_panel.status, "Save", r),
            RunnerRes::LoadResult(r) => set_status(&mut control_panel.status, "Load", r),
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
    data.obstacles = p.obstacles;
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
    primary_window: Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };

    // Only accept input when the sim (primary) window is focused.
    let primary_focused = primary_window.single().map(|w| w.focused).unwrap_or(false);

    let mut zoom = transform.scale.x;

    // Scroll wheel to zoom
    if primary_focused && scroll.delta.y != 0.0 {
        let factor = if scroll.delta.y > 0.0 {
            0.85
        } else {
            1.0 / 0.85
        };
        zoom *= factor;
    }

    // +/- keys to zoom
    if primary_focused && (keyboard.pressed(KeyCode::Equal) || keyboard.pressed(KeyCode::NumpadAdd))
    {
        zoom *= 1.0 - 1.5 * time.delta_secs();
    }
    if primary_focused
        && (keyboard.pressed(KeyCode::Minus) || keyboard.pressed(KeyCode::NumpadSubtract))
    {
        zoom *= 1.0 + 1.5 * time.delta_secs();
    }

    zoom = zoom.clamp(0.01, 100.0);

    // WASD / arrow keys to pan
    let pan_speed = 400.0 * zoom * time.delta_secs();
    if primary_focused && (keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA))
    {
        transform.translation.x -= pan_speed;
    }
    if primary_focused && (keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD))
    {
        transform.translation.x += pan_speed;
    }
    if primary_focused && (keyboard.pressed(KeyCode::ArrowUp) || keyboard.pressed(KeyCode::KeyW)) {
        transform.translation.y += pan_speed;
    }
    if primary_focused && (keyboard.pressed(KeyCode::ArrowDown) || keyboard.pressed(KeyCode::KeyS))
    {
        transform.translation.y -= pan_speed;
    }

    // Middle-mouse drag to pan
    if primary_focused && mouse_button.pressed(MouseButton::Middle) {
        transform.translation.x -= mouse_motion.delta.x * zoom;
        transform.translation.y += mouse_motion.delta.y * zoom;
    }

    // R to reset camera to fit world
    if primary_focused && keyboard.just_pressed(KeyCode::KeyR) {
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
    let nearest = data
        .creatures
        .iter()
        .filter_map(|c| {
            let d2 = (c.position.0 - sim_x).powi(2) + (c.position.1 - sim_y).powi(2);
            (d2 <= radius_world * radius_world).then_some((c, d2))
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((c, _)) = nearest {
        stats.selected_creature_id = Some(c.id);
        runner
            .tx
            .send(RunnerReq::GetNet(c.id))
            .expect("send net request");
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
    shapes.transform.rotation = Quat::IDENTITY;

    // Obstacles (drawn first so creatures render on top).
    shapes.color = Color::from(Srgba::hex(OBSTACLE_COLOR).unwrap());
    data.obstacles.iter().for_each(|o| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(o.position.0, o.position.1),
            data.world_dim,
        );
        shapes.transform.translation = Vec3::new(coords.x, coords.y, -0.5);
        shapes.rect(Vec2::new(o.half_width * 2.0, o.half_height * 2.0));
    });

    data.creatures.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.world_dim,
        );
        if selected_id == Some(c.id) {
            shapes.color = Color::from(Srgba::new(1.0, 1.0, 1.0, 0.9));
            shapes.thickness = 2.0;
            shapes.transform.translation = Vec3::new(coords.x, coords.y, 1.0);
            shapes.transform.rotation = Quat::IDENTITY;
            shapes.circle(c.size + 5.0);
            shapes.thickness = 0.0;
        }
        shapes.color = Color::from(Srgba::hex(CREATURE_COLOR).unwrap());
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        // ngon(3) draws an isoceles triangle with tip pointing up (+y); rotate so tip aligns with `angle`.
        shapes.transform.rotation = Quat::from_rotation_z(c.angle - std::f32::consts::FRAC_PI_2);
        shapes.ngon(3.0, c.size * 0.9);
    });
    shapes.transform.rotation = Quat::IDENTITY;

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
