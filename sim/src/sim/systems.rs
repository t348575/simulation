use std::thread;

use bevy::{
    camera::{visibility::RenderLayers, RenderTarget},
    ecs::schedule::ScheduleLabel,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{PrimaryWindow, WindowClosed, WindowMoved, WindowPosition, WindowRef, WindowResized},
};
use bevy_egui::{egui, EguiContext, EguiGlobalSettings, EguiMultipassSchedule, PrimaryEguiContext};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SystemWindowContextPass;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CreatureExplorerContextPass;
use bevy_vector_shapes::prelude::*;

use super::sim::{
    BasicCreature, FoodKind, Generate, PoisonKind, Runner, RunnerReq, RunnerRes, TerrainMaterial,
    WorldPaint, WorldPaintAction,
};

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
const DEEP_WATER_COLOR: &str = "10213a";
const WATER_COLOR: &str = "174b83";
const SHALLOW_WATER_COLOR: &str = "1e5f74";
const SAND_COLOR: &str = "b99b5f";
const DESERT_COLOR: &str = "c28f45";
const SAVANNA_COLOR: &str = "8f8f3e";
const GRASS_COLOR: &str = "315f38";
const FOREST_COLOR: &str = "173b27";
const RAINFOREST_COLOR: &str = "0c2f21";
const MARSH_COLOR: &str = "315b49";
const TUNDRA_COLOR: &str = "7f8f76";
const ROCK_COLOR: &str = "5f615d";
const SNOW_COLOR: &str = "c6c9c2";
const TERRAIN_MATERIALS: [TerrainMaterial; 13] = [
    TerrainMaterial::DeepWater,
    TerrainMaterial::Water,
    TerrainMaterial::ShallowWater,
    TerrainMaterial::Sand,
    TerrainMaterial::Desert,
    TerrainMaterial::Savanna,
    TerrainMaterial::Grass,
    TerrainMaterial::Forest,
    TerrainMaterial::Rainforest,
    TerrainMaterial::Marsh,
    TerrainMaterial::Tundra,
    TerrainMaterial::Rock,
    TerrainMaterial::Snow,
];

#[derive(Component)]
pub struct SimWorldShape;

pub fn init_runner(mut commands: Commands) {
    let (r, tx, rx) = Runner::new();
    let t = thread::spawn(move || {
        r.run();
    });
    commands.insert_resource(RunnerResource { tx, rx, thread: t });
}

pub fn setup(
    mut control_panel: ResMut<ControlPanel>,
    mut window_query: Query<&mut Window, With<PrimaryWindow>>,
    mut data: ResMut<Simulation>,
    mut commands: Commands,
    mut system_window: ResMut<SystemWindow>,
    cached: Res<CachedSettings>,
    mut egui_global_settings: ResMut<EguiGlobalSettings>,
) {
    // We attach `PrimaryEguiContext` to our own sim camera below so egui
    // renders on the same camera that draws the world. Disable bevy_egui's
    // automatic camera or we'd end up with two cameras on the primary
    // window — the second one clears over egui's output.
    egui_global_settings.auto_create_primary_context = false;
    let mut window = window_query.single_mut().unwrap();

    if let Some(geometry) = cached.primary_window {
        apply_window_geometry(&mut window, geometry);
    }

    // Only set width/height from window if they are "0" or empty
    if control_panel.width == "0" || control_panel.width.is_empty() {
        control_panel.width = window.width().to_string();
    }
    if control_panel.height == "0" || control_panel.height.is_empty() {
        control_panel.height = window.height().to_string();
    }

    data.window_dims = (window.width(), window.height());

    // Primary sim camera. Carries `PrimaryEguiContext` so egui's primary pass
    // renders into the same camera (and same window) as the simulation.
    commands.spawn((Camera2d, SimCamera, PrimaryEguiContext));

    // Spawn system window
    let mut system_window_config = Window {
        title: "System".to_string(),
        resolution: (520u32, 680u32).into(),
        ..default()
    };
    if let Some(geometry) = cached.system_window {
        apply_window_geometry(&mut system_window_config, geometry);
    }
    let window_entity = commands.spawn((system_window_config,)).id();

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
        (
            With<SimWorldShape>,
            Or<(
                With<RectangleComponent>,
                With<DiscComponent>,
                With<RegularPolygonComponent>,
            )>,
        ),
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
                        save_cached_settings(&config, Some(&control_panel));
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
            ui.label("World brush, legend, and spawn-area controls live on the simulation window.");

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
                    runner
                        .tx
                        .send(RunnerReq::LoadSim(
                            path,
                            base_nodes.input_nodes.clone(),
                            base_nodes.output_nodes.clone(),
                        ))
                        .expect("send load");
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
                ui.add_enabled_ui(true, |ui| {
                    if ui
                        .add_sized((w, 26.0), egui::Button::new("📂 Load Creatures"))
                        .on_hover_text("Load creatures into the currently specified world size")
                        .clicked()
                    {
                        let path = std::path::PathBuf::from(&control_panel.save_path);
                        let w = control_panel.width.trim().parse::<f32>().unwrap_or(2000.0).max(100.0);
                        let h = control_panel.height.trim().parse::<f32>().unwrap_or(2000.0).max(100.0);
                        data.world_dim = (w, h);
                        clear_screen(&mut commands, rects);
                        data.creatures.clear();
                        save_cached_settings(&config, Some(&control_panel));
                        runner
                            .tx
                            .send(RunnerReq::LoadCreatures(
                                path,
                                (w, h),
                                config.clone(),
                                base_nodes.input_nodes.clone(),
                                base_nodes.output_nodes.clone(),
                            ))
                            .expect("send load creatures");
                        control_panel.can_create_sim = false;
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
                            save_cached_settings(&config, Some(&control_panel));
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
                    save_cached_settings(&config, Some(&control_panel));
                    control_panel.status = "Config reset to defaults".to_owned();
                }
            });

            ui.separator();

            // ── View ─────────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.checkbox(&mut control_panel.show_signal_field, "Show signal halos")
                    .on_hover_text("Render colored halo around each creature scaled by its emitted voice signal");
            });

            ui.separator();

            // ── Voice & Mating ───────────────────────────────────────────
            egui::CollapsingHeader::new("Voice & Mating")
                .default_open(false)
                .show(ui, |ui| {
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Speak cost");
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.signal_action_cost, 0.0..=0.1)
                                    .step_by(0.001),
                            )
                            .on_hover_text("Energy deducted per tick when a creature emits voice. Lower = cheaper to talk.")
                            .changed();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Listener bonus");
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.signal_listener_bonus, 0.0..=0.1)
                                    .step_by(0.001),
                            )
                            .on_hover_text("Energy gained per tick from being near speakers. Higher = stronger evolutionary pull toward voice.")
                            .changed();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mate voice bonus");
                        changed |= ui
                            .add(egui::Slider::new(&mut config.mate_voice_bonus, 0.0..=0.95))
                            .on_hover_text("If both partners are speaking, multiply mate-energy threshold by (1 - bonus). Higher = talkative pairs mate sooner.")
                            .changed();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Min mate energy");
                        changed |= ui
                            .add(egui::Slider::new(&mut config.min_mate_energy, 0.0..=200.0))
                            .on_hover_text("Floor on energy required to mate.")
                            .changed();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mate cooldown ticks");
                        let mut v = config.mate_cooldown_ticks as i32;
                        if ui
                            .add(egui::Slider::new(&mut v, 0..=600))
                            .on_hover_text("Ticks before a creature can mate again.")
                            .changed()
                        {
                            config.mate_cooldown_ticks = v.max(0) as u16;
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Max births / tick");
                        let mut v = config.max_births_per_tick as i32;
                        if ui
                            .add(egui::Slider::new(&mut v, 1..=200))
                            .changed()
                        {
                            config.max_births_per_tick = v.max(1) as usize;
                            changed = true;
                        }
                    });
                    if changed {
                        let _ = runner.tx.send(RunnerReq::UpdateConfig(config.clone()));
                        save_cached_settings(&config, Some(&control_panel));
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

            // Push the parsed creature-count to the runner whenever it
            // changes, so target_population tracks the text field live
            // (used by mating cap, refill, extinction restart, restart-gen).
            if *sim_state.get() != SimulationState::None {
                if let Ok(parsed) = control_panel.initial_num_creatures.trim().parse::<usize>() {
                    if parsed > 0
                        && control_panel.last_pushed_target_population != Some(parsed)
                    {
                        let _ = runner.tx.send(RunnerReq::SetTargetPopulation(parsed));
                        control_panel.last_pushed_target_population = Some(parsed);
                    }
                }
            }

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
                ui.add(egui::DragValue::new(&mut config.start_energy).speed(1.0));
                ui.label("Max");
                ui.add(egui::DragValue::new(&mut config.max_energy).speed(1.0));
                ui.end_row();
                ui.label("Base cost").on_hover_text("Energy lost per tick regardless of action");
                ui.add(egui::DragValue::new(&mut config.base_energy_cost).speed(0.01));
                ui.label("Move cost").on_hover_text("Additional energy lost per unit of speed");
                ui.add(egui::DragValue::new(&mut config.move_energy_cost).speed(0.01));
                ui.end_row();
                ui.label("Min move").on_hover_text("Energy below which a creature cannot move at all");
                ui.add(egui::DragValue::new(&mut config.min_move_energy).speed(1.0));
                ui.label("Full spd").on_hover_text("Energy level at which movement speed is fully uncapped");
                ui.add(egui::DragValue::new(&mut config.full_speed_energy).speed(1.0));
                ui.end_row();
            });

            section_header(ui, "Creatures");
            config_grid4(ui, "creatures", |ui| {
                ui.label("Ignore max age");
                ui.add(
                    egui::Checkbox::new(&mut config.ignore_max_age, "")
                );
                ui.label("Max age");
                ui.add(egui::DragValue::new(&mut config.max_age).speed(100.0));
                ui.label("Min size");
                ui.add(egui::DragValue::new(&mut config.min_creature_size).speed(0.1));
                ui.end_row();
                ui.label("Max size");
                ui.add(egui::DragValue::new(&mut config.max_creature_size).speed(0.1));
                ui.label("");
                ui.label("");
                ui.end_row();
            });

            section_header(ui, "Initial Genome");
            config_grid4(ui, "initial_genome", |ui| {
                ui.label("Body").on_hover_text("Heritable body diameter used for collision and energy capacity");
                ui.add(egui::DragValue::new(&mut config.initial_genome.body_size).speed(0.1));
                ui.label("Reserve density").on_hover_text("Energy capacity per body area");
                ui.add(egui::DragValue::new(&mut config.initial_genome.reserve_density).speed(0.1));
                ui.end_row();
                ui.label("Metabolism").on_hover_text("Baseline energy burn per tick, scaled by body mass");
                ui.add(egui::DragValue::new(&mut config.initial_genome.metabolism).speed(0.001));
                ui.label("Muscle").on_hover_text("Movement/attack power; stronger bodies cost more to operate");
                ui.add(egui::DragValue::new(&mut config.initial_genome.muscle_power).speed(0.05));
                ui.end_row();
                ui.label("Move eff").on_hover_text("Higher values spend less energy per movement");
                ui.add(egui::DragValue::new(&mut config.initial_genome.move_efficiency).speed(0.05));
                ui.label("Turn agility").on_hover_text("Higher values turn faster and cheaper");
                ui.add(egui::DragValue::new(&mut config.initial_genome.turn_agility).speed(0.05));
                ui.end_row();
                ui.label("Vision").on_hover_text("Heritable sensing distance; longer vision has a maintenance cost");
                ui.add(egui::DragValue::new(&mut config.initial_genome.vision_distance).speed(5.0));
                ui.label("FOV deg");
                let mut genome_fov_deg = config.initial_genome.fov_angle.to_degrees();
                if ui.add(egui::DragValue::new(&mut genome_fov_deg).speed(1.0)).changed() {
                    config.initial_genome.fov_angle = genome_fov_deg.to_radians();
                }
                ui.end_row();
                ui.label("Bite").on_hover_text("Largest food size this genome can eat directly");
                ui.add(egui::DragValue::new(&mut config.initial_genome.bite_size).speed(0.1));
                ui.label("Digestion").on_hover_text("Higher values finish eating faster");
                ui.add(egui::DragValue::new(&mut config.initial_genome.digestion_rate).speed(0.05));
                ui.end_row();
                ui.label("Cold tol");
                ui.add(egui::DragValue::new(&mut config.initial_genome.cold_tolerance).speed(0.05));
                ui.label("Heat tol");
                ui.add(egui::DragValue::new(&mut config.initial_genome.heat_tolerance).speed(0.05));
                ui.end_row();
                ui.label("Water adapt");
                ui.add(egui::DragValue::new(&mut config.initial_genome.water_adaptation).speed(0.05));
                ui.label("Rough adapt");
                ui.add(egui::DragValue::new(&mut config.initial_genome.rough_terrain_adaptation).speed(0.05));
                ui.end_row();
                ui.label("Mate frac").on_hover_text("Energy fraction required before mating");
                ui.add(egui::DragValue::new(&mut config.initial_genome.mate_threshold_frac).speed(0.01));
                ui.label("Child frac").on_hover_text("Parent energy fraction transferred to offspring");
                ui.add(egui::DragValue::new(&mut config.initial_genome.offspring_energy_frac).speed(0.01));
                ui.end_row();
                ui.label("Mut rate");
                ui.add(egui::DragValue::new(&mut config.initial_genome.mutation_rate).speed(0.01));
                ui.label("Mut scale");
                ui.add(egui::DragValue::new(&mut config.initial_genome.mutation_scale).speed(0.01));
                ui.end_row();
            });

            section_header(ui, "Reproduction");
            config_grid4(ui, "repro", |ui| {
                ui.label("Threshold");
                ui.add(egui::DragValue::new(&mut config.mate_threshold).speed(0.01));
                ui.label("Cooldown");
                ui.add(egui::DragValue::new(&mut config.mate_cooldown_ticks).speed(1.0));
                ui.end_row();
                ui.label("Min energy");
                ui.add(egui::DragValue::new(&mut config.min_mate_energy).speed(1.0));
                ui.label("Mate cost");
                ui.add(egui::DragValue::new(&mut config.mate_energy_cost).speed(1.0));
                ui.end_row();
                ui.label("Atmp cost").on_hover_text("Energy lost per tick while the creature is trying to mate");
                ui.add(egui::DragValue::new(&mut config.mate_attempt_cost).speed(0.01));
                ui.label("Child E").on_hover_text("Energy given to each newborn creature");
                ui.add(egui::DragValue::new(&mut config.child_energy).speed(1.0));
                ui.end_row();
                ui.label("Mut rate").on_hover_text("Probability (0–1) that each brain weight mutates per birth");
                ui.add(egui::DragValue::new(&mut config.mutation_rate).speed(0.01));
                ui.label("Mut amt").on_hover_text("Max magnitude of weight change when a mutation occurs");
                ui.add(egui::DragValue::new(&mut config.mutation_amount).speed(0.01));
                ui.end_row();
                ui.label("Max births").on_hover_text("Hard cap on new creatures born per simulation tick");
                ui.add(egui::DragValue::new(&mut config.max_births_per_tick).speed(1.0));
                ui.label("");
                ui.label("");
                ui.end_row();
            });

            section_header(ui, "Food");
            config_grid4(ui, "food", |ui| {
                ui.label("Min size");
                ui.add(egui::DragValue::new(&mut config.min_food_size).speed(0.1));
                ui.label("Max size");
                ui.add(egui::DragValue::new(&mut config.max_food_size).speed(0.1));
                ui.end_row();
                ui.label("E/size").on_hover_text("Energy gained per unit of food size when the food is eaten");
                ui.add(egui::DragValue::new(&mut config.food_energy_per_size).speed(0.5));
                ui.label("Ticks/size").on_hover_text("Eating duration added per unit of food size (larger food takes longer)");
                ui.add(egui::DragValue::new(&mut config.food_eat_ticks_per_size).speed(0.1));
                ui.end_row();
                ui.label("Act cost").on_hover_text("Energy deducted when a creature starts eating a food item");
                ui.add(egui::DragValue::new(&mut config.eat_action_cost).speed(0.1));
                ui.label("Base ticks").on_hover_text("Minimum ticks to finish eating, before size scaling is applied");
                ui.add(egui::DragValue::new(&mut config.eat_action_base_ticks).speed(1.0));
                ui.end_row();
                ui.label("Atmp cost").on_hover_text("Energy lost per tick while the creature is trying to eat");
                ui.add(egui::DragValue::new(&mut config.eat_attempt_cost).speed(0.01));
                ui.label("");
                ui.label("");
                ui.end_row();
            });
            biome_rate_grid(
                ui,
                "food_biome_rates",
                &mut config.food_biome_rates,
                "Food items targeted per loaded terrain tile for each biome",
            );

            section_header(ui, "Poison");
            config_grid4(ui, "poison", |ui| {
                ui.label("Damage");
                ui.add(egui::DragValue::new(&mut config.poison_damage).speed(1.0));
                ui.label("");
                ui.label("");
                ui.end_row();
            });
            biome_rate_grid(
                ui,
                "poison_biome_rates",
                &mut config.poison_biome_rates,
                "Poison items targeted per loaded terrain tile for each biome",
            );

            section_header(ui, "Vision");
            config_grid4(ui, "vision", |ui| {
                ui.label("FOV°").on_hover_text("Field of view cone in degrees (creature only senses food/mate/poison/obstacle inside this cone, centered on facing direction)");
                let mut fov_deg = config.fov_angle.to_degrees();
                if ui.add(egui::DragValue::new(&mut fov_deg).speed(1.0)).changed() {
                    config.fov_angle = fov_deg.to_radians();
                }
                ui.label("Distance").on_hover_text("Maximum sensing range in world units");
                ui.add(egui::DragValue::new(&mut config.vision_distance).speed(5.0));
                ui.end_row();
            });

            section_header(ui, "Obstacles");
            ui.label(egui::RichText::new("Applies on next Create").italics().weak());
            config_grid4(ui, "obstacles", |ui| {
                ui.label("Count").on_hover_text("Number of obstacles to spawn in the world");
                ui.add(egui::DragValue::new(&mut config.num_obstacles).speed(1.0));
                ui.label("");
                ui.label("");
                ui.end_row();
                ui.label("Min size").on_hover_text("Minimum obstacle width/height (world units)");
                ui.add(egui::DragValue::new(&mut config.min_obstacle_size).speed(1.0));
                ui.label("Max size").on_hover_text("Maximum obstacle width/height (world units)");
                ui.add(egui::DragValue::new(&mut config.max_obstacle_size).speed(1.0));
                ui.end_row();
            });

            section_header(ui, "Movement");
            config_grid4(ui, "movement", |ui| {
                ui.label("Turn rate").on_hover_text("Maximum radians per tick when turn outputs are saturated");
                ui.add(egui::DragValue::new(&mut config.max_turn_rate).speed(0.01));
                ui.label("Turn cost").on_hover_text("Energy cost per radian turned");
                ui.add(egui::DragValue::new(&mut config.turn_energy_cost).speed(0.001));
                ui.end_row();
                ui.label("Size speed penalty").on_hover_text("Fraction of speed lost at max size (0 = no penalty, 1 = stop at max size)");
                ui.add(egui::DragValue::new(&mut config.size_speed_penalty).speed(0.01));
                ui.label("Size move cost").on_hover_text("Extra movement cost multiplier at max size (0 = flat, 1 = 2× cost at max size)");
                ui.add(egui::DragValue::new(&mut config.size_move_cost_factor).speed(0.05));
                ui.end_row();
            });

            section_header(ui, "Combat");
            config_grid4(ui, "combat", |ui| {
                ui.label("Range").on_hover_text("Attack reach in world units (only creatures in front are valid targets)");
                ui.add(egui::DragValue::new(&mut config.attack_range).speed(1.0));
                ui.label("Damage").on_hover_text("Energy removed from target per successful attack");
                ui.add(egui::DragValue::new(&mut config.attack_damage).speed(0.5));
                ui.end_row();
                ui.label("Cost").on_hover_text("Energy spent by attacker per attack output activation");
                ui.add(egui::DragValue::new(&mut config.attack_cost).speed(0.05));
                ui.label("Steal").on_hover_text("Fraction of damage gained as attacker energy (0 = no steal, 1 = full steal)");
                ui.add(egui::DragValue::new(&mut config.attack_steal_ratio).speed(0.05));
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
                ui.add(egui::DragValue::new(&mut config.extinction_threshold).speed(0.01));
                ui.label("Top survivor fraction").on_hover_text("Only the top X fraction of survivors (ranked by energy + age) seed the next generation");
                ui.add(
                    egui::DragValue::new(&mut config.next_gen_top_fraction)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
                ui.end_row();
            });
            ui.checkbox(
                &mut config.next_gen_only_mated,
                "Only creatures that have mated",
            )
            .on_hover_text("Restrict next-generation seed pool to creatures that mated at least once. Falls back to all survivors if none qualify.");
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
                    save_cached_settings(&config, Some(&control_panel));
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
            save_cached_settings(&config, Some(&control_panel));
        }
    });
}

fn open_creature_explorer(commands: &mut Commands, explorer: &mut CreatureExplorer) {
    if explorer.window.is_some() {
        return;
    }

    let mut window = Window {
        title: "Creature Explorer".to_string(),
        resolution: (760u32, 540u32).into(),
        ..default()
    };
    if let Some(geometry) = load_cached_settings().creature_explorer_window {
        apply_window_geometry(&mut window, geometry);
    }
    let window_entity = commands.spawn((window,)).id();

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

fn biome_rate_grid(
    ui: &mut egui::Ui,
    id: &str,
    rates: &mut [f32; TERRAIN_MATERIAL_COUNT],
    tooltip: &str,
) {
    egui::Grid::new(id)
        .num_columns(4)
        .spacing([6.0, 4.0])
        .striped(true)
        .min_col_width(72.0)
        .show(ui, |ui| {
            for (i, material) in TERRAIN_MATERIALS.iter().copied().enumerate() {
                ui.label(terrain_material_label(material))
                    .on_hover_text(tooltip);
                ui.add(
                    egui::DragValue::new(&mut rates[i])
                        .speed(0.05)
                        .range(0.0..=100.0),
                );
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
            if TERRAIN_MATERIALS.len() % 2 == 1 {
                ui.label("");
                ui.label("");
                ui.end_row();
            }
        });
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
    spawn_area: Res<SpawnArea>,
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
                data.terrain = p.terrain;
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

                render_world(
                    &mut data,
                    &mut shapes,
                    None,
                    Some(&*spawn_area),
                    control_panel.show_signal_field,
                );
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

fn apply_window_geometry(window: &mut Window, geometry: WindowGeometry) {
    window.set_maximized(false);
    window
        .resolution
        .set(geometry.width.max(100.0), geometry.height.max(100.0));
    window.position.set(IVec2::new(geometry.x, geometry.y));
}

fn window_position(window: &Window) -> Option<IVec2> {
    match window.position {
        WindowPosition::At(pos) => Some(pos),
        _ => None,
    }
}

fn cached_window_geometry(
    entity: Entity,
    cached: &CachedSettings,
    primary_window: Option<Entity>,
    system_window: Option<Entity>,
    creature_explorer_window: Option<Entity>,
) -> Option<WindowGeometry> {
    if Some(entity) == primary_window {
        cached.primary_window
    } else if Some(entity) == system_window {
        cached.system_window
    } else if Some(entity) == creature_explorer_window {
        cached.creature_explorer_window
    } else {
        None
    }
}

fn set_cached_window_geometry(
    entity: Entity,
    geometry: WindowGeometry,
    cached: &mut CachedSettings,
    primary_window: Option<Entity>,
    system_window: Option<Entity>,
    creature_explorer_window: Option<Entity>,
) -> bool {
    if Some(entity) == primary_window {
        cached.primary_window = Some(geometry);
        true
    } else if Some(entity) == system_window {
        cached.system_window = Some(geometry);
        true
    } else if Some(entity) == creature_explorer_window {
        cached.creature_explorer_window = Some(geometry);
        true
    } else {
        false
    }
}

pub fn persist_window_settings(
    mut moved: MessageReader<WindowMoved>,
    mut resized: MessageReader<WindowResized>,
    windows: Query<&Window>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    system_window: Res<SystemWindow>,
    explorer: Res<CreatureExplorer>,
    config: Res<SimulationConfig>,
    control_panel: Res<ControlPanel>,
    mut cached: ResMut<CachedSettings>,
) {
    let primary_window = primary_window.single().ok();
    let system_window = system_window.0;
    let creature_explorer_window = explorer.window;
    let mut changed = false;

    for event in resized.read() {
        let Ok(window) = windows.get(event.window) else {
            continue;
        };
        let existing = cached_window_geometry(
            event.window,
            &cached,
            primary_window,
            system_window,
            creature_explorer_window,
        );
        let pos = window_position(window)
            .map(|pos| (pos.x, pos.y))
            .or_else(|| existing.map(|geometry| (geometry.x, geometry.y)))
            .unwrap_or((0, 0));
        let geometry = WindowGeometry {
            x: pos.0,
            y: pos.1,
            width: event.width,
            height: event.height,
        };
        changed |= set_cached_window_geometry(
            event.window,
            geometry,
            &mut cached,
            primary_window,
            system_window,
            creature_explorer_window,
        );
    }

    for event in moved.read() {
        let Ok(window) = windows.get(event.window) else {
            continue;
        };
        let existing = cached_window_geometry(
            event.window,
            &cached,
            primary_window,
            system_window,
            creature_explorer_window,
        );
        let geometry = WindowGeometry {
            x: event.position.x,
            y: event.position.y,
            width: existing
                .map(|geometry| geometry.width)
                .unwrap_or(window.width()),
            height: existing
                .map(|geometry| geometry.height)
                .unwrap_or(window.height()),
        };
        changed |= set_cached_window_geometry(
            event.window,
            geometry,
            &mut cached,
            primary_window,
            system_window,
            creature_explorer_window,
        );
    }

    if changed {
        cached.config = config.clone();
        cached.world_width = control_panel.width.trim().parse().unwrap_or(2000.0);
        cached.world_height = control_panel.height.trim().parse().unwrap_or(2000.0);
        cached.initial_num_creatures = control_panel
            .initial_num_creatures
            .trim()
            .parse()
            .unwrap_or(2000);
        save_cached_settings_data(&cached);
    }
}

fn clear_screen(
    commands: &mut Commands,
    shapes: Query<
        Entity,
        (
            With<SimWorldShape>,
            Or<(
                With<RectangleComponent>,
                With<DiscComponent>,
                With<RegularPolygonComponent>,
            )>,
        ),
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
        (
            With<SimWorldShape>,
            Or<(
                With<RectangleComponent>,
                With<DiscComponent>,
                With<RegularPolygonComponent>,
            )>,
        ),
    >,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    mut pending_net: ResMut<PendingNet>,
    mut control_panel: ResMut<ControlPanel>,
    spawn_area: Res<SpawnArea>,
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
    data.terrain = p.terrain;
    *stats = p.stats;
    stats.selected_creature_id = selected_id;

    render_world(
        &mut data,
        &mut shapes,
        selected_id,
        Some(&*spawn_area),
        control_panel.show_signal_field,
    );
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
        zoom = 1.0;
        if data.creatures.is_empty() {
            transform.translation = Vec3::ZERO;
        } else {
            let (sum_x, sum_y) = data.creatures.iter().fold((0.0, 0.0), |acc, c| {
                (acc.0 + c.position.0, acc.1 - c.position.1)
            });
            let n = data.creatures.len() as f32;
            transform.translation = Vec3::new(sum_x / n, sum_y / n, 0.0);
        }
    }

    transform.scale = Vec3::splat(zoom);
    view_state.zoom = zoom;
    view_state.pan = transform.translation.truncate();
}

fn convert_bottom_left_to_center_coords(pos: Vec2, dims: (f32, f32)) -> Vec2 {
    let _ = dims;
    Vec2::new(pos.x, -pos.y)
}

pub fn inspect_creature(
    buttons: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    data: Res<Simulation>,
    runner: Res<RunnerResource>,
    mut stats: ResMut<SimulationStats>,
    view_state: Res<ViewState>,
    brush: Res<WorldBrush>,
) {
    if brush.mode != BrushMode::Off {
        return;
    }
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
    let sim_x = world_x;
    let sim_y = -world_y;

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

pub fn simulation_window_overlay(
    mut egui_ctx: Single<&mut EguiContext, With<PrimaryEguiContext>>,
    mut brush: ResMut<WorldBrush>,
    mut spawn_area: ResMut<SpawnArea>,
    runner: Res<RunnerResource>,
    sim_state: Res<State<SimulationState>>,
    control_panel: Res<ControlPanel>,
) {
    let active = !control_panel.can_create_sim && *sim_state.get() != SimulationState::None;
    let ctx = egui_ctx.get_mut();
    egui::SidePanel::left("simulation_tools_panel")
        .resizable(true)
        .default_width(280.0)
        .min_width(220.0)
        .show(ctx, |ui| {
            ui.heading("Tools");
            ui.separator();
            ui.add_enabled_ui(active, |ui| {
                ui.collapsing("World Brush", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(&mut brush.mode, BrushMode::Off, "Off");
                        ui.selectable_value(&mut brush.mode, BrushMode::Terrain, "Terrain");
                        ui.selectable_value(&mut brush.mode, BrushMode::Obstacle, "Obstacle");
                        ui.selectable_value(&mut brush.mode, BrushMode::EraseObstacle, "Erase");
                        ui.selectable_value(&mut brush.mode, BrushMode::SpawnArea, "Spawn area");
                    });
                    ui.add(egui::Slider::new(&mut brush.radius, 10.0..=400.0).text("Radius"));
                    if brush.mode == BrushMode::Terrain {
                        egui::ComboBox::from_label("Material")
                            .selected_text(terrain_material_label(brush.material))
                            .show_ui(ui, |ui| {
                                for material in TERRAIN_MATERIALS {
                                    ui.selectable_value(
                                        &mut brush.material,
                                        material,
                                        terrain_material_label(material),
                                    );
                                }
                            });
                        ui.add(
                            egui::Slider::new(&mut brush.elevation, 0.0..=1.0).text("Elevation"),
                        );
                    }
                    ui.label("Left-click on the world to apply the brush.");
                });

                ui.collapsing("Spawn Area", |ui| {
                    let prev = spawn_area.clone();
                    ui.checkbox(&mut spawn_area.enabled, "Use disc for next generation");
                    ui.add(
                        egui::Slider::new(&mut spawn_area.radius, 50.0..=10_000.0)
                            .logarithmic(true)
                            .text("Radius"),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Center");
                        ui.add(
                            egui::DragValue::new(&mut spawn_area.center.0)
                                .speed(5.0)
                                .prefix("x "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut spawn_area.center.1)
                                .speed(5.0)
                                .prefix("y "),
                        );
                    });
                    ui.label("Pick brush \"Spawn area\" then click the world to move the center.");
                    if *spawn_area != prev {
                        let _ = runner.tx.send(RunnerReq::SetSpawnArea(spawn_area.clone()));
                    }
                });

                ui.collapsing("Legend", |ui| {
                    draw_legend_ui(ui);
                });
            });
        });
}

pub fn paint_world(
    buttons: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    data: Res<Simulation>,
    runner: Res<RunnerResource>,
    view_state: Res<ViewState>,
    brush: Res<WorldBrush>,
    mut spawn_area: ResMut<SpawnArea>,
) {
    if brush.mode == BrushMode::Off || !buttons.pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = q_windows.single() else {
        return;
    };
    if !window.focused {
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Some(position) = cursor_to_sim_position(cursor, &data, &view_state) else {
        return;
    };
    let action = match brush.mode {
        BrushMode::Off => return,
        BrushMode::Terrain => WorldPaintAction::Terrain {
            material: brush.material,
            elevation: brush.elevation,
        },
        BrushMode::Obstacle => WorldPaintAction::AddObstacle,
        BrushMode::EraseObstacle => WorldPaintAction::EraseObstacle,
        BrushMode::SpawnArea => {
            spawn_area.center = position;
            spawn_area.enabled = true;
            let _ = runner.tx.send(RunnerReq::SetSpawnArea(spawn_area.clone()));
            return;
        }
    };
    let _ = runner.tx.send(RunnerReq::PaintWorld(WorldPaint {
        position,
        radius: brush.radius,
        action,
    }));
}

fn cursor_to_sim_position(
    cursor: Vec2,
    data: &Simulation,
    view_state: &ViewState,
) -> Option<(f32, f32)> {
    let (win_w, win_h) = data.window_dims;
    if win_w <= 0.0 || win_h <= 0.0 {
        return None;
    }
    let zoom = view_state.zoom;
    let pan = view_state.pan;
    let world_x = (cursor.x - win_w * 0.5) * zoom + pan.x;
    let world_y = (win_h * 0.5 - cursor.y) * zoom + pan.y;
    Some((world_x, -world_y))
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

fn render_world(
    data: &mut Simulation,
    shapes: &mut ShapeCommands,
    selected_id: Option<usize>,
    spawn_area: Option<&SpawnArea>,
    show_signal_field: bool,
) {
    shapes.thickness = 0.0;
    shapes.transform.rotation = Quat::IDENTITY;

    // tile.position is already the tile center (see terrain_tile_for_coord).
    // Earlier code added an extra half-tile shift here on the assumption it was
    // a bottom-left corner, and inflated rects by 0.5 to mask the resulting
    // seams. Both of those caused adjacent rects to overlap at the same z
    // value, producing tile-edge z-fighting that looked like a localized
    // "shake". Render at the true center with the true size and let edges
    // meet flush.
    data.terrain.iter().for_each(|tile| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(tile.position.0, tile.position.1),
            data.world_dim,
        );
        shapes.color = terrain_color(tile.material, tile.elevation);
        shapes.transform.translation = Vec3::new(coords.x, coords.y, -1.0);
        shapes
            .rect(Vec2::new(tile.size.0, tile.size.1))
            .insert(SimWorldShape);
    });

    // Obstacles (drawn first so creatures render on top).
    shapes.color = Color::from(Srgba::hex(OBSTACLE_COLOR).unwrap());
    data.obstacles.iter().for_each(|o| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(o.position.0, o.position.1),
            data.world_dim,
        );
        shapes.transform.translation = Vec3::new(coords.x, coords.y, -0.5);
        shapes
            .rect(Vec2::new(o.half_width * 2.0, o.half_height * 2.0))
            .insert(SimWorldShape);
    });

    data.creatures.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.world_dim,
        );
        // Signal halo: red for positive voice, blue for negative, alpha by magnitude.
        let mag = c.signal.abs().min(1.0);
        if show_signal_field && mag > 0.02 {
            let (r, g, b) = if c.signal >= 0.0 {
                (1.0, 0.3, 0.2)
            } else {
                (0.2, 0.4, 1.0)
            };
            shapes.color = Color::from(Srgba::new(r, g, b, mag * 0.35));
            shapes.transform.translation = Vec3::new(coords.x, coords.y, -0.25);
            shapes.transform.rotation = Quat::IDENTITY;
            shapes.circle(c.size + 80.0 * mag).insert(SimWorldShape);
        }
        if selected_id == Some(c.id) {
            shapes.color = Color::from(Srgba::new(1.0, 1.0, 1.0, 0.9));
            shapes.thickness = 2.0;
            shapes.transform.translation = Vec3::new(coords.x, coords.y, 1.0);
            shapes.transform.rotation = Quat::IDENTITY;
            shapes.circle(c.size + 5.0).insert(SimWorldShape);
            shapes.thickness = 0.0;
        }
        shapes.color = Color::from(Srgba::hex(CREATURE_COLOR).unwrap());
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        // ngon(3) draws an isoceles triangle with tip pointing up (+y); rotate so tip aligns with `angle`.
        shapes.transform.rotation = Quat::from_rotation_z(c.angle - std::f32::consts::FRAC_PI_2);
        shapes.ngon(3.0, c.size * 0.9).insert(SimWorldShape);
    });
    shapes.transform.rotation = Quat::IDENTITY;

    data.food.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.world_dim,
        );
        shapes.color = resource_food_color(c.kind);
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes.rect(Vec2::new(c.size, c.size)).insert(SimWorldShape);
    });

    data.poison.iter().for_each(|c| {
        let coords = convert_bottom_left_to_center_coords(
            Vec2::new(c.position.0, c.position.1),
            data.world_dim,
        );
        shapes.color = resource_poison_color(c.kind);
        shapes.transform.translation = Vec3::new(coords.x, coords.y, 0.0);
        shapes
            .rect(Vec2::new(POISON_DIM, POISON_DIM))
            .insert(SimWorldShape);
    });

    // Spawn-area circle (next-generation seed disc).
    if let Some(area) = spawn_area {
        if area.enabled && area.radius > 0.0 {
            let coords = convert_bottom_left_to_center_coords(
                Vec2::new(area.center.0, area.center.1),
                data.world_dim,
            );
            shapes.color = Color::from(Srgba::new(1.0, 0.85, 0.2, 0.9));
            shapes.thickness = 2.0;
            shapes.hollow = true;
            shapes.transform.translation = Vec3::new(coords.x, coords.y, 2.0);
            shapes.transform.rotation = Quat::IDENTITY;
            shapes
                .circle(area.radius)
                .insert(SimWorldShape);
            shapes.thickness = 0.0;
            shapes.hollow = false;
        }
    }
}

fn resource_food_color(kind: FoodKind) -> Color {
    let hex = match kind {
        FoodKind::Berries => FOOD_COLOR,
        FoodKind::Fruit => "ffb000",
        FoodKind::Fungus => "b36bff",
        FoodKind::Kelp => "2ec27e",
        FoodKind::CactusFruit => "ff5b8a",
        FoodKind::Lichen => "b0c95a",
        FoodKind::Seeds => "d6b45f",
    };
    Color::from(Srgba::hex(hex).unwrap())
}

fn resource_poison_color(kind: PoisonKind) -> Color {
    let hex = match kind {
        PoisonKind::Nightshade => POISON_COLOR,
        PoisonKind::ToxicMushroom => "ff1f1f",
        PoisonKind::BrineBloom => "00d7ff",
        PoisonKind::ThornPatch => "c46a25",
        PoisonKind::BitterLichen => "879c22",
    };
    Color::from(Srgba::hex(hex).unwrap())
}

fn terrain_color(material: TerrainMaterial, elevation: f32) -> Color {
    let hex = match material {
        TerrainMaterial::DeepWater => DEEP_WATER_COLOR,
        TerrainMaterial::Water => WATER_COLOR,
        TerrainMaterial::ShallowWater => SHALLOW_WATER_COLOR,
        TerrainMaterial::Sand => SAND_COLOR,
        TerrainMaterial::Desert => DESERT_COLOR,
        TerrainMaterial::Savanna => SAVANNA_COLOR,
        TerrainMaterial::Grass => GRASS_COLOR,
        TerrainMaterial::Forest => FOREST_COLOR,
        TerrainMaterial::Rainforest => RAINFOREST_COLOR,
        TerrainMaterial::Marsh => MARSH_COLOR,
        TerrainMaterial::Tundra => TUNDRA_COLOR,
        TerrainMaterial::Rock => ROCK_COLOR,
        TerrainMaterial::Snow => SNOW_COLOR,
    };
    let base = Srgba::hex(hex).unwrap();
    let shade = 0.75 + elevation.clamp(0.0, 1.0) * 0.35;
    Color::from(Srgba::new(
        (base.red * shade).min(1.0),
        (base.green * shade).min(1.0),
        (base.blue * shade).min(1.0),
        1.0,
    ))
}

fn terrain_material_label(material: TerrainMaterial) -> &'static str {
    match material {
        TerrainMaterial::DeepWater => "Deep water",
        TerrainMaterial::Water => "Water",
        TerrainMaterial::ShallowWater => "Shallow water",
        TerrainMaterial::Sand => "Sand",
        TerrainMaterial::Desert => "Desert",
        TerrainMaterial::Savanna => "Savanna",
        TerrainMaterial::Grass => "Grass",
        TerrainMaterial::Forest => "Forest",
        TerrainMaterial::Rainforest => "Rainforest",
        TerrainMaterial::Marsh => "Marsh",
        TerrainMaterial::Tundra => "Tundra",
        TerrainMaterial::Rock => "Rock",
        TerrainMaterial::Snow => "Snow",
    }
}

fn hex_to_egui(hex: &str) -> egui::Color32 {
    let rgba = bevy::color::Srgba::hex(hex).unwrap_or(bevy::color::Srgba::WHITE);
    egui::Color32::from_rgb(
        (rgba.red * 255.0).round() as u8,
        (rgba.green * 255.0).round() as u8,
        (rgba.blue * 255.0).round() as u8,
    )
}

fn legend_row(ui: &mut egui::Ui, color: egui::Color32, label: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, color);
        ui.label(label);
    });
}

fn draw_legend_ui(ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Entities")
        .default_open(true)
        .show(ui, |ui| {
            legend_row(ui, hex_to_egui(CREATURE_COLOR), "Creature");
            legend_row(ui, hex_to_egui(OBSTACLE_COLOR), "Obstacle");
        });

    egui::CollapsingHeader::new("Food")
        .default_open(false)
        .show(ui, |ui| {
            for (kind, label) in [
                (FoodKind::Berries, "Berries"),
                (FoodKind::Fruit, "Fruit"),
                (FoodKind::Fungus, "Fungus"),
                (FoodKind::Kelp, "Kelp"),
                (FoodKind::CactusFruit, "Cactus fruit"),
                (FoodKind::Lichen, "Lichen"),
                (FoodKind::Seeds, "Seeds"),
            ] {
                let bevy::color::Srgba {
                    red, green, blue, ..
                } = resource_food_color(kind).to_srgba();
                let color = egui::Color32::from_rgb(
                    (red * 255.0).round() as u8,
                    (green * 255.0).round() as u8,
                    (blue * 255.0).round() as u8,
                );
                legend_row(ui, color, label);
            }
        });

    egui::CollapsingHeader::new("Poison")
        .default_open(false)
        .show(ui, |ui| {
            for (kind, label) in [
                (PoisonKind::Nightshade, "Nightshade"),
                (PoisonKind::ToxicMushroom, "Toxic mushroom"),
                (PoisonKind::BrineBloom, "Brine bloom"),
                (PoisonKind::ThornPatch, "Thorn patch"),
                (PoisonKind::BitterLichen, "Bitter lichen"),
            ] {
                let bevy::color::Srgba {
                    red, green, blue, ..
                } = resource_poison_color(kind).to_srgba();
                let color = egui::Color32::from_rgb(
                    (red * 255.0).round() as u8,
                    (green * 255.0).round() as u8,
                    (blue * 255.0).round() as u8,
                );
                legend_row(ui, color, label);
            }
        });

    egui::CollapsingHeader::new("Terrain")
        .default_open(false)
        .show(ui, |ui| {
            for material in TERRAIN_MATERIALS {
                let bevy::color::Srgba {
                    red, green, blue, ..
                } = terrain_color(material, 0.5).to_srgba();
                let color = egui::Color32::from_rgb(
                    (red * 255.0).round() as u8,
                    (green * 255.0).round() as u8,
                    (blue * 255.0).round() as u8,
                );
                legend_row(ui, color, terrain_material_label(material));
            }
        });
}
