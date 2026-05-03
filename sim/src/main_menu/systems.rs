use bevy::{prelude::*, window::WindowClosed};
use bevy_egui::{egui, EguiContexts};

use crate::{InspectWindowState, TabState};

pub fn main_menu(
    mut egui_ctx: EguiContexts,
    mut next_tab_state: ResMut<NextState<TabState>>,
    mut next_inspector_state: ResMut<NextState<InspectWindowState>>,
    inspector_state: Res<State<InspectWindowState>>,
) {
    let egui_ctx = egui_ctx.ctx_mut().unwrap();
    egui::Window::new("Main Menu")
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                let button =
                    ui.add_sized((ui.available_width(), 0.0), egui::Button::new("Simulation"));
                if button.clicked() {
                    next_tab_state.set(TabState::Simulation);
                }
            });

            ui.horizontal(|ui| {
                let button = ui.add_sized(
                    (ui.available_width(), 0.0),
                    egui::Button::new("Neural net viewer"),
                );
                if button.clicked() {
                    next_inspector_state.set(match inspector_state.get() {
                        InspectWindowState::Display => InspectWindowState::None,
                        InspectWindowState::None => InspectWindowState::Display,
                    });
                }
            });
        })
        .unwrap();
}

pub fn inspector_exit(
    mut events: MessageReader<WindowClosed>,
    mut next_inspector_state: ResMut<NextState<InspectWindowState>>,
) {
    for _ in events.read() {
        next_inspector_state.set(InspectWindowState::None);
        return;
    }
}
