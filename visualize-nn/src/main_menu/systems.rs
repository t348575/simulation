use bevy::{prelude::*, window::{PrimaryWindow, WindowClosed}};
use bevy_egui::{egui, EguiContext};

use crate::{InspectWindowState, TabState};

pub fn main_menu(
    mut egui_ctx: Query<&mut EguiContext, With<PrimaryWindow>>,
    mut next_tab_state: ResMut<NextState<TabState>>,
    mut next_inspector_state: ResMut<NextState<InspectWindowState>>,
    inspector_state: Res<State<InspectWindowState>>,
) {
    egui::Window::new("Main Menu")
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .show(egui_ctx.single_mut().get_mut(), |ui| {
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
                        InspectWindowState::None => InspectWindowState::Display
                    });
                }
            });
        })
        .unwrap();
}


pub fn inspector_exit(mut events: EventReader<WindowClosed>, mut next_inspector_state: ResMut<NextState<InspectWindowState>>) {
    for event in events.read() {
        next_inspector_state.set(InspectWindowState::None);
        return;
    }
}