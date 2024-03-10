use bevy::prelude::*;

use crate::{InspectWindowState, TabState};

use self::systems::*;

mod systems;

pub struct MainMenuPlugin;

impl Plugin for MainMenuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, main_menu.run_if(in_state(TabState::MainMenu)))
            .add_systems(
                Update,
                inspector_exit.run_if(in_state(InspectWindowState::Display)),
            );
    }
}
