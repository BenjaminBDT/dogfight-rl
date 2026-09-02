pub mod game_app;
pub mod schedules;

use bevy::prelude::Resource;

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct HeadlessMode(pub bool);

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct RenderEnabled(pub bool);

#[derive(Debug, Clone, Copy, Resource, PartialEq, Eq, Default)]
pub enum AppMode {
    #[default]
    Game,
    Observer,
}

#[derive(Debug, Clone, Copy, Resource, PartialEq, Eq, Default)]
pub enum ObserverClientMode {
    #[default]
    RecordedEpisode,
    LiveServer,
}
