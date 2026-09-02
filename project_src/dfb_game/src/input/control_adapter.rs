use crate::input::actions::ControlInput;

pub trait ControlSource {
    fn control_input(&self) -> ControlInput;
}
