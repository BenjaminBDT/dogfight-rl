from __future__ import annotations

import torch

from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.policy_contract import OBS_DIM
from dfb_reinforcement_learning.train.train_distill_bc import copy_overlapping_parameters


def test_copy_overlapping_parameters_inflates_teacher_into_larger_student() -> None:
    obs_dim = OBS_DIM
    teacher = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=256, num_layers=3)
    student = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=512, num_layers=4)

    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.fill_(0.25)
        for parameter in student.parameters():
            parameter.zero_()

    copied = copy_overlapping_parameters(student, teacher.state_dict())
    assert copied > 0

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()

    assert torch.allclose(student_state["shared_stem.0.weight"][:256, :], teacher_state["shared_stem.0.weight"])
    assert torch.allclose(student_state["shared_stem.0.bias"][:256], teacher_state["shared_stem.0.bias"])
    assert torch.allclose(student_state["shared_stem.1.weight"][:256], teacher_state["shared_stem.1.weight"])
    assert torch.allclose(student_state["shared_stem.1.bias"][:256], teacher_state["shared_stem.1.bias"])
    assert torch.allclose(student_state["action_cont_head.weight"][:, :256], teacher_state["action_cont_head.weight"])
    assert torch.allclose(student_state["action_bin_head.weight"][:, :256], teacher_state["action_bin_head.weight"])
    assert torch.allclose(student_state["value_head.weight"][:, :256], teacher_state["value_head.weight"])

    assert torch.count_nonzero(student_state["actor_tower.3.linear1.weight"]).item() == 0
    assert torch.count_nonzero(student_state["critic_tower.3.linear1.weight"]).item() == 0
