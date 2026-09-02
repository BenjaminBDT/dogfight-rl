from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dfb_reinforcement_learning.policy_contract import POLICY_CONTRACT


@dataclass(frozen=True)
class PolicyObservationField:
    name: str
    offset: int
    size: int

    @property
    def value_slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


@dataclass(frozen=True)
class PolicyObservationSchema:
    schema_id: str
    dim: int
    fields: tuple[PolicyObservationField, ...]
    binary_indices: tuple[int, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def field_slices(self) -> dict[str, slice]:
        return {field.name: field.value_slice for field in self.fields}


def _build_observation_schema(payload: dict[str, Any]) -> PolicyObservationSchema:
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise RuntimeError("policy contract observation must be an object")
    raw_fields = observation.get("fields")
    if not isinstance(raw_fields, list):
        raise RuntimeError("policy contract observation.fields must be an array")

    fields: list[PolicyObservationField] = []
    expected_offset = 0
    names: set[str] = set()
    for raw_field in raw_fields:
        if not isinstance(raw_field, dict):
            raise RuntimeError("policy contract observation field must be an object")
        field = PolicyObservationField(
            name=str(raw_field["name"]),
            offset=int(raw_field["offset"]),
            size=int(raw_field["size"]),
        )
        if not field.name or field.name in names:
            raise RuntimeError(f"duplicate or empty observation field: {field.name!r}")
        if field.offset != expected_offset or field.size <= 0:
            raise RuntimeError(f"non-contiguous observation field: {field.name}")
        fields.append(field)
        names.add(field.name)
        expected_offset += field.size

    dim = int(observation["dim"])
    if expected_offset != dim:
        raise RuntimeError(f"observation fields total {expected_offset}, expected {dim}")
    binary_indices = tuple(int(index) for index in observation["binary_indices"])
    if len(set(binary_indices)) != len(binary_indices) or any(index < 0 or index >= dim for index in binary_indices):
        raise RuntimeError("invalid observation binary indices")
    return PolicyObservationSchema(
        schema_id=str(observation["schema_id"]),
        dim=dim,
        fields=tuple(fields),
        binary_indices=binary_indices,
    )


POLICY_OBSERVATION_SCHEMA = _build_observation_schema(POLICY_CONTRACT)
