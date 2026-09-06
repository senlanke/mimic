"""Isaac-style world-frame net normal forces and 1 N contact timing."""

from dataclasses import dataclass

import torch
import warp as wp
from mjlab.sensor.contact_sensor import (
  ContactData, ContactSensor, ContactSensorCfg, _AirTimeState,
)
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5


@wp.kernel
def _sum_normal_forces(
  cone: int,
  frame: wp.array[wp.mat33],
  friction: wp.array[vec5],
  dim: wp.array[int],
  efc_address: wp.array2d[int],
  efc_force: wp.array2d[float],
  njmax: int,
  nacon: wp.array[int],
  geom: wp.array[wp.vec2i],
  worldid: wp.array[int],
  geom_primary: wp.array[int],
  forces: wp.array2d[wp.vec3],
):
  cid = wp.tid()
  if cid >= nacon[0]:
    return
  first = geom_primary[geom[cid][0]]
  second = geom_primary[geom[cid][1]]
  if first < 0 and second < 0:
    return
  world = worldid[cid]
  wrench = contact_force_fn(
    cone, frame, friction, dim, efc_address, efc_force, njmax, nacon,
    world, cid, False,
  )
  normal_force = wp.transpose(frame[cid]) @ wp.vec3(wrench[0], 0.0, 0.0)
  if first >= 0:
    wp.atomic_add(forces, world, first, -normal_force)
  if second >= 0:
    wp.atomic_add(forces, world, second, normal_force)


@dataclass
class AMEContactSensorCfg(ContactSensorCfg):
  def build(self):
    return AMEContactSensor(self)


class AMEContactSensor(ContactSensor):
  @property
  def primary_names(self):
    return self._primary_names

  def edit_spec(self, scene_spec, entities):
    self._primary_names = self._resolve_primary_names(entities, self.cfg.primary)

  def initialize(self, mj_model, model, data, device):
    self._data = data
    self._model = model.struct
    self._device = device
    body_primary = {
      mj_model.body(f"{self.cfg.primary.entity}/{name}").id: index
      for index, name in enumerate(self.primary_names)
    }
    geom_primary = torch.tensor(
      [body_primary.get(int(body), -1) for body in mj_model.geom_bodyid],
      dtype=torch.int32, device=device,
    )
    self._geom_primary = wp.from_torch(geom_primary)
    shape = (data.nworld, len(self.primary_names))
    self._force = torch.zeros((*shape, 3), device=device)
    self._force_wp = wp.from_torch(self._force, dtype=wp.vec3)
    self._history_state = {
      "force": torch.zeros((*shape, self.cfg.history_length, 3), device=device)
    }
    if self.cfg.track_air_time:
      self._air_time_state = _AirTimeState(
        current_air_time=torch.zeros(shape, device=device),
        last_air_time=torch.zeros(shape, device=device),
        current_contact_time=torch.zeros(shape, device=device),
        last_contact_time=torch.zeros(shape, device=device),
        last_time=torch.zeros(data.nworld, device=device),
      )

  def update(self, dt):
    data = self._data.struct
    contact = data.contact
    self._force.zero_()
    wp.launch(
      _sum_normal_forces, dim=data.naconmax,
      inputs=[
        self._model.opt.cone, contact.frame, contact.friction, contact.dim,
        contact.efc_address, data.efc.force, data.njmax, data.nacon,
        contact.geom, contact.worldid, self._geom_primary,
      ],
      outputs=[self._force_wp], device=self._device,
    )
    super().update(dt)

  def _extract_sensor_data(self):
    return ContactData(
      force=self._force,
      found=torch.linalg.vector_norm(self._force, dim=-1) > 1.0,
    )

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self._force[slice(None) if env_ids is None else env_ids] = 0.0


__all__ = ["AMEContactSensorCfg"]
