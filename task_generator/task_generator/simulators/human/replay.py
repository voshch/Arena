"""Backend for human:=replay, a pedestrian whose scenario entry names a track is played from it on the episode clock."""

from __future__ import annotations

import asyncio
import math
import traceback
from collections.abc import Sequence

import attrs
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_simulation_setup.shared.track import Track, TrackState, track_file

from task_generator.constants.rng import stable_int
from task_generator.shared import DynamicObstacle, Orientation, Pose, Position
from task_generator.simulators.human.dummy import DummyHumanSimulator

RATE_HZ = 30.0
WALK_SPEED = 0.1


@attrs.frozen
class Recorded:
    obstacle: DynamicObstacle
    track: Track
    prespawn: bool = False


def pedestrian(name: str, state: TrackState) -> Pedestrian:
    ped = Pedestrian()
    ped.name = name
    ped.id = stable_int(name) & 0xFFFFFFFF
    ped.pose = Pose(Position(state.x, state.y), Orientation.from_yaw(state.yaw)).to_msg()
    ped.twist.linear.x = state.vx
    ped.twist.linear.y = state.vy
    ped.animation_state = Pedestrian.WALKING if math.hypot(state.vx, state.vy) >= WALK_SPEED else Pedestrian.IDLE
    return ped


class ReplayHumanSimulator(DummyHumanSimulator):
    """Recorded stage: a tracked pedestrian is on the bus exactly while its track holds the episode time, the rest stay puppets."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._recorded: dict[str, Recorded] = {}
        self._in_sim: set[str] = set()
        self._episode_start = 0.0
        self._replay_task: asyncio.Task = asyncio.create_task(self._replay_loop())

    def _sim_seconds(self) -> float:
        return self.node.sim_time.to_seconds()

    async def _replay_loop(self) -> None:
        last_error = ""
        try:
            with self.node.sim_time_rate(RATE_HZ) as (done, rate):
                while not done.is_set():
                    await rate.get()
                    try:
                        await self.replay(self._sim_seconds() - self._episode_start)
                    except Exception as e:
                        if repr(e) != last_error:
                            self._logger.error(f"Error in replay tick: {e}\n{traceback.format_exc()}")
                        last_error = repr(e)
                    else:
                        last_error = ""
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Error in replay loop: {e}\n{traceback.format_exc()}")

    async def _keepalive_loop(self) -> None:
        """Publishes nothing, the replay loop publishes the roster."""

    async def replay(self, t: float) -> None:
        """Put every tracked pedestrian at episode time t: on the bus and in the sim while its track holds t, gone otherwise."""
        if not self._recorded:
            self._publish_roster()
            return
        batch = Pedestrians()
        batch.header.frame_id = "map"
        entered: dict[str, DynamicObstacle] = {}
        left: dict[str, DynamicObstacle] = {}
        for name, recorded in self._recorded.items():
            if name not in self._ped_model_uris:
                continue
            state = recorded.track.state(t, hold_start=recorded.prespawn)
            if state is None:
                self._cache.pop(name, None)
                if name in self._in_sim:
                    left[name] = recorded.obstacle
                continue
            ped = pedestrian(name, state)
            self._cache[name] = ped
            batch.pedestrians.append(ped)  # pyright: ignore[reportAttributeAccessIssue]
            if name not in self._in_sim:
                entered[name] = attrs.evolve(recorded.obstacle, pose=Pose.from_msg(ped.pose))
        if left:
            await self._simulator.pedestrian_delete(list(left.values()))
            self._in_sim.difference_update(left)
        if entered:
            await self._simulator.pedestrian_spawn(await self._ensure_spawnable(list(entered.values())))
            self._in_sim.update(entered)
        self._publish_roster()
        if batch.pedestrians:
            await self.relay_pedestrian_update(batch)

    async def _spawn_dynamic_obstacles_impl(
        self,
        obstacles: Sequence[DynamicObstacle],
    ) -> Sequence[DynamicObstacle | None]:
        puppets: list[DynamicObstacle] = []
        origin = self._realizer.get_config()
        for obstacle in obstacles:
            path = track_file(obstacle)
            if path is None:
                puppets.append(obstacle)
                continue
            if not self._recorded:
                self._episode_start = self._sim_seconds()
            level_x, level_y = self._realizer.get_level_origin(obstacle.level)
            self._recorded[obstacle.sim_path] = Recorded(obstacle, Track.load(path).shifted(origin.x + level_x, origin.y + level_y), obstacle.prespawn)
            self._in_sim.add(obstacle.sim_path)
        await super()._spawn_dynamic_obstacles_impl(puppets)
        return obstacles

    async def _remove_pedestrians_impl(self) -> bool:
        self._recorded.clear()
        self._in_sim.clear()
        return await super()._remove_pedestrians_impl()
