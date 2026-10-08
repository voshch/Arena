"""Backend for acoustics:=arena, the arena_auditory stack (propagation, emission, rendering, playback)."""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import Sequence

import attrs
from arena_rclpy_mixins.qos import best_effort
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from task_generator_msgs.msg import AdapterDisplay, AdapterPlugin, RecordedTopic

from task_generator.simulators.acoustics import BaseAcousticsSimulator, PedestrianHearing, SoundEmission

if typing.TYPE_CHECKING:
    from arena_auditory_msgs.msg import ContinuousAudioSourceState, ContinuousHeardSoundState
    from builtin_interfaces.msg import Time as TimeMsg

    from task_generator.simulators.human import BaseHumanSimulator

CONTINUOUS_QOS_DEPTH = 64
_RECORDED_TOPIC_FIELDS = ("key", "topic", "msg_type", "robot_scoped", "throttled", "qos_transient_local", "reliable", "depth", "recorded")


def _numeric_id(source_id: str) -> int:
    digest = hashlib.blake2b(source_id.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


class ArenaAcousticsSimulator(BaseAcousticsSimulator):
    """The propagation node builds its acoustic scene from the map topic, so the map server must be up before episodes start."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        from arena_auditory_msgs.msg import ContinuousAudioSourceState

        from arena_auditory.api import CONTINUOUS_AUDIO_SOURCES

        super().__init__(*args, **kwargs)
        self._source_publisher = self.node.create_publisher(
            ContinuousAudioSourceState,
            self.node.service_namespace(CONTINUOUS_AUDIO_SOURCES),
            best_effort(CONTINUOUS_QOS_DEPTH),
        )
        self._emitted: dict[str, SoundEmission] = {}
        self._retiring: dict[str, tuple[SoundEmission, int]] = {}

    @property
    def requires_map_server(self) -> bool:
        return True

    def displays(self) -> tuple[AdapterDisplay, ...]:
        from arena_auditory.api import ENVIRONMENT_SOURCE_MARKERS, MICROPHONE_MARKERS, PEDESTRIAN_PROPAGATION_MARKERS, ROBOT_PROPAGATION_MARKERS

        auditory_ns = self.node.get_fully_qualified_name()
        return tuple(
            AdapterDisplay(
                name=name,
                topic=f"{auditory_ns}/{topic}",
                topic_type="visualization_msgs/MarkerArray",
                kind=DisplayKind.MARKER_ARRAY,
                style_json=StyleSpec(latched=True).to_json() if topic == MICROPHONE_MARKERS else StyleSpec(enabled=True).to_json(),
                topic_must_exist=False,
                group="Sound Propagation",
            )
            for name, topic in (
                ("Microphones", MICROPHONE_MARKERS),
                ("Environment Audio Sources", ENVIRONMENT_SOURCE_MARKERS),
                ("Pedestrian Heard Sound", PEDESTRIAN_PROPAGATION_MARKERS),
                ("Robot Heard Sound", ROBOT_PROPAGATION_MARKERS),
            )
        )

    def robot_displays(self, robot: str) -> tuple[AdapterDisplay, ...]:
        from arena_robots.audio import ArrayStream, array_stream

        from arena_auditory.api import motor_markers

        auditory_ns = self.node.get_fully_qualified_name()
        return tuple(
            AdapterDisplay(
                name=name,
                topic=f"{auditory_ns}/{topic}",
                topic_type="visualization_msgs/MarkerArray",
                kind=DisplayKind.MARKER_ARRAY,
                style_json=StyleSpec(enabled=True).to_json(),
                topic_must_exist=False,
            )
            for name, topic in (
                ("Motor Sound", motor_markers(robot)),
                ("Mic Levels", array_stream(robot, ArrayStream.LEVELS)),
            )
        )

    def plugins(self) -> tuple[AdapterPlugin, ...]:
        from arena_auditory.api import rviz_plugins

        return tuple(AdapterPlugin(role=role, class_name=class_name, name=name, properties_json=json.dumps(properties)) for role, class_name, name, properties in rviz_plugins(self.node.get_fully_qualified_name()))

    def recorded_topics(self) -> tuple[RecordedTopic, ...]:
        from arena_auditory.api import recorded_topics

        return tuple(RecordedTopic(**dict(zip(_RECORDED_TOPIC_FIELDS, row, strict=True))) for row in recorded_topics())

    def pedestrian_hearing(self, human: BaseHumanSimulator) -> PedestrianHearing:
        from arena_auditory_msgs.msg import ContinuousHeardSoundState

        from arena_auditory.api import CONTINUOUS_HEARD_SOUNDS, ListenerId, ListenerKind

        hearing = PedestrianHearing(human)

        async def on_heard_sound(msg: ContinuousHeardSoundState) -> None:
            try:
                listener = ListenerId.parse(msg.reception.listener_id)
            except ValueError:
                return
            if listener.kind is not ListenerKind.AGENT:
                return
            await hearing.on_heard(int(listener.owner), msg.source.kind, msg.source.id, bool(msg.source.active and msg.reception.audible))

        human.node.create_subscription(
            ContinuousHeardSoundState,
            human.node.service_namespace(CONTINUOUS_HEARD_SOUNDS),
            on_heard_sound,
            best_effort(CONTINUOUS_QOS_DEPTH),
        )
        return hearing

    def emit_sounds(self, sounds: Sequence[SoundEmission]) -> None:
        from arena_auditory.api import INACTIVE_REPEATS

        listed = {sound.entity: sound for sound in sounds}
        for entity, emitted in self._emitted.items():
            if emitted.active and entity not in listed:
                self._retiring[entity] = (attrs.evolve(emitted, active=False), INACTIVE_REPEATS)
        self._emitted = listed
        outgoing = list(sounds)
        for entity, (retired, left) in tuple(self._retiring.items()):
            if entity in listed:
                del self._retiring[entity]
                continue
            outgoing.append(retired)
            if left > 1:
                self._retiring[entity] = (retired, left - 1)
            else:
                del self._retiring[entity]
        stamp = self.node.get_clock().now().to_msg()
        for sound in outgoing:
            self._source_publisher.publish(self.source_msg(sound, stamp))

    def clear_sounds(self) -> None:
        self._emitted.clear()
        self._retiring.clear()

    @staticmethod
    def source_msg(sound: SoundEmission, stamp: TimeMsg) -> ContinuousAudioSourceState:
        """Wire form of one emission."""
        from arena_auditory_msgs.msg import ContinuousAudioSourceState
        from arena_simulation_setup.tree.assets.sound_catalog import WAV_MODELS, AgentKind

        from arena_auditory.api import SourceSpec

        source = SourceSpec(
            id=f"environment:{sound.entity}",
            group_id=sound.group_id,
            kind=sound.kind,
            asset_id=sound.asset_id,
            variant_id=sound.variant_id,
            model="wav_loop" if sound.model in WAV_MODELS else sound.model,
            agent_kind=AgentKind.ENVIRONMENT,
            agent_id=_numeric_id(sound.entity),
            agent_name=sound.name,
            tags=sound.tags,
            position=sound.position,
            yaw_rad=sound.yaw,
            level_db=sound.level_db,
            reference_distance_m=sound.reference_distance_m,
            loop=sound.loop,
            active=sound.active,
            program_start_ns=sound.program_start_ns,
            seed=sound.seed,
        )
        msg = ContinuousAudioSourceState(source=source.to_msg())
        msg.header.stamp = stamp
        msg.header.frame_id = sound.frame_id
        return msg
