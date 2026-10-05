from __future__ import annotations

import csv
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from arena_simulation_setup.acoustics.export_recording import (
    PCM_F32LE,
    AudioBlock,
    _agent_states_pedestrians,
    _merge_pedestrian_sources,
    assemble_audio,
    audio_statistics,
    build_rendered_activity_intervals,
    build_labels,
    clip_audio_chunks,
    occupancy_ray_labels,
    select_map_snapshot,
    transform_robot_trajectory,
    write_csv,
    write_map_snapshot,
    write_parquet,
)


def chunk(index: int, timestamp_ns: int, values: list[tuple[float, float]]) -> AudioBlock:
    payload = b"".join(struct.pack("<ff", *frame) for frame in values)
    return AudioBlock(
        topic="/arena/env_0/jackal/audio/headphones/stereo",
        timestamp_ns=timestamp_ns,
        first_sample_index=index,
        sample_rate=1000,
        channels=2,
        encoding=PCM_F32LE,
        stream_id="rendered",
        channel_names=("left", "right"),
        microphone_frame="robot/microphones",
        payload=payload,
    )


def test_uint64_seed_can_be_written_losslessly(tmp_path: Path) -> None:
    path = tmp_path / "continuous.parquet"
    seed = str(2**64 - 1)

    write_parquet(path, [{"deterministic_seed": seed, "active": True}])

    import pyarrow.parquet as pq
    assert pq.read_table(path).to_pylist() == [
        {"deterministic_seed": seed, "active": True}
    ]


def test_assemble_audio_uses_sample_index_and_reports_gap():
    audio, timing, summary = assemble_audio(
        [
            chunk(0, 1_000_000_000, [(0.1, -0.1), (0.2, -0.2)]),
            chunk(3, 1_003_000_000, [(0.3, -0.3)]),
        ]
    )

    assert audio.shape == (4, 2)
    np.testing.assert_array_equal(audio[2], [0.0, 0.0])
    assert timing[1]["gap_frames_before"] == 1
    assert timing[1]["timestamp_error_ns"] == 0
    assert summary["gap_frames"] == 1


def test_clip_audio_chunks_uses_half_open_simulation_time_window():
    source = chunk(0, 1_000_000_000, [(0.0, 0.0), (0.1, -0.1), (0.2, -0.2), (0.3, -0.3)])

    clipped = clip_audio_chunks([source], 1_001_000_000, 1_003_000_000)

    assert len(clipped) == 1
    assert clipped[0].timestamp_ns == 1_001_000_000
    assert clipped[0].first_sample_index == 1
    np.testing.assert_allclose(np.frombuffer(clipped[0].payload, dtype="<f4").reshape((-1, 2)), [[0.1, -0.1], [0.2, -0.2]])


def test_map_snapshot_selects_latest_preceding_grid_and_is_self_describing(
    tmp_path: Path,
) -> None:
    rows = [
        {"timestamp_ns": 10, "topic": "/arena/env_0/map", "frame_id": "map", "resolution": 0.1, "width": 2, "height": 1, "origin_x": 1.0, "origin_y": 2.0, "origin_z": 0.0, "origin_yaw": 0.0, "data": np.asarray([[0, 100]], dtype=np.int8)},
        {"timestamp_ns": 20, "topic": "/arena/env_0/map", "frame_id": "map", "resolution": 0.1, "width": 2, "height": 1, "origin_x": 3.0, "origin_y": 4.0, "origin_z": 0.0, "origin_yaw": 0.0, "data": np.asarray([[-1, 0]], dtype=np.int8)},
    ]

    selected = select_map_snapshot(rows, 15)
    metadata = write_map_snapshot(tmp_path / "map.npz", selected)

    assert selected["timestamp_ns"] == 10
    assert len(metadata["sha256"]) == 64
    with np.load(tmp_path / "map.npz") as saved:
        np.testing.assert_array_equal(saved["occupancy"], [[0, 100]])
        np.testing.assert_allclose(saved["origin"], [1.0, 2.0, 0.0, 0.0])
        assert saved["frame_id"].item() == "map"


def test_occupancy_ray_labels_detects_occlusion():
    snapshot = {
        "resolution": 1.0,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "origin_yaw": 0.0,
        "width": 4,
        "height": 2,
        "data": np.asarray([[0, 0, 100, 0], [0, 0, 0, 0]], dtype=np.int8),
    }

    blocked = occupancy_ray_labels(snapshot, (0.5, 0.5), (3.5, 0.5))
    clear = occupancy_ray_labels(snapshot, (0.5, 1.5), (3.5, 1.5))

    assert blocked["line_of_sight"] is False
    assert blocked["ray_occupied_cell_count"] > 0
    assert clear["line_of_sight"] is True


def test_robot_odometry_is_transformed_into_map_frame():
    odom = [
        {
            "timestamp_ns": 100,
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0,
            "frame_id": "odom",
            "child_frame_id": "base_link",
            "topic": "/env/odom",
        }
    ]
    transforms = {
        ("map", "odom"): [
            {
                "timestamp_ns": 0,
                "x": 10.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": np.pi / 2,
                "static": True,
            }
        ]
    }

    aligned, provenance = transform_robot_trajectory(odom, transforms, "map", 100_000_000)

    np.testing.assert_allclose([aligned[0]["x"], aligned[0]["y"]], [10.0, 3.0])
    np.testing.assert_allclose([aligned[0]["vx"], aligned[0]["vy"]], [0.0, 1.0], atol=1e-7)
    assert aligned[0]["frame_id"] == "map"
    assert provenance["transform"] == "map->odom"
    np.testing.assert_allclose(
        [aligned[0]["qx"], aligned[0]["qy"], aligned[0]["qz"], aligned[0]["qw"]],
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
    )


def test_rendered_activity_intervals_merge_channels_and_classify_frames():
    records = [
        {
            "event_id": "step-1", "source_id": "agent_7",
            "source_agent_id": 7, "source_agent_name": "agent_7",
            "source_type": "pedestrian", "sound_type": "footstep",
            "asset_id": "footstep", "channel_name": channel,
            "continuous": False, "active": True,
            "start_time_ns": start, "end_time_ns": end,
        }
        for channel, start, end in (
            ("front_left", 1_002_000_000, 1_012_000_000),
            ("front_right", 1_003_000_000, 1_013_000_000),
        )
    ]
    records.extend(
        [
            {
                "event_id": "continuous:motor", "source_id": "motor",
                "source_agent_id": -1, "source_agent_name": "jackal",
                "source_type": "robot", "sound_type": "motor", "asset_id": "",
                "channel_name": "front_left", "continuous": True,
                "active": active, "start_time_ns": timestamp,
                "end_time_ns": timestamp,
            }
            for active, timestamp in ((True, 1_005_000_000), (False, 1_015_000_000))
        ]
    )

    intervals = build_rendered_activity_intervals(
        records,
        capture_start_ns=1_000_000_000,
        capture_end_ns=1_020_000_000,
        sample_rate=1000,
        first_sample_index=100,
    )

    assert len(intervals) == 2
    footstep = next(row for row in intervals if row["sound_type"] == "footstep")
    assert footstep["start_recording_sample_offset"] == 2
    assert footstep["end_recording_sample_offset"] == 13
    assert footstep["channel_names"] == ["front_left", "front_right"]

    audio = np.ones((20, 2), dtype=np.float32) * 0.5
    summary = {"sample_rate": 1000, "first_timestamp_ns": 1_000_000_000, "first_sample_index": 100}
    robot = [{"timestamp_ns": 1_000_000_000, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}]
    labels = build_labels(
        audio, summary, audio, summary, robot, {},
        frame_ms=10, max_pose_gap_ms=20, emit_robot_only=True,
        activity_intervals=intervals,
    )
    assert labels[0]["activity_class"] == "single_pedestrian_plus_motor"
    assert labels[0]["active_pedestrian_ids"] == [7]
    assert labels[1]["activity_class"] == "single_pedestrian_plus_motor"


def test_labels_join_audio_to_interpolated_robot_and_pedestrian_pose():
    audio = np.ones((20, 2), dtype=np.float32) * 0.5
    summary = {
        "sample_rate": 1000,
        "first_timestamp_ns": 1_000_000_000,
        "first_sample_index": 0,
    }
    robot = [
        {"timestamp_ns": 1_000_000_000, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0},
        {"timestamp_ns": 1_020_000_000, "x": 2.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 1.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0},
    ]
    peds = {
        "ped_1": [
            {"timestamp_ns": 1_000_000_000, "pedestrian_id": 1, "pedestrian_name": "ped_1", "x": 0.0, "y": 2.0, "z": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "model_uri": "adult"},
            {"timestamp_ns": 1_020_000_000, "pedestrian_id": 1, "pedestrian_name": "ped_1", "x": 2.0, "y": 2.0, "z": 0.0, "yaw": 0.0, "vx": 1.0, "vy": 0.0, "vz": 0.0, "model_uri": "adult"},
        ]
    }

    labels = build_labels(audio, summary, audio, summary, robot, peds, frame_ms=10, max_pose_gap_ms=20)

    assert len(labels) == 2
    assert labels[1]["robot_x"] == 1.0
    assert labels[1]["pedestrian_x"] == 1.0
    assert labels[1]["range_m"] == 2.0
    assert labels[1]["bearing_robot_rad"] == np.pi / 2
    assert labels[1]["rendered_ch0_rms"] == 0.5
    assert labels[1]["recording_sample_offset"] == 10
    assert labels[1]["recording_time_seconds"] == 0.01


def test_labels_can_emit_robot_audio_rows_without_pedestrian_samples():
    audio = np.ones((10, 2), dtype=np.float32) * 0.25
    summary = {
        "sample_rate": 1000,
        "first_timestamp_ns": 1_000_000_000,
        "first_sample_index": 0,
    }
    robot = [
        {
            "timestamp_ns": 1_000_000_000,
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "yaw": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0,
        }
    ]

    labels = build_labels(
        audio,
        summary,
        audio,
        summary,
        robot,
        {},
        frame_ms=10,
        max_pose_gap_ms=20,
        emit_robot_only=True,
    )

    assert len(labels) == 1
    assert labels[0]["pedestrian_present"] is False
    assert labels[0]["robot_x"] == 1.0


def test_agent_states_are_human_source_pose_samples_in_map_frame():
    header = SimpleNamespace(stamp=SimpleNamespace(sec=7, nanosec=25), frame_id="")
    human = SimpleNamespace(
        agent_id=12,
        kind=0,
        pose=SimpleNamespace(x=1.5, y=-2.0, theta=0.75),
        velocity=SimpleNamespace(x=0.3, y=0.4, z=0.0),
        radius=0.35,
        desired_velocity=1.2,
        agent_type="adult",
        policy="social_force",
    )
    robot = SimpleNamespace(
        agent_id=99,
        kind=1,
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        radius=0.4,
        desired_velocity=0.0,
        agent_type="",
        policy="",
    )

    decoded = _agent_states_pedestrians(SimpleNamespace(header=header, agents=[human, robot]), "/arena/env_0/agent_states", 1)

    assert list(decoded) == ["agent_12"]
    row = decoded["agent_12"][0]
    assert row["timestamp_ns"] == 7_000_000_025
    assert row["frame_id"] == "map"
    assert row["yaw"] == 0.75
    assert row["state_source"] == "agent_states"


def test_arena_pedestrians_are_enriched_from_agent_states_by_shared_id():
    arena = {
        "ped_1": [
            {
                "timestamp_ns": 100,
                "pedestrian_id": 12,
                "pedestrian_name": "ped_1",
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": 0.0,
                "vx": 0.1,
                "vy": 0.0,
                "vz": 0.0,
                "animation_state": 1,
                "model_uri": "adult",
                "radius": None,
                "desired_velocity": None,
                "agent_type": "",
                "policy": "",
                "state_source": "arena_peds",
                "state_source_topic": "/arena_peds",
                "topic": "/arena_peds",
                "frame_id": "map",
            }
        ]
    }
    agents = {
        "agent_12": [
            {
                "timestamp_ns": 100,
                "pedestrian_id": 12,
                "pedestrian_name": "agent_12",
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": 0.0,
                "vx": 0.1,
                "vy": 0.0,
                "vz": 0.0,
                "animation_state": None,
                "model_uri": "",
                "radius": 0.35,
                "desired_velocity": 1.2,
                "agent_type": "adult",
                "policy": "social_force",
                "state_source": "agent_states",
                "state_source_topic": "/agent_states",
                "topic": "/agent_states",
                "frame_id": "map",
            }
        ]
    }

    row = _merge_pedestrian_sources(arena, agents)["ped_1"][0]

    assert row["pedestrian_name"] == "ped_1"
    assert row["model_uri"] == "adult"
    assert row["radius"] == 0.35
    assert row["desired_velocity"] == 1.2
    assert row["state_source"] == "arena_peds+agent_states"
    assert row["state_source_topic"] == ["/arena_peds", "/agent_states"]


def test_audio_statistics_reports_each_microphone_channel():
    audio = np.asarray([[0.0, 0.5], [1.0, -0.5]], dtype=np.float32)

    statistics = audio_statistics(audio, ["front", "rear"])

    assert statistics["peak"] == 1.0
    assert statistics["per_channel"][0]["name"] == "front"
    assert statistics["per_channel"][0]["peak"] == 1.0
    assert statistics["per_channel"][1]["rms"] == 0.5


def test_metadata_csv_preserves_synchronized_scalar_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "0001_meta.csv"
    write_csv(
        path,
        [
            {
                "execution_index": 1,
                "scenario": "hearing_case",
                "timestamp_ns": 1_010_000_000,
                "recording_sample_offset": 480,
                "robot_x": 1.25,
                "pedestrian_x": 2.5,
                "line_of_sight": True,
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "execution_index": "1",
            "scenario": "hearing_case",
            "timestamp_ns": "1010000000",
            "recording_sample_offset": "480",
            "robot_x": "1.25",
            "pedestrian_x": "2.5",
            "line_of_sight": "True",
        }
    ]
