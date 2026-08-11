---
name: project-sensor-topic-binding
description: "Planner manifests declare a sensor TYPE; the DRL adapter binds it to the robot's actual topic"
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-08T03:00:45.441Z
---

Sensor topics are per-robot: 14 robots publish `${namespace}/lidar`, 4 publish `${namespace}/scan`
(mpo700, rbtheron, rbvogui, rbvogui_plus), boxer publishes `${namespace}/front_laser`. Planner
manifests used to hardcode `topic: scan`, so on jackal five planners (height, crowdsurfer, drlvo,
navrep, scope) subscribed to a topic nobody published and silently received nothing -- their
wrappers fall back to an all-max-range scan, i.e. the policy drives blind.

Fixed 2026-08-07: a datasource declares `sensor: <SensorType>` instead of `topic:`, and
`DrlAdapter._bind_sensor_topics` resolves it against `effective_sensors` via
`arena_robots.Sensor.resolve_topic(spec, namespace)`. Unresolvable datasources are dropped with a
warning (plus any alias targeting them). `Pipeline.from_config` raises if an unbound `sensor` key
reaches it, since it would otherwise default the topic to the datasource name and fail silently.

`sensor_needs` did NOT catch this: it validates the type is present and then ignores the topic that
came with it. See [[project-drl-sim-vs-offline-gap]].

**Why:** the failure is invisible -- no error, no missing topic warning, just a policy with no
perception.

**How to apply:** never hardcode a robot sensor topic anywhere. ON ICE (user's call, 2026-08-07):
removing the `${namespace}` placeholder from SensorSpec entirely. Cost is 116 topic lines across 20
model_params.yaml plus catalog.py, node.py, the gz bridge and nav2.py -- and nav2 ships `s.topic`
into `sensors_json` with the placeholder intact for its own substituter, so that path needs rework.
Benefit is hygiene only, zero behaviour change, with a silent-no-data failure mode. Bad ratio;
revisit only if the sensor schema is being touched anyway.
