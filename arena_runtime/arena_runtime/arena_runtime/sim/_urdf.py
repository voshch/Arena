import xml.etree.ElementTree as ET


def transform_urdf_for_bridge(
    urdf_path: str,
    commands_topics: dict[str, str],
    states_topic: str,
    out_path: str,
) -> None:
    """Rewrite the URDF for the Isaac bridge.

    - Swap gz_ros2_control/GazeboSimSystem to JointStateTopicSystem.
    - Split each block into per-command-interface-kind blocks so JointStateTopicSystem
      emits well-formed JointState messages (one kind per topic). Each kind goes to
      its own commands_topics[kind].
    - Add <state_interface name="effort"/> to every joint, because JointStateTopicSystem
      unconditionally writes incoming effort and would otherwise crash.
    """
    bridge_plugin = 'joint_state_topic_hardware_interface/JointStateTopicSystem'
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for plugin_el in root.iter('plugin'):
        if plugin_el.text and plugin_el.text.strip() == 'gz_ros2_control/GazeboSimSystem':
            plugin_el.text = bridge_plugin

    originals = [rc for rc in root.iter('ros2_control') if (h := rc.find('hardware')) is not None and (p := h.find('plugin')) is not None and p.text and p.text.strip() == bridge_plugin]

    for rc in originals:
        parent = next(p for p in root.iter() if rc in list(p))
        name = rc.get('name', 'bridge')
        rc_type = rc.get('type', 'system')

        by_kind: dict[str, list[ET.Element]] = {}
        for joint in rc.findall('joint'):
            cmd = joint.find('command_interface')
            if cmd is None:
                continue
            kind = cmd.get('name')
            if kind not in commands_topics:
                continue
            by_kind.setdefault(kind, []).append(joint)

        rc_idx = list(parent).index(rc)
        parent.remove(rc)

        for offset, (kind, joints) in enumerate(by_kind.items()):
            block = ET.Element('ros2_control', {'name': f'{name}_{kind}', 'type': rc_type})
            hw = ET.SubElement(block, 'hardware')
            ET.SubElement(hw, 'plugin').text = bridge_plugin
            ET.SubElement(hw, 'param', {'name': 'joint_commands_topic'}).text = commands_topics[kind]
            ET.SubElement(hw, 'param', {'name': 'joint_states_topic'}).text = states_topic
            # No anti-spam gating on a sim bridge: publish every CM tick so Isaac sees
            # the very first non-zero command without waiting for state to diverge.
            ET.SubElement(hw, 'param', {'name': 'trigger_joint_command_threshold'}).text = '0.0'
            if kind == 'velocity':
                # physx wraps continuous joints at +-2pi
                ET.SubElement(hw, 'param', {'name': 'sum_wrapped_joint_states'}).text = 'true'
            for joint in joints:
                if not any(si.get('name') == 'effort' for si in joint.findall('state_interface')):
                    ET.SubElement(joint, 'state_interface', {'name': 'effort'})
                # state interfaces default to NaN until the first joint_states message
                # arrives, and a controller activated in that window errors out of its
                # update and wedges the controller_manager, start at zero instead
                for si in joint.findall('state_interface'):
                    if si.find('param[@name="initial_value"]') is None:
                        ET.SubElement(si, 'param', {'name': 'initial_value'}).text = '0.0'
                block.append(joint)
            parent.insert(rc_idx + offset, block)

    tree.write(out_path, encoding='utf-8', xml_declaration=True)
