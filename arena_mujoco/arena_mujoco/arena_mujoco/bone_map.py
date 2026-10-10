"""Wire joint to bone rotation map: each ROS4HRI joint angle turns its bones about an axis in the bone's own neutral frame."""

from __future__ import annotations

BONE_MAP: dict[str, tuple[tuple[str, tuple[float, float, float], float], ...]] = {
    'r_waist': (('LowerBack', (-0.0139, 0.1447, 0.9894), 1.0),),
    'y_waist': (('LowerBack', (0.0026, 0.9895, -0.1447), 1.0),),
    'waist': (('LowerBack', (0.9999, -0.0006, 0.0141), 1.0),),
    'r_spine': (('Spine', (-0.0226, -0.0083, 0.9997), 1.0),),
    'y_spine': (('Spine', (-0.0006, 1.0000, 0.0083), 1.0),),
    'spine': (('Spine', (0.9997, 0.0004, 0.0226), 1.0),),
    'r_chest': (('Spine1', (-0.0273, 0.4378, 0.8987), 1.0),),
    'y_chest': (('Spine1', (-0.0100, 0.8988, -0.4382), 1.0),),
    'chest': (('Spine1', (0.9996, 0.0209, 0.0202), 1.0),),
    'r_head': (('Head', (-0.0131, 0.1115, 0.9937), 1.0),),
    'y_head': (
        ('Neck1', (-0.0001, 0.8652, -0.5015), 0.6),
        ('Head', (-0.0007, 0.9938, -0.1116), 0.4),
    ),
    'p_head': (
        ('Neck', (-1.0000, -0.0042, -0.0073), 0.35),
        ('Head', (-0.9999, -0.0021, -0.0129), 0.65),
    ),
    'l_y_collar': (('LeftShoulder', (0.9797, 0.1839, 0.0800), 1.0),),
    'l_p_collar': (('LeftShoulder', (-0.0655, -0.0831, 0.9944), 1.0),),
    'l_y_shoulder': (('LeftArm', (-0.0593, -0.9803, -0.1883), 1.0),),
    'l_p_shoulder': (('LeftArm', (0.9466, -0.1151, 0.3013), 1.0),),
    'l_r_shoulder': (('LeftArm', (0.0593, 0.9803, 0.1883), 1.0),),
    'l_elbow': (('LeftForeArm', (0.9396, 0.0062, -0.3421), 1.0),),
    'r_y_collar': (('RightShoulder', (0.9832, -0.1643, -0.0791), 1.0),),
    'r_p_collar': (('RightShoulder', (-0.0752, 0.0295, -0.9967), 1.0),),
    'r_y_shoulder': (('RightArm', (0.0739, -0.9805, -0.1820), 1.0),),
    'r_p_shoulder': (('RightArm', (0.9608, 0.1189, -0.2505), 1.0),),
    'r_r_shoulder': (('RightArm', (-0.0739, 0.9805, 0.1820), 1.0),),
    'r_elbow': (('RightForeArm', (0.9253, -0.0049, 0.3791), 1.0),),
    'l_y_hip': (('LeftUpLeg', (0.0046, 0.9978, -0.0659), 1.0),),
    'l_p_hip': (('LeftUpLeg', (0.0414, -0.0661, -0.9970), 1.0),),
    'l_r_hip': (('LeftUpLeg', (-0.9991, 0.0019, -0.0417), 1.0),),
    'l_knee': (('LeftLeg', (-1.0000, 0.0046, -0.0051), 1.0),),
    'r_y_hip': (('RightUpLeg', (-0.0035, 0.9936, -0.1126), 1.0),),
    'r_p_hip': (('RightUpLeg', (0.0497, 0.1126, 0.9924), 1.0),),
    'r_r_hip': (('RightUpLeg', (-0.9988, 0.0021, 0.0497), 1.0),),
    'r_knee': (('RightLeg', (-0.9997, 0.0041, 0.0249), 1.0),),
    'l_y_ankle': (('LeftFoot', (-0.3493, 0.4316, 0.8317), 1.0),),
    'l_ankle': (('LeftFoot', (-0.9370, -0.1697, -0.3054), 1.0),),
    'r_y_ankle': (('RightFoot', (0.3754, 0.4375, 0.8171), 1.0),),
    'r_ankle': (('RightFoot', (-0.9268, 0.1697, 0.3350), 1.0),),
    'l_r_wrist': (('LeftHand', (0.1182, 0.9531, 0.2785), 1.0),),
    'l_wrist': (('LeftHand', (0.1229, -0.2923, 0.9484), 1.0),),
    'r_r_wrist': (('RightHand', (0.1961, -0.9596, -0.2018), 1.0),),
    'r_wrist': (('RightHand', (0.1261, 0.2288, -0.9653), 1.0),),
}

__all__ = ['BONE_MAP']
