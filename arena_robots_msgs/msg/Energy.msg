std_msgs/Header header # stamp/frame copied from the driving joint_states message
float32 total_energy_consumed_wh # time integral of total power since node start
float32 battery_soc_percent # remaining charge against battery_capacity_wh, clamped to [0, 100]
