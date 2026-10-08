std_msgs/Header header  # stamp: detection, frame_id: bearing frame
string robot
string frontend         # bus, srp, seld
string kind             # kinds-table name, onset if class-agnostic
string event_id
float32 azimuth_rad     # CCW from +x of frame_id, NaN = no bearing
float32 elevation_rad   # NaN when unknown
float32 level_db        # received SPL estimate, NaN when unknown
float32 confidence      # 0..1
