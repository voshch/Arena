# arena_cam_msgs

Service of the camera server behind `viz.cam`, see the
[arena_cam README](../../arena_cam/arena_cam/README.md#playing-with-the-episode).

`srv/Play.srv`: `root` names the camera asset to play, `bundle` holds a YAML mapping of
name to spec for every asset it reaches, `params` a YAML mapping of parameters.
`/arena/env_<n>/cam/play` takes all three. `<task generator>/cam/play` reads `root` as a
request and `params`, and builds the bundle itself.
