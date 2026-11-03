# Robots staged for spawn/despawn, applied on the next reset. Latched on state/robots/pending.
RobotDescriptor[] spawn    # queued additions, name + model, ns/frame assigned at reset
RobotDescriptor[] despawn  # queued removals of live robots (full descriptor)
