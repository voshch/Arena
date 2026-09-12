# Animation layer state of one ped (task_generator AnimationManager + GestureLayer)
uint32 id                    # Pedestrian.id
string name                  # Pedestrian.name
string base                  # base animation (idle | walk | run | canned clip)
float32 base_phase           # base cycle phase in rad
AnimationSlot[] slots        # active overlays in blend order
