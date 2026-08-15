bool active
bool paused                     # run kept, stepping suspended
bool ungated                    # gates ignored, free-stepping
float64 target_rtf
float64 measured_rtf            # sim seconds advanced per wall second, recent window
builtin_interfaces/Time tick    # sim time being gated on
LockstepRegistration[] registrations
string[] waiting_on             # hard channels stalling the tick (populated after 1s)
string[] arrived
