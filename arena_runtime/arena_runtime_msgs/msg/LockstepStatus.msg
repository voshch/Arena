bool active
bool paused                     # run kept, stepping suspended
bool ungated                    # gates ignored, free-stepping
float64 target_rtf
float64 measured_rtf            # sim seconds advanced per wall second, recent window
builtin_interfaces/Time tick    # sim time being gated on
LockstepRegistration[] registrations
string[] waiting_on             # hard channels stalling the tick (populated after 1s)
string[] arrived
uint64 gated_ticks              # ticks gated on a hard channel, cumulative per run
LockstepChannelCount[] counts   # per-channel counters, cumulative per run
