# Full semantic snapshot for one env. Latched on state/semantics, republished on any change.
builtin_interfaces/Time stamp   # sim time of this snapshot
uint32 env_id                   # env index this snapshot belongs to
string world                    # active world name
SemanticEntityState[] entities  # every annotated entity in this env
