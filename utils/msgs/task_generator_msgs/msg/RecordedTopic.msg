# One topic the recorder subscribes to.
string key
string topic  # template, {ns} = robot namespace, {tg} = task generator node FQN
string msg_type  # pkg/msg/Type
bool robot_scoped
bool throttled
bool qos_transient_local
bool reliable  # false = best effort
uint32 depth  # 0 = recorder default
bool recorded
