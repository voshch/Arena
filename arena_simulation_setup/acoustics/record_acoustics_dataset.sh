#!/usr/bin/env bash
# Record every acoustics scenario with simulation-time-synchronized ROS audio.

set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORLDS_ROOT="${SCRIPT_DIR}/worlds"
OUTPUT_REL='data/audio_train_set'
SCENARIO_GLOB='*'
WORLD_GLOB='*'
MAX_SCENARIOS=0
DURATION=30
WALL_TIMEOUT=90
READY_TIMEOUT=120
EPISODE_FINALIZE_TIMEOUT=15
SIMULATOR='gazebo'
RUNTIME='auto'
CONTAINER=''
PLAYBACK_DEVICE='auto'
INJECT_REFERENCE_SOUND=0
FORCE=0
RECOVER_INCOMPLETE=0
LIST_ONLY=0
GAZEBO_FULLSCREEN=0
HEADLESS=0
LAUNCH_CLIENT_PID=''
CAPTURE_PID=''
SUPERVISOR_PID_FILE=''
HF_UPLOAD=1
HF_REPO='kameow/arena_audio'
HF_TOKEN_ENV='HF_TOKEN'
DELETE_AFTER_UPLOAD=1
UPLOAD_HELPER="${SCRIPT_DIR}/hf_finalize_upload.py"
UPLOAD_PIDS=()
UPLOAD_LOGS=()
MAX_PARALLEL_UPLOADS=2
UPLOAD_FAIL=0
SCENARIO_FAIL=0
FAILED_SCENARIOS_FILE=''

usage() {
    cat <<'EOF'
Usage: record_acoustics_dataset.sh [options] [-- extra launch arguments]

Records every selected scenario to:
  data/audio_train_set/<scenario>/

The selected PortAudio device plays the Jackal's stereo headphone stream while
the same signal is captured from its ROS AudioFrame source. This excludes
launch/setup noise and preserves exact episode timing. The MCAP also contains
the raw four-microphone stream, /clock, maps, poses, TF, and episode events.
The exporter writes <scenario>_<index>_recording.wav (or FLAC when explicitly
requested) and matching metadata.

Options:
  --worlds-root DIR       Worlds directory (default: acoustics/worlds)
  --output-relative DIR   Output below the Arena workspace (default: data/audio_train_set)
  --scenario-glob GLOB    Scenario-directory glob (default: *, all scenarios)
  --world-glob GLOB       World-directory glob (default: *, all worlds)
  --max-scenarios N       Stop after N selected cases (default: 0, unlimited)
  --duration SEC          Required simulation-time audio duration (default: 30)
  --wall-timeout SEC      Per-capture wall-time watchdog (default: 90)
  --ready-timeout SEC     Simulator/audio startup timeout (default: 120)
  --episode-finalize-timeout SEC  Wait for clean episode cancellation/finalization (default: 15)
  --sim NAME              Arena simulator backend (default: gazebo)
  --runtime MODE          auto, docker, or native (default: auto)
  --container NAME_OR_ID  Arena container (implies docker)
  --playback-device DEV   PortAudio output: auto or a device name (default: auto)
  --reference-sound       Inject an audible calibration event (diagnostics only)
  --headless              Run without the Gazebo GUI (audio capture and playback remain enabled)
  --gazebo-fullscreen     Request fullscreen for the visible Gazebo GUI (optional; needs wmctrl)
  --recover-incomplete    Back up and retry only an incomplete scenario; keep validated runs
  --force                 Move an existing run to a timestamped backup and repeat it
  --hf-repo REPO          Hugging Face dataset repo (default: kameow/arena_audio)
  --hf-token-env NAME     Env var containing HF write token (default: HF_TOKEN)
  --no-hf-upload          Keep recordings local; do not upload
  --keep-local            Upload/verify but do not delete local scenario
  --upload-helper FILE    Finalize/upload helper path
  --max-parallel-uploads N  Maximum concurrent Hub uploads (default: 2)
  --list                  List selected index/world/scenario triples
  -h, --help              Show this help
EOF
}

die() { printf 'record_acoustics_dataset: ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf 'record_acoustics_dataset: %s\n' "$*" >&2; }
is_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

EXTRA_LAUNCH_ARGS=()
while (($#)); do
    case "$1" in
        --worlds-root) WORLDS_ROOT="${2:?missing value}"; shift 2 ;;
        --output-relative) OUTPUT_REL="${2:?missing value}"; shift 2 ;;
        --scenario-glob) SCENARIO_GLOB="${2:?missing value}"; shift 2 ;;
        --world-glob) WORLD_GLOB="${2:?missing value}"; shift 2 ;;
        --max-scenarios) MAX_SCENARIOS="${2:?missing value}"; shift 2 ;;
        --duration) DURATION="${2:?missing value}"; shift 2 ;;
        --wall-timeout) WALL_TIMEOUT="${2:?missing value}"; shift 2 ;;
        --ready-timeout) READY_TIMEOUT="${2:?missing value}"; shift 2 ;;
        --episode-finalize-timeout) EPISODE_FINALIZE_TIMEOUT="${2:?missing value}"; shift 2 ;;
        --sim) SIMULATOR="${2:?missing value}"; shift 2 ;;
        --runtime) RUNTIME="${2:?missing value}"; shift 2 ;;
        --container) CONTAINER="${2:?missing value}"; RUNTIME='docker'; shift 2 ;;
        --playback-device) PLAYBACK_DEVICE="${2:?missing value}"; shift 2 ;;
        --reference-sound) INJECT_REFERENCE_SOUND=1; shift ;;
        --headless) HEADLESS=1; shift ;;
        --gazebo-fullscreen) GAZEBO_FULLSCREEN=1; shift ;;
        --recover-incomplete) RECOVER_INCOMPLETE=1; shift ;;
        --force) FORCE=1; shift ;;
        --hf-repo) HF_REPO="${2:?missing value}"; shift 2 ;;
        --hf-token-env) HF_TOKEN_ENV="${2:?missing value}"; shift 2 ;;
        --no-hf-upload) HF_UPLOAD=0; shift ;;
        --keep-local) DELETE_AFTER_UPLOAD=0; shift ;;
        --upload-helper) UPLOAD_HELPER="${2:?missing value}"; shift 2 ;;
        --max-parallel-uploads) MAX_PARALLEL_UPLOADS="${2:?missing value}"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_LAUNCH_ARGS=("$@"); break ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -d "$WORLDS_ROOT" ]] || die "worlds directory not found: $WORLDS_ROOT"
[[ "$OUTPUT_REL" != /* && "$OUTPUT_REL" != *'..'* ]] || die '--output-relative must be a safe workspace-relative path'
[[ "$OUTPUT_REL" == data || "$OUTPUT_REL" == data/* ]] || die '--output-relative must be below data/'
[[ "$MAX_SCENARIOS" =~ ^[0-9]+$ ]] || die '--max-scenarios must be a non-negative integer'
is_number "$DURATION" || die '--duration must be numeric'
is_number "$WALL_TIMEOUT" || die '--wall-timeout must be numeric'
is_number "$READY_TIMEOUT" || die '--ready-timeout must be numeric'
is_number "$EPISODE_FINALIZE_TIMEOUT" || die '--episode-finalize-timeout must be numeric'
[[ -n "$PLAYBACK_DEVICE" ]] || die '--playback-device must be auto, none, or a device name'
[[ "$MAX_PARALLEL_UPLOADS" =~ ^[1-9][0-9]*$ ]] || die '--max-parallel-uploads must be a positive integer'
[[ -n "$SIMULATOR" ]] || die '--sim must not be empty'
((HEADLESS == 0 || GAZEBO_FULLSCREEN == 0)) || die '--headless and --gazebo-fullscreen cannot be used together'
((RECOVER_INCOMPLETE == 0 || FORCE == 0)) || die '--recover-incomplete and --force cannot be used together'
case "$RUNTIME" in auto|docker|native) ;; *) die '--runtime must be auto, docker, or native' ;; esac

for arg in "${EXTRA_LAUNCH_ARGS[@]}"; do
    case "$arg" in
        sim:=*|world:=*|robot:=*|human:=*|auditory:=*|auditory.playback:=*|microphone_mode:=*|scenario_file:=*|tm_robots:=*|tm_obstacles:=*|task.scenario:=*|task.scenario.file:=*|task.robots:=*|task.obstacles:=*|task.scenario.linger_after_completion:=*|auto_reset:=*|task.auto_reset:=*|env_n:=*|env.n:=*|viz:=*|headless:=*|record_data_dir:=*|record.dir:=*|record.auto:=*)
            die "the script owns launch argument '$arg'"
            ;;
    esac
done

SCENARIO_FILES=()
note "scanning worlds='${WORLD_GLOB}' scenarios='${SCENARIO_GLOB}' under $WORLDS_ROOT"
mapfile -d '' -t scenario_candidates < <(
    find "$WORLDS_ROOT" -type f \
        -path "$WORLDS_ROOT/$WORLD_GLOB/scenarios/$SCENARIO_GLOB/scenario.yaml" \
        -print0 | sort -z
)
for scenario_file in "${scenario_candidates[@]}"; do
    scenario_dir="$(dirname -- "$scenario_file")"
    world_dir="$(dirname -- "$(dirname -- "$scenario_dir")")"
    if [[ "$(basename -- "$scenario_dir")" == hearing__* ]] && ! grep -q '^  waypoint_mode: reverse$' "$scenario_file"; then
        die "legacy generated scenario layout found at $scenario_file; run normalize_acoustics_scenarios --worlds-root '$WORLDS_ROOT' --write"
    fi
    SCENARIO_FILES+=("$scenario_file")
    if ((MAX_SCENARIOS > 0 && ${#SCENARIO_FILES[@]} >= MAX_SCENARIOS)); then break; fi
done
unset scenario_candidates
((${#SCENARIO_FILES[@]})) || die "no scenario.yaml matched world '${WORLD_GLOB}' and scenario '${SCENARIO_GLOB}'"

if ((LIST_ONLY)); then
    execution_number=0
    for scenario_file in "${SCENARIO_FILES[@]}"; do
        execution_number=$((execution_number + 1))
        printf -v run_index '%04d' "$execution_number"
        scenario_dir="$(dirname -- "$scenario_file")"
        world_dir="$(dirname -- "$(dirname -- "$scenario_dir")")"
        printf '%s\t%s\t%s\n' "$run_index" "$(basename -- "$world_dir")" "$(basename -- "$scenario_dir")"
    done
    exit 0
fi

find_arena_container() {
    local -a ids=()
    while IFS= read -r id; do [[ -z "$id" ]] || ids+=("$id"); done \
        < <(docker ps --filter label=com.docker.compose.service=arena --format '{{.ID}}')
    ((${#ids[@]} <= 1)) || die 'multiple Arena containers found; pass --container'
    ((${#ids[@]} == 1)) && printf '%s\n' "${ids[0]}"
}

if [[ "$RUNTIME" == auto ]]; then
    # `source arena -c '...'` already executes this script inside the Arena
    # container.  The Docker CLI and socket may still be mounted there, but
    # the unprivileged arena user is intentionally not a Docker administrator.
    # Treat that environment as native instead of probing the host socket.
    if [[ -e /.dockerenv || "${WORKSPACE_ROOT}" == /opt/arena_ws* ]]; then
        RUNTIME='native'
    else
        if command -v docker >/dev/null; then CONTAINER="$(find_arena_container)"; fi
        [[ -z "$CONTAINER" ]] || RUNTIME='docker'
        [[ "$RUNTIME" != auto ]] || RUNTIME='native'
    fi
fi
if [[ "$RUNTIME" == docker ]]; then
    command -v docker >/dev/null || die 'docker is unavailable'
    [[ -n "$CONTAINER" ]] || CONTAINER="$(find_arena_container)"
    [[ -n "$CONTAINER" ]] || die 'no running Arena container found'
    docker exec "$CONTAINER" true >/dev/null || die "cannot access container $CONTAINER"
    HOST_DATA_ROOT="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/opt/arena_ws/data"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")"
    [[ -n "$HOST_DATA_ROOT" ]] || die 'Arena container has no /opt/arena_ws/data host mount'
    if [[ "$OUTPUT_REL" == data ]]; then OUTPUT_ROOT="$HOST_DATA_ROOT"; else OUTPUT_ROOT="${HOST_DATA_ROOT}/${OUTPUT_REL#data/}"; fi
    ARENA_OUTPUT_ROOT="/opt/arena_ws/${OUTPUT_REL}"
else
    command -v ros2 >/dev/null || die 'ROS 2 is unavailable; source Arena first'
    if [[ -n "${ARENA_DATA_DIR:-}" ]]; then
        if [[ "$OUTPUT_REL" == data ]]; then OUTPUT_ROOT="$ARENA_DATA_DIR"; else OUTPUT_ROOT="${ARENA_DATA_DIR}/${OUTPUT_REL#data/}"; fi
    elif [[ "$(basename -- "$(dirname -- "$WORKSPACE_ROOT")")" == src ]]; then
        OUTPUT_ROOT="$(dirname -- "$(dirname -- "$WORKSPACE_ROOT")")/${OUTPUT_REL}"
    else
        OUTPUT_ROOT="${WORKSPACE_ROOT}/${OUTPUT_REL}"
    fi
    ARENA_OUTPUT_ROOT="$OUTPUT_ROOT"
fi

run_in_arena() {
    if [[ "$RUNTIME" == docker ]]; then
        docker exec -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" -e "GZ_PARTITION=${GZ_PARTITION:-}" \
            "$CONTAINER" bash --norc -c 'cd /opt/arena_ws && source ./source >/dev/null && "$@"' _ "$@"
    else
        (cd "$WORKSPACE_ROOT" && "$@")
    fi
}

refresh_ros_discovery() {
    # ros2cli's long-lived daemon can retain an action briefly after the
    # launch tree has exited.  A fresh daemon distinguishes that stale graph
    # entry from a genuinely running Arena environment in this ROS domain.
    run_in_arena ros2 daemon stop >/dev/null 2>&1 || true
    sleep 1
}

episode_actions() {
    run_in_arena ros2 action list 2>/dev/null | grep '/lifecycle/run_episode$' || true
}

wait_for_environment_shutdown() {
    local actions=''
    refresh_ros_discovery
    for _ in {1..20}; do
        actions="$(episode_actions)"
        [[ -n "$actions" ]] || return 0
        sleep 1
    done
    die "Arena environment survived recorder shutdown: $actions"
}

request_gazebo_fullscreen() {
    local window_id=''
    if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
        note 'Gazebo GUI was requested, but this shell has no desktop display; continuing windowed/headless-display automation'
        return 0
    fi
    if ! command -v wmctrl >/dev/null; then
        note 'Gazebo GUI is enabled, but wmctrl is unavailable; install it with: sudo apt install wmctrl'
        return 0
    fi
    for _ in {1..20}; do
        window_id="$(wmctrl -lx 2>/dev/null | awk 'BEGIN { IGNORECASE=1 } /gazebo|gz[-_ ]?(sim|gui)/ { print $1; exit }')"
        if [[ -n "$window_id" ]]; then
            if wmctrl -ir "$window_id" -b add,fullscreen,maximized_vert,maximized_horz 2>/dev/null; then
                note "Gazebo window $window_id set to fullscreen"
                return 0
            fi
        fi
        sleep 1
    done
    note 'Gazebo GUI is running, but the window manager did not expose a fullscreen-capable Gazebo window'
}

stop_launch() {
    [[ -n "$LAUNCH_CLIENT_PID" ]] || return 0
    if [[ "$RUNTIME" == docker ]]; then
        docker exec "$CONTAINER" bash --norc -c 'test -s "$1" && kill -INT "$(cat "$1")" 2>/dev/null || true' _ "$SUPERVISOR_PID_FILE" >/dev/null 2>&1 || true
    else
        kill -INT -- "-${LAUNCH_CLIENT_PID}" 2>/dev/null || true
    fi
    for _ in {1..40}; do kill -0 "$LAUNCH_CLIENT_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$LAUNCH_CLIENT_PID" 2>/dev/null; then
        note 'graceful shutdown timed out; sending SIGTERM'
        if [[ "$RUNTIME" == docker ]]; then
            docker exec "$CONTAINER" bash --norc -c 'test -s "$1" && kill -TERM "$(cat "$1")" 2>/dev/null || true' _ "$SUPERVISOR_PID_FILE" >/dev/null 2>&1 || true
        else
            kill -TERM -- "-${LAUNCH_CLIENT_PID}" 2>/dev/null || true
        fi
    fi
    wait "$LAUNCH_CLIENT_PID" 2>/dev/null || true
    LAUNCH_CLIENT_PID=''
    cleanup_gazebo_partition
}

cleanup_gazebo_partition() {
    # Gazebo may re-parent its server to PID 1 before the launch supervisor
    # exits.  Such a server is invisible to the ROS graph and used to survive
    # both normal and failed recordings.  Only stop processes carrying this
    # invocation's unique transport partition; never use a broad pkill that
    # could affect another Arena user or ROS domain.
    [[ -n "${GZ_PARTITION:-}" ]] || return 0
    local cleanup_script='partition="$1"
for proc_env in /proc/[0-9]*/environ; do
    pid="${proc_env#/proc/}"; pid="${pid%/environ}"
    [ -r "$proc_env" ] || continue
    tr "\000" "\n" <"$proc_env" 2>/dev/null | grep -Fqx "GZ_PARTITION=$partition" || continue
    command_name="$(cat "/proc/$pid/comm" 2>/dev/null || true)"
    case "$command_name" in gz|ruby) kill -TERM "$pid" 2>/dev/null || true ;; esac
done'
    if [[ "$RUNTIME" == docker ]]; then
        docker exec "$CONTAINER" bash --norc -c "$cleanup_script" _ "$GZ_PARTITION" >/dev/null 2>&1 || true
    else
        bash --norc -c "$cleanup_script" _ "$GZ_PARTITION" >/dev/null 2>&1 || true
    fi
}

cancel_and_finalize_episode() {
    local action_name="$1"
    local cancel_service="${action_name}/_action/cancel_goal"
    local cancel_type='action_msgs/srv/CancelGoal'
    local cancel_request
    local cancel_response=''
    local deadline

    # The recorder deliberately launches scenarios with
    # task.scenario.linger_after_completion:=true.  wait_capture therefore
    # stops after the requested synchronized capture duration while the
    # RunEpisode goal may still be active.  The MCAP exporter requires a
    # terminal EpisodeRecord, so cancel the live goal before shutting Arena
    # down rather than killing the supervisor first.
    cancel_request='{goal_info: {goal_id: {uuid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, stamp: {sec: 0, nanosec: 0}}}'

    # Do not probe this endpoint with `ros2 service type` before calling it.
    # Action protocol services live below the hidden `_action` namespace and
    # some ros2cli/RMW combinations do not report their type through the normal
    # service-introspection path even though the action server is healthy.
    # Call the standard CancelGoal endpoint directly instead.
    note "finalizing episode: cancelling active RunEpisode goal"
    cancel_response="$(
        run_in_arena timeout 10 ros2 service call \
            "$cancel_service" \
            "$cancel_type" \
            "$cancel_request" 2>&1 || true
    )"

    # CancelGoal return_code=0 means ERROR_NONE. This is also safe if the goal
    # became terminal just before our request: Arena still gets the grace
    # period below to flush its terminal EpisodeRecord to MCAP.
    if [[ "$cancel_response" != *'return_code=0'* && "$cancel_response" != *'return_code: 0'* ]]; then
        printf '%s\n' "$cancel_response" >&2
        printf 'record_acoustics_dataset: visible action info follows:\n' >&2
        run_in_arena ros2 action info "$action_name" 2>&1 >&2 || true
        printf 'record_acoustics_dataset: hidden cancel endpoints follow:\n' >&2
        run_in_arena ros2 service list --include-hidden-services 2>/dev/null \
            | grep -F "${action_name}/_action/" >&2 || true
        die 'RunEpisode cancellation was not accepted; local MCAP retained for inspection'
    fi

    deadline=$((SECONDS + ${EPISODE_FINALIZE_TIMEOUT%.*}))
    while ((SECONDS <= deadline)); do
        # Arena's evaluation recorder logs terminal episode records when the
        # action reaches a terminal state.  Prefer that explicit signal when
        # available, but do not require an exact logger wording across Arena
        # versions.  The short grace period also lets queued recorder messages
        # reach the MCAP writer before graceful supervisor shutdown.
        if [[ -s "$launch_log" ]] && grep -Eqi 'terminal EpisodeRecord|EpisodeRecord.*(cancel|canceled|cancelled|complete|completed|terminal)|episode.*(cancel|canceled|cancelled|complete|completed).*record' "$launch_log"; then
            note 'episode terminal record observed before shutdown'
            return 0
        fi
        sleep 0.25
    done

    note "episode cancellation accepted; finalization grace period (${EPISODE_FINALIZE_TIMEOUT}s) elapsed"
}

cleanup() {
    local status=$?
    if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then kill "$CAPTURE_PID" 2>/dev/null || true; fi
    stop_launch
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/.failed_scenarios"
FAILED_SCENARIOS_FILE="$OUTPUT_ROOT/failed_scenarios.tsv"
if [[ ! -s "$FAILED_SCENARIOS_FILE" ]]; then
    printf 'timestamp\texecution_index\tworld\tscenario\texit_code\tlog\n' >"$FAILED_SCENARIOS_FILE"
fi
if ((HF_UPLOAD)); then
    [[ -f "$UPLOAD_HELPER" ]] || die "upload helper not found: $UPLOAD_HELPER"
    [[ -n "${!HF_TOKEN_ENV:-}" ]] || die "$HF_TOKEN_ENV is not set; export a Hugging Face write token before recording"
    mkdir -p "$OUTPUT_ROOT/.upload_logs"
fi
note "runtime=$RUNTIME scenarios=${#SCENARIO_FILES[@]} duration=${DURATION}s simulation time output=$OUTPUT_ROOT"

# Clear graph entries left by a previously interrupted invocation before the
# first exclusivity check. A real environment in this domain is rediscovered.
refresh_ros_discovery

execution_number=0

wait_for_upload_slot() {
    ((${#UPLOAD_PIDS[@]} < MAX_PARALLEL_UPLOADS)) && return 0
    local pid="${UPLOAD_PIDS[0]}" log="${UPLOAD_LOGS[0]}"
    if ! wait "$pid"; then
        UPLOAD_FAIL=1
        printf 'record_acoustics_dataset: ERROR: upload failed; local data retained. Log: %s\n' "$log" >&2
        [[ -s "$log" ]] && tail -n 80 "$log" >&2 || true
    fi
    UPLOAD_PIDS=("${UPLOAD_PIDS[@]:1}")
    UPLOAD_LOGS=("${UPLOAD_LOGS[@]:1}")
}

for scenario_file in "${SCENARIO_FILES[@]}"; do
    execution_number=$((execution_number + 1))
    printf -v run_index '%04d' "$execution_number"
    scenario_dir="$(dirname -- "$scenario_file")"
    scenario_name="$(basename -- "$scenario_dir")"
    world_dir="$(dirname -- "$(dirname -- "$scenario_dir")")"
    world_name="$(basename -- "$world_dir")"
    run_name="${scenario_name}"
    artifact_prefix="${scenario_name}_${run_index}"
    run_host="${OUTPUT_ROOT}/${run_name}"
    run_arena="${ARENA_OUTPUT_ROOT}/${run_name}"
    validation="${run_host}/${artifact_prefix}_validation.json"
    # Gazebo Transport is independent of ROS_DOMAIN_ID. Without a unique
    # partition a new GUI/server can attach to a surviving or unrelated world
    # on the host and display the wrong map and actors.
    export GZ_PARTITION="arena_acoustics_${BASHPID}_${run_index}"

    if ((HF_UPLOAD)); then
        remote_prefix="scenarios/${world_name}/${artifact_prefix}"
        remote_state="$(python3 "$UPLOAD_HELPER" --repo-id "$HF_REPO" \
            --remote-prefix "$remote_prefix" --token-env "$HF_TOKEN_ENV" \
            --world-name "$world_name" --scenario-name "$scenario_name" \
            --execution-index "$execution_number" --check-remote)" \
            || die "could not check Hugging Face state for $remote_prefix"
        if [[ "$remote_state" == "EXISTS" && "$FORCE" -eq 0 ]]; then
            note "SKIP $run_name (already uploaded and finalized)"
            continue
        fi
    fi

    if [[ -s "$validation" && "$FORCE" -eq 0 ]] && grep -q '"valid": true' "$validation"; then
        if ((HF_UPLOAD)); then
            note "UPLOAD retry $run_name (validated locally, missing remotely)"
            upload_log="${OUTPUT_ROOT}/.upload_logs/${world_name}__${artifact_prefix}.log"
            upload_args=(
                "$UPLOAD_HELPER" "$run_host"
                --repo-id "$HF_REPO"
                --remote-prefix "$remote_prefix"
                --token-env "$HF_TOKEN_ENV"
                --world-name "$world_name" --scenario-name "$scenario_name"
                --execution-index "$execution_number"
            )
            if ((DELETE_AFTER_UPLOAD)); then upload_args+=(--delete-after-verify); fi
            if ! python3 "${upload_args[@]}" >"$upload_log" 2>&1; then
                [[ -s "$upload_log" ]] && tail -n 80 "$upload_log" >&2 || true
                die "upload retry failed; local data retained. Log: $upload_log"
            fi
            note "UPLOADED $run_name -> hf://datasets/${HF_REPO}/${remote_prefix}"
        else
            note "SKIP $run_name (validated)"
        fi
        continue
    fi
    failure_log="${OUTPUT_ROOT}/.failed_scenarios/${world_name}__${artifact_prefix}.log"
    set +e
    (
    set -Eeo pipefail
    # Keep launch/capture PIDs local to this attempt. Any failure triggers the
    # inherited cleanup trap without terminating the outer dataset loop.
    trap cleanup EXIT

    existing_actions="$(episode_actions)"
    [[ -z "$existing_actions" ]] || die "another Arena environment is already running; stop it before recording: $existing_actions"
    if [[ -e "$run_host" && ("$FORCE" -eq 1 || "$RECOVER_INCOMPLETE" -eq 1) ]]; then
        backup="${run_host}.backup.$(date +%Y%m%d-%H%M%S)"
        mv -- "$run_host" "$backup"
        note "moved previous incomplete/repeated run to $backup"
    elif [[ -e "$run_host" ]]; then
        die "incomplete existing run at $run_host (use --force after inspecting it)"
    fi
    mkdir -p "$run_host"
    cp -- "$scenario_file" "${run_host}/scenario.yaml"
    launch_log="${run_host}/launch.log"
    waiter_log="${run_host}/capture_wait.json"
    note "START $run_name world=$world_name"

    # Gazebo visibility and fullscreen are separate concerns.  Recording runs
    # Gazebo windowed by default; --gazebo-fullscreen only asks the desktop
    # window manager to enlarge that already-visible window.  Non-GUI
    # backends remain headless.
    if [[ "$SIMULATOR" == gazebo && "$HEADLESS" -eq 0 ]]; then
        display_args=('viz:=false' 'headless:=false')
    else
        display_args=('viz:=false' 'headless:=true')
    fi
    launch_args=(
        "sim:=${SIMULATOR}" "world:=${world_name}" 'robot:=jackal' 'human:=arena' 'auditory:=arena'
        'microphone_mode:=four_mic' "auditory.playback:=${PLAYBACK_DEVICE}"
        'auditory.robot_sound:=true' 'auditory.motor:=procedural'
        # Lift the procedural drivetrain above the four-mic monitor noise
        # floor while retaining headroom for nearby footsteps.
        'auditory.motor.mems_calibration_db:=-28.0'
        'task.robots:=scenario' 'task.obstacles:=scenario' "task.scenario:=${scenario_name}"
        # The explicit ROS parameter override follows parameter files and keeps
        # older installed task configurations from restoring their "default".
        "task.scenario.file:=${scenario_name}"
        'task.scenario.linger_after_completion:=true'
        'task.auto_reset:=false' 'env.n:=1' "${display_args[@]}"
        "record.dir:=${run_arena}" 'record.auto:=true'
        "${EXTRA_LAUNCH_ARGS[@]}"
    )
    if [[ "$RUNTIME" == docker ]]; then
        SUPERVISOR_PID_FILE="/tmp/arena-acoustics-${BASHPID}.pid"
        docker exec -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" -e "GZ_PARTITION=${GZ_PARTITION}" \
            "$CONTAINER" bash --norc -c \
            'cd /opt/arena_ws && source ./source >/dev/null && printf "%s\n" "$BASHPID" > "$1" && shift && exec python3 -m arena_bringup.supervisor "$@"' \
            _ "$SUPERVISOR_PID_FILE" "${launch_args[@]}" >"$launch_log" 2>&1 &
    else
        setsid python3 -m arena_bringup.supervisor "${launch_args[@]}" >"$launch_log" 2>&1 &
    fi
    LAUNCH_CLIENT_PID=$!

    deadline=$((SECONDS + ${READY_TIMEOUT%.*}))
    episode_action=''
    env_namespace=''
    raw_type=''
    rendered_type=''
    map_type=''
    map_state=''
    microphone_node=''
    raw_live=0
    rendered_live=0
    while ((SECONDS <= deadline)); do
        kill -0 "$LAUNCH_CLIENT_PID" 2>/dev/null || { tail -n 80 "$launch_log" >&2; die 'Arena exited during startup'; }
        episode_action="$(run_in_arena ros2 action list 2>/dev/null | grep '/lifecycle/run_episode$' | head -n 1 || true)"
        if [[ -n "$episode_action" ]]; then
            env_namespace="${episode_action%/lifecycle/run_episode}"
            raw_type="$(run_in_arena ros2 topic type "${env_namespace}/jackal/audio/raw_array" 2>/dev/null || true)"
            rendered_type="$(run_in_arena ros2 topic type "${env_namespace}/jackal/audio/headphones/stereo" 2>/dev/null || true)"
            map_type="$(run_in_arena ros2 topic type "${env_namespace}/map" 2>/dev/null || true)"
            map_state="$(run_in_arena ros2 lifecycle get "${env_namespace}/map_server" 2>/dev/null || true)"
            microphone_node="$(run_in_arena ros2 node list 2>/dev/null | grep -x "${env_namespace}/microphone_array_node" || true)"
            if [[ "$raw_type" == 'task_generator_msgs/msg/AudioFrame' && "$raw_live" -eq 0 ]] \
                && run_in_arena timeout 3 ros2 topic echo --once "${env_namespace}/jackal/audio/raw_array" >/dev/null 2>&1; then
                raw_live=1
            fi
            if [[ "$rendered_type" == 'task_generator_msgs/msg/AudioFrame' && "$rendered_live" -eq 0 ]] \
                && run_in_arena timeout 3 ros2 topic echo --once "${env_namespace}/jackal/audio/headphones/stereo" >/dev/null 2>&1; then
                rendered_live=1
            fi
        fi
        if [[ "$raw_type" == 'task_generator_msgs/msg/AudioFrame' \
           && "$rendered_type" == 'task_generator_msgs/msg/AudioFrame' \
           && "$map_type" == 'nav_msgs/msg/OccupancyGrid' \
           && "$map_state" == *active* \
           && -n "$microphone_node" \
           && "$raw_live" -eq 1 \
           && "$rendered_live" -eq 1 ]]; then
            break
        fi
        sleep 1
    done
    [[ -n "$episode_action" ]] || die "episode action not ready; see $launch_log"
    [[ -n "$microphone_node" ]] || die "${env_namespace}/microphone_array_node is not running"
    [[ "$map_type" == 'nav_msgs/msg/OccupancyGrid' && "$map_state" == *active* ]] \
        || die "environment map server is not active at ${env_namespace}/map"
    [[ "$raw_type" == 'task_generator_msgs/msg/AudioFrame' ]] || die "${env_namespace}/jackal/audio/raw_array publisher is missing or has the wrong type"
    [[ "$rendered_type" == 'task_generator_msgs/msg/AudioFrame' ]] || die "${env_namespace}/jackal/audio/headphones/stereo publisher is missing or has the wrong type"
    ((raw_live)) || die "${env_namespace}/jackal/audio/raw_array is declared but did not publish a live frame"
    ((rendered_live)) || die "${env_namespace}/jackal/audio/headphones/stereo is declared but did not publish a live frame"
    if ((GAZEBO_FULLSCREEN)); then request_gazebo_fullscreen; fi

    waiter_args=(
        --namespace "$env_namespace" --robot-name jackal \
        --duration "$DURATION" --wall-timeout "$WALL_TIMEOUT" \
        --run-episode-action "$episode_action" --world "$world_name"
    )
    if ((INJECT_REFERENCE_SOUND)); then waiter_args+=(--inject-reference-sound); fi
    run_in_arena python3 -m arena_simulation_setup.acoustics.wait_capture "${waiter_args[@]}" >"$waiter_log" &
    CAPTURE_PID=$!
    capture_ready=0
    for _ in {1..20}; do
        kill -0 "$CAPTURE_PID" 2>/dev/null || { wait "$CAPTURE_PID" 2>/dev/null || true; die "capture waiter exited before subscribing; see $waiter_log"; }
        if run_in_arena ros2 node list 2>/dev/null | grep -qx '/arena_acoustics_capture_waiter'; then
            capture_ready=1
            break
        fi
        sleep 0.25
    done
    ((capture_ready)) || die 'capture waiter did not become ready before the episode action'
    if ! wait "$CAPTURE_PID"; then
        CAPTURE_PID=''
        [[ ! -s "$waiter_log" ]] || tail -n 20 "$waiter_log" >&2
        die "capture did not reach ${DURATION}s of synchronized raw and rendered audio"
    fi
    CAPTURE_PID=''

    # wait_capture guarantees the requested synchronized audio duration, but
    # the scenario is launched with linger_after_completion=true.  Explicitly
    # cancel the active episode so Arena can write the terminal EpisodeRecord
    # required by export_recording before the launch tree is stopped.
    cancel_and_finalize_episode "$episode_action"

    stop_launch
    wait_for_environment_shutdown

    export_args=(
        "$run_arena" --output "$run_arena" --force --expected-duration "$DURATION"
        --require-activity-annotations
        --artifact-prefix "$artifact_prefix" --execution-index "$execution_number"
        --world-name "$world_name" --scenario-name "$scenario_name"
    )
    # The dummy backend intentionally has no robot odometry. Its useful
    # products are the synchronized rendered audio and HumanSim trajectory.
    if [[ "$SIMULATOR" == dummy ]]; then export_args+=(--basic-audio-pedestrians); fi
    run_in_arena python3 -m arena_simulation_setup.acoustics.export_recording "${export_args[@]}"
    [[ -s "$validation" ]] || die "export did not create $validation"
    ) > >(tee "$failure_log") 2>&1
    scenario_status=$?
    set -e
    LAUNCH_CLIENT_PID=''
    CAPTURE_PID=''
    if ((scenario_status != 0)); then
        SCENARIO_FAIL=$((SCENARIO_FAIL + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "$run_index" "$world_name" \
            "$scenario_name" "$scenario_status" "$failure_log" \
            >>"$FAILED_SCENARIOS_FILE"
        note "FAILED $run_name (exit=$scenario_status); continuing. Log: $failure_log"
        continue
    fi
    rm -f -- "$failure_log"

    if ((HF_UPLOAD)); then
        wait_for_upload_slot
        upload_log="${OUTPUT_ROOT}/.upload_logs/${world_name}__${artifact_prefix}.log"
        remote_prefix="scenarios/${world_name}/${artifact_prefix}"
        upload_args=(
            "$UPLOAD_HELPER" "$run_host"
            --repo-id "$HF_REPO"
            --remote-prefix "$remote_prefix"
            --token-env "$HF_TOKEN_ENV"
            --world-name "$world_name"
            --scenario-name "$scenario_name"
            --execution-index "$execution_number"
        )
        if ((DELETE_AFTER_UPLOAD)); then upload_args+=(--delete-after-verify); fi

        # Upload/finalize in parallel with the next simulation. The helper only
        # removes run_host after the Hub commit and remote checksum/size
        # verification have succeeded.
        python3 "${upload_args[@]}" >"$upload_log" 2>&1 &
        UPLOAD_PIDS+=("$!")
        UPLOAD_LOGS+=("$upload_log")
        note "UPLOAD queued $run_name -> hf://datasets/${HF_REPO}/${remote_prefix}"
    fi

    note "DONE  $run_host"
done

if ((HF_UPLOAD)); then
    for i in "${!UPLOAD_PIDS[@]}"; do
        pid="${UPLOAD_PIDS[$i]}"
        log="${UPLOAD_LOGS[$i]}"
        if ! wait "$pid"; then
            UPLOAD_FAIL=1
            printf 'record_acoustics_dataset: ERROR: upload failed; local data retained. Log: %s\n' "$log" >&2
            [[ -s "$log" ]] && tail -n 80 "$log" >&2 || true
        fi
    done
fi

((UPLOAD_FAIL == 0)) || die 'one or more Hugging Face uploads failed'
((SCENARIO_FAIL == 0)) || die "$SCENARIO_FAIL scenario(s) failed; see $FAILED_SCENARIOS_FILE"
trap - EXIT INT TERM
note 'all selected scenarios have valid synchronized recordings and completed uploads'
