#include "task_generator_gui/task_generator_panel.hpp"
#include "rviz_common/display_context.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "rcl_interfaces/srv/set_parameters.hpp"

#include <chrono>
#include <array>
#include <cstdlib>
#include <exception>
#include <memory>

namespace
{
    struct MotorControlSpec
    {
        const char *name;
        const char *label;
        const char *suffix;
        double minimum;
        double maximum;
        double step;
        double initial;
        int decimals;
        const char *tooltip;
    };

    constexpr std::array<MotorControlSpec, 6> kMotorControlSpecs{{
        {"motor_volume_db", "Volume", " dB", -40.0, 6.0, 0.5, -9.0, 1,
         "Overall motor level. -6 dB is half amplitude."},
        {"motor_frequency_scale", "Frequency", " x", 0.25, 4.0, 0.05, 1.0, 2,
         "Pitch multiplier. Velocity still controls the pitch trajectory."},
        {"motor_tonal_gain_db", "Gear tone", " dB", -24.0, 12.0, 0.5, 0.0, 1,
         "Level of the periodic gear-mesh tones."},
        {"motor_broadband_gain_db", "Mechanical noise", " dB", -40.0, 6.0, 0.5, -12.0, 1,
         "Level of the broadband mechanical-noise layer."},
        {"motor_speed_exponent", "Velocity response", "", 0.25, 3.0, 0.05, 1.5, 2,
         "Higher values make motor level change more strongly with wheel speed."},
        {"motor_velocity_smoothing_sec", "Response smoothing", " s", 0.0, 0.5, 0.005, 0.015, 3,
         "Time used to smooth wheel-velocity changes. Zero is immediate."},
    }};

    bool isMotorTuningParameter(const std::string &name)
    {
        for (const auto &spec : kMotorControlSpecs)
            if (name == spec.name)
                return true;
        return false;
    }

    std::string normalizeNodePath(const std::string &path)
    {
        std::string normalized;
        normalized.reserve(path.size() + 1);
        normalized.push_back('/');
        bool previous_slash = true;
        for (const char character : path)
        {
            if (character == '/')
            {
                if (!previous_slash)
                    normalized.push_back(character);
                previous_slash = true;
                continue;
            }
            normalized.push_back(character);
            previous_slash = false;
        }
        if (normalized.size() > 1 && normalized.back() == '/')
            normalized.pop_back();
        return normalized;
    }

    bool isRuntimeMicrophoneId(const QString &listener_id)
    {
        if (!listener_id.startsWith("microphone"))
            return false;
        const QString suffix = listener_id.mid(10);
        if (suffix.isEmpty())
            return false;
        for (const QChar character : suffix)
            if (!character.isDigit())
                return false;
        return true;
    }
}

namespace task_generator_gui
{
    TaskGeneratorPanel::TaskGeneratorPanel(QWidget *parent) : Panel(parent)
    {
        root_layout = new QVBoxLayout(this);
    }

    TaskGeneratorPanel::~TaskGeneratorPanel() = default;

    void TaskGeneratorPanel::onInitialize()
    {
        node_ptr = getDisplayContext()->getRosNodeAbstraction().lock();
        node = node_ptr->get_raw_node();
        node->get_logger().set_level(rclcpp::Logger::Level::Warn);
    }

    void TaskGeneratorPanel::load(const rviz_common::Config &config)
    {
        rviz_common::Panel::load(config);

        QString result;
        if (config.mapGetString("Target", &result))
            task_generator_node = normalizeNodePath(result.toStdString());
        else
            task_generator_node = "/task_generator_node";

        motor_playback_node = normalizeNodePath(
            task_generator_node + "/robot_sound_node");
        human_playback_node = normalizeNodePath(
            task_generator_node + "/human_sound_playback");
        environment_playback_node = normalizeNodePath(
            task_generator_node + "/environmental_sound_playback");
        propagation_node = normalizeNodePath(
            task_generator_node + "/sound_propagation_node");

        // All clients go on `node` and rviz spins it continuously.
        query_environments_client = node->create_client<task_generator_msgs::srv::QueryEnvironments>(
            task_generator_node + "/query/environments");
        query_parametrizeds_client = node->create_client<task_generator_msgs::srv::QueryParametrizeds>(
            task_generator_node + "/query/parametrizeds");
        query_static_obstacles_client = node->create_client<task_generator_msgs::srv::QueryStaticObstacles>(
            task_generator_node + "/query/static_obstacles");
        query_dynamic_obstacles_client = node->create_client<task_generator_msgs::srv::QueryDynamicObstacles>(
            task_generator_node + "/query/dynamic_obstacles");
        query_scenarios_client = node->create_client<task_generator_msgs::srv::QueryScenarios>(
            task_generator_node + "/query/scenarios");
        query_worlds_client = node->create_client<task_generator_msgs::srv::QueryWorlds>(
            task_generator_node + "/query/worlds");
        query_task_modes_client = node->create_client<task_generator_msgs::srv::QueryTaskModes>(
            task_generator_node + "/query/task_modes");

        reset_episode_client = node->create_client<task_generator_msgs::srv::ResetEpisode>(
            task_generator_node + "/lifecycle/reset_episode");
        pause_client = node->create_client<task_generator_msgs::srv::Pause>(
            task_generator_node + "/lifecycle/pause");

        queue_episode_client = node->create_client<task_generator_msgs::srv::QueueEpisode>(
            task_generator_node + "/config/queue_episode");

        parameters_client = std::make_shared<rclcpp::AsyncParametersClient>(node, task_generator_node);
        motor_playback_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                motor_playback_node);
        human_playback_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                human_playback_node);
        environment_playback_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                environment_playback_node);
        propagation_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                propagation_node);
        set_audio_system_client =
            node->create_client<task_generator_msgs::srv::SetAudioSystem>(
                task_generator_node + "/runtime/set_audio_system");
        remove_microphone_client =
            node->create_client<task_generator_msgs::srv::RemoveMicrophone>(
                task_generator_node + "/runtime/remove_microphone");
        remove_audio_system_client =
            node->create_client<task_generator_msgs::srv::RemoveAudioSystem>(
                task_generator_node + "/runtime/remove_audio_system");

        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.transient_local();
            microphone_listeners_sub =
                node->create_subscription<std_msgs::msg::String>(
                    task_generator_node + "/microphone_listeners",
                    qos,
                    [this](const std_msgs::msg::String::SharedPtr msg)
                    {
                        QMetaObject::invokeMethod(this, [this, data = msg->data]()
                        {
                            updateMicrophoneListeners(data);
                        }, Qt::QueuedConnection);
                    });
        }

        {
            rclcpp::QoS qos(rclcpp::KeepLast(32));
            qos.transient_local();
            audio_system_states_sub = node->create_subscription<
                task_generator_msgs::msg::AudioSystemState>(
                task_generator_node + "/audio_system_states",
                qos,
                [this](
                    const task_generator_msgs::msg::AudioSystemState::SharedPtr msg)
                {
                    QMetaObject::invokeMethod(
                        this,
                        [this, msg]()
                        {
                            if (!audio_systems_tree || !audio_systems_group)
                                return;
                            QSignalBlocker blocker(audio_systems_tree);
                            QTreeWidgetItem *item = nullptr;
                            for (int index = 0;
                                 index < audio_systems_tree->topLevelItemCount();
                                 ++index)
                            {
                                auto candidate =
                                    audio_systems_tree->topLevelItem(index);
                                if (candidate->data(0, Qt::UserRole).toString()
                                    == QString::fromStdString(msg->system_id))
                                {
                                    item = candidate;
                                    break;
                                }
                            }
                            if (msg->emitter_ids.empty())
                            {
                                if (item)
                                    delete item;
                                audio_systems_group->setEnabled(
                                    audio_systems_tree->topLevelItemCount() > 0);
                                return;
                            }
                            if (!item)
                            {
                                item = new QTreeWidgetItem(audio_systems_tree);
                                item->setData(
                                    0,
                                    Qt::UserRole,
                                    QString::fromStdString(msg->system_id));
                                item->setFlags(
                                    item->flags() | Qt::ItemIsUserCheckable);
                            }
                            item->setText(
                                0,
                                QString::fromStdString(msg->system_id));
                            item->setText(
                                1,
                                QString::fromStdString(
                                    msg->sound_type + " / " + msg->asset_id));
                            item->setText(
                                2,
                                QString::number(
                                    static_cast<int>(msg->emitter_ids.size())));
                            item->setCheckState(
                                0,
                                msg->active ? Qt::Checked : Qt::Unchecked);
                            audio_systems_group->setEnabled(true);
                        },
                        Qt::QueuedConnection);
                });
        }

        // Latched paused-state subscription.
        {
            rclcpp::QoS pqos(rclcpp::KeepLast(1));
            pqos.transient_local();
            paused_state_sub = node->create_subscription<std_msgs::msg::Bool>(
                task_generator_node + "/state/paused",
                pqos,
                [this](const std_msgs::msg::Bool::SharedPtr msg)
                {
                    QMetaObject::invokeMethod(this, [this, paused = msg->data]()
                    {
                        paused_state = paused;
                        if (pause_button)
                            pause_button->setText(paused_state ? "Unpause" : "Pause");
                    }, Qt::QueuedConnection);
                });
        }

        // Latched state/episode subscription deduped into history_buffer_.
        {
            rclcpp::QoS qos(rclcpp::KeepLast(20));
            qos.transient_local();
            episode_sub = node->create_subscription<task_generator_msgs::msg::EpisodeRecord>(
                task_generator_node + "/state/episode",
                qos,
                [this](const task_generator_msgs::msg::EpisodeRecord::SharedPtr msg)
                {
                    QMetaObject::invokeMethod(this, [this, msg]()
                    {
                        last_current_episode_ = msg;
                        refreshMotorPlayback();

                        if (next_pending_ && msg->episode_id != next_pending_baseline_id_)
                            clearNextPending();

                        // Dedup history by episode_id: replace existing entry or append.
                        bool found = false;
                        for (auto &entry : history_buffer_)
                        {
                            if (entry.episode_id == msg->episode_id)
                            {
                                entry = *msg;
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                        {
                            history_buffer_.push_back(*msg);
                            if (history_buffer_.size() > kHistoryBufferSize)
                                history_buffer_.pop_front();
                        }

                        refreshHistory();
                    }, Qt::QueuedConnection);
                });
        }

        // Latched state/queue subscription populates widgets when a new queued record arrives.
        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.transient_local();
            queue_sub = node->create_subscription<task_generator_msgs::msg::EpisodeRecord>(
                task_generator_node + "/state/queue",
                qos,
                [this](const task_generator_msgs::msg::EpisodeRecord::SharedPtr msg)
                {
                    QMetaObject::invokeMethod(this, [this, msg]()
                    {
                        last_queued_episode_ = msg;
                        loading_from_queue_ = true;
                        populateFromQueue(*msg);
                        loading_from_queue_ = false;
                        clearDirtyFlags();
                        updateDirtyButtons();
                        refreshHistory();
                    }, Qt::QueuedConnection);
                });
        }

        // Build empty UI shell immediately and populate dropdowns as responses arrive.
        setupUi();

        // Bootstrap queries are gated on service readiness so they survive the
        // race where rviz loads before task_generator_node has advertised.
        whenReady(
            [c = query_worlds_client]() { return c->service_is_ready(); },
            [this]()
            {
                query_worlds_client->async_send_request(
                    std::make_shared<task_generator_msgs::srv::QueryWorlds::Request>(),
                    [this](rclcpp::Client<task_generator_msgs::srv::QueryWorlds>::SharedFuture f)
                    {
                        auto resp = f.get();
                        if (!resp) return;
                        QMetaObject::invokeMethod(this, [this, ids = resp->ids]()
                        {
                            worlds = ids;
                            if (staged_world.empty() && !worlds.empty())
                                staged_world = worlds[0];
                            QSignalBlocker blocker(world_combobox);
                            world_combobox->clear();
                            for (const auto &w : worlds)
                                world_combobox->addItem(QString::fromStdString(w));
                            world_combobox->setCurrentText(QString::fromStdString(staged_world));
                            world_combobox->setEnabled(true);
                        }, Qt::QueuedConnection);
                    });
            });

        whenReady(
            [c = query_task_modes_client]() { return c->service_is_ready(); },
            [this]()
            {
                query_task_modes_client->async_send_request(
                    std::make_shared<task_generator_msgs::srv::QueryTaskModes::Request>(),
                    [this](rclcpp::Client<task_generator_msgs::srv::QueryTaskModes>::SharedFuture f)
                    {
                        auto resp = f.get();
                        if (!resp) return;
                        QMetaObject::invokeMethod(this, [this, obs = resp->obstacles, rob = resp->robots]()
                        {
                            auto title = [](std::string s) -> std::string {
                                if (!s.empty())
                                    s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
                                return s;
                            };

                            obstacles_modes_ = obs;
                            robots_modes_ = rob;

                            {
                                QSignalBlocker b1(obstacles_task_mode_combobox);
                                obstacles_task_mode_combobox->clear();
                                for (const auto &m : obs)
                                    obstacles_task_mode_combobox->addItem(QString::fromStdString(title(m)));
                                if (!obstacles_task_mode.isEmpty())
                                    obstacles_task_mode_combobox->setCurrentText(obstacles_task_mode);
                            }
                            {
                                QSignalBlocker b2(robot_task_mode_combobox);
                                robot_task_mode_combobox->clear();
                                for (const auto &m : rob)
                                    robot_task_mode_combobox->addItem(QString::fromStdString(title(m)));
                                if (!robots_task_mode.isEmpty())
                                    robot_task_mode_combobox->setCurrentText(robots_task_mode);
                            }

                            obstacles_task_mode_combobox->setEnabled(true);
                            robot_task_mode_combobox->setEnabled(true);
                        }, Qt::QueuedConnection);
                    });
            });

        // Parameter event subscription: re-fetch the active param tree when
        // task.<active_mode>.* changes on the task_generator node.
        param_events_sub = node->create_subscription<rcl_interfaces::msg::ParameterEvent>(
            "/parameter_events",
            rclcpp::QoS(10),
            [this, expected_node = task_generator_node](const rcl_interfaces::msg::ParameterEvent::SharedPtr msg)
            {
                if (msg->node == motor_playback_node)
                {
                    bool refresh_tuning = false;
                    auto sync = [this, &refresh_tuning](const auto &parameters)
                    {
                        for (const auto &parameter : parameters)
                        {
                            if (parameter.name == "enable_motor_playback")
                            {
                                const bool enabled = parameter.value.bool_value;
                                QMetaObject::invokeMethod(this, [this, enabled]()
                                {
                                    syncMotorPlaybackCheckbox(enabled, true);
                                }, Qt::QueuedConnection);
                            }
                            else if (isMotorTuningParameter(parameter.name))
                            {
                                refresh_tuning = true;
                            }
                        }
                    };
                    sync(msg->changed_parameters);
                    sync(msg->new_parameters);
                    for (const auto &parameter : msg->deleted_parameters)
                    {
                        if (parameter.name == "enable_motor_playback"
                            || isMotorTuningParameter(parameter.name))
                        {
                            refresh_tuning = true;
                            break;
                        }
                    }
                    if (refresh_tuning)
                        refreshMotorPlayback();
                    for (const auto &parameter : msg->changed_parameters)
                        if (parameter.name.rfind("listener_", 0) == 0)
                            refreshAudioListenerRouting();
                    for (const auto &parameter : msg->new_parameters)
                        if (parameter.name.rfind("listener_", 0) == 0)
                            refreshAudioListenerRouting();
                    return;
                }
                if (msg->node == human_playback_node)
                {
                    refreshAudioListenerRouting();
                    return;
                }
                if (msg->node == environment_playback_node)
                {
                    refreshAudioListenerRouting();
                    refreshAuditoryControls();
                    return;
                }
                if (msg->node == propagation_node)
                {
                    refreshAuditoryControls();
                    return;
                }
                if (msg->node != expected_node) return;

                bool obs_changed = false;
                bool rob_changed = false;
                auto check = [&](const std::string &name)
                {
                    auto obs = obstacles_task_mode.toStdString();
                    for (char &c : obs) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    auto rob = robots_task_mode.toStdString();
                    for (char &c : rob) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    if (!obs.empty() && name.rfind("task." + obs + ".", 0) == 0) obs_changed = true;
                    if (!rob.empty() && name.rfind("task." + rob + ".", 0) == 0) rob_changed = true;
                };
                for (const auto &p : msg->changed_parameters) check(p.name);
                for (const auto &p : msg->new_parameters)     check(p.name);
                for (const auto &p : msg->deleted_parameters) check(p.name);

                if (!obs_changed && !rob_changed) return;
                QMetaObject::invokeMethod(this, [this, obs_changed, rob_changed]()
                {
                    if (obs_changed && obstacles_tree)
                    {
                        auto m = obstacles_task_mode.toStdString();
                        for (char &c : m) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                        dynamic_param_tree_obstacles_->rebuild("task." + m);
                    }
                    if (rob_changed && robots_tree)
                    {
                        auto m = robots_task_mode.toStdString();
                        for (char &c : m) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                        dynamic_param_tree_robots_->rebuild("task." + m);
                    }
                }, Qt::QueuedConnection);
            });

        whenReady(
            [client = motor_playback_parameters_client]()
            {
                return client->service_is_ready();
            },
            [this]() { refreshMotorPlayback(); });
        whenReady(
            [this]()
            {
                return motor_playback_parameters_client->service_is_ready()
                    && human_playback_parameters_client->service_is_ready()
                    && environment_playback_parameters_client->service_is_ready()
                    && propagation_parameters_client->service_is_ready();
            },
            [this]()
            {
                if (audio_listener_selection_pending_)
                    setAudioListenerRouting();
                else
                    refreshAudioListenerRouting();
            });
        whenReady(
            [this]()
            {
                return propagation_parameters_client->service_is_ready()
                    && environment_playback_parameters_client->service_is_ready();
            },
            [this]() { refreshAuditoryControls(); });
    }

    void TaskGeneratorPanel::refreshMotorPlayback()
    {
        if (!motor_playback_parameters_client
            || !motor_playback_parameters_client->service_is_ready())
        {
            QMetaObject::invokeMethod(this, [this]()
            {
                syncMotorPlaybackCheckbox(false, false);
                syncMotorTuningControls({}, false);
            }, Qt::QueuedConnection);
            return;
        }

        std::vector<std::string> names{"enable_motor_playback"};
        for (const auto &spec : kMotorControlSpecs)
            names.emplace_back(spec.name);
        motor_playback_parameters_client->get_parameters(
            names,
            [this](std::shared_future<std::vector<rclcpp::Parameter>> future)
            {
                bool available = false;
                bool enabled = false;
                std::vector<rclcpp::Parameter> parameters;
                try
                {
                    parameters = future.get();
                    if (!parameters.empty()
                        && parameters.front().get_type()
                            == rclcpp::ParameterType::PARAMETER_BOOL)
                    {
                        enabled = parameters.front().as_bool();
                        available = true;
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "reading motor playback parameter failed: %s",
                        exception.what());
                }
                QMetaObject::invokeMethod(
                    this,
                    [this, enabled, available, parameters]()
                {
                    syncMotorPlaybackCheckbox(enabled, available);
                    syncMotorTuningControls(parameters, available);
                },
                    Qt::QueuedConnection);
            });
    }

    void TaskGeneratorPanel::setMotorPlaybackEnabled(bool enabled)
    {
        if (!motor_playback_parameters_client
            || !motor_playback_parameters_client->service_is_ready())
        {
            syncMotorPlaybackCheckbox(false, false);
            return;
        }

        motor_playback_checkbox->setEnabled(false);
        motor_playback_parameters_client->set_parameters(
            {rclcpp::Parameter("enable_motor_playback", enabled)},
            [this](
                std::shared_future<
                    std::vector<rcl_interfaces::msg::SetParametersResult>
                > future)
            {
                try
                {
                    const auto results = future.get();
                    for (const auto &result : results)
                    {
                        if (!result.successful)
                        {
                            RCLCPP_WARN(
                                node->get_logger(),
                                "setting motor playback parameter failed: %s",
                                result.reason.c_str());
                            break;
                        }
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "setting motor playback parameter failed: %s",
                        exception.what());
                }
                refreshMotorPlayback();
            });
    }

    void TaskGeneratorPanel::syncMotorPlaybackCheckbox(
        bool enabled,
        bool available)
    {
        if (!motor_playback_checkbox)
            return;
        QSignalBlocker blocker(motor_playback_checkbox);
        motor_playback_checkbox->setChecked(enabled);
        motor_playback_checkbox->setEnabled(available);
        motor_playback_checkbox->setToolTip(
            available
                ? "Mutes only workstation motor audio. ROS propagation continues."
                : "Waiting for robot_sound_node.");
    }

    void TaskGeneratorPanel::setMotorTuningParameter(
        const std::string &name,
        double value)
    {
        if (!motor_playback_parameters_client
            || !motor_playback_parameters_client->service_is_ready())
        {
            refreshMotorPlayback();
            return;
        }
        motor_playback_parameters_client->set_parameters(
            {rclcpp::Parameter(name, value)},
            [this, name](
                std::shared_future<
                    std::vector<rcl_interfaces::msg::SetParametersResult>
                > future)
            {
                try
                {
                    const auto results = future.get();
                    if (results.empty() || !results.front().successful)
                    {
                        const auto reason = results.empty()
                            ? "no result"
                            : results.front().reason;
                        RCLCPP_WARN(
                            node->get_logger(),
                            "setting %s failed: %s",
                            name.c_str(),
                            reason.c_str());
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "setting %s failed: %s",
                        name.c_str(),
                        exception.what());
                }
                refreshMotorPlayback();
            });
    }

    void TaskGeneratorPanel::syncMotorTuningControls(
        const std::vector<rclcpp::Parameter> &parameters,
        bool available)
    {
        if (motor_tuning_group)
            motor_tuning_group->setEnabled(available);
        for (const auto &parameter : parameters)
        {
            const auto found = motor_tuning_spinboxes.find(parameter.get_name());
            if (found == motor_tuning_spinboxes.end()
                || parameter.get_type()
                    != rclcpp::ParameterType::PARAMETER_DOUBLE)
            {
                continue;
            }
            QSignalBlocker blocker(found->second);
            found->second->setValue(parameter.as_double());
        }
    }

    void TaskGeneratorPanel::resetMotorTuning()
    {
        if (!motor_playback_parameters_client
            || !motor_playback_parameters_client->service_is_ready())
        {
            refreshMotorPlayback();
            return;
        }
        std::vector<rclcpp::Parameter> parameters;
        parameters.reserve(kMotorControlSpecs.size());
        for (const auto &spec : kMotorControlSpecs)
            parameters.emplace_back(spec.name, spec.initial);
        motor_playback_parameters_client->set_parameters(
            parameters,
            [this](
                std::shared_future<
                    std::vector<rcl_interfaces::msg::SetParametersResult>
                > future)
            {
                try
                {
                    const auto results = future.get();
                    for (const auto &result : results)
                    {
                        if (result.successful)
                            continue;
                        RCLCPP_WARN(
                            node->get_logger(),
                            "resetting motor tuning failed: %s",
                            result.reason.c_str());
                        break;
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "resetting motor tuning failed: %s",
                        exception.what());
                }
                refreshMotorPlayback();
            });
    }

    void TaskGeneratorPanel::refreshAudioListenerRouting()
    {
        if (audio_listener_selection_pending_)
        {
            if (motor_playback_parameters_client
                && human_playback_parameters_client
                && environment_playback_parameters_client
                && propagation_parameters_client
                && motor_playback_parameters_client->service_is_ready()
                && human_playback_parameters_client->service_is_ready()
                && environment_playback_parameters_client->service_is_ready()
                && propagation_parameters_client->service_is_ready())
            {
                setAudioListenerRouting();
                return;
            }
            QMetaObject::invokeMethod(this, [this]()
            {
                syncAudioListenerRouting({}, false);
            }, Qt::QueuedConnection);
            return;
        }
        if (!motor_playback_parameters_client
            || !human_playback_parameters_client
            || !environment_playback_parameters_client
            || !propagation_parameters_client
            || !motor_playback_parameters_client->service_is_ready()
            || !human_playback_parameters_client->service_is_ready()
            || !environment_playback_parameters_client->service_is_ready()
            || !propagation_parameters_client->service_is_ready())
        {
            QMetaObject::invokeMethod(this, [this]()
            {
                syncAudioListenerRouting({}, false);
            }, Qt::QueuedConnection);
            return;
        }
        motor_playback_parameters_client->get_parameters(
            {"listener_mode", "listener_id", "listener_ids"},
            [this](std::shared_future<std::vector<rclcpp::Parameter>> future)
            {
                std::vector<rclcpp::Parameter> parameters;
                bool available = false;
                try
                {
                    parameters = future.get();
                    available = parameters.size() == 3;
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "reading audio listener routing failed: %s",
                        exception.what());
                }
                QMetaObject::invokeMethod(
                    this,
                    [this, parameters, available]()
                    {
                        syncAudioListenerRouting(parameters, available);
                    },
                    Qt::QueuedConnection);
            });
    }

    void TaskGeneratorPanel::updateMicrophoneListeners(
        const std::string &data)
    {
        microphone_listener_registry_ = data;
        if (!audio_listener_multiselect || !audio_listener_id_combobox)
            return;
        const auto document = QJsonDocument::fromJson(
            QByteArray::fromStdString(data));
        if (!document.isArray())
        {
            RCLCPP_WARN(
                node->get_logger(),
                "ignoring invalid microphone listener registry");
            return;
        }

        const QStringList selected = audio_listener_multiselect->currentText();
        QStringList next_selected;
        std::set<std::string> listener_ids;
        QStringList listener_items;
        for (const auto &value : document.array())
        {
            if (!value.isString())
                continue;
            const QString listener_id = value.toString();
            if (listener_id.trimmed().isEmpty())
                continue;
            const std::string id = listener_id.toStdString();
            listener_items.append(listener_id);
            listener_ids.insert(id);
            if (microphone_listener_ids_.find(id)
                == microphone_listener_ids_.end())
            {
                next_selected.append(listener_id);
            }
            else if (selected.contains(listener_id))
                next_selected.append(listener_id);
        }
        microphone_listener_ids_ = std::move(listener_ids);

        {
            QSignalBlocker selection_blocker(audio_listener_multiselect);
            QSignalBlocker removal_blocker(audio_listener_id_combobox);
            const QString removal_selection =
                audio_listener_id_combobox->currentText();
            audio_listener_multiselect->clear();
            audio_listener_multiselect->addItems(listener_items);
            audio_listener_multiselect->setCurrentText(next_selected);
            audio_listener_id_combobox->clear();
            audio_listener_id_combobox->addItems(listener_items);
            if (audio_listener_id_combobox->findText(removal_selection) >= 0)
                audio_listener_id_combobox->setCurrentText(removal_selection);
            else if (!listener_items.isEmpty())
                audio_listener_id_combobox->setCurrentIndex(
                    listener_items.size() - 1);
        }
        if (remove_microphone_button)
        {
            remove_microphone_button->setEnabled(
                isRuntimeMicrophoneId(
                    audio_listener_id_combobox->currentText()));
        }
        if (next_selected != selected)
            setAudioListenerRouting();
    }

    void TaskGeneratorPanel::setAudioListenerRouting()
    {
        if (!audio_listener_multiselect)
            return;
        if (!motor_playback_parameters_client->service_is_ready()
            || !human_playback_parameters_client->service_is_ready()
            || !environment_playback_parameters_client->service_is_ready()
            || !propagation_parameters_client->service_is_ready())
        {
            audio_listener_selection_pending_ = true;
            refreshAudioListenerRouting();
            return;
        }
        audio_listener_selection_pending_ = false;
        const QStringList selected = audio_listener_multiselect->currentText();
        QJsonArray selected_json;
        for (const auto &listener_id : selected)
            selected_json.append(listener_id);
        const std::string listener_ids = QJsonDocument(selected_json)
            .toJson(QJsonDocument::Compact)
            .toStdString();
        const std::vector<rclcpp::Parameter> parameters{
            rclcpp::Parameter("listener_mode", "list"),
            rclcpp::Parameter(
                "listener_id",
                selected.isEmpty()
                    ? std::string()
                    : selected.front().toStdString()),
            rclcpp::Parameter("listener_ids", listener_ids),
        };
        audio_listener_group->setEnabled(false);
        auto set_parameters = [this, parameters](
                                  const std::shared_ptr<rclcpp::AsyncParametersClient> &client,
                                  const char *node_name)
        {
            client->set_parameters(
                parameters,
                [this, node_name](
                    std::shared_future<
                        std::vector<rcl_interfaces::msg::SetParametersResult>
                    > future)
                {
                    try
                    {
                        for (const auto &result : future.get())
                        {
                            if (result.successful)
                                continue;
                            RCLCPP_WARN(
                                node->get_logger(),
                                "setting audio listener routing on %s failed: %s",
                                node_name,
                                result.reason.c_str());
                            break;
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "setting audio listener routing on %s failed: %s",
                            node_name,
                            exception.what());
                    }
                    refreshAudioListenerRouting();
                });
        };
        set_parameters(motor_playback_parameters_client, "robot_sound_node");
        set_parameters(human_playback_parameters_client, "human_sound_playback");
        set_parameters(
            environment_playback_parameters_client,
            "environmental_sound_playback");
        propagation_parameters_client->set_parameters(
            {rclcpp::Parameter("active_microphone_ids", listener_ids)},
            [this](auto) { refreshAudioListenerRouting(); });
    }

    void TaskGeneratorPanel::syncAudioListenerRouting(
        const std::vector<rclcpp::Parameter> &parameters,
        bool available)
    {
        if (!audio_listener_group)
            return;
        audio_listener_group->setEnabled(available);
        if (!available || parameters.size() != 3)
            return;
        QSignalBlocker selection_blocker(audio_listener_multiselect);
        const QString mode = QString::fromStdString(parameters[0].as_string());
        QStringList selected;
        if (mode == "all")
        {
            for (const auto &listener_id : microphone_listener_ids_)
                selected.append(QString::fromStdString(listener_id));
        }
        else if (mode == "selected")
        {
            const QString listener_id =
                QString::fromStdString(parameters[1].as_string());
            if (!listener_id.isEmpty())
                selected.append(listener_id);
        }
        else
        {
            const auto document = QJsonDocument::fromJson(
                QByteArray::fromStdString(parameters[2].as_string()));
            if (document.isArray())
            {
                for (const auto &value : document.array())
                {
                    if (value.isString())
                        selected.append(value.toString());
                }
            }
        }
        audio_listener_multiselect->ResetSelection();
        audio_listener_multiselect->setCurrentText(selected);
    }

    void TaskGeneratorPanel::refreshAuditoryControls()
    {
        const bool propagation_available = propagation_parameters_client
            && propagation_parameters_client->service_is_ready();
        const bool playback_available = environment_playback_parameters_client
            && environment_playback_parameters_client->service_is_ready();
        syncAuditoryControls(
            propagation_checkbox && propagation_checkbox->isChecked(),
            environment_playback_checkbox
                && environment_playback_checkbox->isChecked(),
            propagation_available,
            playback_available);
        if (propagation_available)
        {
            propagation_parameters_client->get_parameters(
                {"enable_propagation"},
                [this](
                    std::shared_future<std::vector<rclcpp::Parameter>> future)
                {
                    bool enabled = false;
                    bool available = false;
                    try
                    {
                        const auto parameters = future.get();
                        available = parameters.size() == 1;
                        enabled = available && parameters.front().as_bool();
                    }
                    catch (const std::exception &exception)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "reading propagation state failed: %s",
                            exception.what());
                    }
                    QMetaObject::invokeMethod(
                        this,
                        [this, enabled, available]()
                        {
                            if (!propagation_checkbox)
                                return;
                            QSignalBlocker blocker(propagation_checkbox);
                            propagation_checkbox->setChecked(enabled);
                            propagation_checkbox->setEnabled(available);
                        },
                        Qt::QueuedConnection);
                });
        }
        if (playback_available)
        {
            environment_playback_parameters_client->get_parameters(
                {"enable_environment_playback"},
                [this](
                    std::shared_future<std::vector<rclcpp::Parameter>> future)
                {
                    bool enabled = false;
                    bool available = false;
                    try
                    {
                        const auto parameters = future.get();
                        available = parameters.size() == 1;
                        enabled = available && parameters.front().as_bool();
                    }
                    catch (const std::exception &exception)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "reading environmental playback state failed: %s",
                            exception.what());
                    }
                    QMetaObject::invokeMethod(
                        this,
                        [this, enabled, available]()
                        {
                            if (!environment_playback_checkbox)
                                return;
                            QSignalBlocker blocker(
                                environment_playback_checkbox);
                            environment_playback_checkbox->setChecked(enabled);
                            environment_playback_checkbox->setEnabled(available);
                        },
                        Qt::QueuedConnection);
                });
        }
    }

    void TaskGeneratorPanel::setPropagationEnabled(bool enabled)
    {
        if (!propagation_parameters_client
            || !propagation_parameters_client->service_is_ready())
        {
            refreshAuditoryControls();
            return;
        }
        propagation_checkbox->setEnabled(false);
        propagation_parameters_client->set_parameters(
            {rclcpp::Parameter("enable_propagation", enabled)},
            [this](auto) { refreshAuditoryControls(); });
    }

    void TaskGeneratorPanel::setEnvironmentPlaybackEnabled(bool enabled)
    {
        if (!environment_playback_parameters_client
            || !environment_playback_parameters_client->service_is_ready())
        {
            refreshAuditoryControls();
            return;
        }
        environment_playback_checkbox->setEnabled(false);
        environment_playback_parameters_client->set_parameters(
            {rclcpp::Parameter("enable_environment_playback", enabled)},
            [this](auto) { refreshAuditoryControls(); });
    }

    void TaskGeneratorPanel::syncAuditoryControls(
        bool propagation_enabled,
        bool playback_enabled,
        bool propagation_available,
        bool playback_available)
    {
        if (propagation_checkbox)
        {
            QSignalBlocker blocker(propagation_checkbox);
            propagation_checkbox->setChecked(propagation_enabled);
            propagation_checkbox->setEnabled(propagation_available);
        }
        if (environment_playback_checkbox)
        {
            QSignalBlocker blocker(environment_playback_checkbox);
            environment_playback_checkbox->setChecked(playback_enabled);
            environment_playback_checkbox->setEnabled(playback_available);
        }
    }

    void TaskGeneratorPanel::removeSelectedMicrophone()
    {
        if (!audio_listener_id_combobox
            || !remove_microphone_client
            || !remove_microphone_client->service_is_ready())
            return;
        const std::string listener_id =
            audio_listener_id_combobox->currentText().toStdString();
        if (listener_id.empty())
            return;
        auto request = std::make_shared<
            task_generator_msgs::srv::RemoveMicrophone::Request>();
        request->listener_id = listener_id;
        remove_microphone_client->async_send_request(
            request,
            [this, listener_id](auto future)
            {
                try
                {
                    const auto response = future.get();
                    if (!response->success)
                        RCLCPP_WARN(
                            node->get_logger(),
                            "removing microphone %s failed: %s",
                            listener_id.c_str(), response->error_msg.c_str());
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "removing microphone %s failed: %s",
                        listener_id.c_str(), exception.what());
                }
            });
    }

    void TaskGeneratorPanel::removeSelectedAudioSystem()
    {
        if (!audio_systems_tree
            || !remove_audio_system_client
            || !remove_audio_system_client->service_is_ready())
            return;
        const auto items = audio_systems_tree->selectedItems();
        if (items.isEmpty())
            return;
        const std::string system_id =
            items.front()->data(0, Qt::UserRole).toString().toStdString();
        auto request = std::make_shared<
            task_generator_msgs::srv::RemoveAudioSystem::Request>();
        request->system_id = system_id;
        remove_audio_system_client->async_send_request(
            request,
            [this, system_id](auto future)
            {
                try
                {
                    const auto response = future.get();
                    if (!response->success)
                        RCLCPP_WARN(
                            node->get_logger(),
                            "removing audio source %s failed: %s",
                            system_id.c_str(), response->error_msg.c_str());
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "removing audio source %s failed: %s",
                        system_id.c_str(), exception.what());
                }
            });
    }

    void TaskGeneratorPanel::setAudioSystemActive(
        const std::string &system_id,
        bool active)
    {
        if (system_id.empty()
            || !set_audio_system_client
            || !set_audio_system_client->service_is_ready())
        {
            if (audio_systems_group)
                audio_systems_group->setEnabled(false);
            RCLCPP_WARN(
                node->get_logger(),
                "static audio control is unavailable. Launch with "
                "enable_static_audio_devices:=true");
            return;
        }
        auto request =
            std::make_shared<task_generator_msgs::srv::SetAudioSystem::Request>();
        request->system_id = system_id;
        request->active = active;
        set_audio_system_client->async_send_request(
            request,
            [this, system_id](
                rclcpp::Client<task_generator_msgs::srv::SetAudioSystem>::SharedFuture
                    future)
            {
                try
                {
                    const auto response = future.get();
                    if (!response->success)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "setting static audio system %s failed: %s",
                            system_id.c_str(),
                            response->error_msg.c_str());
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "setting static audio system %s failed: %s",
                        system_id.c_str(),
                        exception.what());
                }
            });
    }

    void TaskGeneratorPanel::whenReady(std::function<bool()> ready_check,
                                       std::function<void()> action,
                                       std::chrono::milliseconds period)
    {
        if (ready_check()) { action(); return; }
        auto holder = std::make_shared<rclcpp::TimerBase::SharedPtr>();
        std::function<void()> tick =
            [holder, check = std::move(ready_check), act = std::move(action)]() mutable
            {
                if (!check()) return;
                if (*holder) (*holder)->cancel();
                holder->reset();
                act();
            };
        *holder = node->create_wall_timer(period, std::move(tick));
    }

    void TaskGeneratorPanel::setupUi()
    {
        // World combobox is disabled until worlds arrive.
        world_combobox = setupComboBoxWithLabel(this->root_layout, QStringList{"Loading..."}, QString("World"));
        world_combobox->setEnabled(false);
        connect(world_combobox, &QComboBox::currentTextChanged, this, &TaskGeneratorPanel::onWorldChanged);

        auto auditory_controls_group = new QGroupBox("Auditory Runtime");
        auto auditory_controls_layout = new QVBoxLayout();
        propagation_checkbox = new QCheckBox(
            "Enable sound propagation");
        propagation_checkbox->setEnabled(false);
        propagation_checkbox->setToolTip(
            "Stops or resumes listener-specific simulated sound propagation.");
        connect(
            propagation_checkbox,
            &QCheckBox::toggled,
            this,
            &TaskGeneratorPanel::setPropagationEnabled);
        environment_playback_checkbox = new QCheckBox(
            "Play radio and alarm audio on this workstation");
        environment_playback_checkbox->setEnabled(false);
        environment_playback_checkbox->setToolTip(
            "Mutes only local output. Propagation and robot hearing continue.");
        connect(
            environment_playback_checkbox,
            &QCheckBox::toggled,
            this,
            &TaskGeneratorPanel::setEnvironmentPlaybackEnabled);
        auditory_controls_layout->addWidget(propagation_checkbox);
        auditory_controls_layout->addWidget(environment_playback_checkbox);
        auditory_controls_group->setLayout(auditory_controls_layout);
        root_layout->addWidget(auditory_controls_group);

        audio_listener_group = new QGroupBox("Microphone Routing");
        audio_listener_group->setEnabled(false);
        auto audio_listener_layout = new QFormLayout();
        audio_listener_multiselect = new MultiSelectComboBox();
        audio_listener_multiselect->SetPlaceHolderText(
            "Select one or more microphones");
        audio_listener_multiselect->SetSearchBarPlaceHolderText(
            "Find microphone");
        audio_listener_id_combobox = new QComboBox();
        connect(
            audio_listener_multiselect,
            &MultiSelectComboBox::selectionChanged,
            this,
            &TaskGeneratorPanel::setAudioListenerRouting);
        audio_listener_layout->addRow(
            "Microphones used",
            audio_listener_multiselect);
        audio_listener_layout->addRow(
            "Microphone to remove",
            audio_listener_id_combobox);
        remove_microphone_button = new QPushButton(
            "Remove selected runtime microphone");
        remove_microphone_button->setEnabled(false);
        connect(
            remove_microphone_button,
            &QPushButton::clicked,
            this,
            &TaskGeneratorPanel::removeSelectedMicrophone);
        connect(
            audio_listener_id_combobox,
            &QComboBox::currentTextChanged,
            this,
            [this](const QString &listener_id)
            {
                remove_microphone_button->setEnabled(
                    isRuntimeMicrophoneId(listener_id));
            });
        audio_listener_layout->addRow(remove_microphone_button);
        if (!microphone_listener_registry_.empty())
            updateMicrophoneListeners(microphone_listener_registry_);
        audio_listener_group->setLayout(audio_listener_layout);
        root_layout->addWidget(audio_listener_group);

        audio_systems_group = new QGroupBox("Static Audio Devices");
        audio_systems_group->setEnabled(false);
        auto audio_systems_layout = new QVBoxLayout();
        audio_systems_tree = new QTreeWidget();
        audio_systems_tree->setColumnCount(3);
        audio_systems_tree->setHeaderLabels(
            QStringList{"Active / system", "Type / asset", "Speakers"});
        audio_systems_tree->setRootIsDecorated(false);
        audio_systems_tree->setToolTip(
            "Check a radio or alarm to start it. Each row may drive several "
            "independent speaker positions.");
        connect(
            audio_systems_tree,
            &QTreeWidget::itemChanged,
            this,
            [this](QTreeWidgetItem *item, int column)
            {
                if (column != 0)
                    return;
                setAudioSystemActive(
                    item->data(0, Qt::UserRole).toString().toStdString(),
                    item->checkState(0) == Qt::Checked);
            });
        audio_systems_layout->addWidget(audio_systems_tree);
        remove_audio_system_button = new QPushButton(
            "Remove selected runtime source");
        remove_audio_system_button->setEnabled(false);
        connect(
            remove_audio_system_button,
            &QPushButton::clicked,
            this,
            &TaskGeneratorPanel::removeSelectedAudioSystem);
        connect(
            audio_systems_tree,
            &QTreeWidget::itemSelectionChanged,
            this,
            [this]()
            {
                const auto items = audio_systems_tree->selectedItems();
                remove_audio_system_button->setEnabled(
                    !items.isEmpty()
                    && items.front()->data(0, Qt::UserRole).toString()
                        .startsWith("runtime_"));
            });
        audio_systems_layout->addWidget(remove_audio_system_button);
        auto audio_hint = new QLabel(
            "Use the Spawn Microphone toolbar button more than once to add "
            "several listeners.");
        audio_hint->setWordWrap(true);
        audio_systems_layout->addWidget(audio_hint);
        audio_systems_group->setLayout(audio_systems_layout);
        root_layout->addWidget(audio_systems_group);

        setupTabs(this->root_layout);

        motor_playback_checkbox = new QCheckBox(
            "Play robot motor audio on this workstation");
        motor_playback_checkbox->setEnabled(false);
        motor_playback_checkbox->setToolTip(
            "Waiting for robot_sound_node.");
        connect(
            motor_playback_checkbox,
            &QCheckBox::toggled,
            this,
            &TaskGeneratorPanel::setMotorPlaybackEnabled);
        root_layout->addWidget(motor_playback_checkbox);

        motor_tuning_group = new QGroupBox("Motor Sound Tuning");
        motor_tuning_group->setEnabled(false);
        auto motor_tuning_layout = new QFormLayout();
        for (const auto &spec : kMotorControlSpecs)
        {
            auto spinbox = new QDoubleSpinBox();
            spinbox->setRange(spec.minimum, spec.maximum);
            spinbox->setSingleStep(spec.step);
            spinbox->setDecimals(spec.decimals);
            spinbox->setSuffix(spec.suffix);
            spinbox->setValue(spec.initial);
            spinbox->setToolTip(spec.tooltip);
            motor_tuning_spinboxes.emplace(spec.name, spinbox);
            connect(
                spinbox,
                &QDoubleSpinBox::editingFinished,
                this,
                [this, name = std::string(spec.name), spinbox]()
                {
                    setMotorTuningParameter(name, spinbox->value());
                });
            motor_tuning_layout->addRow(spec.label, spinbox);
        }
        auto reset_motor_tuning_button = new QPushButton("Reset motor tuning");
        connect(
            reset_motor_tuning_button,
            &QPushButton::clicked,
            this,
            &TaskGeneratorPanel::resetMotorTuning);
        motor_tuning_layout->addRow(reset_motor_tuning_button);
        motor_tuning_group->setLayout(motor_tuning_layout);
        root_layout->addWidget(motor_tuning_group);

        dynamic_param_tree_obstacles_ = std::make_unique<DynamicParamTree>(
            node, parameters_client,
            obstacles_tree,
            &param_widgets_obstacles_, &param_types_obstacles_,
            [this](const std::string &leaf)
            {
                if (loading_from_queue_) return;
                obstacles_params_dirty_ = true;
                mirrorSharedParam(leaf, true);
                updateDirtyButtons();
            },
            [this](const std::string &cat, std::function<void(std::vector<std::string>)> cb)
            {
                fetchCatalog(cat, std::move(cb));
            });

        dynamic_param_tree_robots_ = std::make_unique<DynamicParamTree>(
            node, parameters_client,
            robots_tree,
            &param_widgets_robots_, &param_types_robots_,
            [this](const std::string &leaf)
            {
                if (loading_from_queue_) return;
                robots_params_dirty_ = true;
                mirrorSharedParam(leaf, false);
                updateDirtyButtons();
            },
            [this](const std::string &cat, std::function<void(std::vector<std::string>)> cb)
            {
                fetchCatalog(cat, std::move(cb));
            });

        auto episode_nav_widget = new QWidget();
        auto episode_nav_layout = new QHBoxLayout();

        pause_button   = new QPushButton("Pause");
        discard_button = new QPushButton("Discard");
        queue_button   = new QPushButton("Queue");
        next_button    = new QPushButton("Next");

        discard_button->setEnabled(false);
        queue_button->setEnabled(false);

        connect(pause_button,   &QPushButton::clicked, this, &TaskGeneratorPanel::onPauseClicked);
        connect(discard_button, &QPushButton::clicked, this, &TaskGeneratorPanel::onDiscardClicked);
        connect(queue_button,   &QPushButton::clicked, this, &TaskGeneratorPanel::onQueueClicked);
        connect(next_button,    &QPushButton::clicked, this, &TaskGeneratorPanel::onNextClicked);

        // Fallback re-enable in case the reset is silently dropped or the episode
        // topic stops publishing, so the button does not get stuck.
        next_pending_timeout_ = new QTimer(this);
        next_pending_timeout_->setSingleShot(true);
        connect(next_pending_timeout_, &QTimer::timeout, this, [this]() { clearNextPending(); });

        episode_nav_layout->addWidget(pause_button);
        episode_nav_layout->addWidget(discard_button);
        episode_nav_layout->addWidget(queue_button);
        episode_nav_layout->addWidget(next_button);
        episode_nav_widget->setLayout(episode_nav_layout);
        root_layout->addWidget(episode_nav_widget);

        auto playlist_group  = new QGroupBox("Episode History");
        auto playlist_layout = new QVBoxLayout();

        playlist_table = new QTableWidget(0, 3);
        playlist_table->setHorizontalHeaderLabels(QStringList({"World", "State", "Info"}));
        playlist_table->horizontalHeader()->setStretchLastSection(true);
        playlist_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        playlist_table->setSelectionBehavior(QAbstractItemView::SelectRows);

        playlist_layout->addWidget(playlist_table);
        playlist_group->setLayout(playlist_layout);
        root_layout->addWidget(playlist_group);
    }

    QComboBox *TaskGeneratorPanel::setupComboBoxWithLabel(QLayout *parent, const QStringList &combobox_values, const QString &label)
    {
        auto placeholder_widget = new QWidget();
        auto placeholder_layout = new QHBoxLayout();
        auto label_label = new QLabel(label);
        auto combobox = new QComboBox();

        combobox->addItems(combobox_values);
        placeholder_layout->addWidget(label_label);
        placeholder_layout->addWidget(combobox);
        placeholder_widget->setLayout(placeholder_layout);

        parent->addWidget(placeholder_widget);

        return combobox;
    }

    QTabWidget *TaskGeneratorPanel::setupTabs(QLayout *parent)
    {
        auto tabs = new QTabWidget();
        auto obstacles_tab_widget = new QWidget();
        auto robot_tab_widget     = new QWidget();

        tabs->addTab(obstacles_tab_widget, "Obstacles");
        tabs->addTab(robot_tab_widget, "Robots");

        auto obstacles_tab_layout = new QVBoxLayout();
        auto robot_tab_layout     = new QVBoxLayout();

        // Mode comboboxes: disabled until task_modes response arrives.
        obstacles_task_mode_combobox = setupComboBoxWithLabel(
            obstacles_tab_layout,
            QStringList{"Loading..."},
            QString("Obstacles Task Mode"));
        obstacles_task_mode_combobox->setEnabled(false);
        connect(
            obstacles_task_mode_combobox,
            &QComboBox::currentTextChanged,
            this,
            &TaskGeneratorPanel::onObstaclesTaskModeChanged);

        robot_task_mode_combobox = setupComboBoxWithLabel(
            robot_tab_layout,
            QStringList{"Loading..."},
            QString("Robots Task Mode"));
        robot_task_mode_combobox->setEnabled(false);
        connect(robot_task_mode_combobox,
                &QComboBox::currentTextChanged,
                this,
                &TaskGeneratorPanel::onRobotsTaskModeChanged);

        obstacles_tree = setupTree(obstacles_tab_layout);
        obstacles_tree->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
        obstacles_tree->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
        robots_tree = setupTree(robot_tab_layout);
        robots_tree->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
        robots_tree->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);

        obstacles_tab_widget->setLayout(obstacles_tab_layout);
        robot_tab_widget->setLayout(robot_tab_layout);

        parent->addWidget(tabs);

        return tabs;
    }

    void TaskGeneratorPanel::updateTabs()
    {
        if (obstacles_task_mode_combobox)
            obstacles_task_mode_combobox->setCurrentText(obstacles_task_mode);
        if (robot_task_mode_combobox)
            robot_task_mode_combobox->setCurrentText(robots_task_mode);
        if (world_combobox)
            world_combobox->setCurrentText(QString::fromStdString(staged_world));
    }

    QTreeWidget *TaskGeneratorPanel::setupTree(QLayout *parent)
    {
        auto tree = new QTreeWidget();
        tree->setColumnCount(2);
        tree->setHeaderLabels({"Parameter", "Value"});
        tree->header()->setSectionResizeMode(QHeaderView::Stretch);

        parent->addWidget(tree);

        return tree;
    }

    void TaskGeneratorPanel::onWorldChanged(const QString &text)
    {
        staged_world = text.toStdString();

        getScenarios(staged_world);
        auto obs_mode = obstacles_task_mode.toStdString();
        for (char &c : obs_mode) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        dynamic_param_tree_obstacles_->rebuild("task." + obs_mode);

        auto rob_mode = robots_task_mode.toStdString();
        for (char &c : rob_mode) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        dynamic_param_tree_robots_->rebuild("task." + rob_mode);

        if (!loading_from_queue_)
        {
            world_dirty_ = true;
            updateDirtyButtons();
        }
    }

    void TaskGeneratorPanel::onObstaclesTaskModeChanged(const QString &text)
    {
        obstacles_task_mode = text;
        auto mode = text.toStdString();
        for (char &c : mode) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        dynamic_param_tree_obstacles_->rebuild("task." + mode);

        if (!loading_from_queue_)
        {
            tm_obstacles_dirty_ = true;
            updateDirtyButtons();
        }
    }

    void TaskGeneratorPanel::onRobotsTaskModeChanged(const QString &text)
    {
        robots_task_mode = text;
        auto mode = text.toStdString();
        for (char &c : mode) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        dynamic_param_tree_robots_->rebuild("task." + mode);

        if (!loading_from_queue_)
        {
            tm_robots_dirty_ = true;
            updateDirtyButtons();
        }
    }

    // --- Discard / Queue / Next ---

    void TaskGeneratorPanel::onDiscardClicked()
    {
        if (!last_queued_episode_)
            return;
        loading_from_queue_ = true;
        populateFromQueue(*last_queued_episode_);
        loading_from_queue_ = false;
        clearDirtyFlags();
        updateDirtyButtons();
    }

    void TaskGeneratorPanel::onQueueClicked()
    {
        pushQueueEpisode([](bool) {});
    }

    void TaskGeneratorPanel::onNextClicked()
    {
        if (next_pending_)
            return;

        next_pending_ = true;
        next_pending_baseline_id_ = last_current_episode_ ? last_current_episode_->episode_id : 0;
        next_button->setEnabled(false);
        updateDirtyButtons();
        next_pending_timeout_->start(std::chrono::seconds(30));

        if (isDirty())
        {
            pushQueueEpisode([this](bool ok)
            {
                if (ok)
                    QMetaObject::invokeMethod(this, [this]() { sendResetEpisode(); }, Qt::QueuedConnection);
                else
                    QMetaObject::invokeMethod(this, [this]() { clearNextPending(); }, Qt::QueuedConnection);
            });
        }
        else
        {
            sendResetEpisode();
        }
    }

    void TaskGeneratorPanel::sendResetEpisode()
    {
        auto request = std::make_shared<task_generator_msgs::srv::ResetEpisode::Request>();
        request->seed = -1;
        reset_episode_client->async_send_request(
            request,
            [this](rclcpp::Client<task_generator_msgs::srv::ResetEpisode>::SharedFuture f)
            {
                auto resp = f.get();
                if (resp && !resp->success)
                {
                    RCLCPP_WARN(node->get_logger(),
                                "reset_episode failed: %s", resp->error_msg.c_str());
                    QMetaObject::invokeMethod(this, [this]() { clearNextPending(); }, Qt::QueuedConnection);
                }
            });
    }

    void TaskGeneratorPanel::clearNextPending()
    {
        next_pending_ = false;
        next_pending_baseline_id_ = 0;
        if (next_pending_timeout_)
            next_pending_timeout_->stop();
        if (next_button)
            next_button->setEnabled(true);
        updateDirtyButtons();
    }

    // --- Pause toggle ---

    void TaskGeneratorPanel::onPauseClicked()
    {
        auto request = std::make_shared<task_generator_msgs::srv::Pause::Request>();
        request->action = task_generator_msgs::srv::Pause::Request::TOGGLE;
        pause_client->async_send_request(request);
    }

    // --- Dirty flag management ---

    void TaskGeneratorPanel::clearDirtyFlags()
    {
        obstacles_params_dirty_ = false;
        robots_params_dirty_    = false;
        world_dirty_            = false;
        tm_obstacles_dirty_     = false;
        tm_robots_dirty_        = false;
    }

    void TaskGeneratorPanel::updateDirtyButtons()
    {
        const bool enable = isDirty() && last_queued_episode_ != nullptr && !next_pending_;
        if (discard_button) discard_button->setEnabled(enable);
        if (queue_button)   queue_button->setEnabled(enable);
    }

    // --- Populate widgets from queued record ---

    void TaskGeneratorPanel::populateFromQueue(const task_generator_msgs::msg::EpisodeRecord &rec)
    {
        auto title = [](std::string s) -> std::string {
            if (!s.empty())
                s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
            return s;
        };

        if (!rec.world.empty() && rec.world != staged_world)
        {
            staged_world = rec.world;
            if (world_combobox)
            {
                QSignalBlocker b(world_combobox);
                if (world_combobox->findText(QString::fromStdString(staged_world)) < 0)
                    world_combobox->addItem(QString::fromStdString(staged_world));
                world_combobox->setCurrentText(QString::fromStdString(staged_world));
            }
        }

        auto lower = [](std::string s) {
            for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };

        if (!rec.tm_obstacles.empty())
        {
            const auto new_mode = lower(rec.tm_obstacles);
            const auto cur_mode = lower(obstacles_task_mode.toStdString());
            obstacles_task_mode = QString::fromStdString(title(rec.tm_obstacles));
            if (obstacles_task_mode_combobox)
            {
                QSignalBlocker b(obstacles_task_mode_combobox);
                obstacles_task_mode_combobox->setCurrentText(obstacles_task_mode);
            }
            // Rebuild only on actual mode change. Rebuilds are async and read
            // live param values, which would clobber the staged values carried
            // in rec.<>_params (set below) since staging does not write the
            // live param.
            if (new_mode != cur_mode)
                dynamic_param_tree_obstacles_->rebuild("task." + new_mode);
        }

        if (!rec.tm_robots.empty())
        {
            const auto new_mode = lower(rec.tm_robots);
            const auto cur_mode = lower(robots_task_mode.toStdString());
            robots_task_mode = QString::fromStdString(title(rec.tm_robots));
            if (robot_task_mode_combobox)
            {
                QSignalBlocker b(robot_task_mode_combobox);
                robot_task_mode_combobox->setCurrentText(robots_task_mode);
            }
            if (new_mode != cur_mode)
                dynamic_param_tree_robots_->rebuild("task." + new_mode);
        }

        for (const auto &p : rec.obstacles_params)
        {
            auto it = param_widgets_obstacles_.find(p.name);
            if (it != param_widgets_obstacles_.end())
                DynamicParamTree::setWidgetValueFromParam(it->second, p);
        }
        for (const auto &p : rec.robots_params)
        {
            auto it = param_widgets_robots_.find(p.name);
            if (it != param_widgets_robots_.end())
                DynamicParamTree::setWidgetValueFromParam(it->second, p);
        }
    }

    // --- History table ---

    void TaskGeneratorPanel::refreshHistory()
    {
        if (!playlist_table)
            return;

        auto outcomeLabel = [](uint8_t s) -> QString
        {
            switch (s)
            {
            case task_generator_msgs::msg::EpisodeRecord::QUEUED:   return "QUEUED";
            case task_generator_msgs::msg::EpisodeRecord::RUNNING:  return "RUNNING";
            case task_generator_msgs::msg::EpisodeRecord::SUCCESS:  return "SUCCESS";
            case task_generator_msgs::msg::EpisodeRecord::FAILED:   return "FAILED";
            case task_generator_msgs::msg::EpisodeRecord::SKIPPED:  return "SKIPPED";
            case task_generator_msgs::msg::EpisodeRecord::FATAL:    return "FATAL";
            default:                                                 return "UNKNOWN";
            }
        };

        // Count rows: history + optional current + optional queued preview.
        // Queued row is hidden while running, the latched queue snapshot is stale until next reset.
        bool show_queued = false;
        const bool current_running = last_current_episode_
            && last_current_episode_->outcome_state == task_generator_msgs::msg::EpisodeRecord::RUNNING;
        if (last_queued_episode_ && !current_running)
        {
            if (!last_current_episode_)
            {
                show_queued = true;
            }
            else
            {
                const auto &cur = *last_current_episode_;
                const auto &que = *last_queued_episode_;
                show_queued = (cur.world != que.world
                    || cur.tm_robots != que.tm_robots
                    || cur.tm_obstacles != que.tm_obstacles
                    || cur.robots != que.robots
                    || cur.obstacles_params.size() != que.obstacles_params.size()
                    || cur.robots_params.size() != que.robots_params.size());
            }
        }

        const uint32_t current_id = last_current_episode_ ? last_current_episode_->episode_id : 0;

        int history_rows = 0;
        for (const auto &rec : history_buffer_)
        {
            if (last_current_episode_ && rec.episode_id == current_id) continue;
            ++history_rows;
        }

        int total_rows = history_rows;
        if (last_current_episode_) ++total_rows;
        if (show_queued)           ++total_rows;

        playlist_table->setRowCount(total_rows);

        int row = 0;
        for (const auto &rec : history_buffer_)
        {
            if (last_current_episode_ && rec.episode_id == current_id) continue;
            playlist_table->setItem(row, 0, new QTableWidgetItem(QString::fromStdString(rec.world)));
            playlist_table->setItem(row, 1, new QTableWidgetItem(outcomeLabel(rec.outcome_state)));
            playlist_table->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(rec.outcome_info)));
            ++row;
        }

        if (last_current_episode_)
        {
            const auto &cur = *last_current_episode_;
            auto makeBold = [](const QString &text)
            {
                auto item = new QTableWidgetItem(text);
                QFont f   = item->font();
                f.setBold(true);
                item->setFont(f);
                return item;
            };
            playlist_table->setItem(row, 0, makeBold(QString::fromStdString(cur.world)));
            playlist_table->setItem(row, 1, makeBold(outcomeLabel(cur.outcome_state)));
            playlist_table->setItem(row, 2, makeBold(QString::fromStdString(cur.outcome_info)));
            ++row;
        }

        if (show_queued)
        {
            const auto &que = *last_queued_episode_;
            auto makeItalic = [](const QString &text)
            {
                auto item = new QTableWidgetItem(text);
                QFont f   = item->font();
                f.setItalic(true);
                item->setFont(f);
                return item;
            };
            playlist_table->setItem(row, 0, makeItalic(QString::fromStdString(que.world)));
            playlist_table->setItem(row, 1, makeItalic(QString("(queued)")));
            playlist_table->setItem(row, 2, makeItalic(QString::fromStdString(que.outcome_info)));
        }
    }

} // namespace task_generator_gui

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(task_generator_gui::TaskGeneratorPanel, rviz_common::Panel)
