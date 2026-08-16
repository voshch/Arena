#include "task_generator_gui/auditory_panel.hpp"
#include "rviz_common/display_context.hpp"

#include <QJsonArray>

#include <array>
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

}
namespace task_generator_gui
{
    AuditoryPanel::AuditoryPanel(QWidget *parent) : Panel(parent)
    {
        root_layout = new QVBoxLayout(this);
    }

    AuditoryPanel::~AuditoryPanel() = default;

    void AuditoryPanel::onInitialize()
    {
        node_ptr = getDisplayContext()->getRosNodeAbstraction().lock();
        node = node_ptr->get_raw_node();
        node->get_logger().set_level(rclcpp::Logger::Level::Warn);
    }

    void AuditoryPanel::load(const rviz_common::Config &config)
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
            task_generator_node + "/environment_sound_playback");
        propagation_node = normalizeNodePath(
            task_generator_node + "/sound_propagation_node");

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


        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.transient_local();
            episode_sub = node->create_subscription<task_generator_msgs::msg::EpisodeRecord>(
                task_generator_node + "/state/episode",
                qos,
                [this](const task_generator_msgs::msg::EpisodeRecord::SharedPtr)
                {
                    QMetaObject::invokeMethod(this, [this]()
                    {
                        refreshMotorPlayback();
                    }, Qt::QueuedConnection);
                });
        }

        setupUi();

        param_events_sub = node->create_subscription<rcl_interfaces::msg::ParameterEvent>(
            "/parameter_events",
            rclcpp::QoS(10),
            [this](const rcl_interfaces::msg::ParameterEvent::SharedPtr msg)
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

    void AuditoryPanel::whenReady(std::function<bool()> ready_check,
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

    void AuditoryPanel::refreshMotorPlayback()
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

    void AuditoryPanel::setMotorPlaybackEnabled(bool enabled)
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

    void AuditoryPanel::syncMotorPlaybackCheckbox(
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

    void AuditoryPanel::setMotorTuningParameter(
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

    void AuditoryPanel::syncMotorTuningControls(
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

    void AuditoryPanel::resetMotorTuning()
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

    void AuditoryPanel::refreshAudioListenerRouting()
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
            {"listener_id"},
            [this](std::shared_future<std::vector<rclcpp::Parameter>> future)
            {
                std::vector<rclcpp::Parameter> parameters;
                bool available = false;
                try
                {
                    parameters = future.get();
                    available = parameters.size() == 1;
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

    void AuditoryPanel::updateMicrophoneListeners(
        const std::string &data)
    {
        microphone_listener_registry_ = data;
        if (!audio_listener_id_combobox)
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

        const QString selected = audio_listener_id_combobox->currentText();
        QStringList listener_items;
        for (const auto &value : document.array())
        {
            if (!value.isString())
                continue;
            const QString listener_id = value.toString();
            if (listener_id.trimmed().isEmpty())
                continue;
            listener_items.append(listener_id);
        }

        {
            QSignalBlocker blocker(audio_listener_id_combobox);
            audio_listener_id_combobox->clear();
            audio_listener_id_combobox->addItems(listener_items);
            const int selected_index =
                audio_listener_id_combobox->findText(selected);
            if (selected_index >= 0)
                audio_listener_id_combobox->setCurrentIndex(selected_index);
        }
        if (audio_listener_id_combobox->currentText() != selected)
            setAudioListenerRouting();
    }

    void AuditoryPanel::setAudioListenerRouting()
    {
        if (!audio_listener_id_combobox)
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
        const std::string listener_id =
            audio_listener_id_combobox->currentText().toStdString();
        const std::vector<rclcpp::Parameter> parameters{
            rclcpp::Parameter("listener_id", listener_id),
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
            "environment_sound_playback");
        propagation_parameters_client->set_parameters(
            {rclcpp::Parameter("active_microphone_id", listener_id)},
            [this](auto) { refreshAudioListenerRouting(); });
    }

    void AuditoryPanel::syncAudioListenerRouting(
        const std::vector<rclcpp::Parameter> &parameters,
        bool available)
    {
        if (!audio_listener_group)
            return;
        audio_listener_group->setEnabled(available);
        if (!available || parameters.size() != 1)
            return;
        QSignalBlocker blocker(audio_listener_id_combobox);
        const QString selected =
            QString::fromStdString(parameters.front().as_string());
        const int selected_index =
            audio_listener_id_combobox->findText(selected);
        if (selected_index >= 0)
            audio_listener_id_combobox->setCurrentIndex(selected_index);
    }

    void AuditoryPanel::refreshAuditoryControls()
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
                            "reading environment playback state failed: %s",
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

    void AuditoryPanel::setPropagationEnabled(bool enabled)
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

    void AuditoryPanel::setEnvironmentPlaybackEnabled(bool enabled)
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

    void AuditoryPanel::syncAuditoryControls(
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

    void AuditoryPanel::removeSelectedAudioSystem()
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
            [this, system_id]( rclcpp::Client<task_generator_msgs::srv::RemoveAudioSystem>::SharedFuture future)
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

    void AuditoryPanel::setAudioSystemActive(
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
                "environment audio control is unavailable. Launch with "
                "auditory.static_devices:=[...]");
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
                            "setting environment audio system %s failed: %s",
                            system_id.c_str(),
                            response->error_msg.c_str());
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "setting environment audio system %s failed: %s",
                        system_id.c_str(),
                        exception.what());
                }
            });
    }

    void AuditoryPanel::setupUi()
    {
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
            &AuditoryPanel::setPropagationEnabled);
        environment_playback_checkbox = new QCheckBox(
            "Play environment audio on this workstation");
        environment_playback_checkbox->setEnabled(false);
        environment_playback_checkbox->setToolTip(
            "Mutes only local output. Propagation and robot hearing continue.");
        connect(
            environment_playback_checkbox,
            &QCheckBox::toggled,
            this,
            &AuditoryPanel::setEnvironmentPlaybackEnabled);
        auditory_controls_layout->addWidget(propagation_checkbox);
        auditory_controls_layout->addWidget(environment_playback_checkbox);
        auditory_controls_group->setLayout(auditory_controls_layout);
        root_layout->addWidget(auditory_controls_group);

        audio_listener_group = new QGroupBox("Audio Playback Microphone");
        audio_listener_group->setEnabled(false);
        auto audio_listener_layout = new QFormLayout();
        audio_listener_id_combobox = new QComboBox();
        connect(
            audio_listener_id_combobox,
            &QComboBox::currentTextChanged,
            this,
            [this](const QString &)
            {
                setAudioListenerRouting();
            });
        audio_listener_layout->addRow(
            "Listen through",
            audio_listener_id_combobox);
        if (!microphone_listener_registry_.empty())
            updateMicrophoneListeners(microphone_listener_registry_);
        audio_listener_group->setLayout(audio_listener_layout);
        root_layout->addWidget(audio_listener_group);

        audio_systems_group = new QGroupBox("Environment Audio Sources");
        audio_systems_group->setEnabled(false);
        auto audio_systems_layout = new QVBoxLayout();
        audio_systems_tree = new QTreeWidget();
        audio_systems_tree->setColumnCount(3);
        audio_systems_tree->setHeaderLabels(
            QStringList{"Active / system", "Type / asset", "Speakers"});
        audio_systems_tree->setRootIsDecorated(false);
        audio_systems_tree->setToolTip(
            "Check a source to start it. Each row may drive several "
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
            &AuditoryPanel::removeSelectedAudioSystem);
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
        audio_systems_group->setLayout(audio_systems_layout);
        root_layout->addWidget(audio_systems_group);

        motor_playback_checkbox = new QCheckBox(
            "Play robot motor audio on this workstation");
        motor_playback_checkbox->setEnabled(false);
        motor_playback_checkbox->setToolTip(
            "Waiting for robot_sound_node.");
        connect(
            motor_playback_checkbox,
            &QCheckBox::toggled,
            this,
            &AuditoryPanel::setMotorPlaybackEnabled);
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
            &AuditoryPanel::resetMotorTuning);
        motor_tuning_layout->addRow(reset_motor_tuning_button);
        motor_tuning_group->setLayout(motor_tuning_layout);
        root_layout->addWidget(motor_tuning_group);
    }
} // namespace task_generator_gui

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(task_generator_gui::AuditoryPanel, rviz_common::Panel)
