#ifndef TASK_GENERATOR_GUI_AUDITORY_PANEL_HPP
#define TASK_GENERATOR_GUI_AUDITORY_PANEL_HPP

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/parameter_client.hpp"

#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include "task_generator_msgs/msg/episode_record.hpp"
#include "task_generator_msgs/msg/semantic_snapshot.hpp"
#include "task_generator_msgs/srv/remove_sound.hpp"
#include "task_generator_msgs/srv/remove_microphone.hpp"
#include "task_generator_msgs/srv/set_semantic.hpp"

#include <std_msgs/msg/string.hpp>
#include <rcl_interfaces/msg/parameter_event.hpp>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QJsonDocument>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace task_generator_gui
{

class AuditoryPanel : public rviz_common::Panel
{
    Q_OBJECT

public:
    explicit AuditoryPanel(QWidget *parent = nullptr);
    ~AuditoryPanel() override;

    void onInitialize() override;
    void load(const rviz_common::Config &config) override;

    void whenReady(std::function<bool()> ready_check,
                   std::function<void()> action,
                   std::chrono::milliseconds period = std::chrono::milliseconds(200));

    void refreshMotorPlayback();
    void setMotorPlaybackEnabled(bool enabled);
    void syncMotorPlaybackCheckbox(bool enabled, bool available);
    void setMotorTuningParameter(const std::string &name, double value);
    void syncMotorTuningControls(
        const std::vector<rclcpp::Parameter> &parameters,
        bool available);
    void resetMotorTuning();
    void refreshAudioListenerRouting();
    void setAudioListenerRouting();
    void updateMicrophoneListeners(const std::string &data);
    void selectSideMicrophone(const std::string &listener_id);
    void syncSideMicrophoneButtons();
    void syncAudioListenerRouting(
        const std::vector<rclcpp::Parameter> &parameters,
        bool available);
    void refreshAuditoryControls();
    void setPropagationEnabled(bool enabled);
    void setEnvironmentPlaybackEnabled(bool enabled);
    void syncAuditoryControls(
        bool propagation_enabled,
        bool playback_enabled,
        bool propagation_available,
        bool playback_available);
    void setSoundSounding(const std::string &entity, bool sounding);
    void removeSelectedSound();
    void setArrayParameters(const std::vector<rclcpp::Parameter> &parameters);
    void refreshArrayControls();

protected:
    std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr;
    rclcpp::Node::SharedPtr node;

    std::string task_generator_node;
    std::string motor_playback_node;
    std::string human_playback_node;
    std::string environment_playback_node;
    std::string propagation_node;
    std::string microphone_array_node;

    std::shared_ptr<rclcpp::AsyncParametersClient> motor_playback_parameters_client;
    std::shared_ptr<rclcpp::AsyncParametersClient> human_playback_parameters_client;
    std::shared_ptr<rclcpp::AsyncParametersClient> environment_playback_parameters_client;
    std::shared_ptr<rclcpp::AsyncParametersClient> propagation_parameters_client;
    rclcpp::Client<task_generator_msgs::srv::SetSemantic>::SharedPtr set_semantic_client;
    std::shared_ptr<rclcpp::AsyncParametersClient> microphone_array_parameters_client;
    rclcpp::Client<task_generator_msgs::srv::RemoveMicrophone>::SharedPtr remove_microphone_client;
    rclcpp::Client<task_generator_msgs::srv::RemoveSound>::SharedPtr remove_sound_client;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr microphone_listeners_sub;
    rclcpp::Subscription<task_generator_msgs::msg::SemanticSnapshot>::SharedPtr semantic_snapshot_sub;
    rclcpp::Subscription<task_generator_msgs::msg::EpisodeRecord>::SharedPtr episode_sub;
    rclcpp::Subscription<rcl_interfaces::msg::ParameterEvent>::SharedPtr param_events_sub;
    std::string microphone_listener_registry_;

    QVBoxLayout *root_layout{nullptr};
    QCheckBox *motor_playback_checkbox{nullptr};
    QGroupBox *motor_tuning_group{nullptr};
    QGroupBox *audio_listener_group{nullptr};
    QComboBox *audio_listener_id_combobox{nullptr};
    QPushButton *left_microphone_button{nullptr};
    QPushButton *right_microphone_button{nullptr};
    std::string left_microphone_listener_id_;
    std::string right_microphone_listener_id_;
    bool audio_listener_selection_pending_{false};
    QCheckBox *propagation_checkbox{nullptr};
    QCheckBox *environment_playback_checkbox{nullptr};
    QGroupBox *sounds_group{nullptr};
    QGroupBox *microphone_array_group{nullptr};
    QCheckBox *array_enabled_checkbox{nullptr};
    QCheckBox *headphones_enabled_checkbox{nullptr};
    QCheckBox *array_mute_checkbox{nullptr};
    QCheckBox *array_visualization_checkbox{nullptr};
    QCheckBox *array_tdoa_checkbox{nullptr};
    QDoubleSpinBox *array_master_gain_spinbox{nullptr};
    QDoubleSpinBox *array_monitor_gain_spinbox{nullptr};
    QDoubleSpinBox *array_front_gain_spinbox{nullptr};
    QDoubleSpinBox *array_rear_gain_spinbox{nullptr};
    QComboBox *array_solo_combobox{nullptr};
    QComboBox *array_monitor_combobox{nullptr};
    QTreeWidget *sounds_tree{nullptr};
    QPushButton *remove_sound_button{nullptr};
    std::unordered_map<std::string, QDoubleSpinBox *> motor_tuning_spinboxes;

    void setupUi();
};

} // namespace task_generator_gui

#endif // TASK_GENERATOR_GUI_AUDITORY_PANEL_HPP
