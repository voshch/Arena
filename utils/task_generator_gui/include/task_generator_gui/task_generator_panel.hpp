#ifndef TASK_GENERATOR_GUI_TASK_GENERATOR_PANEL_HPP
#define TASK_GENERATOR_GUI_TASK_GENERATOR_PANEL_HPP

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/parameter_client.hpp"

#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/properties/property_tree_model.hpp>

#include "diagnostic_msgs/msg/key_value.hpp"

#include "arena_runtime_msgs/srv/query_environments.hpp"
#include "arena_runtime_msgs/srv/query_parametrizeds.hpp"
#include "arena_runtime_msgs/srv/query_static_obstacles.hpp"
#include "arena_runtime_msgs/srv/query_dynamic_obstacles.hpp"
#include "arena_runtime_msgs/srv/query_scenarios.hpp"
#include "arena_runtime_msgs/srv/query_task_modes.hpp"
#include "arena_runtime_msgs/srv/query_worlds.hpp"
#include "arena_runtime_msgs/srv/reset_episode.hpp"
#include "arena_runtime_msgs/srv/queue_episode.hpp"

#include "arena_runtime_msgs/msg/episode_record.hpp"

#include "arena_runtime_msgs/msg/hold_registry.hpp"
#include "arena_runtime_msgs/srv/lifecycle_hold.hpp"

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>

#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_event.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>

#include <QLabel>
#include <QPushButton>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QComboBox>
#include <QTreeView>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QTreeView>
#include <QStandardItemModel>
#include <QDir>
#include <QHeaderView>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QFontMetrics>
#include <QTextEdit>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QScrollArea>
#include <QLineEdit>
#include <QSignalBlocker>
#include <QTimer>
#include "Qt-MultiSelectComboBox/MultiSelectComboBox.h"

#include "task_generator_gui/utils/dynamic_param_tree.hpp"

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace task_generator_gui
{
    using rviz_common::properties::PropertyTreeModel;

    class TaskGeneratorPanel
        : public rviz_common::Panel
    {
        Q_OBJECT

    public:
        explicit TaskGeneratorPanel(QWidget *parent = 0);
        ~TaskGeneratorPanel() override;
        void onInitialize() override;
        void load(const rviz_common::Config &config) override;

        void getWorlds();

        void getTMObstaclesParams();
        void getScenarios(const std::string &world_name);

        static constexpr const char *kObstaclesAxis = "obstacles";
        static constexpr const char *kRobotsAxis = "robots";

        // Lower-case mode name, the registered name the env knows the mode by.
        static std::string modeName(const QString &mode);

        // The param namespace the env reports for the axis's mode, empty until query/task_modes answered.
        std::string namespaceOf(const std::string &axis, const std::string &mode) const;

        // Rebuild the tree under the namespace of the axis's mode, nothing while the namespace is unknown.
        void rebuildTree(DynamicParamTree &tree, const std::string &axis, const std::string &mode);

        // Names the axis lists in an axis -> comma-separated names mapping, empty when the axis is absent.
        static std::vector<std::string> modesOf(const std::vector<diagnostic_msgs::msg::KeyValue> &modes,
                                                const std::string &axis);

        // Select the axis's mode and add the tree's params under their full names.
        void addModeToRequest(arena_runtime_msgs::srv::QueueEpisode::Request &req,
                              const std::string &axis, const QString &mode,
                              const std::unordered_map<std::string, QWidget *> &widgets,
                              const std::unordered_map<std::string, uint8_t> &types);
        void getParams();

        // Build a QueueEpisode request from current widget state.
        arena_runtime_msgs::srv::QueueEpisode::Request::SharedPtr buildQueueEpisodeRequest();

        // Send a QueueEpisode request; calls on_done(success) when the response arrives.
        void pushQueueEpisode(std::function<void(bool)> on_done);

        // Populate all widgets from a queued EpisodeRecord; suppresses dirty bumps.
        void populateFromQueue(const arena_runtime_msgs::msg::EpisodeRecord &rec);

        // Clear all dirty flags.
        void clearDirtyFlags();

        // Update enabled state of discard/queue buttons.
        void updateDirtyButtons();

        // Render history_buffer_ + current + queued rows into playlist_table.
        void refreshHistory();

        // Send reset_episode (world field intentionally empty; node resolves from pending overrides).
        void sendResetEpisode();

        // When obstacles and robots task modes match, the leaf maps to the same
        // ROS param. Copy the source widget's current value to its twin in the
        // other tree so the two stay in lockstep as the user edits.
        void mirrorSharedParam(const std::string &leaf, bool from_obstacles);

        // Non-blocking readiness gate: polls `ready_check` on the rviz executor
        // until true, then runs `action` once.
        void whenReady(std::function<bool()> ready_check,
                       std::function<void()> action,
                       std::chrono::milliseconds period = std::chrono::milliseconds(200));

        // Catalog query behind the same readiness gate: yields the response's
        // `ids`, or an empty list if the call fails.
        template <typename Srv>
        void queryIds(const std::shared_ptr<rclcpp::Client<Srv>> &client,
                      std::shared_ptr<typename Srv::Request> request,
                      std::function<void(std::vector<std::string>)> callback);

        // Rebuild both param trees against the current task modes.
        void rebuildParamTrees();

        void setupUi();
        QComboBox *setupComboBoxWithLabel(QLayout *parent, const QStringList &combobox_values, const QString &label);
        QTabWidget *setupTabs(QLayout *Parent);
        void updateTabs();
        QTreeWidget *setupTree(QLayout *parent);
        QWidget *setupMinMaxSpinBox(std::vector<std::int64_t, std::allocator<std::int64_t>> *connected_values);
        MultiSelectComboBox *setupGroupCheckBox(std::vector<std::string> check_box_texts, std::vector<int> *connected_hash_map);

        std::vector<std::string> convert(const QStringList &qList);

        void fetchCatalog(const std::string &catalog_name,
                          std::function<void(std::vector<std::string>)> callback);

    protected:
        std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr;
        rclcpp::Node::SharedPtr node;

        std::string arena_env_node;

        // --- Query service clients (on `node`, spun by rviz executor) ---
        rclcpp::Client<arena_runtime_msgs::srv::QueryEnvironments>::SharedPtr query_environments_client;
        rclcpp::Client<arena_runtime_msgs::srv::QueryParametrizeds>::SharedPtr query_parametrizeds_client;
        rclcpp::Client<arena_runtime_msgs::srv::QueryStaticObstacles>::SharedPtr query_static_obstacles_client;
        rclcpp::Client<arena_runtime_msgs::srv::QueryDynamicObstacles>::SharedPtr query_dynamic_obstacles_client;
        rclcpp::Client<arena_runtime_msgs::srv::QueryScenarios>::SharedPtr query_scenarios_client;
        rclcpp::Client<arena_runtime_msgs::srv::QueryWorlds>::SharedPtr query_worlds_client;
        rclcpp::Client<arena_runtime_msgs::srv::QueryTaskModes>::SharedPtr query_task_modes_client;

        // --- Lifecycle service clients ---
        rclcpp::Client<arena_runtime_msgs::srv::ResetEpisode>::SharedPtr reset_episode_client;
        rclcpp::Client<arena_runtime_msgs::srv::LifecycleHold>::SharedPtr hold_client;
        std::string hold_caller_id;

        // --- Config service clients ---
        rclcpp::Client<arena_runtime_msgs::srv::QueueEpisode>::SharedPtr queue_episode_client;

        std::shared_ptr<rclcpp::AsyncParametersClient> parameters_client;

        // --- state/episode subscription (current, deduped into history_buffer_) ---
        rclcpp::Subscription<arena_runtime_msgs::msg::EpisodeRecord>::SharedPtr episode_sub;
        arena_runtime_msgs::msg::EpisodeRecord::SharedPtr last_current_episode_;

        // --- state/queue subscription (latched, next-to-run slot) ---
        rclcpp::Subscription<arena_runtime_msgs::msg::EpisodeRecord>::SharedPtr queue_sub;
        arena_runtime_msgs::msg::EpisodeRecord::SharedPtr last_queued_episode_;

        // Ordered ring of completed/current records, deduped by episode_id, max 50 entries.
        static constexpr size_t kHistoryBufferSize = 50;
        std::deque<arena_runtime_msgs::msg::EpisodeRecord> history_buffer_;

        // --- parameter_events subscription ---
        rclcpp::Subscription<rcl_interfaces::msg::ParameterEvent>::SharedPtr param_events_sub;

        // --- /arena/state/holders subscription (latched) ---
        rclcpp::Subscription<arena_runtime_msgs::msg::HoldRegistry>::SharedPtr holders_sub;

        std::string staged_world;

        std::vector<std::string> worlds;
        std::vector<std::string> obstacles_modes_;
        std::unordered_map<std::string, std::string> mode_namespaces_;
        std::vector<std::string> robots_modes_;

        std::unordered_map<std::string, QWidget *> param_widgets_obstacles_;
        std::unordered_map<std::string, QWidget *> param_widgets_robots_;
        std::unordered_map<std::string, uint8_t> param_types_obstacles_;
        std::unordered_map<std::string, uint8_t> param_types_robots_;

        std::unique_ptr<DynamicParamTree> dynamic_param_tree_obstacles_;
        std::unique_ptr<DynamicParamTree> dynamic_param_tree_robots_;

        // Dirty-tracking flags: set when the user edits a widget.
        bool obstacles_params_dirty_{false};
        bool robots_params_dirty_{false};
        bool world_dirty_{false};
        bool tm_obstacles_dirty_{false};
        bool tm_robots_dirty_{false};

        inline bool isDirty() const
        {
            return obstacles_params_dirty_ || robots_params_dirty_ || world_dirty_
                || tm_obstacles_dirty_ || tm_robots_dirty_;
        }

        // True while populateFromQueue is running; suppresses dirty bumps from programmatic edits.
        bool loading_from_queue_{false};

        // UI Components
        QVBoxLayout *root_layout;
        QString obstacles_task_mode;
        QString robots_task_mode;
        QTabWidget *tabs;
        QTreeWidget *obstacles_tree;
        QTreeWidget *robots_tree;
        QComboBox *obstacles_task_mode_combobox;
        QComboBox *robot_task_mode_combobox;
        QComboBox *world_combobox;

        QPushButton *discard_button;
        QPushButton *queue_button;
        QPushButton *next_button;

        // Next-button latch: disabled between click and the next state/episode update.
        // baseline_id is the current episode_id at click time, so an arriving
        // record with a different id is the signal that the episode changed.
        bool next_pending_{false};
        uint32_t next_pending_baseline_id_{0};
        QTimer *next_pending_timeout_{nullptr};

        // Re-enable the next button and clear pending state.
        void clearNextPending();

        QPushButton *pause_button{nullptr};
        bool holding{false};

        void setHolding(bool held);

        QTableWidget *playlist_table;

    private Q_SLOTS:
        void onQueueClicked();

        void onWorldChanged(const QString &text);

        void onObstaclesTaskModeChanged(const QString &text);
        void onRobotsTaskModeChanged(const QString &text);

        void onNextClicked();
        void onDiscardClicked();

        void onPauseClicked();
    };
} // namespace task_generator_gui
#endif // TASK_GENERATOR_GUI_TASK_GENERATOR_PANEL_HPP
