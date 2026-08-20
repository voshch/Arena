#ifndef TASK_GENERATOR_GUI_WORLD_GENERATOR_PANEL_HPP
#define TASK_GENERATOR_GUI_WORLD_GENERATOR_PANEL_HPP

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/parameter_client.hpp"

#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "task_generator_msgs/srv/queue_episode.hpp"
#include "task_generator_msgs/srv/reset_episode.hpp"
#include "world_generator_msgs/srv/generate_world.hpp"
#include "world_generator_msgs/msg/alphabet.hpp"

#include <rcl_interfaces/msg/parameter.hpp>

#include "task_generator_gui/utils/dynamic_param_tree.hpp"
#include "task_generator_gui/utils/sketch_canvas.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <climits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

class QResizeEvent;

namespace task_generator_gui
{

class WorldGeneratorPanel : public rviz_common::Panel
{
    Q_OBJECT

public:
    explicit WorldGeneratorPanel(QWidget* parent = nullptr);
    ~WorldGeneratorPanel() override;

    void onInitialize() override;
    void load(const rviz_common::Config& config) override;

protected:
    void resizeEvent(QResizeEvent* event) override;

    std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr;
    rclcpp::Node::SharedPtr node;

    std::string world_generator_node;
    std::string task_generator_node;

    std::shared_ptr<rclcpp::AsyncParametersClient> params_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr generate_client_;
    rclcpp::Client<world_generator_msgs::srv::GenerateWorld>::SharedPtr generate_world_client_;
    rclcpp::Client<task_generator_msgs::srv::QueueEpisode>::SharedPtr queue_episode_client_;
    rclcpp::Client<task_generator_msgs::srv::ResetEpisode>::SharedPtr reset_episode_client_;
    rclcpp::Subscription<world_generator_msgs::msg::Alphabet>::SharedPtr alphabet_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr resetting_sub_;
    rclcpp::TimerBase::SharedPtr algorithms_timer_;

    QComboBox*   algorithm_combobox_;
    QLineEdit*   world_name_edit_;
    QSpinBox*    seed_spin_;
    QTreeWidget* param_tree_;
    bool         busy_{false};
    bool         resetting_{false};
    QPushButton* save_button_;
    QPushButton* generate_button_;
    QPushButton* refresh_button_;
    QCheckBox*   source_toggle_{nullptr};
    QPushButton* clear_button_{nullptr};
    QLabel*      status_label_;
    SketchCanvas* preview_canvas_{nullptr};
    QLabel*      preview_diag_label_;

    std::unordered_map<std::string, QWidget*> param_widgets_;
    std::unordered_map<std::string, uint8_t>  param_types_;

    std::unique_ptr<DynamicParamTree> param_tree_engine_;

    bool preview_in_flight_{false};
    bool preview_dirty_{false};
    bool have_alphabet_{false};
    world_generator_msgs::msg::Alphabet last_alphabet_;

    void setupUi();
    void loadAlgorithms();
    // Apply the generator's episode binding (JSON from generate_world) onto a queue request.
    void applyEpisodeBinding(task_generator_msgs::srv::QueueEpisode::Request& req, const std::string& json);
    // Send a preview GenerateWorld request built from the current widgets. Coalesces while one is in flight.
    void requestPreview();
    // Repopulate the algorithm list, rebuilding the tree when the set or the node changed.
    void applyAlgorithms(const std::set<std::string>& algorithms, bool force);
    // Hand the latched alphabet to the sketch widget if it exists yet.
    void applyAlphabetToSketch();
    // The canvas is the drawing surface. The source row is for pasting and hand-written legends.
    void showSketchSource(bool shown);
    // Save the world from the widgets. With load, also stage it and reset the episode into it.
    void generateWorld(bool load);
    void setBusy(bool busy);
    // Save waits on our own round trip only. Deploy also waits for the task generator to be idle.
    void updateButtons();

private Q_SLOTS:
    void onAlgorithmChanged(const QString& text);
    void onSaveClicked();
    void onGenerateClicked();
    void onRefreshClicked();
};

} // namespace task_generator_gui

#endif // TASK_GENERATOR_GUI_WORLD_GENERATOR_PANEL_HPP
