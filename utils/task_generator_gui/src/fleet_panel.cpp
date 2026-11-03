#include "task_generator_gui/fleet_panel.hpp"
#include "rviz_common/display_context.hpp"

#include <chrono>
#include <memory>
#include <set>

namespace task_generator_gui
{

static void clearLayout(QLayout* layout)
{
    QLayoutItem* item;
    while ((item = layout->takeAt(0)) != nullptr)
    {
        if (item->widget())
            item->widget()->deleteLater();
        delete item;
    }
}

FleetPanel::FleetPanel(QWidget* parent)
: Panel(parent)
{
}

FleetPanel::~FleetPanel() = default;

void FleetPanel::onInitialize()
{
    node_ptr = getDisplayContext()->getRosNodeAbstraction().lock();
    node = node_ptr->get_raw_node();
    node->get_logger().set_level(rclcpp::Logger::Level::Warn);
}

void FleetPanel::load(const rviz_common::Config& config)
{
    rviz_common::Panel::load(config);

    QString result;
    if (config.mapGetString("Target", &result))
        arena_env_node = result.toStdString();
    else
        arena_env_node = "/arena_env_node";

    query_robots_client = node->create_client<arena_runtime_msgs::srv::QueryRobots>(
        arena_env_node + "/query/robots");
    spawn_robot_client = node->create_client<arena_runtime_msgs::srv::SpawnRobot>(
        arena_env_node + "/runtime/spawn_robot");
    despawn_robot_client = node->create_client<arena_runtime_msgs::srv::DespawnRobot>(
        arena_env_node + "/runtime/despawn_robot");

    {
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.transient_local();
        robot_fleet_sub = node->create_subscription<arena_runtime_msgs::msg::RobotFleet>(
            arena_env_node + "/state/robots",
            qos,
            [this](const arena_runtime_msgs::msg::RobotFleet::SharedPtr msg)
            {
                QMetaObject::invokeMethod(this, [this, msg]()
                {
                    last_fleet = msg;
                    rebuildFleet();
                }, Qt::QueuedConnection);
            });
    }

    {
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.transient_local();
        robot_queue_sub = node->create_subscription<arena_runtime_msgs::msg::RobotQueue>(
            arena_env_node + "/state/robots/pending",
            qos,
            [this](const arena_runtime_msgs::msg::RobotQueue::SharedPtr msg)
            {
                QMetaObject::invokeMethod(this, [this, msg]()
                {
                    last_pending = msg;
                    rebuildFleet();
                    rebuildQueue();
                }, Qt::QueuedConnection);
            });
    }

    setupUi();

    whenReady(
        [c = query_robots_client]() { return c->service_is_ready(); },
        [this]()
        {
            query_robots_client->async_send_request(
                std::make_shared<arena_runtime_msgs::srv::QueryRobots::Request>(),
                [this](rclcpp::Client<arena_runtime_msgs::srv::QueryRobots>::SharedFuture f)
                {
                    auto resp = f.get();
                    if (!resp) return;
                    QMetaObject::invokeMethod(this, [this, ids = resp->ids]()
                    {
                        robot_models = ids;
                        if (selected_robot_model.empty() && !robot_models.empty())
                            selected_robot_model = robot_models[0];
                        QSignalBlocker blocker(robot_combobox);
                        robot_combobox->clear();
                        for (const auto& r : robot_models)
                            robot_combobox->addItem(QString::fromStdString(r));
                        robot_combobox->setCurrentText(QString::fromStdString(selected_robot_model));
                        robot_combobox->setEnabled(true);
                    }, Qt::QueuedConnection);
                });
        });
}

void FleetPanel::whenReady(std::function<bool()> ready_check,
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

void FleetPanel::setupUi()
{
    auto* root = new QVBoxLayout(this);

    auto* row = new QHBoxLayout();
    robot_combobox = new QComboBox();
    robot_combobox->addItem("Loading...");
    robot_combobox->setEnabled(false);
    connect(robot_combobox, &QComboBox::currentTextChanged, this, &FleetPanel::onRobotChanged);
    row->addWidget(robot_combobox, 1);
    spawn_robot_button = new QPushButton("Spawn");
    connect(spawn_robot_button, &QPushButton::clicked, this, &FleetPanel::spawnRobotButtonActivated);
    row->addWidget(spawn_robot_button);
    root->addLayout(row);

    fleet_group  = new QGroupBox("Fleet");
    fleet_layout = new QVBoxLayout();
    fleet_group->setLayout(fleet_layout);
    root->addWidget(fleet_group, 1);

    queue_group  = new QGroupBox("Queued");
    queue_layout = new QVBoxLayout();
    queue_group->setLayout(queue_layout);
    queue_group->setVisible(false);
    root->addWidget(queue_group);
}

void FleetPanel::sendDespawn(const std::string& name)
{
    auto req = std::make_shared<arena_runtime_msgs::srv::DespawnRobot::Request>();
    req->name = name;
    despawn_robot_client->async_send_request(
        req,
        [this, name](rclcpp::Client<arena_runtime_msgs::srv::DespawnRobot>::SharedFuture f)
        {
            auto resp = f.get();
            if (resp && !resp->success)
                RCLCPP_WARN(node->get_logger(),
                            "despawn_robot failed (%s): %s",
                            name.c_str(), resp->error_msg.c_str());
        });
}

void FleetPanel::rebuildFleet()
{
    if (!fleet_layout || !last_fleet)
        return;

    std::set<std::string> pending_despawn;
    if (last_pending)
        for (const auto& robot : last_pending->despawn)
            pending_despawn.insert(robot.name);

    clearLayout(fleet_layout);

    for (const auto& state : last_fleet->robots)
    {
        const auto& robot = state.descriptor;
        auto row_widget = new QWidget();
        auto row_layout = new QHBoxLayout();
        row_layout->setContentsMargins(0, 0, 0, 0);
        auto label = new QLabel(
            QString::fromStdString(robot.name) +
            " (" + QString::fromStdString(robot.model) + ")");
        auto btn = new QPushButton();
        std::string name = robot.name;
        if (pending_despawn.count(name))
        {
            btn->setText("pending...");
            btn->setEnabled(false);
        }
        else
        {
            btn->setText("Despawn");
            connect(btn, &QPushButton::clicked, this, [this, name, btn]()
            {
                btn->setEnabled(false);
                sendDespawn(name);
            });
        }
        row_layout->addWidget(label);
        row_layout->addStretch();
        row_layout->addWidget(btn);
        row_widget->setLayout(row_layout);
        fleet_layout->addWidget(row_widget);
    }
    fleet_layout->addStretch();
}

void FleetPanel::rebuildQueue()
{
    if (!queue_layout)
        return;

    clearLayout(queue_layout);

    auto add_row = [this](const QString& prefix, const std::string& name, const std::string& model)
    {
        auto row_widget = new QWidget();
        auto row_layout = new QHBoxLayout();
        row_layout->setContentsMargins(0, 0, 0, 0);
        auto label = new QLabel(
            prefix + " " + QString::fromStdString(name) +
            " (" + QString::fromStdString(model) + ")");
        auto btn = new QPushButton("Cancel");
        std::string n = name;
        connect(btn, &QPushButton::clicked, this, [this, n, btn]()
        {
            btn->setEnabled(false);
            sendDespawn(n);
        });
        row_layout->addWidget(label);
        row_layout->addStretch();
        row_layout->addWidget(btn);
        row_widget->setLayout(row_layout);
        queue_layout->addWidget(row_widget);
    };

    bool empty = !last_pending || (last_pending->spawn.empty() && last_pending->despawn.empty());
    if (last_pending)
    {
        for (const auto& robot : last_pending->spawn)
            add_row("+", robot.name, robot.model);
        for (const auto& robot : last_pending->despawn)
            add_row("-", robot.name, robot.model);
    }
    queue_group->setVisible(!empty);
    queue_layout->addStretch();
}

void FleetPanel::onRobotChanged(const QString& text)
{
    selected_robot_model = text.toStdString();
}

void FleetPanel::spawnRobotButtonActivated()
{
    if (selected_robot_model.empty())
        return;

    auto req = std::make_shared<arena_runtime_msgs::srv::SpawnRobot::Request>();
    req->model    = selected_robot_model;
    req->name     = "";
    req->use_pose = false;

    spawn_robot_client->async_send_request(
        req,
        [this](rclcpp::Client<arena_runtime_msgs::srv::SpawnRobot>::SharedFuture f)
        {
            auto resp = f.get();
            if (resp && !resp->success)
                RCLCPP_WARN(node->get_logger(),
                            "spawn_robot failed: %s", resp->error_msg.c_str());
        });
}

} // namespace task_generator_gui

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(task_generator_gui::FleetPanel, rviz_common::Panel)
