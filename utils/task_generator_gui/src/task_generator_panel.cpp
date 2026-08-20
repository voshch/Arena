#include "task_generator_gui/task_generator_panel.hpp"
#include "rviz_common/display_context.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "rcl_interfaces/srv/set_parameters.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
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
            task_generator_node = result.toStdString();
        else
            task_generator_node = "/task_generator_node";

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

        setupTabs(this->root_layout);

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
