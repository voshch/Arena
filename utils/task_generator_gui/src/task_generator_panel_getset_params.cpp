#include "task_generator_gui/task_generator_panel.hpp"
#include "rviz_common/display_context.hpp"

#include "rcl_interfaces/srv/set_parameters.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <cctype>
#include <sstream>

namespace task_generator_gui
{
    void TaskGeneratorPanel::getWorlds()
    {
        query_worlds_client->async_send_request(
            std::make_shared<arena_runtime_msgs::srv::QueryWorlds::Request>(),
            [this](rclcpp::Client<arena_runtime_msgs::srv::QueryWorlds>::SharedFuture f)
            {
                auto resp = f.get();
                if (!resp) return;
                QMetaObject::invokeMethod(this, [this, ids = resp->ids]()
                {
                    worlds = ids;
                    if (staged_world.empty() && !worlds.empty())
                        staged_world = worlds[0];
                    if (world_combobox)
                    {
                        QSignalBlocker blocker(world_combobox);
                        world_combobox->clear();
                        for (const auto &w : worlds)
                            world_combobox->addItem(QString::fromStdString(w));
                        world_combobox->setCurrentText(QString::fromStdString(staged_world));
                    }
                }, Qt::QueuedConnection);
            });
    }

    void TaskGeneratorPanel::getTMObstaclesParams()
    {
        getScenarios(staged_world);
    }

    void TaskGeneratorPanel::getScenarios(const std::string &world_name)
    {
        (void)world_name;
        // Scenarios are fetched lazily via fetchCatalog("scenarios") in DynamicParamTree::rebuild.
    }

    std::string TaskGeneratorPanel::modeName(const QString &mode)
    {
        auto name = mode.toStdString();
        for (char &c : name)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return name;
    }

    std::string TaskGeneratorPanel::namespaceOf(const std::string &axis, const std::string &mode) const
    {
        const auto it = mode_namespaces_.find(axis + "/" + mode);
        return it == mode_namespaces_.end() ? std::string() : it->second;
    }

    void TaskGeneratorPanel::rebuildTree(DynamicParamTree &tree, const std::string &axis, const std::string &mode)
    {
        const auto prefix = namespaceOf(axis, mode);
        if (!prefix.empty())
            tree.rebuild(prefix);
    }

    std::vector<std::string> TaskGeneratorPanel::modesOf(const std::vector<diagnostic_msgs::msg::KeyValue> &modes,
                                                         const std::string &axis)
    {
        std::vector<std::string> names;
        for (const auto &pair : modes)
        {
            if (pair.key != axis)
                continue;
            std::stringstream stream(pair.value);
            std::string name;
            while (std::getline(stream, name, ','))
                if (!name.empty())
                    names.push_back(name);
        }
        return names;
    }

    void TaskGeneratorPanel::addModeToRequest(arena_runtime_msgs::srv::QueueEpisode::Request &req,
                                              const std::string &axis, const QString &mode,
                                              const std::unordered_map<std::string, QWidget *> &widgets,
                                              const std::unordered_map<std::string, uint8_t> &types)
    {
        if (mode.isEmpty())
            return;

        const auto name   = modeName(mode);
        const auto prefix = namespaceOf(axis, name);
        diagnostic_msgs::msg::KeyValue selection;
        selection.key   = axis;
        selection.value = name;
        req.modes.push_back(selection);

        for (auto param : DynamicParamTree::collectParams(widgets, types))
        {
            param.name = prefix + "." + param.name;
            req.params.push_back(std::move(param));
        }
    }

    void TaskGeneratorPanel::getParams()
    {
        getWorlds();
        updateTabs();
        getTMObstaclesParams();
    }

    arena_runtime_msgs::srv::QueueEpisode::Request::SharedPtr
    TaskGeneratorPanel::buildQueueEpisodeRequest()
    {
        auto req = std::make_shared<arena_runtime_msgs::srv::QueueEpisode::Request>();
        req->action = arena_runtime_msgs::srv::QueueEpisode::Request::MERGE;
        req->world  = staged_world;

        addModeToRequest(*req, kObstaclesAxis, obstacles_task_mode, param_widgets_obstacles_, param_types_obstacles_);
        addModeToRequest(*req, kRobotsAxis, robots_task_mode, param_widgets_robots_, param_types_robots_);

        return req;
    }

    void TaskGeneratorPanel::pushQueueEpisode(std::function<void(bool)> on_done)
    {
        auto req = buildQueueEpisodeRequest();
        queue_episode_client->async_send_request(
            req,
            [this, on_done = std::move(on_done)]
            (rclcpp::Client<arena_runtime_msgs::srv::QueueEpisode>::SharedFuture f)
            {
                auto resp = f.get();
                const bool ok = resp && resp->success;
                if (resp && !ok)
                    RCLCPP_WARN(node->get_logger(),
                                "queue_episode rejected: %s", resp->error_msg.c_str());
                on_done(ok);
            });
    }

    std::vector<std::string> TaskGeneratorPanel::convert(const QStringList &qList)
    {
        std::vector<std::string> result;
        result.reserve(qList.size());
        for (const QString &item : qList)
            result.push_back(item.toStdString());
        return result;
    }

} // namespace task_generator_gui
