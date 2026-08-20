#include "task_generator_gui/world_generator_panel.hpp"
#include "rviz_common/display_context.hpp"
#include "task_generator_gui/utils/sketch_edit.hpp"

#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/parameter_value.hpp>

#include <QByteArray>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QMetaObject>
#include <QPixmap>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QString>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <climits>
#include <set>
#include <string>
#include <vector>

namespace task_generator_gui
{

namespace
{
constexpr int kPreviewMaxHeight = 420;

bool integral(const QJsonValue& value)
{
    return value.isDouble() && value.toDouble() == std::floor(value.toDouble());
}

// Task params are typed, so a binding's counts have to arrive as integers, not strings.
rclcpp::ParameterValue toParameterValue(const QJsonValue& value)
{
    if (value.isBool())   return rclcpp::ParameterValue(value.toBool());
    if (integral(value))  return rclcpp::ParameterValue(static_cast<int64_t>(value.toDouble()));
    if (value.isDouble()) return rclcpp::ParameterValue(value.toDouble());

    if (value.isArray())
    {
        const QJsonArray array = value.toArray();
        if (std::all_of(array.begin(), array.end(), [](const QJsonValue& item) { return integral(item); }))
        {
            std::vector<int64_t> whole;
            for (const auto& item : array) whole.push_back(static_cast<int64_t>(item.toDouble()));
            return rclcpp::ParameterValue(whole);
        }
        std::vector<double> reals;
        for (const auto& item : array) reals.push_back(item.toDouble());
        return rclcpp::ParameterValue(reals);
    }

    return rclcpp::ParameterValue(value.toString().toStdString());
}
} // namespace

WorldGeneratorPanel::WorldGeneratorPanel(QWidget* parent)
: Panel(parent)
{
}

WorldGeneratorPanel::~WorldGeneratorPanel() = default;

void WorldGeneratorPanel::onInitialize()
{
    node_ptr = getDisplayContext()->getRosNodeAbstraction().lock();
    node = node_ptr->get_raw_node();
    node->get_logger().set_level(rclcpp::Logger::Level::Warn);
}

void WorldGeneratorPanel::load(const rviz_common::Config& config)
{
    rviz_common::Panel::load(config);

    QString result;
    if (config.mapGetString("WorldGeneratorTarget", &result))
        world_generator_node = result.toStdString();
    else
        world_generator_node = "/world_generator";

    if (config.mapGetString("Target", &result))
        task_generator_node = result.toStdString();
    else
        task_generator_node = "/task_generator_node";

    params_client_ = std::make_shared<rclcpp::AsyncParametersClient>(node, world_generator_node);

    generate_client_ = node->create_client<std_srvs::srv::Trigger>(
        world_generator_node + "/generate_world");

    generate_world_client_ = node->create_client<world_generator_msgs::srv::GenerateWorld>(
        world_generator_node + "/generate");

    queue_episode_client_ = node->create_client<task_generator_msgs::srv::QueueEpisode>(
        task_generator_node + "/config/queue_episode");

    reset_episode_client_ = node->create_client<task_generator_msgs::srv::ResetEpisode>(
        task_generator_node + "/lifecycle/reset_episode");

    resetting_sub_ = node->create_subscription<std_msgs::msg::Bool>(
        task_generator_node + "/state/resetting",
        rclcpp::QoS(1).transient_local(),
        [this](std_msgs::msg::Bool::SharedPtr msg)
        {
            QMetaObject::invokeMethod(this, [this, resetting = msg->data]()
            {
                resetting_ = resetting;
                updateButtons();
            }, Qt::QueuedConnection);
        });

    alphabet_sub_ = node->create_subscription<world_generator_msgs::msg::Alphabet>(
        world_generator_node + "/alphabet",
        rclcpp::QoS(1).transient_local(),
        [this](world_generator_msgs::msg::Alphabet::SharedPtr msg)
        {
            QMetaObject::invokeMethod(this, [this, msg]()
            {
                last_alphabet_  = *msg;
                have_alphabet_  = true;
                applyAlphabetToSketch();
            }, Qt::QueuedConnection);
        });

    setupUi();

    param_tree_engine_ = std::make_unique<DynamicParamTree>(
        node,
        params_client_,
        param_tree_,
        &param_widgets_,
        &param_types_,
        [this](const std::string&) { requestPreview(); },
        nullptr);

    // The preview must not run on the previous generator's widgets, so it waits for the rebuild.
    param_tree_engine_->whenRebuilt([this]()
    {
        applyAlphabetToSketch();
        requestPreview();
    });

    connect(algorithm_combobox_, &QComboBox::currentTextChanged,
            this, &WorldGeneratorPanel::onAlgorithmChanged);

    loadAlgorithms();
}

void WorldGeneratorPanel::loadAlgorithms()
{
    // The node comes up well after rviz and can restart under the panel, bringing a
    // different parameter set with it, so this keeps polling for the whole session.
    auto in_flight = std::make_shared<bool>(false);
    auto was_ready = std::make_shared<bool>(false);

    algorithms_timer_ = node->create_wall_timer(
        std::chrono::seconds(1),
        [this, in_flight, was_ready]()
        {
            if (!params_client_->service_is_ready())
            {
                *was_ready = false;
                return;
            }
            const bool reappeared = !*was_ready;
            *was_ready            = true;
            if (*in_flight) return;
            *in_flight = true;

            params_client_->list_parameters(
                {"algorithm"}, 10,
                [this, in_flight, reappeared](std::shared_future<rcl_interfaces::msg::ListParametersResult> future)
                {
                    rcl_interfaces::msg::ListParametersResult resp;
                    try { resp = future.get(); } catch (...) { *in_flight = false; return; }

                    std::set<std::string> algos;
                    for (const auto& name : resp.names)
                    {
                        const std::string prefix = "algorithm.";
                        if (name.rfind(prefix, 0) != 0) continue;
                        auto rest = name.substr(prefix.size());
                        auto dot  = rest.find('.');
                        if (dot == std::string::npos) continue;
                        algos.insert(rest.substr(0, dot));
                    }

                    *in_flight = false;
                    if (algos.empty()) return;

                    QMetaObject::invokeMethod(this, [this, algos, reappeared]()
                    {
                        applyAlgorithms(algos, reappeared);
                    }, Qt::QueuedConnection);
                });
        });
}

void WorldGeneratorPanel::applyAlgorithms(const std::set<std::string>& algorithms, bool force)
{
    std::set<std::string> shown;
    for (int index = 0; index < algorithm_combobox_->count(); ++index)
        shown.insert(algorithm_combobox_->itemText(index).toStdString());
    if (shown == algorithms && !force) return;

    std::string selected = algorithm_combobox_->currentText().toStdString();
    if (!algorithms.count(selected))
        selected = *algorithms.begin();

    {
        QSignalBlocker blocker(algorithm_combobox_);
        algorithm_combobox_->clear();
        for (const auto& algorithm : algorithms)
            algorithm_combobox_->addItem(QString::fromStdString(algorithm));
        algorithm_combobox_->setCurrentText(QString::fromStdString(selected));
    }

    param_tree_engine_->rebuild("algorithm." + selected);
}

void WorldGeneratorPanel::applyEpisodeBinding(
    task_generator_msgs::srv::QueueEpisode::Request& req, const std::string& json)
{
    // generate_world returns the generator's episode binding as JSON ({} = no overrides).
    auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(json));
    if (!doc.isObject()) return;
    const QJsonObject obj = doc.object();

    if (obj.contains("tm_robots"))
        req.tm_robots = obj.value("tm_robots").toString().toStdString();
    if (obj.contains("tm_obstacles"))
        req.tm_obstacles = obj.value("tm_obstacles").toString().toStdString();

    auto leaves = [](const QJsonValue& v) -> std::vector<rcl_interfaces::msg::Parameter>
    {
        std::vector<rcl_interfaces::msg::Parameter> out;
        if (!v.isObject()) return out;
        const QJsonObject leaf_obj = v.toObject();
        for (auto it = leaf_obj.begin(); it != leaf_obj.end(); ++it)
        {
            rcl_interfaces::msg::Parameter pm;
            pm.name  = it.key().toStdString();
            pm.value = toParameterValue(it.value()).to_value_msg();
            out.push_back(pm);
        }
        return out;
    };

    auto robots    = leaves(obj.value("robots_params"));
    auto obstacles = leaves(obj.value("obstacles_params"));
    req.robots_params.insert(req.robots_params.end(), robots.begin(), robots.end());
    req.obstacles_params.insert(req.obstacles_params.end(), obstacles.begin(), obstacles.end());
}

void WorldGeneratorPanel::applyAlphabetToSketch()
{
    if (preview_canvas_ == nullptr) return;

    auto  it   = param_widgets_.find("sketch");
    auto* edit = it == param_widgets_.end() ? nullptr : qobject_cast<SketchEdit*>(it->second);

    // The previous generator's editor is gone by now, so the canvas must never keep pointing at it.
    preview_canvas_->setEditor(edit);
    if (source_toggle_ != nullptr) source_toggle_->setVisible(edit != nullptr);
    if (clear_button_ != nullptr) clear_button_->setVisible(edit != nullptr);
    if (edit == nullptr) return;

    if (clear_button_ != nullptr)
        connect(clear_button_, &QPushButton::clicked, edit, &SketchEdit::clearSketch, Qt::UniqueConnection);

    connect(edit, &SketchEdit::statusChanged, status_label_, &QLabel::setText, Qt::UniqueConnection);
    connect(edit, &SketchEdit::cursorMoved, preview_canvas_, &SketchCanvas::setCursorCell, Qt::UniqueConnection);
    connect(preview_canvas_, &SketchCanvas::cellClicked, edit, &SketchEdit::setCell, Qt::UniqueConnection);
    // The stroke lands on the canvas now. The render replaces it a few hundred ms later.
    connect(
        edit, &SketchEdit::sketchEdited, preview_canvas_,
        [this, edit]() { preview_canvas_->setInk(edit->ink()); }, Qt::UniqueConnection);

    showSketchSource(source_toggle_ != nullptr && source_toggle_->isChecked());

    if (!have_alphabet_ || edit->hasAlphabet()) return;
    edit->setAlphabet(last_alphabet_);
}

void WorldGeneratorPanel::showSketchSource(bool shown)
{
    auto it = param_widgets_.find("sketch");
    if (it == param_widgets_.end()) return;

    QList<QTreeWidgetItem*> pending;
    for (int index = 0; index < param_tree_->topLevelItemCount(); ++index)
        pending << param_tree_->topLevelItem(index);
    while (!pending.isEmpty())
    {
        QTreeWidgetItem* item = pending.takeFirst();
        if (param_tree_->itemWidget(item, 1) == it->second)
        {
            item->setHidden(!shown);
            return;
        }
        for (int index = 0; index < item->childCount(); ++index)
            pending << item->child(index);
    }
}

void WorldGeneratorPanel::requestPreview()
{
    if (preview_in_flight_)
    {
        preview_dirty_ = true;
        return;
    }
    preview_in_flight_ = true;
    preview_dirty_     = false;

    auto leaves = DynamicParamTree::collectParams(param_widgets_, param_types_);

    // The sketch rides in the config with every other leaf. The request's own sketch field
    // reads empty as "keep the node parameter", which is the one thing a cleared grid is not.
    QJsonObject config_obj;
    for (const auto& leaf : leaves)
    {
        using PT = rcl_interfaces::msg::ParameterType;
        switch (leaf.value.type)
        {
            case PT::PARAMETER_BOOL:
                config_obj.insert(QString::fromStdString(leaf.name), leaf.value.bool_value);
                break;
            case PT::PARAMETER_INTEGER:
                config_obj.insert(QString::fromStdString(leaf.name), static_cast<double>(leaf.value.integer_value));
                break;
            case PT::PARAMETER_DOUBLE:
                config_obj.insert(QString::fromStdString(leaf.name), leaf.value.double_value);
                break;
            case PT::PARAMETER_STRING:
                config_obj.insert(QString::fromStdString(leaf.name), QString::fromStdString(leaf.value.string_value));
                break;
            default:
                break;
        }
    }

    // The canvas paints its own stand-in between renders, so it needs the same widths.
    preview_canvas_->setWeights(
        config_obj.value("light").toDouble(1.5), config_obj.value("heavy").toDouble(3.0), config_obj.value("double").toDouble(6.0));

    auto req = std::make_shared<world_generator_msgs::srv::GenerateWorld::Request>();
    req->preview_only     = true;
    req->generator        = algorithm_combobox_->currentText().toStdString();
    req->config           = QJsonDocument(config_obj).toJson(QJsonDocument::Compact).toStdString();

    generate_world_client_->async_send_request(
        req,
        [this](rclcpp::Client<world_generator_msgs::srv::GenerateWorld>::SharedFuture f)
        {
            auto resp = f.get();
            QMetaObject::invokeMethod(this, [this, resp]()
            {
                preview_in_flight_ = false;

                if (resp)
                {
                    if (!resp->success)
                    {
                        status_label_->setText(QString::fromStdString(resp->message));
                    }
                    else
                    {
                        QPixmap map;
                        map.loadFromData(resp->png.data(), static_cast<uint>(resp->png.size()), "PNG");
                        preview_canvas_->setPixmap(map);
                        preview_canvas_->setFrame(
                            QPointF(resp->map_origin[0], resp->map_origin[1]), resp->map_resolution,
                            QPointF(resp->grid_origin[0], resp->grid_origin[1]), resp->grid_pitch,
                            resp->grid_size[0], resp->grid_size[1]);

                        auto plural = [](uint16_t n, const char* word)
                        {
                            return QString("%1 %2%3").arg(n).arg(word).arg(n == 1 ? "" : "s");
                        };

                        QString diag = QString("%1  %2  %3  %4 x %5 m  %6 ms")
                            .arg(plural(resp->components, "component"))
                            .arg(plural(resp->islands, "island"))
                            .arg(plural(resp->zones, "zone"))
                            .arg(QString::number(resp->extent[0], 'f', 1))
                            .arg(QString::number(resp->extent[1], 'f', 1))
                            .arg(QString::number(resp->compile_ms, 'f', 0));

                        if (!resp->warnings.empty())
                        {
                            const auto& w = resp->warnings.front();
                            diag += QString(" row %1 col %2: %3")
                                .arg(w.row).arg(w.col).arg(QString::fromStdString(w.text));
                        }

                        preview_diag_label_->setText(diag);
                    }
                }
                else
                {
                    status_label_->setText("No response from generate.");
                }

                if (preview_dirty_)
                    requestPreview();
            }, Qt::QueuedConnection);
        });
}

void WorldGeneratorPanel::resizeEvent(QResizeEvent* event)
{
    Panel::resizeEvent(event);
    if (preview_canvas_ != nullptr) preview_canvas_->update();
}

void WorldGeneratorPanel::setupUi()
{
    auto* root = new QVBoxLayout(this);

    // Algorithm row
    {
        auto* row    = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->addWidget(new QLabel("Algorithm"));
        algorithm_combobox_ = new QComboBox();
        layout->addWidget(algorithm_combobox_);
        root->addWidget(row);
    }

    // World name row
    {
        auto* row    = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->addWidget(new QLabel("World Name"));
        world_name_edit_ = new QLineEdit("generated");
        layout->addWidget(world_name_edit_);
        root->addWidget(row);
    }

    // Seed row
    {
        auto* row    = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->addWidget(new QLabel("Seed (-1 = random)"));
        seed_spin_ = new QSpinBox();
        seed_spin_->setRange(-1, INT_MAX);
        seed_spin_->setValue(-1);
        layout->addWidget(seed_spin_);
        root->addWidget(row);
    }

    // Parameter tree
    param_tree_ = new QTreeWidget();
    param_tree_->setColumnCount(2);
    param_tree_->setHeaderLabels({"Parameter", "Value"});
    param_tree_->header()->setSectionResizeMode(QHeaderView::Stretch);
    root->addWidget(param_tree_);

    // Live map preview
    preview_canvas_ = new SketchCanvas();
    preview_canvas_->setMaximumHeight(kPreviewMaxHeight);
    preview_canvas_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    root->addWidget(preview_canvas_);

    preview_diag_label_ = new QLabel();
    preview_diag_label_->setWordWrap(true);
    root->addWidget(preview_diag_label_);

    {
        auto* row    = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        source_toggle_ = new QCheckBox("Show sketch source");
        connect(source_toggle_, &QCheckBox::toggled, this, &WorldGeneratorPanel::showSketchSource);
        layout->addWidget(source_toggle_);
        layout->addStretch();
        clear_button_ = new QPushButton("Clear sketch");
        layout->addWidget(clear_button_);
        root->addWidget(row);
    }

    refresh_button_ = new QPushButton("Refresh preview");
    connect(refresh_button_, &QPushButton::clicked,
            this, &WorldGeneratorPanel::onRefreshClicked);
    root->addWidget(refresh_button_);

    {
        auto* row    = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        save_button_ = new QPushButton("Save");
        save_button_->setToolTip("Write the world to disk");
        connect(save_button_, &QPushButton::clicked,
                this, &WorldGeneratorPanel::onSaveClicked);
        layout->addWidget(save_button_);
        generate_button_ = new QPushButton("+Deploy");
        connect(generate_button_, &QPushButton::clicked,
                this, &WorldGeneratorPanel::onGenerateClicked);
        layout->addWidget(generate_button_);
        root->addWidget(row);
        updateButtons();
    }

    status_label_ = new QLabel();
    status_label_->setWordWrap(true);
    root->addWidget(status_label_);
}

void WorldGeneratorPanel::onAlgorithmChanged(const QString& text)
{
    preview_diag_label_->setText("loading " + text + "...");
    param_tree_engine_->rebuild("algorithm." + text.toStdString());
}

void WorldGeneratorPanel::onRefreshClicked()
{
    requestPreview();
}

void WorldGeneratorPanel::onSaveClicked()
{
    generateWorld(false);
}

void WorldGeneratorPanel::onGenerateClicked()
{
    generateWorld(true);
}

void WorldGeneratorPanel::setBusy(bool busy)
{
    busy_ = busy;
    updateButtons();
}

void WorldGeneratorPanel::updateButtons()
{
    save_button_->setEnabled(!busy_);
    generate_button_->setEnabled(!busy_ && !resetting_);
    generate_button_->setToolTip(resetting_ ? "Task generator is resetting"
                                            : "Save, then stage it into the task generator and reset the episode into it");
}

void WorldGeneratorPanel::generateWorld(bool load)
{
    const std::string algo   = algorithm_combobox_->currentText().toStdString();
    const std::string target = world_name_edit_->text().toStdString();

    if (target.empty())
    {
        status_label_->setText("World name must not be empty.");
        return;
    }

    setBusy(true);
    status_label_->setText("Setting parameters...");

    auto leaves = DynamicParamTree::collectParams(param_widgets_, param_types_);

    std::vector<rclcpp::Parameter> params;
    params.reserve(leaves.size() + 3);

    for (auto& leaf : leaves)
    {
        auto copy  = leaf;
        copy.name  = "algorithm." + algo + "." + leaf.name;
        params.push_back(rclcpp::Parameter::from_parameter_msg(copy));
    }
    params.emplace_back("generator", algo);
    params.emplace_back("world",     target);
    params.emplace_back("seed",      static_cast<int64_t>(seed_spin_->value()));

    // Every exit below lands here, on the Qt thread.
    auto finish = [this](std::string msg)
    {
        QMetaObject::invokeMethod(this, [this, msg = std::move(msg)]()
        {
            if (!msg.empty()) status_label_->setText(QString::fromStdString(msg));
            setBusy(false);
        }, Qt::QueuedConnection);
    };

    params_client_->set_parameters(
        params,
        [this, target, load, finish](
            std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future)
        {
            auto results = future.get();
            for (const auto& r : results)
            {
                if (!r.successful)
                {
                    finish("Set parameter failed: " + r.reason);
                    return;
                }
            }

            QMetaObject::invokeMethod(this, [this]()
            {
                status_label_->setText("Generating...");
            }, Qt::QueuedConnection);

            auto req = std::make_shared<world_generator_msgs::srv::GenerateWorld::Request>();
            req->preview_only = false;
            req->world        = target;
            generate_world_client_->async_send_request(
                req,
                [this, target, load, finish](rclcpp::Client<world_generator_msgs::srv::GenerateWorld>::SharedFuture f)
                {
                    auto resp = f.get();
                    if (!resp || !resp->success)
                    {
                        finish(resp ? resp->message : "No response from generate_world.");
                        return;
                    }

                    if (!load)
                    {
                        finish("World '" + target + "' saved.");
                        return;
                    }

                    const std::string ok_msg  = "World '" + target + "' saved, deploying...";
                    const std::string binding = resp->episode_binding;
                    // Everything below touches the panel, so it belongs on the Qt thread with the
                    // rest of the callbacks, not on the executor thread this arrived on.
                    QMetaObject::invokeMethod(this, [this, target, ok_msg, binding, finish]()
                    {
                    status_label_->setText(QString::fromStdString(ok_msg));

                    auto qreq = std::make_shared<task_generator_msgs::srv::QueueEpisode::Request>();
                    qreq->action       = task_generator_msgs::srv::QueueEpisode::Request::MERGE;
                    qreq->keep_modules = true;
                    qreq->world        = target;

                    // The generator returns its episode binding (e.g. BARN pins robots to scenario mode).
                    applyEpisodeBinding(*qreq, binding);

                    queue_episode_client_->async_send_request(
                        qreq,
                        [this, target, finish](rclcpp::Client<task_generator_msgs::srv::QueueEpisode>::SharedFuture qf)
                        {
                            auto qresp = qf.get();
                            if (!qresp || !qresp->success)
                            {
                                RCLCPP_WARN(node->get_logger(), "staging generated world into task generator failed");
                                finish("");
                                return;
                            }

                            // Apply the staged world now by resetting the episode.
                            auto rreq = std::make_shared<task_generator_msgs::srv::ResetEpisode::Request>();
                            rreq->world = target;
                            reset_episode_client_->async_send_request(
                                rreq,
                                [this, finish](rclcpp::Client<task_generator_msgs::srv::ResetEpisode>::SharedFuture rf)
                                {
                                    auto rresp = rf.get();
                                    if (!rresp || !rresp->success)
                                        RCLCPP_WARN(node->get_logger(), "applying generated world (reset_episode) failed");
                                    finish("");
                                });
                        });
                    }, Qt::QueuedConnection);
                });
        });
}

} // namespace task_generator_gui

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(task_generator_gui::WorldGeneratorPanel, rviz_common::Panel)
