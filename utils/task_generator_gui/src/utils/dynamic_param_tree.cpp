#include "task_generator_gui/utils/dynamic_param_tree.hpp"

#include "task_generator_gui/utils/sketch_edit.hpp"

#include "Qt-MultiSelectComboBox/MultiSelectComboBox.h"

#include <rcl_interfaces/msg/parameter_type.hpp>

#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QTextEdit>
#include <QPlainTextEdit>
#include <QSignalBlocker>
#include <QMetaObject>
#include <QTimer>

#include <algorithm>
#include <atomic>
#include <limits>
#include <string>
#include <vector>

namespace task_generator_gui
{

namespace
{
using PT = rcl_interfaces::msg::ParameterType;

QWidget *make_int_pair_widget(int64_t v0, int64_t v1)
{
    auto *w   = new QWidget();
    auto *lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(4);
    lay->addWidget(new QLabel("Min"));
    auto *sb0 = new QSpinBox();
    sb0->setRange(0, std::numeric_limits<int>::max());
    sb0->setValue(static_cast<int>(v0));
    lay->addWidget(sb0, 1);
    lay->addWidget(new QLabel("Max"));
    auto *sb1 = new QSpinBox();
    sb1->setRange(0, std::numeric_limits<int>::max());
    sb1->setValue(static_cast<int>(v1));
    lay->addWidget(sb1, 1);
    w->setLayout(lay);
    return w;
}

QWidget *make_float_pair_widget(double v0, double v1)
{
    auto *w   = new QWidget();
    auto *lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(4);
    lay->addWidget(new QLabel("Min"));
    auto *sb0 = new QDoubleSpinBox();
    sb0->setRange(-1e9, 1e9);
    sb0->setValue(v0);
    lay->addWidget(sb0, 1);
    lay->addWidget(new QLabel("Max"));
    auto *sb1 = new QDoubleSpinBox();
    sb1->setRange(-1e9, 1e9);
    sb1->setValue(v1);
    lay->addWidget(sb1, 1);
    w->setLayout(lay);
    return w;
}
} // namespace

// ---------------------------------------------------------------------------

DynamicParamTree::DynamicParamTree(
    rclcpp::Node::SharedPtr node,
    std::shared_ptr<rclcpp::AsyncParametersClient> params_client,
    QTreeWidget* tree,
    std::unordered_map<std::string, QWidget*>* widget_map,
    std::unordered_map<std::string, uint8_t>* type_map,
    ChangeCallback on_changed,
    CatalogFetcher catalog_fetcher)
    : node_(std::move(node))
    , params_client_(std::move(params_client))
    , tree_(tree)
    , widget_map_(widget_map)
    , type_map_(type_map)
    , on_changed_(std::move(on_changed))
    , catalog_fetcher_(std::move(catalog_fetcher))
{
}

// ---------------------------------------------------------------------------

void DynamicParamTree::rebuild(const std::string& namespace_prefix)
{
    retries_ = 5;
    retryRebuild(namespace_prefix);
}

void DynamicParamTree::retryRebuild(const std::string& namespace_prefix)
{
    if (!params_client_ || !tree_)
        return;

    const uint64_t this_gen = ++rebuild_gen_;

    auto state                  = std::make_shared<RebuildState>();
    state->generation           = this_gen;
    state->namespace_prefix     = namespace_prefix;

    params_client_->list_parameters(
        {namespace_prefix}, 10,
        [this, state, this_gen]
        (std::shared_future<rcl_interfaces::msg::ListParametersResult> list_f)
        {
            rcl_interfaces::msg::ListParametersResult list_resp;
            try { list_resp = list_f.get(); } catch (...) { return; }

            if (list_resp.names.empty()) return;

            state->param_names = list_resp.names;

            auto describe_done = std::make_shared<std::atomic<bool>>(false);
            auto get_done      = std::make_shared<std::atomic<bool>>(false);

            auto maybe_fanout = [this, state, this_gen, describe_done, get_done]()
            {
                if (!describe_done->load() || !get_done->load())
                    return;

                if (this_gen != rebuild_gen_) return;

                if (state->descriptors.size() == state->param_names.size()
                    && state->values.size() == state->param_names.size())
                {
                    std::vector<std::string> kept_names;
                    std::vector<rcl_interfaces::msg::ParameterDescriptor> kept_descs;
                    std::vector<rclcpp::Parameter> kept_values;
                    kept_names.reserve(state->param_names.size());
                    kept_descs.reserve(state->descriptors.size());
                    kept_values.reserve(state->values.size());
                    for (size_t i = 0; i < state->param_names.size(); ++i)
                    {
                        if (state->descriptors[i].type == rcl_interfaces::msg::ParameterType::PARAMETER_NOT_SET)
                            continue;
                        kept_names.push_back(state->param_names[i]);
                        kept_descs.push_back(state->descriptors[i]);
                        kept_values.push_back(state->values[i]);
                    }
                    state->param_names = std::move(kept_names);
                    state->descriptors = std::move(kept_descs);
                    state->values      = std::move(kept_values);
                }

                QMetaObject::invokeMethod(tree_, [this, state, this_gen]()
                {
                    if (this_gen != rebuild_gen_) return;
                    buildTreeWidgets(state);
                    fetchCatalogs(this_gen);
                }, Qt::QueuedConnection);
            };

            params_client_->describe_parameters(
                state->param_names,
                [state, describe_done, maybe_fanout]
                (std::shared_future<std::vector<rcl_interfaces::msg::ParameterDescriptor>> f)
                {
                    try { state->descriptors = f.get(); } catch (...) {}
                    describe_done->store(true);
                    maybe_fanout();
                });

            params_client_->get_parameters(
                state->param_names,
                [state, get_done, maybe_fanout]
                (std::shared_future<std::vector<rclcpp::Parameter>> f)
                {
                    try { state->values = f.get(); } catch (...) {}
                    get_done->store(true);
                    maybe_fanout();
                });
        });
}

// ---------------------------------------------------------------------------

void DynamicParamTree::buildTreeWidgets(const std::shared_ptr<RebuildState>& state)
{
    tree_->clear();
    widget_map_->clear();
    type_map_->clear();
    catalog_widgets_.clear();

    // A node that does not know a listed name answers describe/get with an empty list.
    if (state->descriptors.size() != state->param_names.size()
        || state->values.size() != state->param_names.size())
    {
        auto *item = new QTreeWidgetItem(tree_);
        item->setText(0, QString::fromStdString(state->namespace_prefix));
        item->setText(1, retries_ > 0 ? "node did not describe these parameters, retrying"
                                      : "node did not describe these parameters");
        if (retries_ > 0)
        {
            --retries_;
            QTimer::singleShot(2000, &lifetime_, [this, prefix = state->namespace_prefix]() { retryRebuild(prefix); });
        }
        return;
    }

    const std::string prefix = state->namespace_prefix + ".";

    for (size_t i = 0; i < state->param_names.size(); ++i)
    {
        const auto &full_name = state->param_names[i];
        std::string leaf      = full_name;
        if (full_name.rfind(prefix, 0) == 0)
            leaf = full_name.substr(prefix.size());

        const auto &desc    = state->descriptors[i];
        const auto &param   = state->values[i];
        const uint8_t ptype = desc.type;

        std::string label;
        std::string constraints;
        bool is_sketch = false;
        bool is_text   = false;
        {
            std::string rest = desc.additional_constraints;
            while (!rest.empty())
            {
                size_t semi  = rest.find(';');
                std::string token = (semi == std::string::npos) ? rest : rest.substr(0, semi);
                rest = (semi == std::string::npos) ? std::string() : rest.substr(semi + 1);
                if (token == "sketch")
                {
                    is_sketch = true;
                    continue;
                }
                if (token == "text")
                {
                    is_text = true;
                    continue;
                }
                size_t colon = token.find(':');
                if (colon == std::string::npos) continue;
                std::string kind  = token.substr(0, colon);
                std::string value = token.substr(colon + 1);
                if (kind == "label")
                    label = value;
                else
                    constraints = token;
            }
        }

        auto *item = new QTreeWidgetItem(tree_);
        item->setText(0, QString::fromStdString(label.empty() ? leaf : label));
        if (!desc.description.empty())
            item->setToolTip(0, QString::fromStdString(desc.description));

        QWidget *w = nullptr;

        if (constraints == "range:int_pair" && ptype == PT::PARAMETER_INTEGER_ARRAY)
        {
            const auto &arr = param.as_integer_array();
            int64_t v0 = arr.size() > 0 ? arr[0] : 0;
            int64_t v1 = arr.size() > 1 ? arr[1] : 0;
            w = make_int_pair_widget(v0, v1);
        }
        else if (constraints == "range:float_pair" && ptype == PT::PARAMETER_DOUBLE_ARRAY)
        {
            const auto &arr = param.as_double_array();
            double v0 = arr.size() > 0 ? arr[0] : 0.0;
            double v1 = arr.size() > 1 ? arr[1] : 0.0;
            w = make_float_pair_widget(v0, v1);
        }
        else if (constraints.rfind("catalog:", 0) == 0 && ptype == PT::PARAMETER_STRING_ARRAY)
        {
            const std::string catalog_name = constraints.substr(8);
            const auto &selected = param.as_string_array();
            auto *cb = new MultiSelectComboBox();
            for (const auto &entry : seedCatalogItems(catalog_name, selected))
            {
                int checked = std::find(selected.begin(), selected.end(), entry) != selected.end() ? 1 : 0;
                cb->addItem(QString::fromStdString(entry), checked);
            }
            cb->stateChanged(1);
            cb->setProperty("catalog", QString::fromStdString(catalog_name));
            catalog_widgets_[catalog_name].push_back(cb);
            w = cb;
        }
        else if (constraints.rfind("catalog:", 0) == 0 && ptype == PT::PARAMETER_STRING)
        {
            const std::string catalog_name = constraints.substr(8);
            const auto current = param.as_string();
            std::vector<std::string> selected;
            if (!current.empty())
                selected.push_back(current);
            auto *cb = new QComboBox();
            for (const auto &entry : seedCatalogItems(catalog_name, selected))
                cb->addItem(QString::fromStdString(entry));
            cb->setCurrentText(QString::fromStdString(current));
            cb->setProperty("catalog", QString::fromStdString(catalog_name));
            catalog_widgets_[catalog_name].push_back(cb);
            w = cb;
        }
        else if (constraints.rfind("enum:", 0) == 0 && ptype == PT::PARAMETER_STRING)
        {
            auto *cb   = new QComboBox();
            std::string rest = constraints.substr(5);
            size_t start = 0;
            while (start <= rest.size())
            {
                size_t comma = rest.find(',', start);
                std::string tok = rest.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) cb->addItem(QString::fromStdString(tok));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            cb->setCurrentText(QString::fromStdString(param.as_string()));
            w = cb;
        }
        else if (is_sketch && ptype == PT::PARAMETER_STRING)
        {
            auto *se = new SketchEdit();
            se->setSketch(QString::fromStdString(param.as_string()));
            se->setMinimumHeight(160);
            w = se;
        }
        else if (is_text && ptype == PT::PARAMETER_STRING)
        {
            auto *te = new QTextEdit();
            te->setPlainText(QString::fromStdString(param.as_string()));
            te->setMinimumHeight(60);
            te->setLineWrapMode(QTextEdit::NoWrap);
            // Connected here so the generic chain below leaves other QTextEdits on commit-only.
            QObject::connect(te, &QTextEdit::textChanged, tree_, [this, leaf]() { on_changed_(leaf); });
            w = te;
        }
        else if (ptype == PT::PARAMETER_INTEGER && !desc.integer_range.empty())
        {
            auto *sb = new QSpinBox();
            sb->setRange(
                static_cast<int>(desc.integer_range[0].from_value),
                static_cast<int>(desc.integer_range[0].to_value));
            sb->setValue(static_cast<int>(param.as_int()));
            w = sb;
        }
        else if (ptype == PT::PARAMETER_DOUBLE && !desc.floating_point_range.empty())
        {
            auto *sb = new QDoubleSpinBox();
            sb->setRange(desc.floating_point_range[0].from_value,
                         desc.floating_point_range[0].to_value);
            if (desc.floating_point_range[0].step > 0.0)
                sb->setSingleStep(desc.floating_point_range[0].step);
            sb->setValue(param.as_double());
            w = sb;
        }
        else if (ptype == PT::PARAMETER_BOOL)
        {
            auto *cb = new QCheckBox();
            cb->setChecked(param.as_bool());
            w = cb;
        }
        else if (ptype == PT::PARAMETER_INTEGER)
        {
            auto *sb = new QSpinBox();
            sb->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
            sb->setValue(static_cast<int>(param.as_int()));
            w = sb;
        }
        else if (ptype == PT::PARAMETER_DOUBLE)
        {
            auto *sb = new QDoubleSpinBox();
            sb->setRange(-1e9, 1e9);
            sb->setValue(param.as_double());
            w = sb;
        }
        else if (ptype == PT::PARAMETER_STRING)
        {
            const auto val = param.as_string();
            if (val.size() > 80 || desc.description.find("prompt") != std::string::npos
                || leaf.find("prompt") != std::string::npos)
            {
                auto *te = new QTextEdit();
                te->setPlainText(QString::fromStdString(val));
                te->setMinimumHeight(50);
                te->setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
                te->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
                te->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
                te->setLineWrapMode(QTextEdit::WidgetWidth);
                w = te;
            }
            else
            {
                auto *le = new QLineEdit();
                le->setText(QString::fromStdString(val));
                w = le;
            }
        }
        else
        {
            auto *le = new QLineEdit();
            le->setText(QString::fromStdString(param.value_to_string()));
            w = le;
        }

        if (w)
        {
            if (auto *sb = qobject_cast<QSpinBox *>(w))
            {
                QObject::connect(sb, QOverload<int>::of(&QSpinBox::valueChanged), tree_,
                    [this, leaf]()
                    {
                        on_changed_(leaf);
                    });
            }
            else if (auto *dsb = qobject_cast<QDoubleSpinBox *>(w))
            {
                QObject::connect(dsb, QOverload<double>::of(&QDoubleSpinBox::valueChanged), tree_,
                    [this, leaf]()
                    {
                        on_changed_(leaf);
                    });
            }
            else if (auto *le = qobject_cast<QLineEdit *>(w))
            {
                QObject::connect(le, &QLineEdit::editingFinished, tree_,
                    [this, leaf]()
                    {
                        on_changed_(leaf);
                    });
            }
            else if (auto *cb = qobject_cast<QCheckBox *>(w))
            {
                QObject::connect(cb, &QCheckBox::toggled, tree_,
                    [this, leaf](bool)
                    {
                        on_changed_(leaf);
                    });
            }
            else if (auto *combo = qobject_cast<QComboBox *>(w))
            {
                QObject::connect(combo, &QComboBox::currentTextChanged, tree_,
                    [this, leaf](const QString &)
                    {
                        on_changed_(leaf);
                    });
            }
            else if (auto *se = qobject_cast<SketchEdit *>(w))
            {
                QObject::connect(se, &SketchEdit::sketchEdited, tree_,
                    [this, leaf]()
                    {
                        on_changed_(leaf);
                    });
            }
            else
            {
                for (auto *child_sb : w->findChildren<QSpinBox *>())
                {
                    QObject::connect(child_sb, QOverload<int>::of(&QSpinBox::valueChanged), tree_,
                        [this, leaf]()
                        {
                            on_changed_(leaf);
                        });
                }
                for (auto *child_dsb : w->findChildren<QDoubleSpinBox *>())
                {
                    QObject::connect(child_dsb, QOverload<double>::of(&QDoubleSpinBox::valueChanged), tree_,
                        [this, leaf]()
                        {
                            on_changed_(leaf);
                        });
                }
            }

            tree_->setItemWidget(item, 1, w);
            (*widget_map_)[leaf] = w;
            (*type_map_)[leaf]   = ptype;
        }
    }

    if (on_ready_)
        on_ready_();
}

// ---------------------------------------------------------------------------

std::vector<std::string> DynamicParamTree::seedCatalogItems(
    const std::string& catalog, const std::vector<std::string>& current) const
{
    std::vector<std::string> items;
    auto memo = catalog_memo_.find(catalog);
    if (memo != catalog_memo_.end())
        items = memo->second;
    for (const auto &c : current)
        if (std::find(items.begin(), items.end(), c) == items.end())
            items.push_back(c);
    return items;
}

void DynamicParamTree::fetchCatalogs(uint64_t gen)
{
    if (!catalog_fetcher_)
        return;

    for (const auto &entry : catalog_widgets_)
    {
        const std::string cat = entry.first;
        catalog_fetcher_(cat, [this, gen, cat](std::vector<std::string> ids)
        {
            QMetaObject::invokeMethod(tree_, [this, gen, cat, ids = std::move(ids)]()
            {
                catalog_memo_[cat] = ids;
                if (gen != rebuild_gen_) return;
                fillCatalogWidgets(cat, ids);
            }, Qt::QueuedConnection);
        });
    }
}

void DynamicParamTree::fillCatalogWidgets(const std::string& catalog, const std::vector<std::string>& ids)
{
    auto it = catalog_widgets_.find(catalog);
    if (it == catalog_widgets_.end())
        return;

    for (QWidget *w : it->second)
    {
        if (auto *msc = qobject_cast<MultiSelectComboBox *>(w))
        {
            QSignalBlocker blk(msc);
            const QStringList checked = msc->currentText();
            msc->clear();
            for (const auto &entry : ids)
            {
                const QString qs = QString::fromStdString(entry);
                msc->addItem(qs, checked.contains(qs) ? 1 : 0);
            }
            for (const QString &c : checked)
                if (std::find(ids.begin(), ids.end(), c.toStdString()) == ids.end())
                    msc->addItem(c, 1);
            msc->stateChanged(1);
        }
        else if (auto *combo = qobject_cast<QComboBox *>(w))
        {
            QSignalBlocker blk(combo);
            const QString cur = combo->currentText();
            combo->clear();
            for (const auto &entry : ids)
                combo->addItem(QString::fromStdString(entry));
            if (!cur.isEmpty() && combo->findText(cur) < 0)
                combo->addItem(cur);
            combo->setCurrentText(cur);
        }
    }
}

// ---------------------------------------------------------------------------

void DynamicParamTree::setWidgetValueFromParam(QWidget *w, const rcl_interfaces::msg::Parameter &p)
{
    const uint8_t ptype = p.value.type;

    if (ptype == PT::PARAMETER_INTEGER)
    {
        if (auto *sb = qobject_cast<QSpinBox *>(w))
        {
            QSignalBlocker blk(sb);
            sb->setValue(static_cast<int>(p.value.integer_value));
        }
    }
    else if (ptype == PT::PARAMETER_DOUBLE)
    {
        if (auto *dsb = qobject_cast<QDoubleSpinBox *>(w))
        {
            QSignalBlocker blk(dsb);
            dsb->setValue(p.value.double_value);
        }
    }
    else if (ptype == PT::PARAMETER_BOOL)
    {
        if (auto *cb = qobject_cast<QCheckBox *>(w))
        {
            QSignalBlocker blk(cb);
            cb->setChecked(p.value.bool_value);
        }
    }
    else if (ptype == PT::PARAMETER_STRING)
    {
        if (auto *combo = qobject_cast<QComboBox *>(w))
        {
            QSignalBlocker blk(combo);
            const QString text = QString::fromStdString(p.value.string_value);
            if (!text.isEmpty() && combo->findText(text) < 0 && combo->property("catalog").isValid())
                combo->addItem(text);
            combo->setCurrentText(text);
        }
        else if (auto *te = qobject_cast<QTextEdit *>(w))
        {
            QSignalBlocker blk(te);
            te->setPlainText(QString::fromStdString(p.value.string_value));
        }
        else if (auto *se = qobject_cast<SketchEdit *>(w))
        {
            // Before the QPlainTextEdit branch: a SketchEdit is one, and setPlainText kills it.
            QSignalBlocker blk(se);
            se->setSketch(QString::fromStdString(p.value.string_value));
        }
        else if (auto *pte = qobject_cast<QPlainTextEdit *>(w))
        {
            QSignalBlocker blk(pte);
            pte->setPlainText(QString::fromStdString(p.value.string_value));
        }
        else if (auto *le = qobject_cast<QLineEdit *>(w))
        {
            QSignalBlocker blk(le);
            le->setText(QString::fromStdString(p.value.string_value));
        }
    }
    else if (ptype == PT::PARAMETER_INTEGER_ARRAY)
    {
        auto children = w->findChildren<QSpinBox *>();
        const auto &arr = p.value.integer_array_value;
        for (size_t idx = 0; idx < static_cast<size_t>(children.size()) && idx < arr.size(); ++idx)
        {
            QSignalBlocker blk(children[static_cast<int>(idx)]);
            children[static_cast<int>(idx)]->setValue(static_cast<int>(arr[idx]));
        }
    }
    else if (ptype == PT::PARAMETER_DOUBLE_ARRAY)
    {
        auto children = w->findChildren<QDoubleSpinBox *>();
        const auto &arr = p.value.double_array_value;
        for (size_t idx = 0; idx < static_cast<size_t>(children.size()) && idx < arr.size(); ++idx)
        {
            QSignalBlocker blk(children[static_cast<int>(idx)]);
            children[static_cast<int>(idx)]->setValue(arr[idx]);
        }
    }
}

// ---------------------------------------------------------------------------

std::vector<rcl_interfaces::msg::Parameter> DynamicParamTree::collectParams(
    const std::unordered_map<std::string, QWidget*>& widget_map,
    const std::unordered_map<std::string, uint8_t>& type_map)
{
    std::vector<rcl_interfaces::msg::Parameter> result;

    for (const auto &[leaf, w] : widget_map)
    {
        auto it = type_map.find(leaf);
        if (it == type_map.end())
            continue;
        const uint8_t ptype = it->second;

        rcl_interfaces::msg::Parameter p;
        p.name = leaf;

        if (ptype == PT::PARAMETER_INTEGER_ARRAY)
        {
            auto children = w->findChildren<QSpinBox *>();
            if (children.size() >= 2)
            {
                p.value.type = PT::PARAMETER_INTEGER_ARRAY;
                p.value.integer_array_value = {
                    static_cast<int64_t>(children[0]->value()),
                    static_cast<int64_t>(children[1]->value())};
            }
            else
                continue;
        }
        else if (ptype == PT::PARAMETER_DOUBLE_ARRAY)
        {
            auto children = w->findChildren<QDoubleSpinBox *>();
            if (children.size() >= 2)
            {
                p.value.type = PT::PARAMETER_DOUBLE_ARRAY;
                p.value.double_array_value = {children[0]->value(), children[1]->value()};
            }
            else
                continue;
        }
        else if (ptype == PT::PARAMETER_STRING_ARRAY)
        {
            auto *cb = qobject_cast<MultiSelectComboBox *>(w);
            if (!cb) continue;
            p.value.type = PT::PARAMETER_STRING_ARRAY;
            for (const QString &s : cb->currentText())
                p.value.string_array_value.push_back(s.toStdString());
        }
        else if (ptype == PT::PARAMETER_STRING)
        {
            p.value.type = PT::PARAMETER_STRING;
            if (auto *cb = qobject_cast<QComboBox *>(w))
                p.value.string_value = cb->currentText().toStdString();
            else if (auto *te = qobject_cast<QTextEdit *>(w))
                p.value.string_value = te->toPlainText().toStdString();
            else if (auto *pte = qobject_cast<QPlainTextEdit *>(w))
                p.value.string_value = pte->toPlainText().toStdString();
            else if (auto *le = qobject_cast<QLineEdit *>(w))
                p.value.string_value = le->text().toStdString();
            else
                continue;
        }
        else if (ptype == PT::PARAMETER_BOOL)
        {
            auto *cb = qobject_cast<QCheckBox *>(w);
            if (!cb) continue;
            p.value.type = PT::PARAMETER_BOOL;
            p.value.bool_value = cb->isChecked();
        }
        else if (ptype == PT::PARAMETER_INTEGER)
        {
            auto *sb = qobject_cast<QSpinBox *>(w);
            if (!sb) continue;
            p.value.type = PT::PARAMETER_INTEGER;
            p.value.integer_value = sb->value();
        }
        else if (ptype == PT::PARAMETER_DOUBLE)
        {
            auto *sb = qobject_cast<QDoubleSpinBox *>(w);
            if (!sb) continue;
            p.value.type = PT::PARAMETER_DOUBLE;
            p.value.double_value = sb->value();
        }
        else
        {
            if (auto *le = qobject_cast<QLineEdit *>(w))
            {
                p.value.type = PT::PARAMETER_STRING;
                p.value.string_value = le->text().toStdString();
            }
            else
                continue;
        }

        result.push_back(std::move(p));
    }

    return result;
}

} // namespace task_generator_gui
