#ifndef TASK_GENERATOR_GUI_UTILS_DYNAMIC_PARAM_TREE_HPP
#define TASK_GENERATOR_GUI_UTILS_DYNAMIC_PARAM_TREE_HPP

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/parameter_client.hpp"

#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>

#include <QTreeWidget>
#include <QWidget>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace task_generator_gui
{

struct RebuildState
{
    uint64_t generation{0};
    std::string namespace_prefix;

    std::vector<std::string>                              param_names;
    std::vector<rcl_interfaces::msg::ParameterDescriptor> descriptors;
    std::vector<rclcpp::Parameter>                        values;
};

class DynamicParamTree
{
public:
    using CatalogFetcher = std::function<void(const std::string& catalog,
                            std::function<void(std::vector<std::string>)> cb)>;
    using ChangeCallback = std::function<void(const std::string& leaf)>;
    using ReadyCallback  = std::function<void()>;

    DynamicParamTree(rclcpp::Node::SharedPtr node,
                     std::shared_ptr<rclcpp::AsyncParametersClient> params_client,
                     QTreeWidget* tree,
                     std::unordered_map<std::string, QWidget*>* widget_map,
                     std::unordered_map<std::string, uint8_t>* type_map,
                     ChangeCallback on_changed,
                     CatalogFetcher catalog_fetcher = nullptr);

    void rebuild(const std::string& namespace_prefix);
    // Same, minus the retry budget reset, so a retry cannot renew its own budget.
    void retryRebuild(const std::string& namespace_prefix);
    // Fires once the widgets match the requested namespace, never on a failed rebuild.
    void whenRebuilt(ReadyCallback callback) { on_ready_ = std::move(callback); }

    static std::vector<rcl_interfaces::msg::Parameter> collectParams(
        const std::unordered_map<std::string, QWidget*>& widget_map,
        const std::unordered_map<std::string, uint8_t>& type_map);

    static void setWidgetValueFromParam(QWidget* w, const rcl_interfaces::msg::Parameter& p);

private:
    void buildTreeWidgets(const std::shared_ptr<RebuildState>& state);
    std::vector<std::string> seedCatalogItems(const std::string& catalog,
                                              const std::vector<std::string>& current) const;
    void fetchCatalogs(uint64_t gen);
    void fillCatalogWidgets(const std::string& catalog, const std::vector<std::string>& ids);

    rclcpp::Node::SharedPtr                           node_;
    std::shared_ptr<rclcpp::AsyncParametersClient>    params_client_;
    QTreeWidget*                                      tree_;
    std::unordered_map<std::string, QWidget*>*        widget_map_;
    std::unordered_map<std::string, uint8_t>*         type_map_;
    ChangeCallback                                    on_changed_;
    ReadyCallback                                     on_ready_;
    CatalogFetcher                                    catalog_fetcher_;

    uint64_t rebuild_gen_{0};
    int retries_{0};
    // Timer context, so a pending retry dies with us. tree_ cannot serve: it outlives us.
    QObject lifetime_;
    std::unordered_map<std::string, std::vector<std::string>> catalog_memo_;
    std::unordered_map<std::string, std::vector<QWidget*>>    catalog_widgets_;
};

} // namespace task_generator_gui

#endif // TASK_GENERATOR_GUI_UTILS_DYNAMIC_PARAM_TREE_HPP
