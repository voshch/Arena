#ifndef TASK_GENERATOR_GUI_UTILS_SKETCH_CANVAS_HPP
#define TASK_GENERATOR_GUI_UTILS_SKETCH_CANVAS_HPP

#include "task_generator_gui/utils/sketch_edit.hpp"

#include <QPixmap>
#include <QPointer>
#include <QPointF>
#include <QPainterPath>
#include <QRectF>
#include <QWidget>

#include <array>

namespace task_generator_gui
{

// The generated map, drawn on directly. Keys are forwarded to the editor that owns the sketch,
// so the document stays the only copy of the drawing.
class SketchCanvas : public QWidget
{
    Q_OBJECT

public:
    explicit SketchCanvas(QWidget* parent = nullptr);

    void setPixmap(const QPixmap& pixmap);
    // World frame of the map and of the cell grid. Pitch 0 means there is no grid.
    void setFrame(QPointF map_origin, double map_resolution, QPointF grid_origin, double pitch, int rows, int cols);
    void setCursorCell(int row, int col);
    void setEditor(QWidget* editor) { editor_ = editor; }
    bool hasGrid() const { return pitch_ > 0.0 && rows_ > 0 && cols_ > 0; }
    // The drawing as the editor holds it, painted here so a stroke shows without a round trip.
    // The rendered map replaces it when it arrives. Until then this is what the panel shows.
    void setInk(Ink ink);
    // Corridor width in metres per arm weight, indexed by weight. Full spans a whole cell.
    void setWeights(double light, double heavy, double twin);

Q_SIGNALS:
    void cellClicked(int row, int col);

protected:
    void paintEvent(QPaintEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    // What the widget shows: the whole grid when there is one, else the map. Built with top()
    // as the lowest world y, so bottom() is the highest.
    QRectF  world() const;
    double  scale() const;
    QPointF toWidget(QPointF point) const;
    QPointF toWorld(QPointF point) const;
    // Free space as the sketch grammar defines it: a quad per arm out to the shared boundary,
    // a box across the orthogonal arms and a diamond across the diagonal ones.
    QPainterPath inkPath() const;

    QPixmap  pixmap_;
    Ink      ink_;
    std::array<double, 5> widths_{0.0, 1.5, 3.0, 6.0, 0.0};
    QPointF  map_origin_;
    double   map_resolution_{0.0};
    QPointF  grid_origin_;
    double   pitch_{0.0};
    int      rows_{0};
    int      cols_{0};
    int      cursor_row_{0};
    int      cursor_col_{0};
    QPointer<QWidget> editor_;
};

} // namespace task_generator_gui

#endif // TASK_GENERATOR_GUI_UTILS_SKETCH_CANVAS_HPP
