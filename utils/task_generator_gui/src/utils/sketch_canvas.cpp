#include "task_generator_gui/utils/sketch_canvas.hpp"

#include <QApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QColor>
#include <QPainter>

#include <algorithm>
#include <cmath>

namespace task_generator_gui
{

namespace
{
constexpr double kMargin = 6.0;

// Directions in the alphabet's order, N NE E SE S SW W NW, as (row, col) steps.
constexpr int kStepRow[8]  = {-1, -1, 0, 1, 1, 1, 0, -1};
constexpr int kStepCol[8]  = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int kOpposite[8] = {4, 5, 6, 7, 0, 1, 2, 3};
}

SketchCanvas::SketchCanvas(QWidget* parent)
    : QWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setMinimumHeight(160);
}

void SketchCanvas::setPixmap(const QPixmap& pixmap)
{
    pixmap_ = pixmap;
    // The render has caught up, so the locally painted stand-in has nothing left to say.
    ink_ = {};
    update();
}

void SketchCanvas::setFrame(QPointF map_origin, double map_resolution, QPointF grid_origin, double pitch, int rows, int cols)
{
    map_origin_     = map_origin;
    map_resolution_ = map_resolution;
    grid_origin_    = grid_origin;
    pitch_          = pitch;
    rows_           = rows;
    cols_           = cols;
    update();
}

void SketchCanvas::setCursorCell(int row, int col)
{
    if (row == cursor_row_ && col == cursor_col_) return;
    cursor_row_ = row;
    cursor_col_ = col;
    update();
}

void SketchCanvas::setInk(Ink ink)
{
    ink_ = std::move(ink);
    update();
}

void SketchCanvas::setWeights(double light, double heavy, double twin)
{
    widths_[1] = light;
    widths_[2] = heavy;
    widths_[3] = twin;
    update();
}

// ---------------------------------------------------------------------------

QRectF SketchCanvas::world() const
{
    if (hasGrid())
        return QRectF(grid_origin_.x(), grid_origin_.y(), cols_ * pitch_, rows_ * pitch_);
    if (pixmap_.isNull() || map_resolution_ <= 0.0) return QRectF();
    return QRectF(map_origin_.x(), map_origin_.y(), pixmap_.width() * map_resolution_, pixmap_.height() * map_resolution_);
}

double SketchCanvas::scale() const
{
    const QRectF extent = world();
    if (extent.isEmpty()) return 1.0;
    const double factor = std::min((width() - 2 * kMargin) / extent.width(), (height() - 2 * kMargin) / extent.height());
    return factor > 0.0 ? factor : 1.0;
}

QPointF SketchCanvas::toWidget(QPointF point) const
{
    const QRectF extent = world();
    const double factor = scale();
    const double left   = (width() - extent.width() * factor) / 2.0;
    const double top    = (height() - extent.height() * factor) / 2.0;
    // world y grows up, widget y grows down
    return QPointF(left + (point.x() - extent.left()) * factor, top + (extent.bottom() - point.y()) * factor);
}

QPointF SketchCanvas::toWorld(QPointF point) const
{
    const QRectF extent = world();
    const double factor = scale();
    const double left   = (width() - extent.width() * factor) / 2.0;
    const double top    = (height() - extent.height() * factor) / 2.0;
    return QPointF(extent.left() + (point.x() - left) / factor, extent.bottom() - (point.y() - top) / factor);
}

QPainterPath SketchCanvas::inkPath() const
{
    QPainterPath path;
    if (!hasGrid() || ink_.cells.empty()) return path;

    const auto at = [this](int row, int col) -> const Arms*
    {
        if (row < 0 || col < 0 || row >= ink_.rows || col >= ink_.cols) return nullptr;
        const Arms& arms = ink_.cells[static_cast<size_t>(row) * ink_.cols + col];
        return std::any_of(arms.begin(), arms.end(), [](uint8_t weight) { return weight != 0; }) ? &arms : nullptr;
    };
    const auto metres = [this](uint8_t weight)
    { return weight == 4 ? pitch_ : (weight < widths_.size() ? widths_[weight] : 0.0); };

    // The generator's rule, so what is painted here is what a render would come back with: an arm
    // reaches only a neighbour that is there, both sides of a boundary take the wider of the two,
    // and a glyph with nothing at all to reach draws itself across its whole cell.
    std::vector<std::array<double, 8>> link(ink_.cells.size(), std::array<double, 8>{});
    for (int row = 0; row < ink_.rows; ++row)
    {
        for (int col = 0; col < ink_.cols; ++col)
        {
            const Arms* arms = at(row, col);
            if (arms == nullptr) continue;
            const size_t here = static_cast<size_t>(row) * ink_.cols + col;

            bool alone = true;
            for (int side = 0; side < 8; ++side)
                if ((*arms)[side] != 0 && at(row + kStepRow[side], col + kStepCol[side]) != nullptr) alone = false;

            for (int side = 0; side < 8; ++side)
            {
                if ((*arms)[side] == 0) continue;
                double      width = metres((*arms)[side]);
                const int   away  = kOpposite[side];
                const Arms* other = at(row + kStepRow[side], col + kStepCol[side]);
                if (other == nullptr)
                {
                    if (alone) link[here][side] = width;
                    continue;
                }
                if ((*other)[away] != 0) width = std::max(width, metres((*other)[away]));
                const size_t there = static_cast<size_t>(row + kStepRow[side]) * ink_.cols + col + kStepCol[side];
                link[there][away]  = width;
                link[here][side]   = width;
            }
        }
    }

    for (int row = 0; row < ink_.rows && row < rows_; ++row)
    {
        for (int col = 0; col < ink_.cols && col < cols_; ++col)
        {
            const std::array<double, 8>& axis = link[static_cast<size_t>(row) * ink_.cols + col];
            if (std::none_of(axis.begin(), axis.end(), [](double width) { return width > 0.0; })) continue;

            const QPointF centre(grid_origin_.x() + (col + 0.5) * pitch_,
                                 grid_origin_.y() + (rows_ - 1 - row + 0.5) * pitch_);

            for (int side = 0; side < 8; ++side)
            {
                if (axis[side] <= 0.0) continue;
                const QPointF reach(kStepCol[side] * pitch_ / 2.0, -kStepRow[side] * pitch_ / 2.0);
                const double  length = std::hypot(reach.x(), reach.y());
                const QPointF across(-reach.y() / length * axis[side] / 2.0, reach.x() / length * axis[side] / 2.0);
                const QPointF far(centre.x() + reach.x(), centre.y() + reach.y());
                path.addPolygon(QPolygonF(QVector<QPointF>{
                    toWidget(centre + across), toWidget(far + across), toWidget(far - across), toWidget(centre - across)}));
            }

            const double vertical   = std::max(axis[0], axis[4]);
            const double horizontal = std::max(axis[2], axis[6]);
            const double diagonal   = std::max({axis[1], axis[3], axis[5], axis[7]});
            if (vertical > 0.0 || horizontal > 0.0)
            {
                const double across = vertical > 0.0 ? vertical : horizontal;
                const double along  = horizontal > 0.0 ? horizontal : vertical;
                path.addPolygon(QPolygonF(QVector<QPointF>{
                    toWidget(QPointF(centre.x() - across / 2, centre.y() - along / 2)),
                    toWidget(QPointF(centre.x() + across / 2, centre.y() - along / 2)),
                    toWidget(QPointF(centre.x() + across / 2, centre.y() + along / 2)),
                    toWidget(QPointF(centre.x() - across / 2, centre.y() + along / 2))}));
            }
            if (diagonal > 0.0)
            {
                // the largest diamond that stays inside a band of this width
                const double reach = diagonal / std::sqrt(2.0);
                path.addPolygon(QPolygonF(QVector<QPointF>{
                    toWidget(QPointF(centre.x() + reach, centre.y())),
                    toWidget(QPointF(centre.x(), centre.y() + reach)),
                    toWidget(QPointF(centre.x() - reach, centre.y())),
                    toWidget(QPointF(centre.x(), centre.y() - reach))}));
            }
        }
    }
    return path.simplified();
}

// ---------------------------------------------------------------------------

void SketchCanvas::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    // Unknown is black on the rendered map, so a cell with nothing in it must not read as free.
    painter.fillRect(rect(), Qt::black);

    const QRectF extent = world();
    if (extent.isEmpty()) return;

    if (!pixmap_.isNull() && map_resolution_ > 0.0)
    {
        const QSizeF size(pixmap_.width() * map_resolution_ * scale(), pixmap_.height() * map_resolution_ * scale());
        const QPointF top_left = toWidget(QPointF(map_origin_.x(), map_origin_.y() + pixmap_.height() * map_resolution_));
        painter.setRenderHint(QPainter::SmoothPixmapTransform, size.width() < pixmap_.width());
        painter.drawPixmap(QRectF(top_left, size), pixmap_, pixmap_.rect());
    }

    if (!hasGrid()) return;

    // Drawn over the map, not instead of it: this is only ever ahead of the render, never behind.
    const QPainterPath ink = inkPath();
    if (!ink.isEmpty())
    {
        painter.setPen(Qt::NoPen);
        painter.setBrush(Qt::white);
        painter.drawPath(ink);
    }

    const QPointF near = toWidget(extent.topLeft());
    const QPointF far  = toWidget(extent.bottomRight());

    painter.setPen(QPen(QColor(70, 70, 70), 0.0, Qt::DotLine));
    for (int col = 0; col <= cols_; ++col)
    {
        const double x = toWidget(QPointF(grid_origin_.x() + col * pitch_, extent.top())).x();
        painter.drawLine(QPointF(x, near.y()), QPointF(x, far.y()));
    }
    for (int row = 0; row <= rows_; ++row)
    {
        const double y = toWidget(QPointF(extent.left(), grid_origin_.y() + row * pitch_)).y();
        painter.drawLine(QPointF(near.x(), y), QPointF(far.x(), y));
    }

    if (cursor_row_ >= rows_ || cursor_col_ >= cols_) return;

    const QPointF corner(grid_origin_.x() + cursor_col_ * pitch_, grid_origin_.y() + (rows_ - 1 - cursor_row_) * pitch_);
    painter.setPen(QPen(palette().highlight().color(), 2.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(QRectF(toWidget(QPointF(corner.x(), corner.y() + pitch_)), toWidget(QPointF(corner.x() + pitch_, corner.y()))));
}

void SketchCanvas::keyPressEvent(QKeyEvent* event)
{
    if (editor_.isNull())
    {
        QWidget::keyPressEvent(event);
        return;
    }
    QKeyEvent forwarded(event->type(), event->key(), event->modifiers(), event->text());
    QApplication::sendEvent(editor_, &forwarded);
    event->accept();
}

void SketchCanvas::mousePressEvent(QMouseEvent* event)
{
    setFocus(Qt::MouseFocusReason);
    if (!hasGrid()) return;

    const QPointF point = toWorld(QPointF(event->pos()));
    const int     col   = static_cast<int>(std::floor((point.x() - grid_origin_.x()) / pitch_));
    const int     row   = rows_ - 1 - static_cast<int>(std::floor((point.y() - grid_origin_.y()) / pitch_));
    // A click in the margin must not pad the sketch out to meet it.
    if (row >= 0 && col >= 0 && row < rows_ && col < cols_)
        Q_EMIT cellClicked(row, col);
}

} // namespace task_generator_gui
