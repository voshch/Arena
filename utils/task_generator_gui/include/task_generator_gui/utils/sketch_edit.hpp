#ifndef TASK_GENERATOR_GUI_UTILS_SKETCH_EDIT_HPP
#define TASK_GENERATOR_GUI_UTILS_SKETCH_EDIT_HPP

#include <world_generator_msgs/msg/alphabet.hpp>

#include <QPlainTextEdit>
#include <QString>
#include <QStringList>

#include <array>
#include <map>
#include <vector>

namespace task_generator_gui
{

// Arm weight per direction, in the order N NE E SE S SW W NW.
using Arms = std::array<uint8_t, 8>;

// A drawing as arm weights per cell, row-major.
struct Ink
{
    int               rows{0};
    int               cols{0};
    std::vector<Arms> cells;
};

// Grid editor for box-drawing sketches. Arrows navigate, shift+arrow draws, punctuation
// toggles a through-connection in place. Glyph lookup comes from the generator, never from here.
class SketchEdit : public QPlainTextEdit
{
    Q_OBJECT

public:
    explicit SketchEdit(QWidget* parent = nullptr);

    void setAlphabet(const world_generator_msgs::msg::Alphabet& alphabet);
    bool hasAlphabet() const { return !by_arms_.empty(); }
    // Put the caret on a grid cell, for a view that lets you point at one.
    void setCell(int row, int col);
    // Seed the document. Never QPlainTextEdit::setPlainText: that scrolls the caret into view,
    // and a widget nobody has shown yet has no layout to scroll.
    void setSketch(const QString& text);
    // What is drawn right now, for a view that paints it without waiting for a render.
    Ink ink() const;

public Q_SLOTS:
    // Empty the grid and its legend. Directives stay: they configure, they do not draw.
    void clearSketch();

Q_SIGNALS:
    void sketchEdited();
    void statusChanged(const QString& text);
    void cursorMoved(int row, int col);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    // Every paste route lands here, so pasted text can never run through the drawing grammar.
    void insertFromMimeData(const QMimeData* source) override;

private:
    struct Cursor { int row; int col; };

    // Every edit ends the same way: shrink the grid, tell the panel if the document changed, report.
    void commit(int revision);
    bool handleArrow(int direction, Qt::KeyboardModifiers mods);
    bool handleCharacter(const QString& text);

    Cursor cell() const;
    // Moves our own caret only. Asking Qt to move the widget's caret scrolls it into view, which
    // needs a laid-out document, which a source view nobody has shown does not have.
    void moveTo(Cursor at);
    bool accepts(QChar character) const;
    QString glyphAt(Cursor at) const;
    Arms armsAt(Cursor at) const;
    // Canonical glyph for an arm vector, or nullptr when the alphabet has none.
    const QString* glyphFor(const Arms& arms) const;
    // Character to write for these arms: the alphabet's glyph if it has one, else a legend
    // symbol. Unicode runs out long before the grammar does.
    QString characterFor(const Arms& arms);
    // Declares an unspelled arm vector in the `!legend:` block and returns its symbol.
    QString mint(const Arms& arms);
    void readLegend();
    void writeLegend();
    void writeText(Cursor at, const QString& text);
    void writeArms(Cursor at, const Arms& arms);
    // Grow the grid so none of these directions points outside it. True when the caret shifted.
    bool grow(Cursor& at, const std::vector<int>& directions);
    bool link(Cursor at, int direction, uint8_t weight);
    // Writes the whole stroke, then steps one cell along `advance` (-1 stays put).
    void toggleThrough(const std::vector<int>& directions, int advance);
    void clearCell(Cursor at);
    // Replace the whole sketch with pasted text, canonicalising shorthand. Refuses on an
    // unresolvable character rather than dropping it.
    void loadSketch(const QString& text);
    // Shrink the grid back to the drawing plus the caret. True when anything went.
    bool trim();
    // Drop minted entries no longer used by the grid. Only lines the editor itself wrote match.
    bool sweepLegend(const QStringList& rows);
    void growTop();
    void growLeft();
    void reportStatus(const QString& detail = QString());

    std::map<QString, Arms> by_glyph_;
    std::map<Arms, QString> by_arms_;
    std::map<QString, Arms> by_alias_;
    QString void_chars_;
    uint8_t pen_weight_{1};
    // The caret is ours, not the widget's: a hidden QPlainTextEdit cannot service setTextCursor.
    Cursor  caret_{0, 0};

    // The leading `!` block, split so an entry can be added without disturbing the rest.
    std::map<QChar, Arms> legend_;
    QStringList legend_lines_;
    QStringList directives_;
};

} // namespace task_generator_gui

#endif // TASK_GENERATOR_GUI_UTILS_SKETCH_EDIT_HPP
