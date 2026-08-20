#include "task_generator_gui/utils/sketch_edit.hpp"

#include <QClipboard>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMimeData>
#include <QRegularExpression>
#include <QTextBlock>
#include <QTextCursor>
#include <QTextDocument>

#include <algorithm>
#include <climits>

namespace task_generator_gui
{

namespace
{
// Row/col offsets per direction, matching N NE E SE S SW W NW.
constexpr int kDeltaRow[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
constexpr int kDeltaCol[8] = {0, 1, 1, 1, 0, -1, -1, -1};

const char* kWeightNames[5] = {"none", "light", "heavy", "double", "full"};

const Arms kVoid{0, 0, 0, 0, 0, 0, 0, 0};
const Arms kFull{4, 4, 4, 4, 4, 4, 4, 4};

// Grid rows are the lines after the leading '!' directive block.
int firstGridRow(const QTextDocument* doc)
{
    QTextBlock block = doc->begin();
    int index        = 0;
    while (block.isValid() && block.text().startsWith(QLatin1Char('!')))
    {
        block = block.next();
        ++index;
    }
    return index;
}

QTextBlock ensureRow(QTextDocument* doc, int row)
{
    const int want = firstGridRow(doc) + row;
    QTextCursor tc(doc);
    while (doc->blockCount() <= want)
    {
        tc.movePosition(QTextCursor::End);
        tc.insertBlock();
    }
    return doc->findBlockByNumber(want);
}

void ensureCol(const QTextBlock& block, int col)
{
    const int length = block.text().length();
    if (length > col) return;
    QTextCursor tc(block);
    tc.movePosition(QTextCursor::EndOfBlock);
    tc.insertText(QString(col - length + 1, QLatin1Char(' ')));
}

const QStringList kDirections = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};

// `  a: {sockets: [NE, SW], weight: heavy}`, the one form the editor writes.
bool parseLegendEntry(const QString& line, QChar& symbol, Arms& arms)
{
    static const QRegularExpression kEntry(QStringLiteral(R"(^\s+(\S)\s*:\s*\{(.*)\}\s*$)"));
    static const QRegularExpression kSockets(QStringLiteral(R"(sockets\s*:\s*[\[{]([^\]}]*)[\]}])"));
    static const QRegularExpression kWeight(QStringLiteral(R"(weight\s*:\s*(\w+))"));

    const auto entry = kEntry.match(line);
    if (!entry.hasMatch()) return false;

    symbol = entry.captured(1).at(0);
    arms   = kVoid;

    const auto sockets = kSockets.match(entry.captured(2));
    if (!sockets.hasMatch()) return true;

    uint8_t weight     = 1;
    const auto named   = kWeight.match(entry.captured(2));
    for (int index = 1; named.hasMatch() && index < 5; ++index)
        if (named.captured(1) == QLatin1String(kWeightNames[index])) weight = static_cast<uint8_t>(index);

    for (const QString& socket : sockets.captured(1).split(QLatin1Char(','), Qt::SkipEmptyParts))
    {
        // A socket is either a bare direction taking the entry's weight, or `direction: weight`.
        const QStringList pair  = socket.split(QLatin1Char(':'));
        const int         index = kDirections.indexOf(pair.first().trimmed());
        if (index < 0) continue;

        arms[index] = weight;
        for (int named = 1; pair.size() > 1 && named < 5; ++named)
            if (pair.at(1).trimmed() == QLatin1String(kWeightNames[named])) arms[index] = static_cast<uint8_t>(named);
    }
    return true;
}

QString formatLegendEntry(QChar symbol, const Arms& arms)
{
    QStringList sockets;
    uint8_t     weight = 0;
    bool        mixed  = false;
    for (int direction = 0; direction < 8; ++direction)
    {
        if (arms[direction] == 0) continue;
        sockets << kDirections[direction];
        mixed  = mixed || (weight != 0 && weight != arms[direction]);
        weight = arms[direction];
    }
    if (!mixed)
        return QStringLiteral("  %1: {sockets: [%2], weight: %3}")
            .arg(symbol)
            .arg(sockets.join(QStringLiteral(", ")), QLatin1String(kWeightNames[weight == 0 ? 1 : weight]));

    QStringList pairs;
    for (int direction = 0; direction < 8; ++direction)
        if (arms[direction] != 0)
            pairs << QStringLiteral("%1: %2").arg(kDirections[direction], QLatin1String(kWeightNames[arms[direction]]));
    return QStringLiteral("  %1: {sockets: {%2}}").arg(symbol).arg(pairs.join(QStringLiteral(", ")));
}
} // namespace

// ---------------------------------------------------------------------------

SketchEdit::SketchEdit(QWidget* parent)
    : QPlainTextEdit(parent)
{
    setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    setLineWrapMode(QPlainTextEdit::NoWrap);
    setFocusPolicy(Qt::StrongFocus);
    setTabChangesFocus(true);

    connect(this, &QPlainTextEdit::cursorPositionChanged, this, [this]()
    {
        if (!isVisible()) return;
        const QTextCursor tc  = textCursor();
        const int         row = tc.blockNumber() - firstGridRow(document());
        caret_                = {row < 0 ? 0 : row, tc.positionInBlock()};
        Q_EMIT cursorMoved(caret_.row, caret_.col);
    });
}

void SketchEdit::setCell(int row, int col)
{
    moveTo({row, col});
}

void SketchEdit::setAlphabet(const world_generator_msgs::msg::Alphabet& alphabet)
{
    // The message says uint8 and one string. The grid needs a weight this table names and exactly
    // one code unit per cell. Anything else is dropped here rather than indexed later.
    const auto usable = [](const QString& glyph, const Arms& arms)
    {
        if (glyph.size() != 1) return false;
        return std::all_of(arms.begin(), arms.end(), [](uint8_t weight) { return weight < std::size(kWeightNames); });
    };

    by_glyph_.clear();
    by_arms_.clear();

    by_glyph_[QStringLiteral(" ")] = kVoid;
    by_arms_[kVoid]                = QStringLiteral(" ");

    for (const auto& entry : alphabet.entries)
    {
        const QString glyph = QString::fromStdString(entry.glyph);
        if (!usable(glyph, entry.arms)) continue;
        by_glyph_[glyph]     = entry.arms;
        by_arms_[entry.arms] = glyph;
    }

    by_alias_.clear();
    for (const auto& entry : alphabet.aliases)
    {
        const QString glyph = QString::fromStdString(entry.glyph);
        if (usable(glyph, entry.arms)) by_alias_[glyph] = entry.arms;
    }

    void_chars_ = QString::fromStdString(alphabet.void_chars);
}

bool SketchEdit::accepts(QChar character) const
{
    if (character.isSpace()) return true;
    if (void_chars_.contains(character)) return true;
    if (by_glyph_.find(QString(character)) != by_glyph_.end()) return true;
    if (by_alias_.find(QString(character)) != by_alias_.end()) return true;

    for (QTextBlock block = document()->begin();
         block.isValid() && block.text().startsWith(QLatin1Char('!'));
         block = block.next())
    {
        if (block.text().contains(character)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------

void SketchEdit::insertFromMimeData(const QMimeData* source)
{
    if (source->hasText()) loadSketch(source->text());
}

void SketchEdit::loadSketch(const QString& text)
{
    QStringList directives;
    QStringList rows;
    std::map<QChar, Arms> pasted;
    bool inside = false;
    for (const QString& line : text.split(QLatin1Char('\n')))
    {
        if (!line.startsWith(QLatin1Char('!')))
        {
            rows << line;
            continue;
        }
        directives << line;
        const QString body = line.mid(1);
        if (body.trimmed() == QLatin1String("legend:")) { inside = true; continue; }
        if (!inside || body.isEmpty() || !body.at(0).isSpace()) { inside = false; continue; }
        QChar symbol;
        Arms  arms;
        if (parseLegendEntry(body, symbol, arms)) pasted[symbol] = arms;
        else if (const auto declared = body.indexOf(QLatin1Char(':')); declared > 0) pasted[body.trimmed().at(0)] = kVoid;
    }

    QStringList canonical;
    for (int row = 0; row < rows.size(); ++row)
    {
        QString out;
        for (int col = 0; col < rows[row].size(); ++col)
        {
            const QChar   character = rows[row].at(col);
            const QString one(character);
            if (character.isSpace() || void_chars_.contains(character))
            {
                out += character;
            }
            else if (by_glyph_.count(one) != 0 || pasted.count(character) != 0)
            {
                out += character;
            }
            else if (const auto alias = by_alias_.find(one); alias != by_alias_.end())
            {
                const QString* glyph = glyphFor(alias->second);
                out += glyph != nullptr ? *glyph : one;
            }
            else
            {
                reportStatus(QStringLiteral("row %1 col %2: %3 is not a glyph and the pasted legend does not declare it")
                                 .arg(row).arg(col).arg(character));
                return;
            }
        }
        canonical << out;
    }

    QStringList lines = directives;
    lines += canonical;
    // setPlainText resets the widget cursor, which scrolls, which is what killed rviz when hidden.
    QTextCursor whole(document());
    whole.select(QTextCursor::Document);
    whole.insertText(lines.join(QLatin1Char('\n')));
    caret_ = {0, 0};

    readLegend();
    trim();
    Q_EMIT sketchEdited();
    reportStatus();
}

void SketchEdit::setSketch(const QString& text)
{
    QTextCursor whole(document());
    whole.select(QTextCursor::Document);
    whole.insertText(text);
    caret_ = {0, 0};
    readLegend();
}

void SketchEdit::clearSketch()
{
    QTextCursor whole(document());
    whole.select(QTextCursor::Document);
    whole.removeSelectedText();
    legend_.clear();
    legend_lines_.clear();
    writeLegend();
    caret_ = {0, 0};
    trim();
    Q_EMIT sketchEdited();
    reportStatus();
}

Ink SketchEdit::ink() const
{
    Ink ink;
    const int first = firstGridRow(document());
    for (int index = first; index < document()->blockCount(); ++index)
    {
        ++ink.rows;
        ink.cols = std::max(ink.cols, static_cast<int>(document()->findBlockByNumber(index).text().size()));
    }
    ink.cells.assign(static_cast<size_t>(ink.rows) * ink.cols, kVoid);
    for (int row = 0; row < ink.rows; ++row)
        for (int col = 0; col < ink.cols; ++col)
            ink.cells[static_cast<size_t>(row) * ink.cols + col] = armsAt({row, col});
    return ink;
}

void SketchEdit::commit(int revision)
{
    trim();
    if (document()->revision() != revision) Q_EMIT sketchEdited();
    reportStatus();
}

bool SketchEdit::handleArrow(int direction, Qt::KeyboardModifiers mods)
{
    const int revision = document()->revision();

    if (mods == Qt::NoModifier)
    {
        // Moving off the drawing pads the grid to reach the caret, which is an edit like any
        // other. Past the top-left corner that padding goes in front, so the grid grows both ways.
        Cursor      at          = cell();
        QTextCursor edit_cursor = textCursor();
        edit_cursor.beginEditBlock();
        if (at.row + kDeltaRow[direction] < 0) { growTop(); ++at.row; }
        if (at.col + kDeltaCol[direction] < 0) { growLeft(); ++at.col; }
        moveTo({at.row + kDeltaRow[direction], at.col + kDeltaCol[direction]});
        edit_cursor.endEditBlock();
        commit(revision);
        return true;
    }
    if (mods != Qt::ShiftModifier && mods != Qt::AltModifier) return false;

    Cursor      at          = cell();
    QTextCursor edit_cursor = textCursor();
    edit_cursor.beginEditBlock();
    if (at.row + kDeltaRow[direction] < 0) { growTop(); ++at.row; }
    if (at.col + kDeltaCol[direction] < 0) { growLeft(); ++at.col; }

    const bool ok = link(at, direction, mods == Qt::ShiftModifier ? pen_weight_ : 0);
    if (ok) moveTo({at.row + kDeltaRow[direction], at.col + kDeltaCol[direction]});
    edit_cursor.endEditBlock();

    // link() reports why it refused, so leave its message standing.
    if (ok) commit(revision);
    return true;
}

bool SketchEdit::handleCharacter(const QString& text)
{
    const QChar ch = text.at(0);

    // arms of the stroke, then the direction it advances in. A junction stays put
    static const std::map<QChar, std::pair<std::vector<int>, int>> kThrough = {
        {QLatin1Char('-'),  {{2, 6}, 2}},
        {QLatin1Char('|'),  {{0, 4}, 4}},
        {QLatin1Char('+'),  {{0, 2, 4, 6}, -1}},
        {QLatin1Char('/'),  {{1, 5}, 1}},
        {QLatin1Char('\\'), {{3, 7}, 3}},
        {QLatin1Char('X'),  {{1, 3, 5, 7}, -1}},
        {QLatin1Char('x'),  {{1, 3, 5, 7}, -1}},
    };

    if (const auto through = kThrough.find(ch); through != kThrough.end())
    {
        toggleThrough(through->second.first, through->second.second);
        return true;
    }

    if (ch >= QLatin1Char('1') && ch <= QLatin1Char('4'))
    {
        pen_weight_ = static_cast<uint8_t>(ch.digitValue());
        reportStatus();
        return true;
    }

    if (!ch.isPrint()) return false;
    if (!accepts(ch))
    {
        reportStatus(QStringLiteral("%1 is not a sketch glyph, declare it in !legend first").arg(ch));
        return true;
    }

    const int    revision    = document()->revision();
    const Cursor at          = cell();
    const bool   erases      = void_chars_.contains(ch);
    QTextCursor  edit_cursor = textCursor();
    edit_cursor.beginEditBlock();
    if (erases) clearCell(at);
    writeText(at, text);
    // An erase stays put. Anything you draw by hand reads left to right.
    moveTo(erases ? at : Cursor{at.row, at.col + 1});
    edit_cursor.endEditBlock();

    commit(revision);
    return true;
}

void SketchEdit::keyPressEvent(QKeyEvent* event)
{
    // A hidden source field has nothing selected to copy, so bare copy takes the whole sketch.
    if (event->matches(QKeySequence::Copy) && !textCursor().hasSelection())
    {
        QGuiApplication::clipboard()->setText(toPlainText());
        event->accept();
        return;
    }
    if (event->matches(QKeySequence::Paste))
    {
        // Not paste(): that hands off to Qt, which scrolls the caret into view afterwards.
        loadSketch(QGuiApplication::clipboard()->text());
        event->accept();
        return;
    }

    readLegend();

    // Hand-editing the directive block only happens in the visible widget.
    if (isVisible() && textCursor().block().text().startsWith(QLatin1Char('!')))
    {
        QPlainTextEdit::keyPressEvent(event);
        return;
    }

    int direction = -1;
    switch (event->key())
    {
    case Qt::Key_Up:    direction = 0; break;
    case Qt::Key_Right: direction = 2; break;
    case Qt::Key_Down:  direction = 4; break;
    case Qt::Key_Left:  direction = 6; break;
    default: break;
    }

    if (direction >= 0 && handleArrow(direction, event->modifiers() & ~Qt::KeypadModifier))
    {
        event->accept();
        return;
    }

    if (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace)
    {
        const int    revision    = document()->revision();
        const Cursor at          = cell();
        QTextCursor  edit_cursor = textCursor();
        edit_cursor.beginEditBlock();
        clearCell(at);
        edit_cursor.endEditBlock();
        moveTo(at);
        commit(revision);
        event->accept();
        return;
    }

    if (event->text().length() == 1 && handleCharacter(event->text()))
    {
        event->accept();
        return;
    }

    // Home, PageDown, Ctrl+Z and friends land here, and Qt's handling of them scrolls the caret
    // into view. A source view nobody has shown has no layout to scroll, so it dies in Qt instead.
    if (!isVisible())
    {
        event->accept();
        return;
    }

    QPlainTextEdit::keyPressEvent(event);
}

// ---------------------------------------------------------------------------

bool SketchEdit::grow(Cursor& at, const std::vector<int>& directions)
{
    bool top  = false;
    bool left = false;
    for (int direction : directions)
    {
        top  = top || at.row + kDeltaRow[direction] < 0;
        left = left || at.col + kDeltaCol[direction] < 0;
    }
    if (top)
    {
        growTop();
        ++at.row;
    }
    if (left)
    {
        growLeft();
        ++at.col;
    }
    return top || left;
}

SketchEdit::Cursor SketchEdit::cell() const
{
    return caret_;
}

void SketchEdit::moveTo(Cursor at)
{
    if (at.row < 0) at.row = 0;
    if (at.col < 0) at.col = 0;

    ensureCol(ensureRow(document(), at.row), at.col);
    caret_ = at;
    Q_EMIT cursorMoved(caret_.row, caret_.col);
}

void SketchEdit::growTop()
{
    QTextCursor tc(document()->findBlockByNumber(firstGridRow(document())));
    tc.movePosition(QTextCursor::StartOfBlock);
    tc.insertText(QStringLiteral("\n"));
}

void SketchEdit::growLeft()
{
    const int first = firstGridRow(document());
    for (int index = first; index < document()->blockCount(); ++index)
    {
        QTextCursor tc(document()->findBlockByNumber(index));
        tc.movePosition(QTextCursor::StartOfBlock);
        tc.insertText(QStringLiteral(" "));
    }
}

// ---------------------------------------------------------------------------

QString SketchEdit::glyphAt(Cursor at) const
{
    if (at.row < 0 || at.col < 0) return QStringLiteral(" ");

    const QTextBlock block = document()->findBlockByNumber(firstGridRow(document()) + at.row);
    if (!block.isValid()) return QStringLiteral(" ");

    const QString text = block.text();
    if (at.col >= text.length()) return QStringLiteral(" ");
    return text.mid(at.col, 1);
}

Arms SketchEdit::armsAt(Cursor at) const
{
    const QString glyph = glyphAt(at);
    const auto    it    = by_glyph_.find(glyph);
    if (it != by_glyph_.end()) return it->second;

    const auto declared = legend_.find(glyph.at(0));
    if (declared != legend_.end()) return declared->second;
    return kVoid;
}

void SketchEdit::readLegend()
{
    legend_.clear();
    legend_lines_.clear();
    directives_.clear();

    bool inside = false;
    for (QTextBlock block = document()->begin();
         block.isValid() && block.text().startsWith(QLatin1Char('!'));
         block = block.next())
    {
        const QString line = block.text().mid(1);
        if (line.trimmed() == QLatin1String("legend:"))
        {
            inside = true;
            continue;
        }
        // An indented line under `legend:` is one of its entries. Anything else ends the block.
        if (inside && !line.isEmpty() && line.at(0).isSpace())
        {
            legend_lines_ << line;
            QChar symbol;
            Arms  arms;
            if (parseLegendEntry(line, symbol, arms)) legend_[symbol] = arms;
            continue;
        }
        inside = false;
        directives_ << line;
    }
}

void SketchEdit::writeLegend()
{
    QString text;
    for (const QString& line : directives_)
        text += QLatin1Char('!') + line + QLatin1Char('\n');
    if (!legend_lines_.isEmpty())
    {
        text += QStringLiteral("!legend:\n");
        for (const QString& line : legend_lines_)
            text += QLatin1Char('!') + line + QLatin1Char('\n');
    }

    const int   first = firstGridRow(document());
    QTextCursor tc(document()->findBlockByNumber(0));
    tc.movePosition(QTextCursor::StartOfBlock);
    if (first < document()->blockCount())
        tc.setPosition(document()->findBlockByNumber(first).position(), QTextCursor::KeepAnchor);
    else
        tc.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
    tc.insertText(text);
}

QString SketchEdit::mint(const Arms& arms)
{
    static const QString kPool = QStringLiteral("abcdefghijklmnopqrstuvwyzABCDEFGHIJKLMNOPQRSTUVWYZ");

    QChar chosen;
    for (const QChar candidate : kPool)
    {
        if (legend_.count(candidate) != 0 || by_alias_.count(QString(candidate)) != 0) continue;
        if (by_glyph_.count(QString(candidate)) != 0) continue;
        chosen = candidate;
        break;
    }
    if (chosen.isNull())
    {
        reportStatus(QStringLiteral("every legend symbol is taken, clear one to draw that"));
        return QString();
    }

    legend_[chosen] = arms;
    legend_lines_ << formatLegendEntry(chosen, arms);
    writeLegend();
    return QString(chosen);
}

QString SketchEdit::characterFor(const Arms& arms)
{
    if (const QString* exact = glyphFor(arms)) return *exact;

    for (const auto& [symbol, declared] : legend_)
        if (declared == arms) return QString(symbol);

    return mint(arms);
}

const QString* SketchEdit::glyphFor(const Arms& arms) const
{
    const auto it = by_arms_.find(arms);
    return it == by_arms_.end() ? nullptr : &it->second;
}

void SketchEdit::writeText(Cursor at, const QString& text)
{
    QTextBlock block = ensureRow(document(), at.row);
    ensureCol(block, at.col);

    QTextCursor tc(block);
    tc.setPosition(block.position() + at.col);
    tc.setPosition(block.position() + at.col + 1, QTextCursor::KeepAnchor);
    tc.insertText(text);
}

void SketchEdit::writeArms(Cursor at, const Arms& arms)
{
    const QString character = characterFor(arms);
    if (!character.isEmpty())
        writeText(at, character);
}

bool SketchEdit::link(Cursor at, int direction, uint8_t weight)
{
    if (by_arms_.empty())
    {
        Q_EMIT statusChanged(QStringLiteral("waiting for the glyph alphabet from the world generator"));
        return false;
    }

    const Cursor to{at.row + kDeltaRow[direction], at.col + kDeltaCol[direction]};
    if (to.row < 0 || to.col < 0) return false;

    Arms here       = armsAt(at);
    here[direction] = weight;

    Arms there                 = armsAt(to);
    there[(direction + 4) % 8] = weight;

    writeArms(at, here);
    writeArms(to, there);
    return true;
}

void SketchEdit::toggleThrough(const std::vector<int>& directions, int advance)
{
    if (by_arms_.empty())
    {
        reportStatus(QStringLiteral("waiting for the glyph alphabet from the world generator"));
        return;
    }

    Cursor at = cell();
    if (grow(at, directions))
        moveTo(at);

    const Arms own     = armsAt(at);
    bool       all_set = true;
    for (int direction : directions)
        all_set = all_set && own[direction] != 0;
    const uint8_t weight = all_set ? 0 : pen_weight_;

    // The whole stroke lands at once: no single-arm glyph exists for a diagonal.
    Arms here = own;
    for (int direction : directions)
        here[direction] = weight;

    QTextCursor edit_cursor = textCursor();
    edit_cursor.beginEditBlock();
    const QString character = characterFor(here);
    if (character.isEmpty())
    {
        edit_cursor.endEditBlock();
        return;
    }
    writeText(at, character);
    for (int direction : directions)
    {
        const Cursor to{at.row + kDeltaRow[direction], at.col + kDeltaCol[direction]};
        if (to.row < 0 || to.col < 0) continue;

        Arms there                 = armsAt(to);
        there[(direction + 4) % 8] = weight;
        // A neighbour that cannot show the stub keeps its glyph. The run links up once it is drawn too.
        if (glyphFor(there) != nullptr)
            writeArms(to, there);
    }
    edit_cursor.endEditBlock();

    // Advancing along the stroke is what makes ---- a corridor and //// a diagonal.
    if (advance >= 0)
        at = {at.row + kDeltaRow[advance], at.col + kDeltaCol[advance]};
    moveTo(at);
    trim();

    Q_EMIT sketchEdited();
    reportStatus();
}

bool SketchEdit::trim()
{
    const int first = firstGridRow(document());
    if (first >= document()->blockCount()) return false;

    QStringList rows;
    for (int index = first; index < document()->blockCount(); ++index)
        rows << document()->findBlockByNumber(index).text();

    int top    = rows.size();
    int bottom = -1;
    int left   = INT_MAX;
    int right  = -1;
    for (int row = 0; row < rows.size(); ++row)
        for (int col = 0; col < rows[row].size(); ++col)
        {
            // '.' clears a cell and is printed, so blankness is the alphabet's call, not isSpace's.
            if (rows[row].at(col).isSpace() || void_chars_.contains(rows[row].at(col))) continue;
            top    = std::min(top, row);
            bottom = std::max(bottom, row);
            left   = std::min(left, col);
            right  = std::max(right, col);
        }

    const Cursor at = cell();
    if (bottom < 0)
    {
        // An inkless grid has no cell to stand on, so the caret's cell becomes the world's one
        // full cell. Every route to blank ends here: delete, paste, clear.
        if (const QString* full = glyphFor(kFull); full != nullptr)
        {
            writeText(at, *full);
            return trim();
        }
        top  = bottom = at.row;
        left = right = at.col;
    }
    else
    {
        top    = std::min(top, at.row);
        bottom = std::max(bottom, at.row);
        left   = std::min(left, at.col);
        right  = std::max(right, at.col);
    }

    QStringList kept;
    for (int row = top; row <= bottom && row < rows.size(); ++row)
        kept << rows[row].mid(left, right - left + 1);

    bool          changed = false;
    const QString text    = kept.join(QLatin1Char('\n'));
    if (text != rows.join(QLatin1Char('\n')))
    {
        QTextCursor tc(document()->findBlockByNumber(first));
        tc.movePosition(QTextCursor::StartOfBlock);
        tc.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
        tc.insertText(text);
        moveTo({at.row - top, at.col - left});
        changed = true;
    }
    return sweepLegend(kept) || changed;
}

bool SketchEdit::sweepLegend(const QStringList& rows)
{
    if (legend_lines_.isEmpty()) return false;

    const QString used = rows.join(QString());
    QStringList   kept;
    for (const QString& line : legend_lines_)
    {
        QChar symbol;
        Arms  arms;
        const bool minted = parseLegendEntry(line, symbol, arms) && arms != kVoid && line == formatLegendEntry(symbol, arms);
        if (!minted || used.contains(symbol)) kept << line;
    }
    if (kept.size() == legend_lines_.size()) return false;

    legend_lines_ = kept;
    writeLegend();
    return true;
}

void SketchEdit::clearCell(Cursor at)
{
    writeArms(at, kVoid);

    for (int direction = 0; direction < 8; ++direction)
    {
        const Cursor neighbour{at.row + kDeltaRow[direction], at.col + kDeltaCol[direction]};
        if (neighbour.row < 0 || neighbour.col < 0) continue;

        Arms arms = armsAt(neighbour);
        if (arms[(direction + 4) % 8] == 0) continue;

        arms[(direction + 4) % 8] = 0;
        // Erasing must never mint: a neighbour that cannot spell the leftover keeps its glyph,
        // and the arm it still declares reaches an empty cell, so it carries nothing.
        if (glyphFor(arms) != nullptr)
            writeArms(neighbour, arms);
    }
}

// ---------------------------------------------------------------------------

void SketchEdit::reportStatus(const QString& detail)
{
    const Cursor at = cell();
    QString text    = QStringLiteral("row %1, col %2  pen %3").arg(at.row).arg(at.col).arg(kWeightNames[pen_weight_]);
    if (!detail.isEmpty())
        text += QStringLiteral("  ") + detail;
    Q_EMIT statusChanged(text);
}

} // namespace task_generator_gui
