"""App-wide dark theme: Fusion style + palette + QSS, dark title bar on Windows."""
import sys

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QWidget

# core colors
BG = "#15181e"          # window background
PANEL = "#1c2027"       # side panel / dialogs
FIELD = "#232833"       # inputs, lists
BORDER = "#2e3542"
TEXT = "#dfe5ee"
TEXT_DIM = "#8b94a5"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6ba0ff"
SELECTION = "#2a4a80"
CANVAS_BG = "#101318"

_QSS = f"""
* {{
    outline: none;
}}
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget {{
    color: {TEXT};
    font-size: 10pt;
}}
QToolTip {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 8px;
}}

/* ---------- toolbar ---------- */
QToolBar {{
    background: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    spacing: 2px;
}}
QToolBar::separator {{
    background: {BORDER};
    width: 1px;
    margin: 6px 8px;
}}
QToolButton {{
    background: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 11px;
}}
QToolButton:hover {{
    background: rgba(255, 255, 255, 0.07);
    border-color: {BORDER};
}}
QToolButton:pressed {{
    background: rgba(255, 255, 255, 0.12);
}}
QToolButton[accent="true"] {{
    color: {ACCENT_HOVER};
    font-weight: 600;
}}
QToolBar QLabel {{
    color: {TEXT_DIM};
    padding: 0 2px 0 8px;
}}

/* ---------- inputs ---------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {SELECTION};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: #3d4657;
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {SELECTION};
    padding: 4px;
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
}}
QPushButton:hover {{
    background: #2b3140;
    border-color: #3d4657;
}}
QPushButton:pressed {{
    background: #242a36;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    background: {PANEL};
}}
QPushButton:default, QPushButton[accent="true"] {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton:default:hover, QPushButton[accent="true"]:hover {{
    background: {ACCENT_HOVER};
}}
QPushButton[accent="true"]:disabled {{
    background: {FIELD};
    border-color: {BORDER};
    color: {TEXT_DIM};
}}
QLabel#sectionLabel {{
    color: {TEXT_DIM};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel[dim="true"] {{
    color: {TEXT_DIM};
}}

/* ---------- lists ---------- */
QListWidget {{
    background: {PANEL};
    border: none;
    padding: 4px;
}}
QListWidget::item {{
    color: {TEXT};
    border-radius: 6px;
    padding: 6px 8px;
    margin: 1px 2px;
}}
QListWidget::item:hover {{
    background: rgba(255, 255, 255, 0.05);
}}
QListWidget::item:selected {{
    background: {SELECTION};
    color: white;
}}

/* ---------- status bar ---------- */
QStatusBar {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QStatusBar QLabel {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 2px 10px;
    margin: 2px 4px 2px 0;
    color: {TEXT};
}}
QStatusBar::item {{
    border: none;
}}

/* ---------- misc ---------- */
QProgressBar {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 14px;
    text-align: center;
    color: {TEXT};
    font-size: 8pt;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}
QPlainTextEdit, QTextEdit {{
    background: {CANVAS_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {SELECTION};
    font-family: Consolas, monospace;
}}
QCheckBox {{
    spacing: 8px;
}}
QSplitter::handle {{
    background: {BORDER};
    width: 1px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle {{
    background: #39414f;
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{
    background: #4a5364;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QMenu {{
    background: {PANEL};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {SELECTION};
}}
QLabel#panelHeader {{
    color: {TEXT_DIM};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 8px 10px 4px 10px;
}}
QLabel#panelCount {{
    color: {TEXT_DIM};
    font-size: 9pt;
    padding: 8px 10px 4px 10px;
}}
QWidget#sidePanel {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
"""


class _DarkTitleBarFilter(QObject):
    """Force dark title bars on Windows for every top-level window."""

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Show and isinstance(obj, QWidget)
                and obj.isWindow()):
            _dark_title_bar(obj)
        return False


def _dark_title_bar(widget):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass


def folder_icon(size=48):
    """Crisp two-tone folder glyph — replaces the blurry 📂 emoji."""
    from PySide6.QtCore import QRectF
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 24.0
    p.setPen(Qt.NoPen)
    # back tab
    p.setBrush(QColor("#d9a53f"))
    p.drawRoundedRect(QRectF(2 * s, 4.5 * s, 9 * s, 7 * s), 1.5 * s, 1.5 * s)
    # body
    p.setBrush(QColor("#f0c05a"))
    p.drawRoundedRect(QRectF(2 * s, 7 * s, 20 * s, 12.5 * s), 2 * s, 2 * s)
    # subtle front lip
    p.setBrush(QColor("#f6cf7d"))
    p.drawRoundedRect(QRectF(2 * s, 7 * s, 20 * s, 3.5 * s), 2 * s, 2 * s)
    p.end()
    return QIcon(pix)


def dot_icon(color, size=12):
    """Small round color swatch used in lists/combos."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pix)


def apply_theme(qapp):
    qapp.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(FIELD))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(FIELD))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(SELECTION))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(PANEL))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_DIM))
    qapp.setPalette(pal)

    font = QFont("Segoe UI", 10)
    qapp.setFont(font)
    qapp.setStyleSheet(_QSS)

    filt = _DarkTitleBarFilter(qapp)
    qapp.installEventFilter(filt)
