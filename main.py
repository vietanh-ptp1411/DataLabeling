import os
import sys

from PySide6.QtWidgets import QApplication


def resource_path(rel):
    """Path to bundled resource — works from source and from PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def main():
    from PySide6.QtGui import QIcon

    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme

    if sys.platform == "win32":
        # own AppUserModelID so the taskbar shows our icon, not python's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "MVA.DataLabeling")

    qapp = QApplication(sys.argv)
    apply_theme(qapp)
    icon_file = resource_path(os.path.join("assets", "icon.ico"))
    if os.path.exists(icon_file):
        qapp.setWindowIcon(QIcon(icon_file))
    win = MainWindow()
    win.show()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
