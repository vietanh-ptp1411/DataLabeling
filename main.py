import multiprocessing
import os
import sys

# Windowed (no-console) builds have sys.stdout/stderr = None; libraries that
# write progress bars (ultralytics/tqdm) crash on .write(). Give them a sink.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="ignore")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="ignore")

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
    win.showMaximized()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    # required in frozen builds: torch DataLoader workers spawn new processes
    # that would otherwise re-launch the whole GUI
    multiprocessing.freeze_support()
    main()
