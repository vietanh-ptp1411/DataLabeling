import sys

from PySide6.QtWidgets import QApplication


def main():
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme

    qapp = QApplication(sys.argv)
    apply_theme(qapp)
    win = MainWindow()
    win.show()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
