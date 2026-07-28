import sys

from PySide6.QtWidgets import QApplication


def main():
    from app.ui.main_window import MainWindow

    qapp = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
