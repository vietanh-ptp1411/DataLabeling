from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout)

from app.i18n import tr
from app.models.label_class import LabelClass
from app.ui import theme


class ManageClassesDialog(QDialog):
    """Add/remove label classes; removal cascades across all annotations."""

    def __init__(self, classes, store, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Quản lý Classes"))
        self.classes = classes
        self.store = store
        layout = QVBoxLayout(self)
        self.listw = QListWidget()
        layout.addWidget(self.listw)
        row = QHBoxLayout()
        add_btn = QPushButton(tr("Thêm..."))
        del_btn = QPushButton(tr("Xóa"))
        close_btn = QPushButton(tr("Đóng"))
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)
        add_btn.clicked.connect(self.add_class)
        del_btn.clicked.connect(self.delete_class)
        close_btn.clicked.connect(self.accept)
        self.refresh()

    def refresh(self):
        self.listw.clear()
        for c in self.classes:
            item = QListWidgetItem(theme.dot_icon(c.color, 14), c.name)
            self.listw.addItem(item)

    def add_class(self):
        name, ok = QInputDialog.getText(self, tr("Thêm class"),
                                        tr("Tên class:"))
        name = name.strip()
        if not ok or not name:
            return
        if any(c.name == name for c in self.classes):
            QMessageBox.warning(self, tr("Trùng tên"),
                                tr("Class '{name}' đã tồn tại.").format(
                                    name=name))
            return
        self.classes.append(LabelClass(name))
        self.refresh()

    def delete_class(self):
        row = self.listw.currentRow()
        if row < 0:
            return
        cls = self.classes[row]
        used = sum(
            sum(1 for b in ann.boxes if b.class_name == cls.name)
            + sum(1 for p in ann.polygons if p.class_name == cls.name)
            for ann in self.store.values())
        msg = tr("Xóa class '{name}'?").format(name=cls.name)
        if used:
            msg += tr("\nClass đang được dùng bởi {n} nhãn — xóa sẽ gỡ khỏi "
                      "TẤT CẢ ảnh.").format(n=used)
        if QMessageBox.question(self, tr("Xác nhận"), msg) != QMessageBox.Yes:
            return
        for ann in self.store.values():
            ann.boxes = [b for b in ann.boxes if b.class_name != cls.name]
            ann.polygons = [p for p in ann.polygons if p.class_name != cls.name]
        del self.classes[row]
        self.refresh()
