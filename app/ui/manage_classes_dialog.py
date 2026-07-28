from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout)

from app.models.label_class import LabelClass
from app.ui import theme


class ManageClassesDialog(QDialog):
    """Add/remove label classes; removal cascades across all annotations."""

    def __init__(self, classes, store, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quản lý Classes")
        self.classes = classes
        self.store = store
        layout = QVBoxLayout(self)
        self.listw = QListWidget()
        layout.addWidget(self.listw)
        row = QHBoxLayout()
        add_btn = QPushButton("Thêm...")
        del_btn = QPushButton("Xóa")
        close_btn = QPushButton("Đóng")
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
        name, ok = QInputDialog.getText(self, "Thêm class", "Tên class:")
        name = name.strip()
        if not ok or not name:
            return
        if any(c.name == name for c in self.classes):
            QMessageBox.warning(self, "Trùng tên", f"Class '{name}' đã tồn tại.")
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
        msg = f"Xóa class '{cls.name}'?"
        if used:
            msg += f"\nClass đang được dùng bởi {used} nhãn — xóa sẽ gỡ khỏi TẤT CẢ ảnh."
        if QMessageBox.question(self, "Xác nhận", msg) != QMessageBox.Yes:
            return
        for ann in self.store.values():
            ann.boxes = [b for b in ann.boxes if b.class_name != cls.name]
            ann.polygons = [p for p in ann.polygons if p.class_name != cls.name]
        del self.classes[row]
        self.refresh()
