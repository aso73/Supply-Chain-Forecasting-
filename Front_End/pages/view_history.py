
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class ViewHistoryPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel("View History Page")
        layout.addWidget(label)
        self.setLayout(layout)

""" """