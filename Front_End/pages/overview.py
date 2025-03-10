from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class OverviewPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel("Overview Page")
        layout.addWidget(label)
        self.setLayout(layout)
