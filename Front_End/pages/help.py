from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class HelpPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel("Help Page")
        layout.addWidget(label)
        self.setLayout(layout)