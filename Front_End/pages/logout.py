from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class LogoutPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel("Logout Page")
        layout.addWidget(label)
        self.setLayout(layout)