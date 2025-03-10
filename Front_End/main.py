from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QStackedWidget, QHBoxLayout, QSpacerItem, QSizePolicy, QLabel
from PyQt6.QtCore import Qt
import sys

# Import separate page modules
from pages.overview import OverviewPage
from pages.predict_sales import PredictSalesPage
from pages.product_forecast import ProductForecastPage
from pages.add_sales import AddSalesPage
from pages.view_history import ViewHistoryPage
from pages.settings import SettingsPage
from pages.logout import LogoutPage
from pages.help import HelpPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predicting Application")
        self.showFullScreen()
        self.setStyleSheet("background-color: #18162e; color: #bdbdbd;")
        

        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Navigation Bar
        nav_bar = QHBoxLayout()
        menu_icon = QLabel("📌")  # Placeholder for left corner icon
        title = QLabel("SUPPLY CHAIN FORECASTING FOR INVENTORY MANAGEMENT")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff;")
        quit_button = QPushButton("❌")
        quit_button.setFixedSize(40, 40)
        quit_button.clicked.connect(self.close)
        
        nav_bar.addWidget(menu_icon)
        nav_bar.addStretch()
        nav_bar.addWidget(title)
        nav_bar.addStretch()
        nav_bar.addWidget(quit_button)

        # Main content layout
        content_layout = QHBoxLayout()

        # Sidebar
        
        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sidebar.addItem(QSpacerItem(50, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        self.buttons = []
        self.stack = QStackedWidget()
        
        top_menu_items = [
            (" Overview ", OverviewPage, "📊"),
            (" Predict Sales ", PredictSalesPage, "📈"),
            (" Product Demand Forecast ", ProductForecastPage, "🔍"),
            (" Add Today's Sales ", AddSalesPage, "➕"),
            (" View Data History ", ViewHistoryPage, "📜")
        ]
        
        bottom_menu_items = [
            (" Settings ", SettingsPage, "⚙️"),
            (" Log Out ", LogoutPage, "🚪"),
            (" Help ", HelpPage, "❓")
        ]
        button_style = """
            QPushButton {
                text-align: left; 
                padding: 5px; 
                font-size: 14px;
                border: 2px solid #18162e;
                background-color: #18162e;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #2c2a4a;
                border-radius: 10px;
                color: white;
            }
        """
        for index, (name, page_class, icon) in enumerate(top_menu_items):
            btn = QPushButton(f"{icon} {name}")
            btn.setFixedHeight(40)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(lambda _, idx=index: self.switch_page(idx))
            self.sidebar.addWidget(btn)
            self.buttons.append(btn)
            
            page = page_class()
            page.setStyleSheet("background-color: #100e24;")
            self.stack.addWidget(page)
        
        # Spacer to push bottom items down
        self.sidebar.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        for index, (name, page_class, icon) in enumerate(bottom_menu_items, start=len(top_menu_items)):
            btn = QPushButton(f"{icon} {name}")
            btn.setFixedHeight(40)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(lambda _, idx=index: self.switch_page(idx))
            self.sidebar.addWidget(btn)
            self.buttons.append(btn)
            
            page = page_class()
            page.setStyleSheet("background-color: #100e24;")
            self.stack.addWidget(page)
        
        self.sidebar.addItem(QSpacerItem(50, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        content_layout.addLayout(self.sidebar)
        content_layout.addWidget(self.stack)
        
        main_layout.addLayout(nav_bar)
        main_layout.addLayout(content_layout)
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget)
        self.switch_page(0)  # Set default page

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())