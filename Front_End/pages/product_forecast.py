from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QRadioButton, QComboBox, QButtonGroup,
    QFormLayout, QHBoxLayout, QPushButton, QFrame, QSizePolicy, QSpacerItem, QProgressBar, QDialog
)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pages.DEMAND_FORECAST import DemandForecastingModel

class ProductForecastPage(QWidget):
    def __init__(self):
        super().__init__()
        self.model = DemandForecastingModel()

        # Main layout
        main_layout = QHBoxLayout()

        # Left frame for form
        self.form_frame = QFrame()
        self.form_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        form_layout = QFormLayout(self.form_frame)

        # Year input
        self.year_input = QLineEdit()
        self.year_input.setPlaceholderText("Enter Year")
        self.year_input.setFixedWidth(80)
        form_layout.addRow(QLabel("Enter Year:"), self.year_input)

        # Radio buttons
        self.radio_month = QRadioButton("Forecast Demand for a Month")
        self.radio_product = QRadioButton("Forecast Demand for a Product")
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_month)
        self.radio_group.addButton(self.radio_product)
        form_layout.addRow(self.radio_month)
        form_layout.addRow(self.radio_product)

        # Dropdowns
        self.month_label = QLabel("Select Month:")
        self.month_dropdown = QComboBox()
        self.month_dropdown.addItems(["January", "February", "March", "April", "May", "June", 
                                      "July", "August", "September", "October", "November", "December"])
        self.month_dropdown.setFixedWidth(120)
        self.month_label.setVisible(False)
        self.month_dropdown.setVisible(False)

        self.product_label = QLabel("Select Product:")
        self.product_dropdown = QComboBox()
        self.product_dropdown.setFixedWidth(120)
        self.product_mapping = {
            "Fresh Produce": "P0001", "Bakery and Confectionery": "P0002", "Meat and Poultry": "P0003",
            "Seafood": "P0004", "Snacks and Beverages": "P0005", "Cleaning Supplies": "P0006",
            "Laundry and Detergents": "P0007", "Kitchenware and Utensils": "P0008", "Home Decor": "P0009",
            "Furniture and Furnishings": "P0010", "Men’s Clothing": "P0011", "Women’s Clothing": "P0012",
            "Kids’ Clothing": "P0013", "Shoes and Footwear": "P0014", "Bags and Luggage": "P0015",
            "Skincare and Cosmetics": "P0016", "Hair Care Products": "P0017", "Perfumes and Deodorants": "P0018",
            "Baby Care": "P0019", "Pet Supplies": "P0020"
        }
        self.product_dropdown.addItems(self.product_mapping.keys())
        self.product_label.setVisible(False)
        self.product_dropdown.setVisible(False)

        form_layout.addRow(self.month_label, self.month_dropdown)
        form_layout.addRow(self.product_label, self.product_dropdown)

        # Loader (Progress Bar)
        self.loader = QProgressBar()
        self.loader.setRange(0, 0)
        self.loader.setVisible(False)
        form_layout.addRow(self.loader)

        # Forecast Button
        self.forecast_button = QPushButton("Forecast Demand")
        self.forecast_button.setFixedWidth(200)
        self.forecast_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.forecast_button.clicked.connect(self.forecast_demand)
        form_layout.addRow(self.forecast_button)

        # Connect radio buttons to toggle visibility
        self.radio_month.toggled.connect(self.toggle_dropdowns)
        self.radio_product.toggled.connect(self.toggle_dropdowns)

        # Add form frame to the main layout
        main_layout.addWidget(self.form_frame, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Right frame for graph display
        self.graph_frame = QFrame()
        self.graph_layout = QVBoxLayout(self.graph_frame)
        self.canvas = FigureCanvas(plt.figure(figsize=(6,4)))
        self.canvas.mpl_connect("button_press_event", self.show_graph_popup)
        self.graph_layout.addWidget(self.canvas)
        self.graph_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.graph_frame)

        # Set main layout
        self.setLayout(main_layout)

    def toggle_dropdowns(self):
        if self.radio_month.isChecked():
            self.month_label.setVisible(True)
            self.month_dropdown.setVisible(True)
            self.product_label.setVisible(False)
            self.product_dropdown.setVisible(False)
        elif self.radio_product.isChecked():
            self.month_label.setVisible(False)
            self.month_dropdown.setVisible(False)
            self.product_label.setVisible(True)
            self.product_dropdown.setVisible(True)

    def forecast_demand(self):
        year = self.year_input.text()
        if not year.isdigit():
            return  # Handle invalid input case
        year = int(year)

        self.loader.setVisible(True)
        QTimer.singleShot(2000, lambda: self.run_forecast(year))  # Simulate loading time

    def run_forecast(self, year):
        self.loader.setVisible(False)
        data = None
        if self.radio_month.isChecked():
            month_index = self.month_dropdown.currentIndex() + 1  # Convert to numerical month
            data = self.model.predict_demand_month(month_index, year)
        elif self.radio_product.isChecked():
            product_name = self.product_dropdown.currentText()
            product_id = self.product_mapping[product_name]
            data = self.model.predict_demand_product(product_id, year)

        if isinstance(data, list) and len(data) > 0:
            self.plot_graph(data)

    def plot_graph(self, data):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(range(len(data)), data, marker='o', linestyle='-', color='b')
        ax.set_title("Forecasted Demand")
        ax.set_xlabel("Time")
        ax.set_ylabel("Demand")
        ax.grid(True)
        self.canvas.draw()
        self.update()

    def show_graph_popup(self, event):
        popup = QDialog(self)
        popup.setWindowTitle("Detailed Forecast Graph")
        layout = QVBoxLayout()
        canvas = FigureCanvas(self.canvas.figure)
        layout.addWidget(canvas)
        close_button = QPushButton("Close")
        close_button.clicked.connect(popup.close)
        layout.addWidget(close_button)
        popup.setLayout(layout)
        popup.exec()
