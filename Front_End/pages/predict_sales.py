from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QComboBox, QPushButton, QDateEdit, QFormLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QProgressBar, QDialog
from PyQt6.QtCore import QDate, Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pages.UNIT_Sold import UnitSold
import matplotlib.pyplot as plt
import pandas as pd

class PredictWorker(QThread):
    result = pyqtSignal(pd.DataFrame)
    
    def __init__(self, store_id, product_id, start_date, end_date):
        super().__init__()
        self.store_id = store_id
        self.product_id = product_id
        self.start_date = start_date
        self.end_date = end_date
    
    def run(self):
        sales_model = UnitSold(self.store_id, self.product_id)
        sales_model.train_model()
        predicted_df = sales_model.predict_future(self.start_date, self.end_date)
        self.result.emit(predicted_df)

class FullScreenGraph(QDialog):
    def __init__(self, figure):
        super().__init__()
        self.setWindowTitle("Full Screen Graph")
        self.setGeometry(100, 100, 1200, 800)
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(figure)
        layout.addWidget(self.canvas)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        self.setLayout(layout)

class PredictSalesPage(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Create frame for inputs
        input_frame = QFrame()
        input_layout = QFormLayout()

        # Starting Date Selector
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate())
        self.start_date_edit.setFixedWidth(150)
        input_layout.addRow(QLabel("Starting Date:"), self.start_date_edit)

        # Ending Date Selector
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate().addDays(7))
        self.end_date_edit.setFixedWidth(150)
        input_layout.addRow(QLabel("Ending Date:"), self.end_date_edit)

        # Store ID Dropdown
        self.store_dropdown = QComboBox()
        self.store_dropdown.addItems(["S001", "S002", "S003", "S004", "S005"])
        self.store_dropdown.setFixedWidth(150)
        input_layout.addRow(QLabel("Select Store ID:"), self.store_dropdown)

        # Product ID Dropdown
        self.product_dropdown = QComboBox()
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
        self.product_dropdown.setFixedWidth(150)
        input_layout.addRow(QLabel("Select Product ID:"), self.product_dropdown)

        # Predict Button
        self.predict_button = QPushButton("Predict Sales")
        self.predict_button.setFixedSize(180, 35)
        self.predict_button.setStyleSheet("""
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
        self.predict_button.clicked.connect(self.predict_sales)
        input_layout.addRow(self.predict_button)

        # Loading Indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_layout.addRow(self.progress_bar)

        # Set input frame layout
        input_frame.setLayout(input_layout)
        left_layout.addWidget(input_frame)

        # Create Table for Displaying Predictions
        self.table = QTableWidget()
        left_layout.addWidget(self.table)
        main_layout.addLayout(left_layout, 1)

        # Create Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots(figsize=(16, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedHeight(700)
        self.canvas.mousePressEvent = self.open_full_screen
        right_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout, 2)

        self.setLayout(main_layout)

    def predict_sales(self):
        store_id = self.store_dropdown.currentText()
        product_name = self.product_dropdown.currentText()
        product_id = self.product_mapping[product_name]
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")

        self.progress_bar.setRange(0, 0)
        self.worker = PredictWorker(store_id, product_id, start_date, end_date)
        self.worker.result.connect(self.update_ui)
        self.worker.start()

    def update_ui(self, predicted_df):
        self.progress_bar.setRange(0, 1)
        self.ax.clear()
        self.ax.plot(predicted_df['Date'], predicted_df['Predicted_Units_Sold'], linestyle='-', color='b', marker='o')
        self.ax.set_title("Predicted Sales", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Date", fontsize=12)
        self.ax.set_ylabel("Units Sold", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
        self.table.setRowCount(len(predicted_df))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Date", "Predicted Units Sold"])
        for row, (date, value) in enumerate(zip(predicted_df['Date'], predicted_df['Predicted_Units_Sold'])):
            self.table.setItem(row, 0, QTableWidgetItem(str(date)))
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def open_full_screen(self, event):
        dialog = FullScreenGraph(self.figure)
        dialog.exec()
