import sys
import pandas as pd
import os
import random
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QLineEdit, QFrame, QDateEdit, 
    QGridLayout, QScrollArea, QHBoxLayout, QPushButton, QMessageBox
)
from PyQt6.QtCore import QDate


class AddSalesPage(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()

        # Frame 1: Date and Store Selection
        frame1 = QFrame()
        frame1_layout = QVBoxLayout()

        date_layout = QHBoxLayout()
        date_label = QLabel("Date:")
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())

        date_layout.addWidget(date_label)
        date_layout.addWidget(self.date_picker)
        date_layout.addStretch()  

        store_layout = QHBoxLayout()
        store_label = QLabel("Store:")
        self.store_dropdown = QComboBox()
        self.store_dropdown.addItems(["S001", "S002", "S003", "S004", "S005"])

        store_layout.addWidget(store_label)
        store_layout.addWidget(self.store_dropdown)
        store_layout.addStretch()

        frame1_layout.addLayout(date_layout)
        frame1_layout.addLayout(store_layout)
        frame1.setLayout(frame1_layout)

        # Frame 2: Split into Two Columns
        frame2 = QFrame()
        frame2_layout = QGridLayout()

        frame2_left = QFrame()
        left_layout = QVBoxLayout()

        frame2_right = QFrame()
        right_layout = QVBoxLayout()

        categories = [
            "Fresh Produce", "Bakery and Confectionery", "Meat and Poultry", "Seafood", "Snacks and Beverages", 
            "Cleaning Supplies", "Laundry and Detergents", "Kitchenware and Utensils", "Home Decor", "Furniture and Furnishings",
            "Men’s Clothing", "Women’s Clothing", "Kids’ Clothing", "Shoes and Footwear", "Bags and Luggage", 
            "Skincare and Cosmetics", "Hair Care Products", "Perfumes and Deodorants", "Baby Care", "Pet Supplies"
        ]

        self.input_fields = {}
        for index, category in enumerate(categories):
            label = QLabel(category)
            input_field = QLineEdit()
            input_field.setPlaceholderText("Units Sold")
            input_field.setText("0")
            input_field.setFixedWidth(140)

            category_layout = QHBoxLayout()
            category_layout.addWidget(label)
            category_layout.addWidget(input_field)
            category_layout.setSpacing(5)

            self.input_fields[category] = input_field

            if index < 10:
                left_layout.addLayout(category_layout)
            else:
                right_layout.addLayout(category_layout)

        frame2_left.setLayout(left_layout)
        frame2_right.setLayout(right_layout)

        # Add to Grid Layout
        frame2_layout.addWidget(frame2_left, 0, 0)
        frame2_layout.addWidget(QLabel(""), 0, 1)  # Empty Column
        frame2_layout.addWidget(frame2_right, 0, 2)

        frame2.setLayout(frame2_layout)

        # Scrollable Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(frame2)

        # "ADD SALES +" Button (Centered)
        self.add_sales_button = QPushButton("ADD SALES +")
        self.add_sales_button.setStyleSheet("font-size: 16px; background-color: #007BFF; color: white; padding: 8px; border-radius: 5px;")
        self.add_sales_button.clicked.connect(self.add_sales)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.add_sales_button)
        button_layout.addStretch()

        # Add Frames to Main Layout
        main_layout.addWidget(frame1)
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def add_sales(self):
        """ Fetches values, validates input, and saves to CSV. """
        date = self.date_picker.date().toString("dd-MM-yyyy")
        store_id = self.store_dropdown.currentText()
        sales_data = []

        for category, input_field in self.input_fields.items():
            units_sold = input_field.text().strip()
            if units_sold.isdigit() and int(units_sold) > 0:
                product_id = f"P{str(len(sales_data) + 1).zfill(4)}"  # Generate Product ID
                inventory_level = random.randint(10, 100)  # Random Inventory Level
                units_ordered = random.randint(1, 20)  # Random Units Ordered
                price = random.randint(50, 500)  # Random Price
                discount = random.randint(0, 20)  # Random Discount

                sales_data.append([
                    date, store_id, product_id, category, 
                    inventory_level,
                    int(units_sold), 
                    units_ordered, 
                    10,  # Demand Forecast (dummy value)
                    price, 
                    discount
                ])

        if not sales_data:
            self.show_message("INVALID! PLEASE ENTER UNITS SOLD PER CATEGORY", "red")
            return

        # Save to CSV
        file_name = "Back-End\Data\Data_Set.csv"
        columns = ["Date", "Store ID", "ProductId", "Category", "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast", "Price", "Discount"]

        try:
            file_exists = os.path.isfile(file_name)

            df_new = pd.DataFrame(sales_data, columns=columns)
            df_new.to_csv(file_name, mode="a", index=False, header=not file_exists)

            self.show_message("SALES ADDED SUCCESSFULLY!", "green")
            
            # Reset input fields to 0
            for input_field in self.input_fields.values():
                input_field.setText("0")

        except Exception as e:
            self.show_message(f"Error: {e}", "red")

    def show_message(self, message, color):
        """ Displays a popup message in the given color. """
        msg_box = QMessageBox()
        msg_box.setText(message)
        msg_box.setStyleSheet(f"color: {color}; font-size: 14px;")
        msg_box.setWindowTitle("Notification")
        msg_box.exec()
